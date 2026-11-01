"""Train PG-BSA with covariance synthesis, angular supervision, and resumable AdamW updates."""

from __future__ import annotations
import os

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key, "1")
import argparse, hashlib, json, math, random, time
from pathlib import Path
from collections import Counter

PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = PACKAGE / "assets/training_initial.pth"


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        help="JSON training options; explicit command-line flags take precedence",
    )
    p.add_argument(
        "--weights", type=Path, default=DEFAULT_WEIGHTS, help="Initialization checkpoint"
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--train-data",
        type=Path,
        help="Dataset root or manifest; omit for online synthesis",
    )
    p.add_argument(
        "--validation-data", type=Path, help="Dataset root or manifest; uses the validation split"
    )
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--gradient-clip", type=float, default=5.0)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--minimum-lr-ratio", type=float, default=0.10)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--steps",
        type=int,
        default=6000,
        help="Total optimizer updates across all supervised stages",
    )
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=202610100101)
    p.add_argument("--mode", choices=("refiner-scratch", "full-scratch"), default="refiner-scratch")
    p.add_argument("--stages", default="route_original,route_adapted,route_close")
    p.add_argument("--lr", type=float, default=3.0e-4)
    p.add_argument("--owner-lr-scale", type=float, default=0.33)
    p.add_argument("--head-lr-scale", type=float, default=0.10)
    p.add_argument("--head-start-fraction", type=float, default=0.35)
    p.add_argument("--anchor-start", type=float, default=0.95)
    p.add_argument("--anchor-end", type=float, default=0.25)
    p.add_argument("--anchor-decay-fraction", type=float, default=0.70)
    p.add_argument("--candidate-loss-weight", type=float, default=0.25)
    p.add_argument("--offset-loss-weight", type=float, default=0.25)
    p.add_argument("--peak-consistency-weight", type=float, default=0.25)
    p.add_argument("--calibration-batches", type=int, default=6)
    p.add_argument("--validation-every", type=int, default=1000)
    p.add_argument("--validation-batches", type=int, default=6)
    p.add_argument("--validation-batch-size", type=int, default=32)
    p.add_argument("--validation-seed", type=int, default=202610109999)
    p.add_argument("--snapshot-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument(
        "--resume", type=Path, help="Paired Adam/RNG state; matching weights restored automatically"
    )
    p.add_argument(
        "--stop-after", type=int, default=0, help="Pause at a global step for continuation testing"
    )
    return p


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8", newline="\n")
    tmp.replace(path)


def source_binding():
    return {p.relative_to(PACKAGE).as_posix(): sha(p) for p in sorted(PACKAGE.rglob("*.py"))}


def parse_args(argv=None):
    p = parser()
    preliminary, _ = p.parse_known_args(argv)
    if preliminary.config is not None:
        config = json.loads(preliminary.config.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("Training configuration must be a JSON object")
        allowed = {
            a.dest: a
            for a in p._actions
            if a.dest
            not in (
                "help",
                "config",
                "output",
                "weights",
                "resume",
                "stop_after",
                "train_data",
                "validation_data",
            )
        }
        unknown = set(config) - set(allowed)
        if unknown:
            raise ValueError("Unknown training options: " + ", ".join(sorted(unknown)))
        for key, value in config.items():
            action = allowed[key]
            if action.type is int and (isinstance(value, bool) or not isinstance(value, int)):
                raise ValueError("Expected integer: " + key)
            if action.type is float and (
                isinstance(value, bool) or not isinstance(value, (int, float))
            ):
                raise ValueError("Expected number: " + key)
            if key in ("device", "stages", "mode") and not isinstance(value, str):
                raise ValueError("Expected string: " + key)
            if action.choices is not None and value not in action.choices:
                raise ValueError("Invalid choice: " + key)
        p.set_defaults(**config)
    return p.parse_args(argv)


def reset_selected(container, selected, torch, nn, SharedDeltaLinear):
    initialized = {}

    def apply(p, policy, fn, override=False):
        if p is None or id(p) not in selected or id(p) in initialized and not override:
            return
        with torch.no_grad():
            fn(p)
        initialized[id(p)] = policy

    for m in container.modules():
        if isinstance(m, SharedDeltaLinear):
            apply(m.delta_left, "zero_lowrank_left", lambda p: p.zero_())
            apply(
                m.delta_right,
                "random_lowrank_right",
                lambda p: nn.init.kaiming_uniform_(p, a=math.sqrt(5)),
            )
            apply(m.bias, "zero_bias", lambda p: p.zero_())
        elif isinstance(m, nn.MultiheadAttention):
            apply(m.in_proj_weight, "xavier_qkv", nn.init.xavier_uniform_)
            apply(m.in_proj_bias, "zero_attention_bias", lambda p: p.zero_())
        elif isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            apply(m.weight, "random_kaiming", lambda p: nn.init.kaiming_uniform_(p, a=math.sqrt(5)))
            apply(m.bias, "zero_bias", lambda p: p.zero_())
        elif isinstance(m, (nn.LayerNorm, nn.modules.batchnorm._BatchNorm)):
            apply(m.weight, "unit_norm", lambda p: p.fill_(1.0))
            apply(m.bias, "zero_norm_bias", lambda p: p.zero_())
            if isinstance(m, nn.modules.batchnorm._BatchNorm) and any(
                id(p) in selected for p in m.parameters(recurse=False)
            ):
                m.reset_running_stats()
    for name, p in container.named_parameters():
        if id(p) not in selected or id(p) in initialized:
            continue
        if name.endswith(".tau"):
            apply(p, "tau_minus2", lambda p: p.fill_(-2.0))
        elif name.endswith(".gate_cov"):
            apply(p, "cov_gate_one", lambda p: p.fill_(1.0))
        elif name.endswith(".gate_lag"):
            apply(p, "lag_gate_half", lambda p: p.fill_(0.5))
        elif "correction_gate_logit" in name:
            apply(p, "correction_gate09", lambda p: p.fill_(math.atanh(0.9)))
        else:
            apply(p, "random_normal_std002", lambda p: nn.init.normal_(p, 0.0, 0.02))
    for m in container.modules():
        if type(m).__name__ == "AngularContextAdapter":
            for field in ("relative_bias", "output_projection"):
                child = getattr(m, field, None)
                if child is not None:
                    apply(child.weight, "zero_adapter_residual_weight", lambda p: p.zero_(), True)
                    apply(
                        getattr(child, "bias", None),
                        "zero_adapter_residual_bias",
                        lambda p: p.zero_(),
                        True,
                    )
        if isinstance(m, SharedDeltaLinear):
            m.retie()
    if set(initialized) != selected:
        raise RuntimeError("Selected tensors escaped reset")
    return {
        name: initialized[id(p)] for name, p in container.named_parameters() if id(p) in selected
    }


def configure_trainable(model, root, mode):
    refiners = (
        model.engine.original,
        model.engine.dual.localizer,
        model.engine.dual.close_pair_localizer,
    )
    frozen = {id(p) for r in refiners for p in r.base.parameters()}
    frozen.update(id(p) for p in model.retrieval.parameters())
    frozen.update(id(p) for p in model.engine.generator.parameters())
    heads = {
        id(p)
        for m in (model.engine.bundle.scorer, model.engine.bundle.offset)
        for p in m.parameters()
    }
    refine = {id(p) for r in refiners for p in r.parameters() if id(p) not in frozen}
    all_ids = {id(p) for p in root.parameters()}
    selected = all_ids if mode == "full-scratch" else refine | heads
    for p in root.parameters():
        p.requires_grad_(id(p) in selected)
    return refine, heads, frozen, all_ids if mode == "full-scratch" else refine


def set_training(model, mode, training):
    for root in model.roots().values():
        for m in root.modules():
            m.training = bool(training)
    if training and mode == "refiner-scratch":
        for r in (
            model.engine.original.base,
            model.engine.dual.localizer.base,
            model.engine.dual.close_pair_localizer.base,
            model.retrieval,
            model.engine.generator,
        ):
            for m in r.modules():
                m.training = False


def frozen_digest(model, torch):
    h = hashlib.sha256()
    for prefix, r in [
        ("owner", model.engine.original.base),
        ("retrieval", model.retrieval),
        ("close", model.engine.dual.close_pair_localizer.base),
        ("proposal", model.engine.generator),
    ]:
        for name, v in r.state_dict().items():
            h.update((prefix + "." + name).encode())
            h.update(v.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def validation_bank(recipe, args, torch):
    if args.validation_data is None:
        g = recipe.make_generator(args.validation_seed, args.device, split="development")
    else:
        from .data import make_dataset_generator

        g = make_dataset_generator(
            args.validation_data, device=args.device, split="development", seed=args.validation_seed
        )
    bank = []
    modes = ("ordinary", "adapted", "close_covered", "retrieval")
    for i in range(args.validation_batches):
        mode = modes[i % 4]
        values = recipe.generate_batch(args.validation_batch_size, args.device, g, mode=mode)
        bank.append((mode, tuple(v.detach().cpu() for v in values)))
    return bank, g.provenance()


def validate(model, bank, metrics, args, torch, np):
    started = time.perf_counter()
    modes = [(m, m.training) for r in model.roots().values() for m in r.modules()]
    set_training(model, args.mode, False)
    allrows = []
    bymode = {}
    try:
        with torch.inference_mode():
            for mode, values in bank:
                x, angles, active, _, _, _ = values
                pred = model.predict(x.to(args.device))
                rows = []
                for a, b, p in zip(angles.numpy(), active.numpy(), pred):
                    truth = a[b]
                    rows.append(metrics.metric_row(truth, p, tolerance=0.5, cutoff=5.0))
                rows = np.stack(rows)
                allrows.append(rows)
                bymode.setdefault(mode, []).append(rows)
    finally:
        for m, flag in modes:
            m.training = flag

    def summary(rows):
        index = {name: i for i, name in enumerate(metrics.FIELDS)}
        return {
            name: float(rows[:, index[name]].mean())
            for name in ("scene_f1", "full_recovery", "ospa2")
        }

    return dict(
        scenes=sum(len(r) for r in allrows),
        noise_scenes=int(sum((r[:, 0] == 0).sum() for r in allrows)),
        matching_tolerance_degrees=0.5,
        ospa_cutoff_degrees=5.0,
        metric_fields=list(metrics.FIELDS),
        metrics=summary(np.concatenate(allrows)),
        by_mode={k: summary(np.concatenate(v)) for k, v in bymode.items()},
        seconds=time.perf_counter() - started,
        policy="Fixed independent development scenes; labels are used only to compute metrics.",
        selection_metrics=["scene_f1", "full_recovery", "ospa2"],
    )


def main(argv=None):
    args = parse_args(argv)
    output = args.output.resolve()
    if output.suffix.lower() not in (".pth", ".pt"):
        raise ValueError("Output checkpoint must use a .pth or .pt extension")
    if args.stop_after < 0 or args.warmup_steps < 1:
        raise ValueError("Invalid stop position or warmup duration")
    if not 0 <= args.minimum_lr_ratio <= 1 or args.weight_decay < 0 or args.gradient_clip <= 0:
        raise ValueError("Invalid optimizer settings")
    numeric = (
        args.lr,
        args.owner_lr_scale,
        args.head_lr_scale,
        args.head_start_fraction,
        args.anchor_start,
        args.anchor_end,
        args.anchor_decay_fraction,
        args.candidate_loss_weight,
        args.offset_loss_weight,
        args.peak_consistency_weight,
        args.weight_decay,
        args.gradient_clip,
        args.minimum_lr_ratio,
    )
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("Training parameters must be finite")
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError("Preserved existing final artifact")
    if args.steps < 1 or args.batch_size < 4 or args.lr <= 0 or args.calibration_batches < 1:
        raise ValueError("Invalid budget/learning rate")
    if not 0 < args.anchor_end <= args.anchor_start <= 1 or not 0 < args.anchor_decay_fraction <= 1:
        raise ValueError("Invalid anchor curriculum")
    if not 0 <= args.head_start_fraction <= 1 or args.head_lr_scale < 0 or args.owner_lr_scale <= 0:
        raise ValueError("Invalid optimizer groups")
    if args.seed == args.validation_seed:
        raise ValueError("Train/development streams must be disjoint")
    if (
        min(args.log_every, args.validation_batches, args.validation_batch_size) < 1
        or min(args.validation_every, args.snapshot_every) < 0
    ):
        raise ValueError("Invalid logging/validation settings")
    from . import recipe, resume as resume_api, metrics
    import numpy as np
    import torch
    from torch import nn
    from .._core import PGBSA
    from .._core.shared_delta import SharedDeltaLinear, export_shared_delta_payload

    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed % 2**32)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "mha"):
        torch.backends.mha.set_fastpath_enabled(False)
    stages = tuple(s.strip() for s in args.stages.split(",") if s.strip())
    if not stages or any(
        s
        not in (
            "route_original",
            "route_adapted",
            "route_close",
            "coarse_original",
            "coarse_retrieval",
        )
        for s in stages
    ):
        raise ValueError("Unknown supervised stage")
    if args.mode == "refiner-scratch" and any(s.startswith("coarse_") for s in stages):
        raise ValueError("Frozen encoder mode cannot train coarse stages")
    if args.mode == "full-scratch":
        stages = tuple(dict.fromkeys(("coarse_original", "coarse_retrieval", *stages)))
    immutable = {
        key: getattr(args, key)
        for key in (
            "steps",
            "batch_size",
            "seed",
            "mode",
            "lr",
            "owner_lr_scale",
            "head_lr_scale",
            "head_start_fraction",
            "anchor_start",
            "anchor_end",
            "anchor_decay_fraction",
            "candidate_loss_weight",
            "offset_loss_weight",
            "peak_consistency_weight",
            "calibration_batches",
            "validation_every",
            "validation_batches",
            "validation_batch_size",
            "validation_seed",
            "snapshot_every",
            "log_every",
            "weight_decay",
            "gradient_clip",
            "warmup_steps",
            "minimum_lr_ratio",
        )
    }
    immutable["stages"] = stages
    for key in ("train_data", "validation_data"):
        path = getattr(args, key)
        if path is not None:
            path = path / "manifest.json" if path.is_dir() else path
        immutable[key + "_sha256"] = sha(path) if path is not None else None
    options_sha = hashlib.sha256(json.dumps(immutable, sort_keys=True).encode()).hexdigest()
    source_before = source_binding()
    resume = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    weights = (
        (args.resume.parent / resume["weights_checkpoint"]).resolve()
        if resume
        else args.weights.resolve()
    )
    input_weights_sha = sha(args.weights)
    if resume is not None and resume["loop_state"].get("input_weights_sha256") != input_weights_sha:
        raise ValueError("Resume initialization checkpoint differs from --weights")
    template = torch.load(weights, map_location="cpu", weights_only=True)
    if template.get("format") != "pgbsa_shared_delta_v3":
        raise ValueError("Expected canonical shared-delta topology checkpoint")
    output.parent.mkdir(parents=True, exist_ok=True)
    runtime = dict(
        compact_frozen_outputs=False,
        fold_frozen_layers=False,
        cached_steering=False,
        efficient_attention=False,
        dense_owner_sdpa=False,
        shared_delta_ffn_chunk=0,
        fixed_query_projection=False,
        graph_compact_refiner=False,
        graph_expanded_candidates=False,
        grid_kv_stride=1,
    )
    model = PGBSA(
        args.device,
        weights,
        graph_encoders=False,
        search_workers=1,
        cpu_threads=1,
        runtime_options=runtime,
    )
    root = recipe.registered_roots(model)
    refine_ids, head_ids, frozen_ids, reset_ids = configure_trainable(model, root, args.mode)
    selected = [(n, p) for n, p in root.named_parameters() if p.requires_grad]
    before_frozen = frozen_digest(model, torch) if args.mode == "refiner-scratch" else None
    initialization = None
    if not resume:
        policies = reset_selected(root, reset_ids, torch, nn, SharedDeltaLinear)
        initialization = dict(
            mode=args.mode,
            reset_parameter_tensors=len(reset_ids),
            reset_parameters=sum(p.numel() for p in root.parameters() if id(p) in reset_ids),
            reset_policy=policies,
            trained_encoders_reused=args.mode == "refiner-scratch",
            all_network_weights_random=args.mode == "full-scratch",
            distillation=False,
        )
        if args.mode == "full-scratch":
            stats = recipe.feature_normalization(model)
            for name in stats:
                stats[name].fill_(1.0 if name.endswith("_std") else 0.0)
            recipe.apply_feature_normalization(model, stats)
    set_training(model, args.mode, True)
    owner_ids = {id(p) for p in model.engine.original.base.parameters()}
    groups = []
    for name, ids, scale in [
        ("refiner", refine_ids, 1.0),
        ("heads", head_ids, args.head_lr_scale),
        ("owner", owner_ids, args.owner_lr_scale),
        ("other", set(), 1.0),
    ]:
        if name == "other":
            previous = {id(p) for group in groups for p in group["params"]}
            values = [p for _, p in selected if id(p) not in previous]
        else:
            values = [p for _, p in selected if id(p) in ids]
        if values:
            groups.append(dict(params=values, lr=args.lr * scale, group_name=name, lr_scale=scale))
    if sum(len(g["params"]) for g in groups) != len(selected) or len(
        {id(p) for g in groups for p in g["params"]}
    ) != len(selected):
        raise RuntimeError("Optimizer ownership overlaps or is incomplete")
    optimizer = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    if args.train_data is None:
        generator = recipe.make_generator(args.seed, model.device, split="train")
    else:
        from .data import make_dataset_generator

        generator = make_dataset_generator(
            args.train_data, device=model.device, split="train", seed=args.seed
        )
    bank, bank_provenance = validation_bank(recipe, args, torch)
    bank_sha = hashlib.sha256(
        b"".join(
            v.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            for _, batch in bank
            for v in batch
        )
    ).hexdigest()
    model._candidate_weight = args.candidate_loss_weight
    model._offset_weight = args.offset_loss_weight
    model._proposal_weight = 1.0
    model._coarse_offset_weight = 0.25
    model._coarse_peak_consistency_weight = args.peak_consistency_weight
    calibration = []
    validations = []
    visits = Counter()
    present = set()
    nonzero = set()
    start = 0
    prior_seconds = 0.0
    last_rows = []
    head_started = args.mode == "full-scratch"
    try:
        if resume:
            loop = resume_api.restore(
                resume,
                root,
                optimizer,
                generator,
                weights_path=weights,
                source_sha256=source_before,
                options_sha256=options_sha,
                device=model.device,
            )
            start = loop["completed_steps"]
            if start >= args.steps:
                raise ValueError("Resume state has already completed the requested schedule")
            visits.update(loop["stage_visits"])
            initialization = loop["initialization"]
            calibration = loop["calibration"]
            validations = loop["validations"]
            prior_seconds = loop["seconds"]
            present.update(loop["gradient_present"])
            nonzero.update(loop["gradient_nonzero"])
            head_started = loop["head_started"]
            last_rows = loop["last_rows"]
            if loop["validation_bank_sha256"] != bank_sha:
                raise RuntimeError("Development bank differs on continuation")
            if args.mode == "refiner-scratch" and loop["frozen_state_sha256"] != before_frozen:
                raise RuntimeError("Frozen encoder state differs on continuation")
        else:
            model._preserve_normalization_function = args.mode == "refiner-scratch"
            calibration.append(
                recipe.calibrate_features(
                    model, generator, args.batch_size, args.calibration_batches, 0
                )
            )
            calibration[-1][
                "head_function_preserved_by_affine_reparameterization"
            ] = model._preserve_normalization_function
        started = time.perf_counter()
        log_path = output.with_suffix(".training.jsonl")
        if log_path.exists() and not resume:
            raise FileExistsError(log_path)

        def record(completed):
            return dict(
                configuration=immutable,
                from_scratch=args.mode == "full-scratch",
                refinement_from_scratch=True,
                mode=args.mode,
                initialization=initialization,
                seed=args.seed,
                steps=args.steps,
                completed_steps=completed,
                batch_size=args.batch_size,
                stage_visits=dict(visits),
                stages=stages,
                common_owner_frozen=args.mode == "refiner-scratch",
                retrieval_proposal_frozen=args.mode == "refiner-scratch",
                no_count_branch=True,
                count_accuracy_selection=False,
                distillation=False,
                candidate_geometry_detached=True,
                candidate_heads_start_fraction=args.head_start_fraction,
                close_native_branch="Two inferred routes; the deployed training stratum includes 25% compact three-source scenes.",
                source_sha256=source_before,
                input_weights_sha256=input_weights_sha,
                compact_refinement=template.get("compact_refinement"),
                refinement_bottleneck=template.get("refinement_bottleneck"),
                calibration=calibration,
                validations=validations,
                gradient_present_names=sorted(present),
                gradient_nonzero_names=sorted(nonzero),
                generator=generator.provenance(),
                validation_generator=bank_provenance,
                validation_bank_sha256=bank_sha,
                training_log=log_path.name,
                last_training_rows=last_rows,
                seconds=prior_seconds + time.perf_counter() - started,
                selection_policy="Final scheduled update; development metrics are recorded without checkpoint ranking.",
            )

        def export(path, info):
            result = export_shared_delta_payload(
                template,
                recipe.models(model),
                dict(
                    generator=model.engine.generator,
                    scorer=model.engine.bundle.scorer,
                    offset=model.engine.bundle.offset,
                ),
            )
            result["stats"] = recipe.feature_normalization(model)
            result.setdefault("metadata", {}).update(
                training_status=args.mode,
                training=info,
                compact_refinement_trained=info["completed_steps"] > 0,
                neural_weights_unchanged=False,
                distillation_used=False,
            )
            path = Path(path)
            if path.exists():
                raise FileExistsError(path)
            temp = path.with_name(path.name + ".tmp")
            torch.save(result, temp)
            temp.replace(path)
            return result

        def save_pair(completed, info):
            checkpoint = output.with_name(output.stem + f"_step{completed:06d}" + output.suffix)
            export(checkpoint, info)
            loop = dict(
                completed_steps=completed,
                stage_visits=dict(visits),
                calibrated=True,
                schedule_total_steps=args.steps,
                training_config=immutable,
                input_weights_sha256=input_weights_sha,
                initialization=initialization,
                calibration=calibration,
                validations=validations,
                seconds=info["seconds"],
                gradient_present=sorted(present),
                gradient_nonzero=sorted(nonzero),
                head_started=head_started,
                last_rows=last_rows,
                validation_bank_sha256=bank_sha,
                frozen_state_sha256=before_frozen,
            )
            payload = resume_api.capture(
                root,
                optimizer,
                generator,
                weights_path=checkpoint,
                source_sha256=source_before,
                options_sha256=options_sha,
                loop_state=loop,
                device=model.device,
            )
            state_path = checkpoint.with_suffix(".resume.pth")
            resume_api.atomic_save(state_path, payload)
            return checkpoint, state_path

        for index in range(start, args.steps):
            completed = index + 1
            progress = index / max(args.steps - 1, 1)
            fraction = min(progress / args.anchor_decay_fraction, 1.0)
            anchor = args.anchor_end + (args.anchor_start - args.anchor_end) * 0.5 * (
                1 + math.cos(math.pi * fraction)
            )
            if not head_started and progress >= args.head_start_fraction:
                model._preserve_normalization_function = True
                calibration.append(
                    recipe.calibrate_features(
                        model, generator, args.batch_size, args.calibration_batches, index
                    )
                )
                calibration[-1]["head_function_preserved_by_affine_reparameterization"] = True
                head_started = True
            schedule = min(completed / min(args.warmup_steps, args.steps), 1.0) * (
                args.minimum_lr_ratio
                + (1.0 - args.minimum_lr_ratio)
                * 0.5
                * (1 + math.cos(math.pi * completed / args.steps))
            )
            for group in optimizer.param_groups:
                scale = group["lr_scale"]
                if group["group_name"] == "heads" and not head_started:
                    scale = 0.0
                group["lr"] = args.lr * scale * schedule
            stage = stages[index % len(stages)]
            optimizer.zero_grad(set_to_none=True)
            loss, parts, _ = recipe.stage_loss(
                model,
                stage,
                args.batch_size,
                generator,
                anchor,
                stage == "route_adapted" and visits[stage] % 2 == 1,
            )
            visits[stage] += 1
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Nonfinite loss " + stage)
            loss.backward()
            pairs = [(name, p.grad) for name, p in selected if p.grad is not None]
            present.update(name for name, _ in pairs)
            if index < 10 or completed % args.log_every == 0:
                maxima = torch.stack([g.detach().abs().amax() for _, g in pairs]).cpu().tolist()
                nonzero.update(name for (name, _), v in zip(pairs, maxima) if v > 0)
            norm = torch.nn.utils.clip_grad_norm_([p for _, p in selected], args.gradient_clip)
            if not bool(torch.isfinite(norm)):
                raise RuntimeError("Nonfinite gradient norm")
            # Withheld heads have no Adam moments/weight decay before their first training phase.
            if not head_started:
                for name, p in selected:
                    if id(p) in head_ids:
                        p.grad = None
            optimizer.step()
            row = dict(
                step=completed,
                stage=stage,
                anchor_fraction=anchor,
                loss=float(loss.detach()),
                gradient_norm_before_clip=float(norm),
                learning_rates={g["group_name"]: g["lr"] for g in optimizer.param_groups},
                parts={k: float(v) for k, v in parts.items()},
            )
            last_rows.append(row)
            last_rows = last_rows[-20:]
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            info = record(completed)
            if completed == 1 or completed % args.log_every == 0 or completed == args.steps:
                status = dict(
                    complete=False,
                    latest=row,
                    generated_scenes=generator.samples_generated,
                    stage_visits=dict(visits),
                    seconds=info["seconds"],
                )
                save_json(output.with_suffix(".progress.json"), status)
                print(json.dumps(status, allow_nan=False), flush=True)
            saved_before_validation = False
            should_stop = (
                args.stop_after and completed >= args.stop_after and completed < args.steps
            )
            if (
                should_stop
                or args.snapshot_every
                and completed % args.snapshot_every == 0
                and completed < args.steps
            ):
                expected = source_before
                if source_binding() != expected:
                    raise RuntimeError("Training sources changed during execution")
                if args.mode == "refiner-scratch" and frozen_digest(model, torch) != before_frozen:
                    raise RuntimeError("Frozen encoders/proposal changed")
                checkpoint, state_path = save_pair(completed, record(completed))
                saved_before_validation = True
                if should_stop:
                    save_json(
                        output.with_suffix(".progress.json"),
                        dict(
                            complete=False,
                            paused_at=completed,
                            checkpoint=checkpoint.name,
                            resume=state_path.name,
                        ),
                    )
                    print(json.dumps(dict(paused_at=completed, resume=state_path.name)), flush=True)
                    return
            if args.validation_every and (
                completed % args.validation_every == 0 or completed == args.steps
            ):
                try:
                    evaluation = validate(model, bank, metrics, args, torch, np)
                except Exception:
                    if not saved_before_validation:
                        save_pair(completed, record(completed))
                    raise
                evaluation["step"] = completed
                validations.append(evaluation)
                save_json(
                    output.with_suffix(".validation.json"),
                    dict(development_only=True, results=validations),
                )
                print(json.dumps(dict(validation=evaluation), allow_nan=False), flush=True)
        expected = source_before
        if source_binding() != expected:
            raise RuntimeError("Training sources changed during execution")
        if args.mode == "refiner-scratch" and frozen_digest(model, torch) != before_frozen:
            raise RuntimeError("Frozen encoders/proposal changed")
        model._preserve_normalization_function = True
        calibration.append(
            recipe.calibrate_features(
                model, generator, args.batch_size, args.calibration_batches, args.steps
            )
        )
        calibration[-1]["head_function_preserved_by_affine_reparameterization"] = True
        info = record(args.steps)
        export(output, info)
        checkpoint, state_path = save_pair(args.steps, info)
        check = PGBSA(
            "cpu",
            output,
            graph_encoders=False,
            search_workers=1,
            cpu_threads=1,
            runtime_options=runtime,
        )
        try:
            actual = recipe.registered_roots(check).state_dict()
            expected = root.state_dict()
            if set(actual) != set(expected) or any(
                not torch.equal(actual[k], expected[k].detach().cpu()) for k in actual
            ):
                raise RuntimeError("Strict canonical roundtrip differs")
            for key, value in recipe.feature_normalization(model).items():
                if not torch.equal(value, recipe.feature_normalization(check)[key]):
                    raise RuntimeError("Strict normalization roundtrip differs")
        finally:
            check.close()
        report = dict(
            complete=True,
            checkpoint=output.name,
            checkpoint_sha256=sha(output),
            final_resume=state_path.name,
            mode=args.mode,
            parameters_uncompacted=model.parameter_count(),
            trainable_parameters=sum(p.numel() for _, p in selected),
            frozen_state_unchanged=args.mode == "refiner-scratch",
            strict_state_roundtrip=True,
            strict_normalization_roundtrip=True,
            training=info,
        )
        save_json(output.with_suffix(".json"), report)
        save_json(
            output.with_suffix(".progress.json"),
            dict(complete=True, checkpoint=output.name, steps=args.steps),
        )
        print(
            json.dumps(dict(complete=True, output=str(output), seconds=info["seconds"])), flush=True
        )
    finally:
        model.close()


if __name__ == "__main__":
    main()
