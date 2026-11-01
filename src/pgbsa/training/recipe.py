"""Supervised PG-BSA route losses, candidate calibration, and curriculum helpers."""

from __future__ import annotations
import hashlib
import json
from collections import Counter
from contextlib import contextmanager
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from .._core.angular_adapter import AngularContextAdapter, set_context as _legacy_set_context
from .._core.preprocessing import adaptive_diagonal_loading
from .._core.networks import flatten_supports
from .._core.decoder import close_pair_scene_mask
from .._core.gpu_candidates import build_gpu_candidates
from .._core.heads import replace_profiles
from .losses import two_grid_labels, route_loss
from .data import make_generator, generate_batch
from .peak_loss import coarse_peak_consistency

ROUTE_STAGES = ("route_original", "route_adapted", "route_close")


def set_context(root, encoded, sb, si):
    if isinstance(getattr(root, "support_transformer", None), AngularContextAdapter):
        _legacy_set_context(root, encoded, sb, si)


def registered_roots(model):
    root = nn.Module()
    for name, child in model.roots().items():
        root.add_module(name, child)
    return root


def iter_conv_ffn_modules(model):
    return (m for m in registered_roots(model).modules() if hasattr(m, "conv_dw"))


def models(model):
    e = model.engine
    return dict(
        original=e.original,
        localizer=e.dual.localizer,
        retrieval=model.retrieval,
        close_pair=e.dual.close_pair_localizer,
    )


def encode_close(root, x, supports):
    # The close-pair path encodes the complete covariance observation.
    return root(x)


def anchors(root, angles, active, counts, generator, close=False):
    grid = root.theta_grid
    step = grid[1] - grid[0]
    jitter = (
        torch.multinomial(
            angles.new_tensor([0.15, 0.70, 0.15]),
            len(angles),
            replacement=True,
            generator=generator.torch_rng,
        )
        - 1
    )
    index = (
        torch.floor((angles + jitter[:, None] * 0.5 - grid[0]) / step + 0.5)
        .long()
        .clamp(0, len(grid) - 1)
    )
    if close:
        index = index[:, :2].clone()
        truth = angles[:, :2].clone()
        single = counts == 1
        center = torch.floor((angles[single, 0] - grid[0]) / step + 0.5).long()
        index[single, 0] = (center - 1).clamp(0, len(grid) - 1)
        index[single, 1] = (center + 1).clamp(0, len(grid) - 1)
        truth[single, 1] = truth[single, 0]
        order = index.argsort(-1)
        supports = [row.sort().values for row in index]
        truth = truth.gather(1, order).reshape(-1)
    else:
        supports = [index[j, active[j]] for j in range(len(angles))]
        truth = angles[active]
    sb, si = flatten_supports(supports, angles.device)
    return supports, sb, si, truth


def generate_deployed_batch(n, device, generator, mode):
    return generate_batch(n, device, generator, mode="close_covered" if mode == "close" else mode)


def apply_feature_normalization(model, normalization):
    e = model.engine
    preserve = getattr(model, "_preserve_normalization_function", False)
    with torch.no_grad():
        for name, mkey, skey in [("scorer", "mean", "std"), ("offset", "om", "os")]:
            oldm, olds = getattr(e, mkey), getattr(e, skey)
            nm = normalization[name + "_mean"].to(model.device).reshape_as(oldm)
            ns = normalization[name + "_std"].to(model.device).reshape_as(olds)
            if not torch.isfinite(nm).all() or not torch.isfinite(ns).all() or not (ns > 0).all():
                raise ValueError("Invalid fresh normalization")
            if preserve:
                first = getattr(e.bundle, name).net[0]
                weight = first.weight.detach().clone()
                first.bias.add_(F.linear(((nm - oldm) / olds).reshape(-1), weight))
                first.weight.mul_((ns / olds).reshape(1, -1))
            setattr(e, mkey, nm)
            setattr(e, skey, ns)
            setattr(e.bundle, name + "_mean", nm.cpu().numpy().copy())
            setattr(e.bundle, name + "_std", ns.cpu().numpy().copy())


def tensor_sha(value):
    value = value.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(json.dumps(list(value.shape)).encode())
    h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _masked_mean(value, mask):
    weight = mask.to(value.dtype)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def coarse_angle_loss(root, encoded, angles, active, *, peak_consistency_weight=0.25):
    """Combine coarse localization, peak geometry, and peak-consistency losses."""
    logits = encoded["location_logits"].float()
    grid = root.theta_grid.float()
    angles = angles.float()
    active = active.bool()
    target, offset, left, right = two_grid_labels(grid, angles, active)
    probability = logits.sigmoid()
    batch, bins = logits.shape
    spatial = 1 / torch.cos(torch.deg2rad(grid)).clamp_min(0.3)
    spatial /= spatial.mean()
    pixels = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pixels = pixels * (target - probability).abs().square() * spatial[None]
    denominator = torch.where(target.sum(1) > 0, target.sum(1), target.new_full((batch,), bins))
    location = (pixels.sum(1) / denominator.clamp_min(1.0)).mean() + 0.03 * probability.mean()
    core = torch.maximum(probability.gather(1, left), probability.gather(1, right))
    step = grid[1] - grid[0]
    nearest = torch.floor(((angles - grid[0]) / step).clamp(0, bins - 1) + 0.5).long()
    pair_mask = active[:, :, None] & active[:, None, :]
    not_self = ~torch.eye(active.shape[1], device=angles.device, dtype=torch.bool)[None]
    unresolved = pair_mask & not_self & ((nearest[:, :, None] - nearest[:, None, :]).abs() < 2)
    resolvable = active & ~unresolved.any(-1)
    other = angles[:, None, :]
    query = angles[:, :, None]
    lower = torch.where(active[:, None, :] & (other < query), other, -torch.inf).amax(-1)
    upper = torch.where(active[:, None, :] & (other > query), other, torch.inf).amin(-1)
    lb = torch.where(torch.isfinite(lower), (angles + lower) / 2, angles - step)
    ub = torch.where(torch.isfinite(upper), (angles + upper) / 2, angles + step)

    def interpolate(value):
        coordinate = ((value - grid[0]) / step).clamp(0, bins - 1)
        lo = coordinate.floor().long()
        hi = (lo + 1).clamp_max(bins - 1)
        return probability.gather(1, lo) + (coordinate - lo) * (
            probability.gather(1, hi) - probability.gather(1, lo)
        )

    outside = torch.maximum(interpolate(lb), interpolate(ub))
    minimum = torch.as_tensor(
        getattr(root, "count_min_threshold", 0.05), device=logits.device
    ).float()
    relative = torch.as_tensor(
        getattr(root, "count_relative_alpha", 0.10), device=logits.device
    ).float()
    cutoff = torch.maximum(minimum, relative * probability.amax(1).detach())
    prominence = F.relu(0.03 - (core - outside))
    strength = F.relu(0.03 - (core - cutoff[:, None]))
    selector = _masked_mean(
        torch.maximum(strength, torch.where(resolvable, prominence, torch.zeros_like(prominence))),
        active,
    )
    sorted_angles, order = torch.where(active, angles, torch.inf).sort(1)
    sorted_active = active.gather(1, order)
    safe = torch.where(sorted_active, sorted_angles, torch.zeros_like(sorted_angles))
    ng = torch.floor(((safe - grid[0]) / step).clamp(0, bins - 1) + 0.5).long()
    peaks = probability.gather(1, ng)
    gaps = safe[:, 1:] - safe[:, :-1]
    pairs = (
        sorted_active[:, :-1]
        & sorted_active[:, 1:]
        & (gaps > 0)
        & (gaps <= 2.25)
        & ((ng[:, 1:] - ng[:, :-1]) >= 2)
    )
    valley = _masked_mean(
        F.relu(
            interpolate((safe[:, 1:] + safe[:, :-1]) / 2)
            + 0.05
            - torch.minimum(peaks[:, :-1], peaks[:, 1:])
        ),
        pairs,
    )
    window = torch.zeros_like(logits).scatter_add_(1, nearest, active.to(logits.dtype))
    window = F.max_pool1d(window[:, None], 3, stride=1, padding=1)[:, 0] > 0
    false_peak = _masked_mean(probability.square(), ~window)
    peak_loss, peak_parts = coarse_peak_consistency(root, logits, angles, active)
    loss = (
        location
        + 4.0 * selector
        + 0.75 * valley
        + 0.08 * false_peak
        + float(peak_consistency_weight) * peak_loss
    )
    parts = {
        k: v.detach()
        for k, v in dict(
            location=location, selector=selector, pair_valley=valley, false_peak=false_peak
        ).items()
    }
    parts.update(peak_parts)
    return loss, parts


def feature_normalization(model):
    return {
        name: torch.as_tensor(getattr(model.engine.bundle, name), dtype=torch.float32)
        .detach()
        .cpu()
        .clone()
        for name in ("scorer_mean", "scorer_std", "offset_mean", "offset_std")
    }


def set_training(model, training):
    # Set module flags directly so shared encoder modules follow the selected training mode.
    for root in model.roots().values():
        for module in root.modules():
            module.training = bool(training)


@contextmanager
def eval_for_routing(model):
    modules = list(registered_roots(model).modules())
    modes = [m.training for m in modules]
    for module in modules:
        module.training = False
    try:
        yield
    finally:
        for module, mode in zip(modules, modes):
            module.training = mode


def candidate_features(model, ge, supports, routes):
    raw, features, _, _, _ = build_gpu_candidates(model.retrieval.theta_grid, ge, supports, routes)
    adjusted = replace_profiles(
        features,
        features[:, :, 14] * 70.0,
        routes["base_angle"].detach(),
        routes["route_base_angle"].detach(),
        routes["fine_logits"].detach().softmax(-1),
        routes["wide_logits"].detach().softmax(-1),
    )
    return raw.detach(), features.detach(), adjusted.detach()


def candidate_loss(model, ge, supports, routes, angles, active):
    """Independent head supervision on detached discrete candidate samples."""
    raw, features, adjusted = candidate_features(model, ge, supports, routes)
    sb, _ = flatten_supports(supports, model.device)
    difference = (
        (raw[:, :, None] - angles[sb, None, :]).abs().masked_fill(~active[sb, None, :], torch.inf)
    )
    distance, which = difference.min(-1)
    truth = angles[sb].gather(1, which)
    target = torch.exp(-0.5 * (distance / 0.15).square())
    engine = model.engine
    scores = engine.scorer((features - engine.mean) / engine.std)
    expanded_scores = engine.scorer((adjusted - engine.mean) / engine.std)
    scoring = 0.5 * (
        F.binary_cross_entropy_with_logits(scores, target)
        + F.binary_cross_entropy_with_logits(expanded_scores, target)
    )
    offset = engine.bundle.offset((features - engine.om) / engine.os)
    correction = _masked_mean(
        F.smooth_l1_loss(offset / 0.05, (truth - raw) / 0.05, beta=0.5, reduction="none"),
        distance <= 0.35,
    )
    return model._candidate_weight * scoring + model._offset_weight * correction, dict(
        candidate_scoring=scoring.detach(), candidate_offset=correction.detach()
    )


@torch.no_grad()
def calibrate_features(model, generator, batch, batches, global_step):
    """Estimate candidate normalization from sampled training observations."""
    sums = {}
    squares = {}
    counts = Counter()
    before = generator.samples_generated
    with eval_for_routing(model):
        for i in range(batches):
            stage = ROUTE_STAGES[i % 3]
            root, mode, loaded = {
                "route_original": (model.engine.original, "ordinary", False),
                "route_adapted": (model.engine.dual.localizer, "adapted", False),
                "route_close": (model.engine.dual.close_pair_localizer, "close", True),
            }[stage]
            x, angles, active, k, _, _ = generate_batch(batch, model.device, generator, mode=mode)
            loaded_x = adaptive_diagonal_loading(x)[0]
            ge = model.retrieval(loaded_x)
            supports, sb, si, _ = anchors(
                root, angles, active, k, generator, close=stage == "route_close"
            )
            encoded = encode_close(root, loaded_x, supports) if loaded else root(x)
            if stage == "route_adapted":
                set_context(root, encoded, sb, si)
            routes = root.refine_routes(
                encoded, sb, si, detach_base=False, build_reconstruction=False
            )
            _, features, adjusted = candidate_features(model, ge, supports, routes)
            for name, values in [
                (
                    "scorer",
                    torch.cat(
                        (
                            features.reshape(-1, features.shape[-1]),
                            adjusted.reshape(-1, adjusted.shape[-1]),
                        ),
                        0,
                    ),
                ),
                ("offset", features.reshape(-1, features.shape[-1])),
            ]:
                values = values.double()
                sums[name] = sums.get(name, 0.0) + values.sum(0)
                squares[name] = squares.get(name, 0.0) + values.square().sum(0)
                counts[name] += len(values)
    normalization = {}
    for name in ("scorer", "offset"):
        mean = sums[name] / counts[name]
        std = (squares[name] / counts[name] - mean.square()).clamp_min(1.0e-4).sqrt()
        normalization[name + "_mean"] = mean.float().cpu()
        normalization[name + "_std"] = std.float().cpu()
    apply_feature_normalization(model, normalization)
    return dict(
        global_step=global_step,
        batches=batches,
        batch_size=batch,
        generated_scenes=generator.samples_generated - before,
        feature_rows=dict(counts),
        scorer_inputs="equal raw initial and adjusted expanded candidate features",
        offset_inputs="raw candidate features",
        minimum_std=0.01,
        training_stream_only=True,
        data_source="dataset" if hasattr(generator, "dataset_identity") else "online",
        dataset_identity=getattr(generator, "dataset_identity", None),
        sha256={name: tensor_sha(value) for name, value in normalization.items()},
    )


def deployed_route_loss(model, stage, batch, generator, anchor_fraction, expanded):
    """Mix label-derived anchors and inferred routes using the current curriculum fraction."""
    engine = model.engine
    root, mode, loaded = {
        "route_original": (engine.original, "ordinary", False),
        "route_adapted": (engine.dual.localizer, "adapted", False),
        "route_close": (engine.dual.close_pair_localizer, "close", True),
    }[stage]
    n_anchor = min(batch, max(0, int(round(batch * anchor_fraction))))
    n_deploy = batch - n_anchor
    terms = []
    parts = {}
    all_counts = []
    if n_anchor:
        x, angles, active, counts, _, _ = generate_batch(
            n_anchor, model.device, generator, mode=mode
        )
        all_counts.append(counts)
        raw_x = x
        if loaded:
            x = adaptive_diagonal_loading(x)[0]
        supports, sb, si, truth = anchors(
            root, angles, active, counts, generator, close=stage == "route_close"
        )
        encoded = encode_close(root, x, supports) if loaded else root(x)
        if stage == "route_adapted":
            set_context(root, encoded, sb, si)
        routes = root.refine_routes(encoded, sb, si, detach_base=False, build_reconstruction=False)
        anchor_loss, anchor_parts = route_loss(root, routes, truth)
        with eval_for_routing(model), torch.no_grad():
            ge = model.retrieval(adaptive_diagonal_loading(raw_x)[0])
        heads, head_parts = candidate_loss(model, ge, supports, routes, angles, active)
        anchor_loss = anchor_loss + heads
        anchor_parts.update(head_parts)
        terms.append(anchor_fraction * anchor_loss)
        parts.update({"anchor_" + k: v for k, v in anchor_parts.items()})
        parts["anchor_scenes"] = float(n_anchor)
    if n_deploy:
        x, angles, active, counts, _, _ = generate_deployed_batch(
            n_deploy, model.device, generator, mode=mode
        )
        all_counts.append(counts)
        with eval_for_routing(model), torch.no_grad():
            loaded_x, ge, supports, _, _ = model._retrieve(x)
            special = close_pair_scene_mask(engine.dual, supports, 3.5, model.device).cpu().tolist()
            eligible = [
                j
                for j, s in enumerate(supports)
                if (
                    len(s) >= 11
                    if stage == "route_adapted"
                    else (
                        special[j] if stage == "route_close" else 0 < len(s) < 11 and not special[j]
                    )
                )
            ]
        parts["deployed_generated_scenes"] = float(n_deploy)
        parts["deployed_eligible_scenes"] = float(len(eligible))
        parts["deployed_skipped_scenes"] = float(n_deploy - len(eligible))
        parts["deployed_expanded"] = float(expanded and stage == "route_adapted")
        if eligible:
            ix = torch.as_tensor(eligible, device=model.device, dtype=torch.long)
            ss = [supports[j] for j in eligible]
            with torch.no_grad():
                if expanded and stage == "route_adapted":
                    ss = model._expand_supports(model._slice_encoding(ge, eligible), ss)
            inputs = (loaded_x if loaded else x).index_select(0, ix)
            encoded = encode_close(root, inputs, ss) if loaded else root(inputs)
            sb, si = flatten_supports(ss, model.device)
            if stage == "route_adapted":
                set_context(root, encoded, sb, si)
            routes = root.refine_routes(
                encoded, sb, si, detach_base=False, build_reconstruction=False
            )
            heads, head_parts = candidate_loss(
                model,
                model._slice_encoding(ge, eligible),
                ss,
                routes,
                angles.index_select(0, ix),
                active.index_select(0, ix),
            )
            terms.append((1.0 - anchor_fraction) * heads)
            parts.update({"deployed_" + k: v for k, v in head_parts.items()})
            # Match inferred routes to labels after selecting the branch and supports.
            origins = routes["route_base_angle"].detach().float().cpu().numpy()
            truths = angles.index_select(0, ix).detach().float().cpu().numpy()
            masks = active.index_select(0, ix).detach().cpu().numpy()
            halfspan = float(root.fine_offsets.abs().max())
            if getattr(root, "wide_offsets", None) is not None:
                halfspan += float(root.wide_offsets.abs().max())
            picked = []
            targets = []
            cursor = 0
            for j, support in enumerate(ss):
                truth = truths[j, masks[j]]
                origin = origins[cursor : cursor + len(support)]
                if len(truth) and len(origin):
                    cost = np.abs(origin[:, None] - truth[None])
                    penalized = np.where(cost <= halfspan + 1.0e-6, cost, 1.0e6 + cost)
                    rows, cols = linear_sum_assignment(penalized)
                    valid = cost[rows, cols] <= halfspan + 1.0e-6
                    picked.extend((cursor + rows[valid]).tolist())
                    targets.extend(truth[cols[valid]].tolist())
                cursor += len(support)
            parts["close_true_k3_eligible_scenes"] = (
                float((counts.index_select(0, ix) == 3).sum()) if stage == "route_close" else 0.0
            )
            parts["deployed_total_routes"] = float(cursor)
            parts["deployed_matched_routes"] = float(len(picked))
            parts["deployed_unmatched_routes"] = float(cursor - len(picked))
            if picked:
                indices = torch.as_tensor(picked, device=model.device, dtype=torch.long)
                matched = {
                    k: routes[k].index_select(0, indices)
                    for k in (
                        "conditional_angle",
                        "route_base_angle",
                        "base_angle",
                        "fine_logits",
                        "wide_logits",
                    )
                }
                truth = torch.as_tensor(targets, device=model.device, dtype=torch.float32)
                deploy_loss, deploy_parts = route_loss(root, matched, truth)
                terms.append((1.0 - anchor_fraction) * deploy_loss)
                parts.update({"deployed_" + k: v for k, v in deploy_parts.items()})
            else:
                parts["deployed_zero_matched"] = 1.0
        else:
            parts["deployed_zero_matched"] = 1.0
    if not terms:
        raise RuntimeError("No routes available for supervised stage " + stage)
    return sum(terms), parts, torch.cat(all_counts)


def stage_loss(model, stage, batch, generator, anchor_fraction, expanded):
    if stage == "coarse_original":
        x, angles, active, counts, _, _ = generate_batch(
            batch, model.device, generator, mode="retrieval"
        )
        owner = model.engine.original
        encoded = owner(x)
        loss, parts = coarse_angle_loss(
            owner,
            encoded,
            angles,
            active,
            peak_consistency_weight=model._coarse_peak_consistency_weight,
        )
        target, offset, _, _ = two_grid_labels(owner.theta_grid, angles, active)
        pixels = F.smooth_l1_loss(
            encoded["base_offset"].float() / 0.05, offset / 0.05, beta=0.5, reduction="none"
        )
        offset_loss = (pixels * target).sum() / target.sum().clamp_min(1.0)
        loss = loss + model._coarse_offset_weight * offset_loss
        parts["coarse_base_offset"] = offset_loss.detach()
        return loss, parts, counts
    if stage == "coarse_retrieval":
        x, angles, active, counts, _, _ = generate_batch(
            batch, model.device, generator, mode="retrieval"
        )
        encoded = model.retrieval(adaptive_diagonal_loading(x)[0])
        loss, parts = coarse_angle_loss(
            model.retrieval,
            encoded,
            angles,
            active,
            peak_consistency_weight=model._coarse_peak_consistency_weight,
        )
        target = two_grid_labels(model.retrieval.theta_grid, angles, active)[0]
        proposal = F.binary_cross_entropy_with_logits(
            model.engine.generator(encoded["location_logits"], encoded["grid_feature"]), target
        )
        loss = loss + model._proposal_weight * proposal
        parts["proposal"] = proposal.detach()
        return loss, parts, counts
    return deployed_route_loss(model, stage, batch, generator, anchor_fraction, expanded)


def actual_fine_gates(model):
    values = {
        name: p.detach().tanh()
        for name, p in registered_roots(model).named_parameters()
        if name.endswith("correction_gate_logit")
    }
    names = list(values)
    cpu = torch.stack([values[n] for n in names]).cpu().tolist()
    return dict(zip(names, cpu))
