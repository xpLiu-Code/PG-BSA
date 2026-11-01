"""Paired model, optimizer, and random-stream state for step-boundary continuation."""

from __future__ import annotations
import copy, hashlib, json, os, random
from pathlib import Path
import numpy as np
import torch

FORMAT = "pgbsa_scratch_resume_state_v1"
REQUIRED_LOOP = (
    "completed_steps",
    "stage_visits",
    "calibrated",
    "schedule_total_steps",
    "training_config",
)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(8 << 20), b""):
            h.update(part)
    return h.hexdigest()


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def parameter_identity(root, optimizer):
    named = list(root.named_parameters())
    names = {id(p): n for n, p in named}
    aliases = {}
    for n, p in root.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(p), []).append(n)
    descriptors = [
        dict(
            name=n,
            aliases=aliases[id(p)],
            shape=list(p.shape),
            dtype=str(p.dtype),
            requires_grad=p.requires_grad,
        )
        for n, p in named
    ]
    groups = []
    for group in optimizer.param_groups:
        group_names = []
        for p in group["params"]:
            if id(p) not in names:
                raise ValueError("Optimizer parameter is outside the registered roots")
            group_names.append(names[id(p)])
        groups.append(group_names)
    layout = dict(parameters=descriptors, optimizer_groups=groups)
    layout["sha256"] = hashlib.sha256(json.dumps(layout, sort_keys=True).encode()).hexdigest()
    return layout


def generator_state(generator):
    return dict(
        dataset_identity=getattr(generator, "dataset_identity", None),
        seed=int(generator.seed),
        split=generator.split,
        device_type=generator.device.type,
        low_snapshot_fraction=float(generator.low_snapshot_fraction),
        low_snr_fraction=float(generator.low_snr_fraction),
        numpy_bit_generator=type(generator.rng.bit_generator).__name__,
        numpy_state=copy.deepcopy(generator.rng.bit_generator.state),
        torch_state=generator.torch_rng.get_state().cpu().clone(),
        samples_generated=int(generator.samples_generated),
        last_metadata=copy.deepcopy(generator.last_metadata),
    )


def global_rng_state(device):
    device = torch.device(device)
    numpy_state = np.random.get_state()
    out = dict(
        python=random.getstate(),
        torch_cpu=torch.get_rng_state().clone(),
        numpy_legacy=dict(
            kind=numpy_state[0],
            keys=torch.from_numpy(numpy_state[1].copy()),
            position=int(numpy_state[2]),
            has_gauss=int(numpy_state[3]),
            cached_gaussian=float(numpy_state[4]),
        ),
        device_type=device.type,
    )
    if device.type == "cuda":
        # Only the trainer's device; never initialize an unrelated CUDA device.
        out["torch_cuda"] = torch.cuda.get_rng_state(device).cpu().clone()
    return out


def runtime_identity(device):
    device = torch.device(device)
    out = dict(
        torch=str(torch.__version__),
        numpy=str(np.__version__),
        device_type=device.type,
        cuda_version=torch.version.cuda,
        cudnn_version=torch.backends.cudnn.version() if device.type == "cuda" else None,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,
    )
    if device.type == "cuda":
        out["device_name"] = torch.cuda.get_device_name(device)
    return out


def capture(
    root, optimizer, generator, *, weights_path, source_sha256, options_sha256, loop_state, device
):
    missing = [k for k in REQUIRED_LOOP if k not in loop_state]
    if missing:
        raise ValueError("Missing continuation positions: " + ",".join(missing))
    if int(loop_state["completed_steps"]) < 0:
        raise ValueError("Invalid completed step")
    weights_path = Path(weights_path)
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)
    return dict(
        format=FORMAT,
        weights_checkpoint=weights_path.name,
        weights_sha256=sha256(weights_path),
        source_sha256=copy.deepcopy(source_sha256),
        options_sha256=options_sha256,
        parameter_identity=parameter_identity(root, optimizer),
        optimizer=cpu_tree(optimizer.state_dict()),
        generator=generator_state(generator),
        global_rng=global_rng_state(device),
        loop_state=cpu_tree(loop_state),
        runtime=runtime_identity(device),
        boundary="After optimizer.step, before any next-step calibration/sampling/dropout",
    )


def atomic_save(path, payload):
    path = Path(path)
    if path.exists():
        raise FileExistsError("Refusing to replace existing resume state: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + str(os.getpid()))
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with temporary.open("xb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        # publish after the complete file is durable; caller uses one writer per run.
        if path.exists():
            raise FileExistsError(path)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return sha256(path)


def restore(
    payload,
    root,
    optimizer,
    generator,
    *,
    weights_path,
    source_sha256,
    options_sha256,
    device,
    strict_runtime=True,
):
    if payload.get("format") != FORMAT:
        raise ValueError("Unrecognized resume state")
    if payload["weights_sha256"] != sha256(weights_path):
        raise ValueError("Weights/state pair mismatch")
    if payload["source_sha256"] != source_sha256 or payload["options_sha256"] != options_sha256:
        raise ValueError("Source/options mismatch")
    if payload["parameter_identity"] != parameter_identity(root, optimizer):
        raise ValueError("Parameter/alias/optimizer order mismatch")
    if strict_runtime and payload["runtime"] != runtime_identity(device):
        raise ValueError("Runtime differs; exact continuation not established")
    state = payload["generator"]
    if state.get("dataset_identity") != getattr(generator, "dataset_identity", None):
        raise ValueError("Dataset identity differs from the saved training stream")
    for key in ("seed", "split", "low_snapshot_fraction", "low_snr_fraction"):
        if getattr(generator, key) != state[key]:
            raise ValueError("Generator config mismatch: " + key)
    if generator.device.type != state["device_type"]:
        raise ValueError("Generator CPU/CUDA stream mismatch")
    if type(generator.rng.bit_generator).__name__ != state["numpy_bit_generator"]:
        raise ValueError("NumPy bit-generator mismatch")
    optimizer.load_state_dict(payload["optimizer"])
    generator.rng.bit_generator.state = copy.deepcopy(state["numpy_state"])
    generator.torch_rng.set_state(state["torch_state"].cpu())
    generator.samples_generated = state["samples_generated"]
    generator.last_metadata = copy.deepcopy(state["last_metadata"])
    rng = payload["global_rng"]
    legacy = rng["numpy_legacy"]
    random.setstate(rng["python"])
    np.random.set_state(
        (
            legacy["kind"],
            legacy["keys"].cpu().numpy().astype(np.uint32),
            legacy["position"],
            legacy["has_gauss"],
            legacy["cached_gaussian"],
        )
    )
    torch.set_rng_state(rng["torch_cpu"].cpu())
    if torch.device(device).type == "cuda":
        torch.cuda.set_rng_state(rng["torch_cuda"].cpu(), device)
    return copy.deepcopy(payload["loop_state"])
