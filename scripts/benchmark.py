"""Measure end-to-end inference latency after warm-up on a covariance batch."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import scipy
import torch

# Allow direct execution from a source checkout as well as an installed package.
_source = Path(__file__).resolve().parents[1] / "src"
if _source.is_dir():
    sys.path.insert(0, str(_source))

from pgbsa import create_model


def _load_fixture(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            if "in_mat" not in payload:
                raise ValueError("NPZ input must contain an 'in_mat' array.")
            array = np.asarray(payload["in_mat"])
    elif path.suffix.lower() == ".npy":
        array = np.load(path, allow_pickle=False)
    else:
        raise ValueError("Input must be an .npy file or an .npz file containing in_mat.")
    if array.ndim != 4 or tuple(array.shape[1:]) != (2, 16, 16) or len(array) == 0:
        raise ValueError("Input must have nonempty shape [N, 2, 16, 16].")
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError("Input must be a real floating-point array with real/imaginary channels.")
    if not np.isfinite(array).all():
        raise ValueError("Input contains NaN or infinity.")
    return array


def _summary(values_ms: list[float], batch_size: int) -> dict:
    values = np.asarray(values_ms, dtype=np.float64)
    batch = {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
    }
    return {
        "total_batch_ms": batch,
        "per_sample_ms": {key: value / batch_size for key, value in batch.items()},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help=".npy array or .npz containing in_mat, shape [N, 2, 16, 16]")
    parser.add_argument("--device", default="cuda:0", help="cpu or cuda:<index>")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("benchmark.json"))
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.repeats < 1 or args.warmup < 0:
        parser.error("batch-size and repeats must be positive; warmup must be nonnegative.")
    if args.output.resolve() == args.input.resolve():
        parser.error("Output must differ from the input file.")
    if args.output.exists():
        parser.error("Output already exists; choose a new output path.")
    try:
        device = torch.device(args.device)
        if device.type not in ("cpu", "cuda"):
            raise ValueError("Only cpu and cuda devices are supported.")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable in this environment; use --device cpu.")
        fixture = _load_fixture(args.input)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))

    # Repeat in fixture order when N < B; take the first B scenes when N >= B.
    indices = np.arange(args.batch_size, dtype=np.int64) % len(fixture)
    batch = torch.from_numpy(np.ascontiguousarray(fixture[indices], dtype=np.float32))
    if not torch.isfinite(batch).all():
        parser.error("Input cannot be represented as finite float32 values.")

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    with create_model(device=str(device)) as model:
        def timed_predict() -> float:
            synchronize()
            started = time.perf_counter()
            predictions = model.predict(batch)
            # Materialize CPU NumPy outputs inside the measured interval.
            cpu_outputs = [
                prediction.detach().cpu().numpy() if torch.is_tensor(prediction)
                else np.asarray(prediction)
                for prediction in predictions
            ]
            synchronize()
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if len(cpu_outputs) != args.batch_size:
                raise RuntimeError("Prediction batch size differs from the input batch size.")
            return elapsed_ms

        warmup_ms = [timed_predict() for _ in range(args.warmup)]
        samples_ms = [timed_predict() for _ in range(args.repeats)]
        report = {
            "protocol": "user_fixture_end_to_end_warm_inference",
            "timing_scope": (
                "predict on a CPU float32 tensor, including input transfer, "
                "CPU NumPy output materialization, and final CUDA synchronization; "
                "excludes file loading, tensor preparation, and model loading"
            ),
            "input": str(args.input),
            "fixture_scenes": len(fixture),
            "batch_size": args.batch_size,
            "fixture_indices": indices.tolist(),
            "input_dtype": "float32",
            "device": str(device),
            "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "deployed_parameters": int(model.parameter_count()),
            "versions": {"torch": str(torch.__version__), "scipy": scipy.__version__,
                         "numpy": np.__version__},
            "warmup_calls": args.warmup,
            "measured_calls": args.repeats,
            "warmup_total_batch_ms": warmup_ms,
            "measured_total_batch_ms": samples_ms,
            **_summary(samples_ms, args.batch_size),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
