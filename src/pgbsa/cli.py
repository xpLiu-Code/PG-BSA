"""Infer angles from normalized covariance files using PG-BSA."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .api import _sha256, create_model, verify_release


def _load_input(path):
    import numpy as np
    import torch

    suffix = path.suffix.lower()
    if suffix == ".npy":
        value = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            if "in_mat" not in data.files:
                raise ValueError("NPZ input must contain the array 'in_mat'.")
            value = data["in_mat"]
    elif suffix in (".pt", ".pth"):
        value = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(value, dict):
            if "in_mat" not in value:
                raise ValueError("PyTorch input dictionary must contain 'in_mat'.")
            value = value["in_mat"]
    else:
        raise ValueError("Input must be .npy, .npz, .pt or .pth.")
    if isinstance(value, np.ndarray):
        if value.dtype != np.float32:
            raise ValueError("Input must use float32; automatic dtype conversion is disabled.")
        # Copy handles read-only arrays and negative strides consistently.
        value = torch.from_numpy(np.array(value, copy=True, order="C"))
    if (not torch.is_tensor(value) or value.dtype != torch.float32
            or value.ndim != 4 or tuple(value.shape[1:]) != (2, 16, 16)
            or value.shape[0] < 1):
        raise ValueError("Input must be a nonempty float32 tensor of shape [B, 2, 16, 16].")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("Input must contain only finite values; NaN and infinity are rejected.")
    return value.contiguous()


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Check bundled files without importing numerical dependencies")
    parser.add_argument("--input", type=Path, help="Float32 normalized covariance .npy/.npz(in_mat)/.pt/.pth")
    parser.add_argument("--output", type=Path, help="New JSON result path; an existing file is never overwritten")
    parser.add_argument("--device", default="cpu", help="cpu, cuda:0, cuda:1, ... (default: cpu)")
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint exported by pgbsa-train; defaults to bundled weights")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--search-workers", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--no-graphs", action="store_true", help="Disable all encoder, refiner and candidate CUDA graphs")
    return parser


def main(argv=None):
    """Run the command and return its exit code; failures have concise messages."""
    parser = _parser()
    args = parser.parse_args(argv)
    for name in ("batch_size", "search_workers", "cpu_threads"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.verify_only:
        if args.input is not None or args.output is not None or args.checkpoint is not None:
            parser.error("--verify-only cannot be combined with --input, --output or --checkpoint")
        try:
            manifest = verify_release()
        except Exception as exc:
            print(f"PG-BSA: {exc}", file=sys.stderr)
            return 2
        print(json.dumps({"verified": True, "release_version": manifest["release_version"],
                          "deployed_parameters": manifest["deployed_parameters"],
                          "python_source_files": len(manifest["source_sha256"])}, indent=2))
        return 0
    if args.input is None or args.output is None:
        parser.error("Inference requires both --input and --output")
    if args.output.exists():
        parser.error("Output already exists; choose a new output path")
    model = None
    try:
        if not args.input.is_file():
            raise ValueError(f"Input file does not exist: {args.input}")
        input_digest = _sha256(args.input)
        x = _load_input(args.input)
        if _sha256(args.input) != input_digest:
            raise RuntimeError("Input file changed while it was being loaded.")
        predictions = []
        model = create_model(args.device, checkpoint=args.checkpoint, graph_encoders=not args.no_graphs,
                             search_workers=args.search_workers, cpu_threads=args.cpu_threads)
        for start in range(0, len(x), args.batch_size):
            predictions.extend(model.predict(x[start:start + args.batch_size]))
        result = {
            "model": model.metadata,
            "diagnostics": model.diagnostics(),
            "input_sha256": input_digest,
            "input_shape": list(x.shape),
            "predictions": [dict(sample_id=index, angles_deg=angles.tolist())
                            for index, angles in enumerate(predictions)],
        }
        # Serialize before creating the output, then use exclusive creation.
        text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(text + "\n")
        print(f"Wrote {len(predictions)} predictions to {args.output}")
        return 0
    except Exception as exc:
        print(f"PG-BSA: {exc}", file=sys.stderr)
        return 2
    finally:
        if model is not None:
            model.close()


if __name__ == "__main__":
    raise SystemExit(main())
