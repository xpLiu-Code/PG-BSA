# Usage

## Environment

Create a virtual environment with Python 3.10-3.12:

```bash
python -m venv .venv
```

Activate it with `source .venv/bin/activate` on Linux or macOS, or `.venv\Scripts\Activate.ps1` in Windows PowerShell.

Choose one PyTorch build. For CPU inference:

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

For the CUDA 12.4 build on a compatible NVIDIA system:

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

Then install this repository from its root:

```bash
python -m pip install -e .
pgbsa-infer --verify-only
```

The package pins NumPy 1.26.4, SciPy 1.13.1, Numba 0.59.1, and PyTorch 2.6.0. The direct L-BFGS-B interface depends on the SciPy solver ABI; retaining the pinned environment avoids changing that execution path.

## Input preparation

For a complex snapshot matrix `Y` with shape `[16, L]`:

```python
import numpy as np
import torch

Y = np.asarray(Y, dtype=np.complex128)
if Y.ndim != 2 or Y.shape[0] != 16 or Y.shape[1] == 0:
    raise ValueError("Y must have shape [16, L] with L > 0.")
if not np.isfinite(Y).all():
    raise ValueError("Y contains non-finite values.")

R = Y @ Y.conj().T / Y.shape[1]
trace = np.trace(R).real
if trace <= 0:
    raise ValueError("The covariance trace must be positive.")
R /= trace

x = np.stack((R.real, R.imag), axis=0).astype(np.float32)
covariance = torch.from_numpy(x[None])
```

For several scenes, stack their two-channel arrays along the leading batch axis. Preserve sensor order and the steering convention in [Model and inference](model.md#observation-model).

| Property | Required value |
| --- | --- |
| Shape | `[B, 2, 16, 16]`, with `B >= 1` for the CLI |
| Data type | `float32` |
| Channel 0 | Real part of the covariance |
| Channel 1 | Imaginary part of the covariance |
| Covariance | Finite, Hermitian, positive semidefinite, unit trace |

The CLI checks shape, data type, and finite values. The caller is responsible for the covariance normalization and physical validity.

## Python API

```python
from pgbsa import create_model, verify_release

manifest = verify_release()

with create_model(
    device="cpu",
    cpu_threads=1,
    search_workers=4,
    graph_encoders=True,
) as model:
    predictions = model.predict(covariance)
```

`verify_release()` checks the bundled sources, pretrained checkpoint, training initialization checkpoint, and inference options against the file manifest. It requires only the Python standard library. `create_model()` loads the numerical dependencies and the pretrained model.

To evaluate a checkpoint produced by training, pass `checkpoint="runs/refiner/model.pth"` to `create_model()`, or add `--checkpoint runs/refiner/model.pth` to the inference command. The architecture and tensor layout must match the compact model.

`predict()` returns a list of NumPy arrays, one per scene. Each array contains sorted angle estimates in degrees. Use `len(predictions[i])` to obtain the estimated source count for scene `i`.

Use `device="cuda:0"` for the first CUDA device. `cpu_threads` controls PyTorch CPU threads; `search_workers` controls the CPU search pool. CUDA inference still uses CPU routines for physical decoding. `graph_encoders=False` disables encoder, refiner, and candidate CUDA graphs.

Reuse a model instance across batches. The context manager calls `close()` to release its search pool and graph caches. When constructing the model without `with`, call `model.close()` when finished.

## Command-line interface

```bash
pgbsa-infer --input examples/sample_covariances.npz --output predictions.json --device cpu
```

`python -m pgbsa` provides the same interface. Supported input files are:

- `.npy`: a covariance array with the shape and data type specified above.
- `.npz`: an archive containing the array under the key `in_mat`.
- `.pt` or `.pth`: a float32 tensor, or a dictionary containing that tensor under `in_mat`, loaded with `weights_only=True`.

The result JSON includes `predictions`, model metadata, runtime diagnostics, the input shape, and the input file hash. Each prediction contains `sample_id` and `angles_deg`. Existing output files are not overwritten.

| Option | Default | Meaning |
| --- | --- | --- |
| `--device` | `cpu` | CPU or CUDA device |
| `--checkpoint` | Bundled weights | Exported training checkpoint |
| `--batch-size` | `32` | Number of scenes per inference call |
| `--cpu-threads` | `1` | PyTorch CPU thread count |
| `--search-workers` | `4` | CPU search worker count |
| `--no-graphs` | Disabled | Disable CUDA graph execution |
| `--verify-only` | Disabled | Check bundled files without inference |

Run `pgbsa-infer --help` for the full command syntax.

## Tests and timing

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

The tests check covariance generation and labels, saved-dataset loading, training gradients, exact CPU continuation after a saved boundary, checkpoint reloads, and inference from another working directory. The inference regression uses three saved inputs and the angle tolerance recorded in `examples/expected_predictions.json`.

After intentional source edits, refresh the source inventory before verification:

```bash
python scripts/update_manifest.py
pgbsa-infer --verify-only
python -m pytest
```

Keep `assets/inference.json` paired with the supplied checkpoint. Updating the manifest records file hashes; it does not establish that a changed inference configuration remains compatible with the pretrained weights.

To time a supplied workload:

```bash
python scripts/benchmark.py --input examples/sample_covariances.npz --device cpu --batch-size 1 --warmup 10 --repeats 7 --output benchmark.json
```

The script reports warm-up calls, all measured calls, and batch and per-sample summaries. It includes input transfer and CPU output materialization, while excluding file loading, tensor preparation, and model loading. If the fixture contains fewer scenes than the requested batch size, the script repeats scenes in their existing order.

Initial calls can include Numba compilation and CUDA graph capture. Report hardware, software versions, batch size, input workload, and warm-up settings with timing results.
