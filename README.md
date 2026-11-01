# PG-BSA: Physics-Guided Beam-Domain Sparse Attention for High-Resolution Underdetermined DOA Estimation With an Unknown Number of Sources

PG-BSA estimates a variable-length set of directions of arrival from the sample covariance of a 16-element uniform linear array. It combines physics-guided beam-domain features, sparse angular attention, local angle refinement, and covariance-based set recovery. The source count is inferred from the observation.

This repository contains the current inference implementation, its pretrained weights, and a small set of regression examples. The deployed network has **2,367,509 parameters**. 

## Installation

Use Python 3.10, 3.11, or 3.12 in a virtual environment. Run the following commands from the repository root for CPU inference:

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
pgbsa-infer --verify-only
```

The reference numerical dependencies are NumPy 1.26.4, SciPy 1.13.1, and Numba 0.59.1. See [Usage](docs/usage.md) for CUDA installation and runtime options.

## Run inference

```bash
pgbsa-infer --input examples/sample_covariances.npz --output predictions.json --device cpu
```

The command writes one angle array per input scene. Each array is sorted in ascending order and expressed in degrees; its length is the estimated source count. Noise-only observations may produce an empty array.

```python
import numpy as np
import torch
from pgbsa import create_model

with np.load("examples/sample_covariances.npz", allow_pickle=False) as data:
    covariance = torch.from_numpy(data["in_mat"].copy())

with create_model(device="cpu") as model:
    predictions = model.predict(covariance)

for angles_deg in predictions:
    print(len(angles_deg), angles_deg)
```

Inputs are finite `torch.float32` tensors of shape `[B, 2, 16, 16]`, where the two channels contain the real and imaginary parts of a Hermitian, unit-trace covariance matrix. The model does not require the true source count, source angles, SNR, or snapshot count. See [Input preparation](docs/usage.md#input-preparation) for conversion from complex snapshots.

## Repository structure

| Path | Contents |
| --- | --- |
| `src/pgbsa/` | Public Python API and command-line interface |
| `src/pgbsa/_core/` | Neural inference and covariance-based decoding |
| `src/pgbsa/assets/` | Pretrained checkpoint, inference options, and file manifest |
| `examples/` | Sample covariance inputs and an inference example |
| `tests/` | Integrity, portability, and prediction regression checks |
| `scripts/benchmark.py` | Warm inference timing on a supplied input batch |
| `docs/model.md` | Signal model and inference stages |
| `docs/usage.md` | Installation, API, file formats, and timing |

## Tests

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

The saved examples check numerical regression and package portability. They do not constitute a separate accuracy benchmark.

## Rights

See [NOTICE.md](NOTICE.md) for the code and weight usage notice, and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the licences of included third-party adaptations.
