# Training

## Supplied checkpoint and initialization

The supplied weights were trained with `refiner-scratch`: pretrained covariance encoders and candidate proposal weights are retained, local refinement parameters are reset, and the refinement and candidate calibration modules are trained for 6,000 updates. The matching initialization checkpoint is bundled as `src/pgbsa/assets/training_initial.pth`.

| Component | Initialization | Updated in `refiner-scratch` |
| --- | --- | --- |
| Shared covariance encoder | Pretrained | No |
| Retrieval and close-pair encoders | Pretrained | No |
| Candidate proposal generator | Pretrained | No |
| Shared refinement core and route adapters | Random initialization | Yes |
| Candidate confidence and angle correction heads | Pretrained | From 35% of the schedule |

The training graph has 2,380,568 unique parameters. Of these, 276,823 refinement parameters are reset and 353,337 parameters are optimized, including the candidate heads. Inference compaction reduces the deployed count to 2,367,509. The trainer verifies that the frozen tensors remain unchanged.

## Run the checkpoint's training recipe

From the repository root, after [installation](usage.md#environment):

```bash
pgbsa-train --config configs/training_refiner.json --output runs/refiner/model.pth
```

The default device is `cuda:0`. Add `--device cpu` to run on CPU. `python -m pgbsa.training.train` and `python scripts/train.py` provide the same interface. Command-line options override the JSON configuration. The output must be a new `.pth` or `.pt` checkpoint path.

Fresh labeled covariance batches are generated online. The three supervised stages cycle through `route_original`, `route_adapted`, and `route_close`; each receives 2,000 updates in the 6,000-update schedule. Ground-truth angles construct training targets and matched routes. Inference takes covariance tensors only.

| Stage | Network path and supervision |
| --- | --- |
| `route_original` | Ordinary-scene refinement from the raw covariance |
| `route_adapted` | Dense-scene refinement, including retrieved and expanded angular supports |
| `route_close` | Close-pair refinement from the diagonally loaded covariance |
| `coarse_original` | Coarse localization and angle offsets of the shared encoder; `full-scratch` only |
| `coarse_retrieval` | Retrieval encoder and candidate proposal supervision; `full-scratch` only |

The route stages mix label-derived starting locations with supports produced by the current network. The mixture gradually shifts toward the network's own supports. Loss computation matches these supports to the source-angle labels after routing.

## Training parameters

[configs/training_refiner.json](../configs/training_refiner.json) contains the training schedule associated with the supplied weights. [configs/model.json](../configs/model.json) records the checkpoint architecture; [inference.json](../src/pgbsa/assets/inference.json) contains deployment options.

| Parameter | Value |
| --- | --- |
| Optimizer | AdamW |
| Total updates | 6,000 |
| Batch size | 128 |
| Base learning rate | 0.0003 |
| Candidate head learning-rate multiplier | 0.10 |
| Candidate head activation | 35% of schedule |
| Weight decay | 0.0001 |
| Gradient norm limit | 5.0 |
| Linear warmup | First 200 updates |
| Cosine learning-rate floor | 0.10 of initial rate |
| Label-anchored route fraction | 0.95 to 0.25 over first 70% of schedule |
| Candidate confidence loss weight | 0.25 |
| Candidate angle correction loss weight | 0.25 |
| Feature calibration | 6 batches per calibration |
| Development evaluation | Every 1,000 updates; 6 batches of 32 scenes |
| Checkpoint interval | 1,000 updates |
| Training seed | 202610100101 |
| Development seed | 202610109999 |

The cosine decay is multiplied by the warmup factor. The candidate heads have neither optimizer updates nor weight decay before their activation. Their feature normalization is calibrated initially, when head training starts, and at the end; affine reparameterization preserves the current head function when normalization changes.

Each route loss combines final-angle and intermediate-center smooth-L1 terms with wide- and fine-grid cross entropy. Angular residuals are scaled by 0.05 degrees, the smooth-L1 transition is 0.5, and each grid cross-entropy term has weight 0.1. Targets outside a route's representable interval are masked. Candidate losses use matched angular targets. Every second adapted-route visit includes expanded candidate supports. The definitions are in [losses.py](../src/pgbsa/training/losses.py) and [recipe.py](../src/pgbsa/training/recipe.py).

### Configuration reference

JSON fields use underscores; the corresponding command-line options use hyphens, for example `batch_size` and `--batch-size`. Explicit command-line values override the JSON fields. The two supplied configurations differ in `mode` and `steps`.

| JSON fields | Meaning |
| --- | --- |
| `device` | PyTorch training device, normally `cuda:0` or `cpu` |
| `mode` | `refiner-scratch` retains pretrained encoders; `full-scratch` initializes every learned parameter |
| `steps`, `batch_size` | Total optimizer updates and scenes requested per update |
| `stages` | Comma-separated cycle of supervised stages; the two coarse stages are prepended in `full-scratch` |
| `seed`, `validation_seed` | Separate training and fixed-development random streams |
| `lr`, `weight_decay` | AdamW base learning rate and weight decay |
| `owner_lr_scale`, `head_lr_scale` | Learning-rate multipliers for the shared encoder and candidate heads; the encoder remains frozen in `refiner-scratch` |
| `head_start_fraction` | Fraction of the schedule before candidate-head updates begin in `refiner-scratch`; heads are active immediately in `full-scratch` |
| `warmup_steps`, `minimum_lr_ratio` | Linear warmup duration and cosine-decay floor; warmup is capped at the total update count |
| `gradient_clip` | Maximum joint gradient norm before each optimizer update |
| `anchor_start`, `anchor_end`, `anchor_decay_fraction` | Initial/final fractions of label-anchored routes and the schedule fraction over which they decay |
| `candidate_loss_weight`, `offset_loss_weight` | Weights of candidate-confidence and angle-correction losses |
| `peak_consistency_weight` | Peak-consistency loss weight in the coarse stages of `full-scratch` |
| `calibration_batches` | Batches used for each candidate-feature normalization calibration |
| `validation_every` | Development evaluation interval; `0` disables evaluation |
| `validation_batches`, `validation_batch_size` | Size of the fixed development bank |
| `snapshot_every` | Periodic checkpoint interval; `0` disables intermediate snapshots, while the final model and resume pair are still saved |
| `log_every` | Console and progress-summary interval; the JSONL training log still records every update |

Path and continuation options are passed on the command line:

| Option | Meaning |
| --- | --- |
| `--config` | JSON file containing the fields above |
| `--output` | New final `.pth` or `.pt` path |
| `--weights` | Initialization checkpoint; defaults to the bundled `training_initial.pth` |
| `--train-data` | Dataset root or manifest; samples its training split, or synthesizes online when omitted |
| `--validation-data` | Dataset root or manifest; samples its validation split, or synthesizes the development bank when omitted |
| `--resume` | Paired resume-state file from a previous update boundary |
| `--stop-after` | Stop after this update and save state, preserving the configured total schedule; `0` runs to completion |

Run `pgbsa-train --help` for the accepted argument syntax. Training batches must contain at least four scenes.

## Train from saved datasets

Generate the labeled splits described in [Data](data.md), then supply their manifest:

```bash
pgbsa-generate-data --config configs/data.json --output data/synthetic
pgbsa-train --config configs/training_refiner.json --train-data data/synthetic --validation-data data/synthetic --output runs/offline/model.pth
```

The loader samples from the training split by curriculum mode and uses the validation split for development evaluation. The test split is reserved for evaluation. The saved-dataset option uses the same loss functions but draws from a finite dataset; it does not replay the original online random stream. The default dataset configuration contains every mode needed by calibration and both training modes.

## Train all network components

```bash
pgbsa-train --config configs/training_full_scratch.json --output runs/full/model.pth
```

`full-scratch` resets all learned parameters and feature-normalization statistics, then trains the retained encoders, proposal generator, refinement modules, and candidate heads. The bundled checkpoint supplies the architecture and tensor layout. Coarse localization and retrieval stages are added to the three route stages. The shared encoder learning-rate multiplier is 0.33; the proposal BCE weight is 1.0, the coarse offset weight is 0.25, and the peak consistency weight is 0.25. The provided configuration uses 20,000 updates.

The bundled `model.pth` corresponds to `training_refiner.json`. The `training_full_scratch.json` configuration trains a new model with every learned parameter initialized.

## Checkpoints and continuation

For `--output runs/refiner/model.pth`, the trainer writes:

| File | Contents |
| --- | --- |
| `model.pth` | Final model weights and feature-normalization statistics |
| `model_step001000.pth` | Periodic model checkpoint |
| `model_step001000.resume.pth` | Paired optimizer, RNG, schedule, and dataset state |
| `model.training.jsonl` | Per-update losses, learning rates, and gradient norms |
| `model.validation.json` | Development metrics |
| `model.json` | Completed run configuration and integrity checks |
| `model.progress.json` | Latest progress |

Resume with the same total schedule and configuration, keeping the paired model and resume files together:

```bash
pgbsa-train --config configs/training_refiner.json --output runs/refiner/model.pth --resume runs/refiner/model_step001000.resume.pth
```

For an offline run, also pass the same `--train-data` and `--validation-data` paths. Resumption checks the model checksum, source code, optimizer layout, configuration, random streams, and dataset identity. A moved run directory is supported; changing the training code or schedule requires a new run. `--stop-after N` saves a resumable boundary without changing the intended total updates.

The final scheduled checkpoint is exported; development results do not rank checkpoints. Export includes a strict reload check of all model tensors and normalization statistics.

## Development metrics

The development bank is created once and reused at each evaluation. Its mode order is `ordinary`, `adapted`, `close_covered`, and `retrieval`, repeating as needed. The default six batches therefore contain two ordinary batches, two adapted batches, one close-covered batch, and one retrieval batch. Overall metrics are averages over scenes, with separate summaries for each mode.

Let $K$ and $P$ be the true and predicted counts, and $M$ the maximum number of one-to-one angular matches within 0.5 degrees. The metrics in `model.validation.json` are:

| Field | Definition |
| --- | --- |
| `scene_f1` | $2M/(K+P)$; 1 when both angle sets are empty |
| `full_recovery` | 1 exactly when $K=P=M$, otherwise 0 |
| `ospa2` | Order-2 optimal subpattern assignment distance in degrees, with a 5-degree cutoff and cardinality penalty; 0 when both sets are empty |

OSPA uses the minimum sum of capped squared angular distances, adds $5^2\lvert K-P\rvert$, divides by $\max(K,P)$, and takes the square root. Definitions and additional per-scene quantities are in [metrics.py](../src/pgbsa/training/metrics.py). Inference receives covariance only; labels enter the metric calculation afterward.

## Evaluate a trained checkpoint

```bash
pgbsa-infer --checkpoint runs/refiner/model.pth --input examples/sample_covariances.npz --output predictions-trained.json --device cpu
```

The Python equivalent is `create_model(checkpoint="runs/refiner/model.pth", device="cpu")`. The bundled pretrained checkpoint remains available when `checkpoint` is omitted.

To score a trained model on every shard of a generated test split:

```python
from pathlib import Path
import json
import numpy as np
import torch
from pgbsa import create_model
from pgbsa.training.metrics import FIELDS, metric_row

dataset = Path("data/synthetic")
manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
rows = []
with create_model(device="cpu", checkpoint="runs/refiner/model.pth") as model:
    for entry in manifest["splits"]["test"]["files"]:
        with np.load(dataset / entry["path"], allow_pickle=False) as data:
            covariance = data["in_mat"]
            labels = data["angles_deg"]
            masks = data["active"]
            for start in range(0, len(covariance), 32):
                stop = start + 32
                predictions = model.predict(torch.from_numpy(covariance[start:stop]))
                for angles, active, prediction in zip(
                    labels[start:stop], masks[start:stop], predictions
                ):
                    rows.append(metric_row(angles[active], prediction, tolerance=0.5, cutoff=5.0))

values = np.stack(rows)
summary = {name: float(values[:, FIELDS.index(name)].mean())
           for name in ("scene_f1", "full_recovery", "ospa2")}
print(json.dumps({"scenes": len(rows), **summary}, indent=2))
```

These are scene-averaged metrics for that generated test split. Record its manifest and the checkpoint hash with the results.

## Reproducibility

Keep the initialization checkpoint, training configuration, seed, device type, batch size, stage order, and numerical dependencies fixed. Training records the settings and source hashes. CPU and CUDA random streams and floating-point kernels can differ; reproducing a run requires matching the execution environment as well as the seed.
