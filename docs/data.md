# Synthetic covariance datasets

## Signal and noise model

The simulator generates narrowband far-field observations for a 16-element uniform linear array with half-wavelength spacing:

$$
Y=A(\Theta)S+N,\qquad
a_m(\theta)=\exp(-j\pi m\sin\theta).
$$

Active sources are independent proper complex Gaussians with expected power 1. Noise is spatially white proper complex Gaussian with variance $\sigma_n^2=10^{-\mathrm{SNR}_{\mathrm{dB}}/10}$ per sensor. The specified SNR is **per source**. For $K>0$, total signal-to-noise power at a sensor is $K/\sigma_n^2$; the source signals are not divided by $\sqrt K$.

For each scene, the simulator forms $R=YY^H/L$ without subtracting a sample mean, then divides by $\operatorname{tr}(R)$. Real and imaginary parts are stored as separate float32 channels. Labels contain the angles, source count, snapshot count, and nominal per-source SNR. Only covariance is supplied to inference.

| Property | Sampling rule |
| --- | --- |
| Number of sources | 0 to 20, according to curriculum mode |
| Source angles | Within -65 to +65 degrees |
| Snapshot counts | 16, 32, 64, 128, 256, 512, 1000 |
| Snapshot strata | 65% drawn from 16-256; 35% from 512-1000 |
| Per-source SNR | -10, -5, 0, 5, 10, 20 dB |
| SNR strata | 15% drawn from -10/-5; 85% from 0/5/10/20 |
| Source powers | Equal expected power, 1 per active source |
| Covariance shape | `[N, 2, 16, 16]` |
| Label capacity | 20 angles per scene, with an active mask |

For a noise-only scene, `snr_db=0` is a bookkeeping placeholder and noise variance is 1. No source SNR is defined for that scene.

## Curriculum modes

| Mode | Scene mix |
| --- | --- |
| `ordinary` | Single sources, pairs, moderate counts, and compact triples |
| `adapted` | 11-20 sources, with extra samples at 12, 18, and 20 |
| `close` | Single sources and pairs separated by 0.5-3.5 degrees |
| `close_covered` | Compact triples plus single sources and close pairs |
| `retrieval` | Noise-only scenes, single sources, pairs, and counts up to 20 |

Geometries include random spread, a randomly pinned minimum gap, close pairs, compact groups, and a local cluster with background sources. The exact reservation fractions and spacing ranges are recorded in the generator's `provenance()` output and each dataset manifest. Fractions are rounded within each batch, so the batch size is part of the sampling protocol.

## Generate labeled splits

```bash
pgbsa-generate-data --config configs/data.json --output data/synthetic
```

`python scripts/generate_dataset.py` and `python -m pgbsa.training.data` provide the same interface. CPU generation is the default; add `--device cuda:0` for CUDA. The output directory must not already exist.

[configs/data.json](../configs/data.json) controls split sizes, independent seeds, simulation batch size, shard size, curriculum mode order, and the low-snapshot and low-SNR fractions. Its default sizes are 24,576 training, 3,072 validation, and 3,072 test scenes. These are reusable example dataset sizes, not the number of observations used in the original online training run.

| Configuration field | Meaning |
| --- | --- |
| `batch_size` | Scenes generated per simulator call; affects rounded curriculum quotas and random-stream order |
| `samples_per_shard` | Maximum scenes per NPZ shard; must be a positive multiple of `batch_size` |
| `mode_sequence` | Curriculum modes cycled across simulator calls; the sequence continues across shard boundaries |
| `low_snapshot_fraction` | Probability of drawing from the 16-256 snapshot stratum |
| `low_snr_fraction` | Probability of drawing from the -10/-5 dB stratum |
| `splits.<name>.seed` | Independent random seed for `train`, `validation`, or `test` |
| `splits.<name>.samples` | Positive number of scenes in that split |

The three named splits are required. Their last simulator call may use a partial batch. The default mode sequence covers all training and calibration paths; a custom sequence must retain every mode required by the intended training run. Sensor geometry, source-power assumptions, available snapshot counts, and available SNR values are defined in `src/pgbsa/training/data.py`.

```text
data/synthetic/
  manifest.json
  train/part-00000.npz
  validation/part-00000.npz
  test/part-00000.npz
```

Additional shards are numbered consecutively. The manifest records the configuration, simulator checksum, dependency versions, device, array schema, and each shard's checksum. Existing datasets are not overwritten. The three splits must have distinct seeds.

## Stored arrays

| Key | Shape and type | Meaning |
| --- | --- | --- |
| `in_mat` | `[N,2,16,16]`, float32 | Real/imaginary normalized covariance |
| `angles_deg` | `[N,20]`, float32 | Sorted active angles, followed by zero padding |
| `active` | `[N,20]`, bool | Valid source slots; zero degrees can be a valid angle |
| `source_count` | `[N]`, int64 | Number of active sources |
| `snapshots` | `[N]`, int64 | Number of simulated snapshots |
| `snr_db` | `[N]`, float32 | Nominal per-source SNR |
| `source_power` | `[N,20]`, float32 | Expected source variance; 0 for inactive slots |
| `noise_variance` | `[N]`, float32 | Expected noise variance per sensor |
| `geometry_family` | `[N]`, Unicode | Sampled angular geometry |
| `mode` | `[N]`, Unicode | Curriculum mode |

Load a shard without object deserialization:

```python
import numpy as np

with np.load("data/synthetic/train/part-00000.npz", allow_pickle=False) as data:
    covariance = data["in_mat"]
    truth_angles = data["angles_deg"][0, data["active"][0]]
```

The same NPZ file can be passed to `pgbsa-infer`; it reads only `in_mat`. To train on the full dataset, pass `--train-data data/synthetic --validation-data data/synthetic` to `pgbsa-train`. That loader verifies shard checksums and samples scenes of the requested curriculum mode from the designated split.

Sampling is without replacement within each batch when the mode pool is large enough; smaller pools use replacement. Separate batches may revisit scenes. Training follows an update schedule rather than an epoch traversal of the saved dataset.

## Online sampling API

```python
from pgbsa.training.data import make_generator, generate_batch

generator = make_generator(seed=202610100101, device="cpu", split="train")
covariance, angles, active, count, snapshots, snr = generate_batch(
    128, "cpu", generator, mode="ordinary"
)
```

The trainer uses this simulator directly when `--train-data` is omitted. Reproducing a stream requires the same seed, device, dependency versions, batch sizes, and sequence of modes. Exporting a separate dataset with the same seed does not reproduce a training stream whose calibration and stage calls consume random draws in a different order.
