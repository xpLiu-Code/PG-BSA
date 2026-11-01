# Model and inference

## Observation model

The pretrained model uses a 16-element uniform linear array with half-wavelength spacing. The sources are narrowband and in the far field. Angles are measured from broadside, with the steering convention

$$
a_n(\theta)=\exp\!\left(-j\pi n\sin\theta\right),
\qquad n=0,\ldots,15.
$$

The implementation accepts and returns angles in degrees and converts them to radians for trigonometric operations. The coarse angular grid covers -70 to +70 degrees. Local refinement produces continuous angle estimates.

For a snapshot matrix $Y\in\mathbb C^{16\times L}$, form

$$
R=\frac{YY^H}{L},\qquad
\widehat R=\frac{R}{\operatorname{tr}(R)}.
$$

The input contains the real and imaginary parts of $\widehat R$. Normalization removes the total power scale while retaining spatial correlations.

## Inference stages

1. **Beam-domain encoding.** Steering-conditioned covariance features and spatial-lag features form a representation indexed by direction.
2. **Sparse support retrieval.** Angular attention and learned shrinkage identify candidate regions. The number of retrieved regions depends on the observation.
3. **Local refinement.** Wide and fine steering queries refine each region. Regional context, candidate confidence, and angle corrections support the subsequent set selection. The deployed network retains a close-pair branch.
4. **Set recovery.** Spacing-constrained selection produces an initial angle set. Covariance fitting guides angle replacement, source-count revision, and eligible continuous angle refinement.

The output is a sorted set of angles. Its cardinality is the estimated source count; the implementation supports up to 20 output sources. No true source count is supplied to the decoder.

## Covariance fitting

For a nonempty trial set $\Theta$ containing $k\geq1$ directions, the physical decoder uses the equal-power covariance model

$$
C(\Theta,f)=\frac{f}{16k}A(\Theta)A(\Theta)^H
+\frac{1-f}{16}I,
$$

where $A(\Theta)$ contains the corresponding steering vectors. The scalar $f$ is the signal fraction of the model's total power; $1-f$ is its noise fraction. It is fitted from the observed covariance and the current angle hypothesis.

The fitting objective is

$$
\ell(\Theta,f)=\log\det C(\Theta,f)
+\operatorname{tr}\!\left(C(\Theta,f)^{-1}\widehat R\right).
$$

Each fixed-count coordinate-search pass first fits $f$, then holds it fixed while updating angles. The source-count refinement routines can jointly optimize angles and the power fraction. The scalar is an internal fitting parameter, not a supplied SNR value or a neural network output.

## Implementation map

Paths below are relative to `src/pgbsa/_core/`.

| Component | Main files |
| --- | --- |
| Covariance preparation and support retrieval | `preprocessing.py`, `networks.py` |
| Candidate confidence and decoding | `heads.py`, `decoder.py` |
| Local refinement and regional interaction | `compact_refinement.py`, `angular_adapter.py` |
| Fixed-count covariance search | `rank_update_search.py`, `bounded_spectral.py` |
| Global and local source-count refinement | `fast_cardinality_refinement.py`, `local_cardinality_refinement.py` |
| Conditional angle refinement | `cpu_angle_polish.py` |
| End-to-end inference | `pipeline.py` |

The public factory loads `assets/model.pth` and `assets/inference.json`. It also accepts an exported training checkpoint through `create_model(checkpoint=...)`. Source files under `_core` are internal implementation details; application code should import from `pgbsa`.

## Architecture parameters

[configs/model.json](../configs/model.json) exports the architecture and decoder settings stored in the pretrained checkpoint. The loader constructs the architecture from the checkpoint itself; editing this reference file does not change a model.

| Parameter | Value |
| --- | --- |
| Array | 16 sensors, half-wavelength spacing |
| Coarse angular grid | -70 to +70 degrees, 0.5-degree spacing |
| Encoder feature width | 128 |
| Shared low-rank correction rank | 16 |
| Compact refinement width | 96 |
| Wide query offsets | -0.7 to +0.7 degrees, 0.1-degree spacing |
| Fine query offsets | -0.35 to +0.35 degrees, 0.05-degree spacing |
| Wide covariance attention passes | 1 |
| Regional attention passes | 1 |
| Candidate feature dimension | 44 |
| Confidence head hidden width | 160 |
| Angle correction head hidden width | 128 |
| Maximum source count | 20 |

The checkpoint retains constructor fields such as `fine_dim=256` and `support_layers=2` for assembly compatibility. The `compact_refinement` specification replaces that refinement topology with the 96-channel module above. Training retains 2,380,568 parameters; inference removes unused frozen outputs and folds eligible layers to obtain 2,367,509 deployed parameters.

## Scope

The repository provides the compact model's training and inference paths, its initialization and final weights, and a simulator for labeled covariance datasets. The physical decoder assumes an ideal ULA, spatially white noise, and uncorrelated equal-power sources. The generation protocol and the supplied checkpoint's training schedule are described in [Data](data.md) and [Training](training.md).
