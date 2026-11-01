"""Observation-only covariance loading and retrieval."""
from __future__ import annotations
import torch
from .networks import flatten_supports

@torch.no_grad()
def adaptive_diagonal_loading(
    in_mat: torch.Tensor,
    *,
    target_noise_floor: float = 0.020,
    maximum_loading: float = 0.35,
    floor_eigenvalues: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Raise only abnormally low normalized covariance noise floors.

    The input covariance has unit trace.  Loading therefore mixes it with
    I/N and preserves unit trace.  The loading coefficient is inferred per
    scene from the mean of the smallest Hermitian eigenvalues.
    """
    if in_mat.ndim != 4 or in_mat.shape[1] != 2:
        raise ValueError("in_mat must have shape [B,2,N,N]")
    if not 0.0 <= target_noise_floor < 1.0:
        raise ValueError("target_noise_floor must be in [0, 1)")
    if not 0.0 <= maximum_loading < 1.0:
        raise ValueError("maximum_loading must be in [0, 1)")
    sensor_count = int(in_mat.shape[-1])
    count = min(max(int(floor_eigenvalues), 1), sensor_count)
    covariance = torch.complex(in_mat[:, 0].float(), in_mat[:, 1].float())
    covariance = 0.5 * (covariance + covariance.conj().transpose(-2, -1))
    eigenvalue = torch.linalg.eigvalsh(covariance).real
    floor = eigenvalue[:, :count].mean(dim=1).clamp_min(0.0)
    identity_floor = 1.0 / float(sensor_count)
    denominator = (identity_floor - floor).clamp_min(1.0e-8)
    loading = ((float(target_noise_floor) - floor) / denominator).clamp(
        0.0, float(maximum_loading)
    )
    identity = torch.eye(
        sensor_count, device=in_mat.device, dtype=covariance.dtype
    ) / float(sensor_count)
    loaded = (
        (1.0 - loading)[:, None, None] * covariance
        + loading[:, None, None] * identity[None, :, :]
    )
    return torch.stack([loaded.real, loaded.imag], dim=1).to(in_mat.dtype), loading


@torch.no_grad()
def retrieve(dual, x):
    loaded, _ = adaptive_diagonal_loading(
        x, target_noise_floor=.020, maximum_loading=.35)
    ret = dual.high_snapshot_retrieval
    # The encoder uses covariance alone, so snapshot conditioning is disabled.
    ge = ret(loaded)
    supports = ret.predicted_supports(ge, max_sources=20)
    sb, si = flatten_supports(supports, x.device)
    return loaded, ge, supports, sb, si
