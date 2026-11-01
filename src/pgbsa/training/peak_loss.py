"""Straight-through coarse peak consistency with unresolved-angle masking."""

from __future__ import annotations
import torch
from torch.nn import functional as F


def coarse_peak_consistency(
    root, logits, angles, active, *, prominence_temperature=0.02, strength_temperature=0.03
):
    """Return straight-through peak loss and detached diagnostics.

    Equal scene weight over resolvable scenes. An all-noise scene is resolvable
    with target zero. For truths less than two nearest-grid indices apart the count
    objective is omitted, since distinct local maxima may be unrepresentable.
    """
    if (
        logits.ndim != 2
        or angles.ndim != 2
        or active.shape != angles.shape
        or angles.shape[0] != logits.shape[0]
    ):
        raise ValueError("Expected logits[B,G], angles/active[B,S]")
    if prominence_temperature <= 0 or strength_temperature <= 0:
        raise ValueError("Positive soft-peak temperatures required")
    probability = logits.float().sigmoid()
    grid = root.theta_grid.to(device=logits.device, dtype=torch.float32)
    if grid.ndim != 1 or grid.numel() != logits.shape[1] or grid.numel() < 2:
        raise ValueError("Coarse grid shape mismatch")
    active = active.bool()
    angles = angles.float()
    step = grid[1] - grid[0]
    minimum = torch.as_tensor(
        getattr(root, "count_min_threshold", 0.05), device=logits.device, dtype=torch.float32
    )
    relative = torch.as_tensor(
        getattr(root, "count_relative_alpha", 0.10), device=logits.device, dtype=torch.float32
    )
    cutoff = torch.maximum(minimum, relative * probability.amax(1).detach())
    left = F.pad(probability[:, :-1], (1, 0), value=-1.0)
    right = F.pad(probability[:, 1:], (0, 1), value=-1.0)
    soft_local = torch.sigmoid((probability - left) / prominence_temperature) * torch.sigmoid(
        (probability - right) / prominence_temperature
    )
    soft_selected = soft_local * torch.sigmoid(
        (probability - cutoff[:, None]) / strength_temperature
    )
    soft_count = soft_selected.sum(1)
    hard_count = (
        ((probability >= left) & (probability >= right) & (probability >= cutoff[:, None]))
        .sum(1)
        .to(probability.dtype)
    )
    surrogate = hard_count + soft_count - soft_count.detach()
    target = active.sum(1).to(probability.dtype)
    safe_angles = torch.where(active, angles, grid[0])
    nearest = torch.floor(((safe_angles - grid[0]) / step).clamp(0, grid.numel() - 1) + 0.5).long()
    pair_mask = active[:, :, None] & active[:, None, :]
    not_self = ~torch.eye(active.shape[1], device=active.device, dtype=torch.bool)[None]
    unresolved = pair_mask & not_self & ((nearest[:, :, None] - nearest[:, None, :]).abs() < 2)
    valid = ~unresolved.any(-1).any(-1)
    per_scene = F.smooth_l1_loss(surrogate, target, beta=1.0, reduction="none")
    mask = valid.to(per_scene.dtype)
    loss = (per_scene * mask).sum() / mask.sum().clamp_min(1.0)

    def masked(value):
        return (value * mask).sum() / mask.sum().clamp_min(1.0)

    parts = dict(
        coarse_peak_consistency=loss.detach(),
        coarse_peak_supervised_scenes=valid.sum().detach(),
        coarse_peak_unresolved_truth_scenes=(~valid).sum().detach(),
        coarse_peak_hard_mean=masked(hard_count).detach(),
        coarse_peak_soft_mean=masked(soft_count).detach(),
        coarse_peak_true_mean=masked(target).detach(),
        coarse_peak_cutoff_mean=masked(cutoff.expand_as(target)).detach(),
        coarse_peak_surrogate_abs_error=masked((hard_count - target).abs()).detach(),
    )
    return loss, parts
