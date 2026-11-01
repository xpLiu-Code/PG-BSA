"""Angular interpolation targets and supervised local refinement losses."""

import torch
from torch.nn import functional as F


def _masked_mean(values, mask):
    weight = mask.to(values.dtype)
    return (values * weight).sum() / weight.sum().clamp_min(1.0)


def two_grid_labels(theta_grid, angles, active):
    """Max-normalized two-bin angle targets, with first-source tie breaking."""
    grid = theta_grid.float()
    angles = angles.float()
    active = active.bool()
    b, sources = angles.shape
    g = grid.numel()
    step = grid[1] - grid[0]
    coordinate = ((angles - grid[0]) / step).clamp(0, g - 1)
    left = coordinate.floor().long()
    right = (left + 1).clamp_max(g - 1)
    fraction = coordinate - left
    raw = torch.stack((1 - fraction, fraction), -1)
    weights = raw / raw.amax(-1, keepdim=True).clamp_min(1e-07)
    weights *= active[..., None]
    index = torch.stack((left, right), -1).reshape(b, -1)
    weight = weights.reshape(b, -1)
    location = angles.new_zeros((b, g)).scatter_reduce_(
        1, index, weight, reduce="amax", include_self=True
    )
    slots = torch.arange(sources * 2, device=angles.device)[None].expand(b, -1)
    eligible = (weight > 1e-07) & (weight == location.gather(1, index))
    selected = torch.where(eligible, slots, sources * 2)
    winner = torch.full((b, g), sources * 2, device=angles.device, dtype=torch.long)
    winner.scatter_reduce_(1, index, selected, reduce="amin", include_self=True)
    truth = angles.repeat_interleave(2, dim=1).gather(1, winner.clamp_max(sources * 2 - 1))
    offset = torch.where(winner < sources * 2, truth - grid[None], torch.zeros_like(truth))
    return (location, offset, left, right)


def soft_bin_targets(residual, offsets):
    offsets = offsets.float()
    residual = residual.float()
    coordinate = ((residual - offsets[0]) / (offsets[1] - offsets[0])).clamp(0, offsets.numel() - 1)
    left = coordinate.floor().long()
    right = (left + 1).clamp_max(offsets.numel() - 1)
    fraction = coordinate - left
    target = residual.new_zeros((residual.numel(), offsets.numel()))
    target.scatter_add_(1, left[:, None], (1 - fraction)[:, None])
    target.scatter_add_(1, right[:, None], fraction[:, None])
    return target


def route_loss(model, rr, truthflat, *, angle_scale=0.05):
    """Angle supervision for ordinary, adapted, and close-pair routes.

    Combines final Huber + .25 smooth MAE,
    .1 fine CE, .1 wide CE, .5 center Huber + .125 center smooth MAE.
    CE ignores targets outside its actual bin support; final-angle loss ignores
    routes outside the combined wide+fine representable interval. Both route
    hypotheses in a single-source scene receive the same target angle.
    """
    truth = truthflat.float().reshape(-1)
    pred = rr["conditional_angle"].float().reshape(-1)
    if truth.shape != pred.shape:
        raise ValueError("truthflat must align one-to-one with refined routes")
    zero = pred.sum() * 0.0
    if not truth.numel():
        return (zero, {"routes": truth.new_tensor(0), "angle": zero.detach()})
    fine_offsets = model.fine_offsets.float()
    wide_offsets = getattr(model, "wide_offsets", None)
    has_wide = (
        wide_offsets is not None
        and rr.get("wide_logits") is not None
        and (rr["wide_logits"].numel() > 0)
    )
    origin = rr.get("route_base_angle", rr["base_angle"]).float()
    wide_res = truth - origin
    lo = fine_offsets.min()
    hi = fine_offsets.max()
    if has_wide:
        wide_offsets = wide_offsets.float()
        lo = lo + wide_offsets.min()
        hi = hi + wide_offsets.max()
    reachable = (
        torch.isfinite(truth)
        & (wide_res.detach() >= lo - 1e-06)
        & (wide_res.detach() <= hi + 1e-06)
    )
    if not torch.isfinite(truth).all():
        raise ValueError("Pass finite matched truth angles only")
    scaled = (pred - truth) / float(angle_scale)
    angle = _masked_mean(
        F.smooth_l1_loss(scaled, torch.zeros_like(scaled), beta=0.5, reduction="none")
        + 0.25 * torch.sqrt(scaled.square() + 0.0001),
        reachable,
    )
    center_error = (rr["base_angle"].float() - truth) / float(angle_scale)
    center = _masked_mean(
        0.5
        * F.smooth_l1_loss(center_error, torch.zeros_like(center_error), beta=0.5, reduction="none")
        + 0.125 * torch.sqrt(center_error.square() + 0.0001),
        reachable,
    )
    fine_res = truth - rr["base_angle"].float()
    fine_mask = (
        reachable
        & (fine_res.detach() >= fine_offsets.min() - 1e-06)
        & (fine_res.detach() <= fine_offsets.max() + 1e-06)
    )
    target = soft_bin_targets(fine_res.detach(), fine_offsets)
    fine_ce = _masked_mean(
        -(target * F.log_softmax(rr["fine_logits"].float(), -1)).sum(-1), fine_mask
    )
    wide_ce = zero
    if has_wide:
        wide_mask = (
            reachable
            & (wide_res.detach() >= wide_offsets.min() - 1e-06)
            & (wide_res.detach() <= wide_offsets.max() + 1e-06)
        )
        target = soft_bin_targets(wide_res.detach(), wide_offsets)
        wide_ce = _masked_mean(
            -(target * F.log_softmax(rr["wide_logits"].float(), -1)).sum(-1), wide_mask
        )
    loss = angle + center + 0.1 * fine_ce + 0.1 * wide_ce
    parts = {
        "angle": angle,
        "center": center,
        "fine_ce": fine_ce,
        "wide_ce": wide_ce,
        "routes": reachable.sum(),
        "fine_routes": fine_mask.sum(),
        "unreachable_routes": (~reachable).sum(),
    }
    return (loss, {k: v.detach() for k, v in parts.items()})
