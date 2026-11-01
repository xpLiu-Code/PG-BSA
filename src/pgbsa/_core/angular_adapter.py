"""Angle-aware local/global support attention for dense predicted routes."""
from __future__ import annotations
import math
from typing import Iterator
import torch
from torch import nn

class AngularContextAdapter(nn.Module):
    """Support attention with local differences and global context.

    Local weights depend on angular distance. The weighted differences
    sum_j A_ij (v_j - v_i) cancel identical neighboring responses. A learned
    gate combines these differences with global attention and support context.
    Zero initialization of the output projection preserves the base
    Transformer at initialization. Eligibility uses predicted route counts."""

    def __init__(self, backbone: nn.Module, width: int, heads: int = 4,
                 radius_deg: float = 2.0, minimum_routes: int = 11):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        if radius_deg <= 0 or minimum_routes < 1:
            raise ValueError("radius_deg and minimum_routes must be positive")
        self.backbone = backbone
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self.width, self.heads = int(width), int(heads)
        self.radius_deg, self.minimum_routes = float(radius_deg), int(minimum_routes)
        self.input_norm = nn.LayerNorm(width)
        self.context_norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.relative_bias = nn.Linear(2, heads, bias=False)
        nn.init.zeros_(self.relative_bias.weight)
        self.fusion_gate = nn.Linear(3 * width + 2, width)
        self.output_projection = nn.Linear(2 * width, width)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        # Per-batch context is excluded from checkpoint buffers.
        self._angles: torch.Tensor | None = None
        self._counts: torch.Tensor | None = None
        self._padding: torch.Tensor | None = None

    def train(self, mode: bool = True):
        super().train(mode)
        # Evaluation mode disables dropout in the frozen Transformer.
        self.backbone.eval()
        return self

    def adapter_parameters(self) -> Iterator[nn.Parameter]:
        for name, parameter in self.named_parameters():
            if not name.startswith("backbone."):
                yield parameter

    def set_route_context(self, angles: torch.Tensor, counts: torch.Tensor,
                          padding: torch.Tensor) -> None:
        self._angles = angles.detach()
        self._counts = counts.detach()
        self._padding = padding.detach()

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None,
                src_key_padding_mask: torch.Tensor | None = None,
                is_causal: bool | None = None) -> torch.Tensor:
        if self._angles is None or self._counts is None or self._padding is None:
            raise RuntimeError("Call set_context before refine_routes")
        if x.shape[:2] != self._angles.shape:
            raise ValueError("Stale route context: call set_context for this batch")
        angles = self._angles.to(x.device, dtype=torch.float32)
        padding = self._padding.to(x.device)
        if src_key_padding_mask is not None:
            supplied = src_key_padding_mask.to(x.device)
            supplied = supplied if supplied.dtype == torch.bool else supplied < 0
            if not torch.equal(supplied, padding):
                raise ValueError("Route context and Transformer padding disagree")
        if mask is not None or is_causal:
            raise ValueError("The reference support context is noncausal and unmasked")
        base = self.backbone(x, src_key_padding_mask=padding)
        batch, routes, width = x.shape
        normalized = self.input_norm(x)
        q, k, v = self.qkv(normalized).reshape(
            batch, routes, 3, self.heads, width // self.heads
        ).permute(2, 0, 3, 1, 4).unbind(0)
        distance = angles[:, None, :] - angles[:, :, None]
        relative = torch.stack((distance / 140.0, distance.abs() / 140.0), -1)
        bias = self.relative_bias(relative.to(x.dtype)).permute(0, 3, 1, 2)
        scores = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(width // self.heads)
        scores = scores + bias.float()
        keys = ~padding[:, None, None, :]
        local_allowed = keys & (distance.abs() <= self.radius_deg + 1e-6)[:, None]
        # Finite sentinel and masking after aggregation handle padded queries.
        global_weight = torch.softmax(scores.masked_fill(~keys, -1e4), -1).to(v.dtype)
        local_weight = torch.softmax(scores.masked_fill(~local_allowed, -1e4), -1).to(v.dtype)
        global_context = (global_weight @ v).transpose(1, 2).reshape(batch, routes, width)
        local_difference = (local_weight @ v - v).transpose(1, 2).reshape(batch, routes, width)
        local_density = (local_allowed[:, 0].sum(-1).to(x.dtype) - 1).clamp_min(0)
        local_density = local_density / self._counts.to(x.device, x.dtype)[:, None].clamp_min(1)
        count_feature = self._counts.to(x.device, x.dtype)[:, None].expand(-1, routes) / 20.0
        original_context = self.context_norm(base)
        gate = torch.sigmoid(self.fusion_gate(torch.cat((
            local_difference, global_context, original_context,
            local_density[..., None], count_feature[..., None],
        ), -1)))
        coupled = gate * local_difference + (1.0 - gate) * global_context
        residual = self.output_projection(torch.cat((coupled, original_context), -1))
        enabled = (~padding) & (self._counts.to(x.device)[:, None] >= self.minimum_routes)
        # Selection prevents nonfinite disabled residuals from entering the output.
        return base + torch.where(enabled[..., None], residual, torch.zeros_like(residual))


def iter_adapters(localizer: nn.Module) -> Iterator[AngularContextAdapter]:
    for name in ("support_transformer", "precision_support_transformer"):
        value = getattr(localizer, name, None)
        if isinstance(value, AngularContextAdapter):
            yield value


def attach_adapter(localizer: nn.Module, *, radius_deg: float = 2.0,
                   minimum_routes: int = 11, heads: int = 4,
                   attach_precision: bool = False) -> AngularContextAdapter:
    """Wrap the wide/shared context; optionally wrap dual-precision context.

    Attach before loading an adapted state_dict. To train only the adapter,
    freeze localizer first, then construct the optimizer from
    ``adapter.adapter_parameters()`` (or all ``iter_adapters`` if requested).
    """
    names = ["support_transformer"]
    if attach_precision and getattr(localizer, "precision_support_transformer", None) is not None:
        names.append("precision_support_transformer")
    for name in names:
        existing = getattr(localizer, name)
        if isinstance(existing, AngularContextAdapter):
            raise ValueError(f"{name} is already adapted")
        reference = next(existing.parameters())
        wrapper = AngularContextAdapter(existing, int(localizer.d_model), heads,
                                        radius_deg, minimum_routes)
        wrapper.to(device=reference.device, dtype=reference.dtype)
        setattr(localizer, name, wrapper)
    return localizer.support_transformer


def set_context(localizer: nn.Module, encoded: dict[str, torch.Tensor],
                support_batch: torch.Tensor, support_index: torch.Tensor) -> None:
    """Set angular context in the route order used by refine_routes."""
    adapters = list(iter_adapters(localizer))
    if not adapters:
        return
    batch_size = int(encoded["location_logits"].shape[0])
    if support_batch.numel() != support_index.numel():
        raise ValueError("support_batch and support_index lengths differ")
    if support_batch.numel() and bool((support_batch[1:] < support_batch[:-1]).any()):
        raise ValueError("Reference refine_routes requires scene-grouped supports")
    counts = torch.bincount(support_batch, minlength=batch_size)
    maximum = int(counts.max().item()) if batch_size else 0
    starts = torch.cumsum(counts, 0) - counts
    slots = torch.arange(support_batch.numel(), device=support_batch.device)
    slots = slots - torch.repeat_interleave(starts, counts)
    angles = encoded["location_logits"].new_zeros((batch_size, maximum), dtype=torch.float32)
    if support_index.numel():
        route_angle = localizer.theta_grid[support_index].float()
        if "base_offset" in encoded:
            route_angle = route_angle + encoded["base_offset"][support_batch, support_index].float()
        angles[support_batch, slots] = route_angle.detach()
    padding = torch.arange(maximum, device=counts.device)[None, :] >= counts[:, None]
    active = counts > 0
    for adapter in adapters:
        adapter.set_route_context(angles[active], counts[active], padding[active])
