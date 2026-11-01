"""Covariance/grid encoders and shared route refinement networks."""
from __future__ import annotations
import math
import copy
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

DEFAULT_MODEL_CONFIG: Dict[str, Any] = {
    "sensor_count": 16,
    "theta_min": -70.0,
    "theta_max": 70.0,
    "coarse_step": 0.5,
    "fine_step": 0.05,
    "fine_span": 0.35,
    "d_lambda": 0.5,
    "d_model": 128,
    "fine_dim": 256,
    "support_layers": 2,
    "max_sources": 20,
    "use_route_residual": False,
    "route_residual_span": 1.0,
    "use_wide_cascade": False,
    "wide_span": 1.05,
    "wide_step": 0.15,
    "recenter_grid_context_blend": 0.0,
    "use_dual_precision": False,
    "use_route_offset_residual": False,
    "route_offset_span": 0.5,
    "route_offset_context_radius": 0,
    "interpolate_route_context": False,
    "soft_route_power": 0.0,
    "use_snapshot_conditioning": False,
}


def covariance_to_lags(in_mat: torch.Tensor) -> torch.Tensor:
    """Convert [B,2,N,N] real/imag covariance channels to ULA lags."""
    covariance = torch.complex(in_mat[:, 0], in_mat[:, 1])
    _, sensor_count, _ = covariance.shape
    lag_values = []
    for lag in range(-(sensor_count - 1), sensor_count):
        diagonal = []
        for row in range(sensor_count):
            column = row - lag
            if 0 <= column < sensor_count:
                diagonal.append(covariance[:, row, column])
        lag_values.append(torch.stack(diagonal, dim=1).mean(dim=1))
    lags = torch.stack(lag_values, dim=1)
    return torch.stack([lags.real, lags.imag], dim=1)


def configure_encoder_optimizations(
    model: nn.Module, *, vectorized_lags: bool = True,
    cached_steering: bool = True, efficient_attention: bool = True,
) -> None:
    """Configure optional encoder kernels before CUDA graph capture.

    The options preserve weight values and checkpoint keys. Derived buffers
    are nonpersistent."""
    model._encoder_options = dict(
        vectorized_lags=bool(vectorized_lags),
        cached_steering=bool(cached_steering),
        efficient_attention=bool(efficient_attention),
    )
    model.base.cov_cross._efficient_attention = bool(efficient_attention)
    device = model.base_steering.device
    n = model.N
    indices = torch.zeros((2*n-1, n), dtype=torch.long, device=device)
    valid = torch.zeros((2*n-1, n), dtype=torch.bool, device=device)
    for out, lag in enumerate(range(-(n-1), n)):
        entries = [row*n + row-lag for row in range(n) if 0 <= row-lag < n]
        indices[out, :len(entries)] = torch.tensor(entries, device=device)
        valid[out, :len(entries)] = True
    for name, value in (('_lag_indices', indices), ('_lag_valid', valid),
                        ('_lag_counts', valid.sum(-1)), ('_cached_base_query', None)):
        model.register_buffer(name, value, persistent=False)
    model._cached_base_query_key = None


def vectorized_covariance_to_lags(in_mat: torch.Tensor, model: nn.Module) -> torch.Tensor:
    """Gather all ULA diagonals in one operation and reduce their valid entries."""
    values = in_mat.flatten(-2)[:, :, model._lag_indices]
    values = values.masked_fill(~model._lag_valid, 0.)
    return values.sum(-1) / model._lag_counts


def _steering_query(model: nn.Module, in_mat: torch.Tensor) -> torch.Tensor:
    fixed = getattr(model, '_frozen_base_query', None)
    if fixed is not None:
        return fixed.expand(len(in_mat), -1, -1)
    options = getattr(model, '_encoder_options', {})
    projection = model.base.steer_proj
    parameters = tuple(projection.parameters())
    cache_allowed = (options.get('cached_steering', False) and not model.training
                     and not torch.is_grad_enabled()
                     and not torch.is_autocast_enabled(in_mat.device.type)
                     and not model.base_steering.is_inference()
                     and all(not p.requires_grad and not p.is_inference() for p in parameters))
    if cache_allowed:
        # Version checks invalidate after load_state_dict or in-place weight
        # updates. Device/dtype changes also rebuild the tiny constant token.
        key = (in_mat.device, in_mat.dtype, model.base_steering.data_ptr(),
               model.base_steering._version,
               tuple((p.data_ptr(), p._version) for p in parameters))
        if model._cached_base_query_key != key:
            # A warm-up builds normal tensors even if the caller uses
            # inference_mode. Such buffers also work in no_grad graph replay.
            with torch.inference_mode(False), torch.no_grad():
                token = model.base_steering.T.unsqueeze(0).to(in_mat.dtype)
                model._cached_base_query = projection(token)
            model._cached_base_query_key = key
        return model._cached_base_query.expand(len(in_mat), -1, -1)
    token = model.base_steering.T.unsqueeze(0).expand(len(in_mat), -1, -1).to(in_mat.dtype)
    return projection(token)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 281, position_step: float = 1.0) -> None:
        super().__init__()
        encoding = torch.zeros(max_len, d_model)
        position = (torch.arange(max_len, dtype=torch.float32) * float(position_step)).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer("pe", encoding.unsqueeze(0))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.pe[:, : value.shape[1]]


class SoftShrinkage(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.tau = nn.Parameter(torch.full((1, 1, channels), -2.0))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        threshold = torch.sigmoid(self.tau) * 0.1
        return torch.sign(value) * F.relu(torch.abs(value) - threshold)


class PhysicsCrossAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int = 4) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )

    def forward(
        self, query: torch.Tensor, key_value: torch.Tensor
    ) -> torch.Tensor:
        attended, _ = self.cross_attn(
            query, key_value, key_value, need_weights=False
        )
        query = self.norm1(query + attended)
        return self.norm2(query + self.mlp(query))


class BasePhysicsCrossAttention(PhysicsCrossAttention):
    """Optional SDPA path avoids producing discarded attention weights."""

    def forward(
        self, query: torch.Tensor, key_value: torch.Tensor
    ) -> torch.Tensor:
        attended, _ = self.cross_attn(
            query, key_value, key_value,
            need_weights=not getattr(self, '_efficient_attention', False),
        )
        query = self.norm1(query + attended)
        return self.norm2(query + self.mlp(query))


class PhysicsBeamEncoder(nn.Module):
    """Physics-guided covariance and spatial-lag encoder on an angular grid."""

    def __init__(
        self,
        sensor_count: int,
        grid_count: int,
        theta_grid: np.ndarray,
        d_model: int = 128,
        nhead: int = 8,
        grid_layers: int = 4,
        offset_scale: float = 0.5,
    ) -> None:
        super().__init__()
        self.N = int(sensor_count)
        self.K = int(grid_count)
        self.offset_scale = float(offset_scale)

        self.steer_proj = nn.Sequential(
            nn.Linear(2 * self.N, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.cov_proj = nn.Sequential(
            nn.Conv1d(
                2 * self.N, d_model, kernel_size=3, padding=1
            ),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        covariance_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=256,
            batch_first=True,
            activation="gelu",
        )
        self.cov_encoder = nn.TransformerEncoder(
            covariance_layer, num_layers=2
        )
        self.cov_cross = BasePhysicsCrossAttention(
            d_model, nhead=4
        )

        self.lag_proj = nn.Sequential(
            nn.Conv1d(2, d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        self.lag_to_grid = nn.Sequential(
            nn.Linear(2 * self.N - 1, self.K),
            nn.GELU(),
        )

        self.gate_cov = nn.Parameter(torch.tensor(1.0))
        self.gate_lag = nn.Parameter(torch.tensor(0.5))
        self.fusion_norm = nn.LayerNorm(d_model)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        physical_step = float(theta_grid[1] - theta_grid[0]) / 0.5
        self.pos_encoder = PositionalEncoding(d_model, max_len=self.K, position_step=physical_step)
        grid_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            batch_first=True,
            activation="gelu",
        )
        self.grid_transformer = nn.TransformerEncoder(
            grid_layer, num_layers=grid_layers
        )
        self.shrinkage = SoftShrinkage(d_model)
        self.loc_head = nn.Linear(d_model, 1)
        self.offset_head = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Tanh(),
        )


def select_calibrated_peaks(
    local_probability: np.ndarray,
    min_threshold: float,
    relative_alpha: float,
    max_sources: int,
) -> np.ndarray:
    """Threshold local maxima without count-head or top-k truncation."""
    del max_sources  # Kept only for checkpoint/API compatibility.
    values = np.asarray(local_probability, dtype=np.float32)
    peak_index = np.flatnonzero(values > 1.0e-8)
    if peak_index.size == 0:
        return np.empty(0, dtype=np.int64)
    threshold = max(
        float(min_threshold),
        float(relative_alpha * values[peak_index].max()),
    )
    selected = np.flatnonzero(values >= threshold)
    return selected.astype(np.int64)


class BeamDOANetwork(nn.Module):
    """Grid encoding with routed local refinement and support interaction.

    Coarse detection uses calibrated grid maxima. Wide and fine query stages
    refine the selected directions and exchange features across supports."""

    def __init__(
        self,
        sensor_count: int = 16,
        theta_min: float = -70.0,
        theta_max: float = 70.0,
        coarse_step: float = 0.5,
        fine_step: float = 0.05,
        fine_span: float = 0.35,
        d_lambda: float = 0.5,
        d_model: int = 128,
        fine_dim: int = 256,
        support_layers: int = 2,
        max_sources: int = 20,
        count_min_threshold: float = 0.05,
        count_relative_alpha: float = 0.10,
        use_route_residual: bool = False,
        route_residual_span: float = 1.0,
        use_wide_cascade: bool = False,
        wide_span: float = 1.05,
        wide_step: float = 0.15,
        recenter_grid_context_blend: float = 0.0,
        use_dual_precision: bool = False,
        use_route_offset_residual: bool = False,
        route_offset_span: float = 0.5,
        route_offset_context_radius: int = 0,
        interpolate_route_context: bool = False,
        soft_route_power: float = 0.0,
        use_snapshot_conditioning: bool = False,
    ) -> None:
        super().__init__()
        del fine_dim  # Kept in the checkpoint configuration for compatibility.
        theta_grid = np.arange(
            theta_min,
            theta_max + 1.0e-6,
            coarse_step,
            dtype=np.float32,
        )
        fine_offsets = np.arange(
            -fine_span,
            fine_span + 1.0e-6,
            fine_step,
            dtype=np.float32,
        )
        self.N = int(sensor_count)
        self.K = int(theta_grid.size)
        self.M = int(fine_offsets.size)
        self.d_lambda = float(d_lambda)
        self.d_model = int(d_model)
        self.max_sources = int(max_sources)
        self.use_snapshot_conditioning = bool(use_snapshot_conditioning)
        self.use_route_residual = False
        self.route_residual_span = float(route_residual_span)
        self.use_wide_cascade = False
        self.wide_span = float(wide_span)
        self.wide_step = float(wide_step)
        self.recenter_grid_context_blend = float(
            recenter_grid_context_blend
        )
        if not 0.0 <= self.recenter_grid_context_blend <= 1.0:
            raise ValueError(
                "recenter_grid_context_blend must be in [0, 1]"
            )

        self.register_buffer(
            "theta_grid", torch.tensor(theta_grid, dtype=torch.float32)
        )
        self.register_buffer(
            "fine_offsets",
            torch.tensor(fine_offsets, dtype=torch.float32),
        )
        self.register_buffer(
            "array_position",
            torch.arange(sensor_count, dtype=torch.float32),
        )
        steering = self._steering(self.theta_grid)
        self.register_buffer(
            "base_steering",
            torch.cat([steering.real.T, steering.imag.T], dim=0),
        )
        self.register_buffer(
            "count_min_threshold",
            torch.tensor(float(count_min_threshold)),
        )
        self.register_buffer(
            "count_relative_alpha",
            torch.tensor(float(count_relative_alpha)),
        )

        self.base = PhysicsBeamEncoder(
            sensor_count=self.N,
            grid_count=self.K,
            theta_grid=theta_grid,
            d_model=d_model,
            nhead=8,
            grid_layers=4,
            offset_scale=min(float(coarse_step), 0.5),
        )
        if self.use_snapshot_conditioning:
            self.snapshot_conditioner = nn.Sequential(
                nn.Linear(1, d_model),
                nn.GELU(),
                nn.Linear(d_model, 2 * d_model),
            )
            self.snapshot_logit_affine = nn.Sequential(
                nn.Linear(1, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 2),
            )
            # Zero initialization preserves the base network before conditioning.
            nn.init.zeros_(self.snapshot_conditioner[-1].weight)
            nn.init.zeros_(self.snapshot_conditioner[-1].bias)
            nn.init.zeros_(self.snapshot_logit_affine[-1].weight)
            nn.init.zeros_(self.snapshot_logit_affine[-1].bias)
        else:
            self.snapshot_conditioner = None
            self.snapshot_logit_affine = None

        self.local_steer_proj = nn.Sequential(
            nn.Linear(2 * self.N, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.offset_embed = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.local_cross = PhysicsCrossAttention(d_model, nhead=4)
        support_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=4 * d_model,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.support_transformer = nn.TransformerEncoder(
            support_layer, num_layers=support_layers
        )
        self.local_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.power_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.noise_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.correction_gate_logit = nn.Parameter(torch.tensor(0.0))
        self.route_residual_head: nn.Module | None = None
        self.route_blend_head: nn.Module | None = None
        self.wide_head: nn.Module | None = None
        self.precision_local_steer_proj: nn.Module | None = None
        self.precision_offset_embed: nn.Module | None = None
        self.precision_local_cross: nn.Module | None = None
        self.precision_support_transformer: nn.Module | None = None
        self.precision_local_head: nn.Module | None = None
        self.precision_correction_gate_logit: nn.Parameter | None = None
        self.use_dual_precision = False
        self.route_offset_head: nn.Module | None = None
        self.use_route_offset_residual = False
        self.route_offset_span = float(route_offset_span)
        self.route_offset_context_radius = int(
            route_offset_context_radius
        )
        self.interpolate_route_context = bool(interpolate_route_context)
        self.soft_route_power = float(soft_route_power)
        if self.soft_route_power < 0.0:
            raise ValueError("soft_route_power must be nonnegative")
        self.register_buffer(
            "wide_offsets",
            torch.arange(
                -wide_span,
                wide_span + 0.5 * wide_step,
                wide_step,
                dtype=torch.float32,
            ),
            persistent=False,
        )
        if use_route_residual:
            self.enable_route_residual(route_residual_span)
        if use_wide_cascade:
            self.enable_wide_cascade(wide_span, wide_step)
        if use_dual_precision:
            self.enable_dual_precision()
        if use_route_offset_residual:
            self.enable_route_offset_residual(
                route_offset_span, route_offset_context_radius
            )

    def enable_route_residual(self, span: float = 1.0) -> None:
        """Enable a bounded continuous correction without adding candidates."""
        self.route_residual_span = float(span)
        if self.route_residual_head is None:
            self.route_residual_head = nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.GELU(),
                nn.Linear(self.d_model, 1),
                nn.Tanh(),
            ).to(self.theta_grid.device)
            self.route_blend_head = nn.Sequential(
                nn.Linear(self.d_model, self.d_model // 2),
                nn.GELU(),
                nn.Linear(self.d_model // 2, 1),
            ).to(self.theta_grid.device)
            nn.init.zeros_(self.route_residual_head[-2].weight)
            nn.init.zeros_(self.route_residual_head[-2].bias)
            nn.init.zeros_(self.route_blend_head[-1].weight)
            nn.init.constant_(self.route_blend_head[-1].bias, -3.0)
        self.use_route_residual = True

    def enable_wide_cascade(
        self, span: float = 1.05, step: float = 0.15
    ) -> None:
        """Add a 15-atom wide physical pass before the original fine pass."""
        offsets = torch.arange(
            -float(span),
            float(span) + 0.5 * float(step),
            float(step),
            device=self.theta_grid.device,
            dtype=torch.float32,
        )
        if offsets.numel() != 15:
            raise ValueError(
                "wide cascade must use exactly 15 atoms; "
                f"got {offsets.numel()}"
            )
        self.wide_span = float(span)
        self.wide_step = float(step)
        self.wide_offsets = offsets
        if self.wide_head is None:
            self.wide_head = nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.GELU(),
                nn.Linear(self.d_model, 1),
            ).to(self.theta_grid.device)
            self.wide_head.load_state_dict(self.local_head.state_dict())
        self.use_wide_cascade = True

    def enable_dual_precision(self) -> None:
        """Split robust recentering from the final high-resolution branch."""
        if self.precision_local_head is None:
            self.precision_local_steer_proj = copy.deepcopy(
                self.local_steer_proj
            )
            self.precision_offset_embed = copy.deepcopy(self.offset_embed)
            self.precision_local_cross = copy.deepcopy(self.local_cross)
            self.precision_support_transformer = copy.deepcopy(
                self.support_transformer
            )
            self.precision_local_head = copy.deepcopy(self.local_head)
            self.precision_correction_gate_logit = nn.Parameter(
                self.correction_gate_logit.detach().clone()
            )
        self.use_dual_precision = True

    def enable_route_offset_residual(
        self, span: float = 0.5, context_radius: int = 0
    ) -> None:
        """Add a zero-initialized nonlinear correction to the base offset."""
        self.route_offset_span = float(span)
        context_radius = int(context_radius)
        if context_radius < 0:
            raise ValueError("route offset context radius must be nonnegative")
        self.route_offset_context_radius = context_radius
        if self.route_offset_head is None:
            self.route_offset_head = nn.Sequential(
                nn.Linear(
                    (2 * context_radius + 1) * self.d_model,
                    self.d_model,
                ),
                nn.GELU(),
                nn.Linear(self.d_model, 1),
                nn.Tanh(),
            ).to(self.theta_grid.device)
            nn.init.zeros_(self.route_offset_head[-2].weight)
            nn.init.zeros_(self.route_offset_head[-2].bias)
        self.use_route_offset_residual = True

    def _steering(self, angle: torch.Tensor) -> torch.Tensor:
        phase = (
            -1j
            * 2.0
            * math.pi
            * self.d_lambda
            * self.array_position
            * torch.sin(torch.deg2rad(angle))[..., None]
        )
        return torch.exp(phase)

    def _snapshot_coordinate(
        self, snapshots: torch.Tensor | float | int, batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        value = torch.as_tensor(
            snapshots, dtype=torch.float32, device=device
        ).reshape(-1)
        if value.numel() == 1:
            value = value.expand(batch_size)
        if value.numel() != batch_size:
            raise ValueError(
                f"Expected {batch_size} snapshot values, got {value.numel()}"
            )
        value = value.clamp(20.0, 1000.0)
        low = math.log(20.0)
        span = math.log(1000.0) - low
        return (2.0 * (torch.log(value) - low) / span - 1.0)[:, None]

    def encode_base(
        self,
        in_mat: torch.Tensor,
        snapshots: torch.Tensor | float | int | None = None,
    ) -> Dict[str, torch.Tensor]:
        if in_mat.ndim != 4 or tuple(in_mat.shape[1:]) != (
            2,
            self.N,
            self.N,
        ):
            raise ValueError(
                f"Expected [B,2,{self.N},{self.N}], got {tuple(in_mat.shape)}"
            )
        batch_size = in_mat.shape[0]
        query = _steering_query(self, in_mat)

        covariance_input = torch.cat(
            [in_mat[:, 0], in_mat[:, 1]], dim=1
        )
        covariance_feature = self.base.cov_proj(
            covariance_input
        ).permute(0, 2, 1)
        del covariance_input
        covariance_feature = self.base.cov_encoder(covariance_feature)
        covariance_grid = self.base.cov_cross(
            query, covariance_feature
        )

        if getattr(self, '_encoder_options', {}).get('vectorized_lags', False):
            lag = vectorized_covariance_to_lags(in_mat, self)
        else:
            lag = covariance_to_lags(in_mat)
        lag_feature = self.base.lag_proj(lag)
        lag_grid = self.base.lag_to_grid(
            lag_feature
        ).permute(0, 2, 1)

        del lag, lag_feature
        feature = (
            torch.sigmoid(self.base.gate_cov) * covariance_grid
            + torch.sigmoid(self.base.gate_lag) * lag_grid
        )
        del covariance_grid, lag_grid
        feature = self.base.fusion_norm(feature)
        feature = feature + self.base.fusion_mlp(feature)
        snapshot_coordinate = None
        if self.use_snapshot_conditioning:
            if snapshots is None:
                raise ValueError(
                    "snapshots are required by the conditioned retrieval model"
                )
            snapshot_coordinate = self._snapshot_coordinate(
                snapshots, batch_size, in_mat.device
            )
            film = self.snapshot_conditioner(snapshot_coordinate)
            scale, shift = film.chunk(2, dim=1)
            feature = feature * (
                1.0 + 0.25 * torch.tanh(scale)[:, None, :]
            ) + 0.25 * shift[:, None, :]
        feature = self.base.pos_encoder(feature)
        feature = self.base.grid_transformer(feature)
        feature = self.base.shrinkage(feature)

        role = getattr(self, '_encoder_output_role', None)
        location_logits = (self.base.loc_head(feature).squeeze(-1)
                           if role != 'localization' else
                           in_mat.new_empty((batch_size, 0)))
        if self.use_snapshot_conditioning:
            affine = self.snapshot_logit_affine(snapshot_coordinate)
            log_scale, bias = affine.unbind(dim=1)
            location_logits = location_logits * torch.exp(
                0.25 * torch.tanh(log_scale)
            )[:, None] + bias[:, None]
        if role == 'retrieval':
            # Consume the frozen grid now; retain only the small proposal scores.
            return {'location_logits': location_logits,
                    'proposal_scores': self._proposal_generator(location_logits, feature)}
        base_offset = (
            self.base.offset_head(feature).squeeze(-1)
            * self.base.offset_scale
        )
        covariance = torch.complex(in_mat[:, 0], in_mat[:, 1])
        return {
            "location_logits": location_logits,
            "base_offset": base_offset,
            "grid_feature": feature,
            "covariance_feature": covariance_feature,
            "covariance": covariance,
        }

    def forward(
        self,
        in_mat: torch.Tensor,
        snapshots: torch.Tensor | float | int | None = None,
    ) -> Dict[str, torch.Tensor]:
        return self.encode_base(in_mat, snapshots=snapshots)

    @torch.no_grad()
    def predicted_supports(
        self,
        encoded: Dict[str, torch.Tensor],
        max_sources: int | None = None,
    ) -> List[torch.Tensor]:
        maximum = self.max_sources if max_sources is None else max_sources
        probability = torch.sigmoid(encoded["location_logits"])
        pooled = F.max_pool1d(
            probability.unsqueeze(1),
            kernel_size=3,
            stride=1,
            padding=1,
        ).squeeze(1)
        local = probability * (probability == pooled)
        local_np = local.detach().cpu().numpy()
        minimum = float(self.count_min_threshold)
        relative = float(self.count_relative_alpha)
        selected = [select_calibrated_peaks(row, min_threshold=minimum,
                    relative_alpha=relative, max_sources=int(maximum)) for row in local_np]
        lengths = [len(row) for row in selected]
        # All selected indices cross the device boundary once. Scene views
        # retain their original increasing index order and empty scenes.
        flat = np.concatenate(selected) if sum(lengths) else np.empty(0, np.int64)
        packed = torch.as_tensor(flat, dtype=torch.long, device=probability.device)
        return list(packed.split(lengths))

    @staticmethod
    def _route_slots(
        support_batch: torch.Tensor,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        counts = torch.bincount(
            support_batch, minlength=batch_size
        )
        maximum = int(counts.max().item())
        starts = torch.cumsum(counts, dim=0) - counts
        slot = (
            torch.arange(
                support_batch.numel(),
                device=support_batch.device,
            )
            - torch.repeat_interleave(starts, counts)
        )
        return slot.long(), counts.long(), maximum

    def refine_routes(
        self,
        encoded: Dict[str, torch.Tensor],
        support_batch: torch.Tensor,
        support_index: torch.Tensor,
        *,
        detach_base: bool = True,
        build_reconstruction: bool = False,
    ) -> Dict[str, torch.Tensor]:
        batch_size = encoded["location_logits"].shape[0]
        if support_index.numel() == 0:
            empty = encoded["location_logits"].new_empty((0, self.M))
            return {
                "fine_logits": empty,
                "fine_probability": empty,
                "base_angle": empty.new_empty((0,)),
                "route_base_angle": empty.new_empty((0,)),
                "route_offset_residual": empty.new_empty((0,)),
                "wide_logits": empty,
                "wide_residual": empty.new_empty((0,)),
                "raw_residual": empty.new_empty((0,)),
                "conditional_angle": empty.new_empty((0,)),
                "route_residual": empty.new_empty((0,)),
                "route_blend": empty.new_empty((0,)),
                "R_hat": encoded["covariance"].new_zeros(
                    batch_size, self.N, self.N
                ),
                "source_power": empty.new_empty((0,)),
                "noise_power": empty.new_zeros((batch_size,)),
                "local_candidate_count": 0,
            }

        grid_feature = encoded["grid_feature"]
        covariance_feature = encoded["covariance_feature"]
        base_offset = encoded["base_offset"]
        covariance = encoded["covariance"]
        if detach_base:
            grid_feature = grid_feature.detach()
            covariance_feature = covariance_feature.detach()
            base_offset = base_offset.detach()

        grid_context = grid_feature[support_batch, support_index]
        route_offset_residual = torch.zeros(
            support_index.numel(),
            dtype=grid_context.dtype,
            device=grid_context.device,
        )
        if self.use_route_offset_residual:
            if self.route_offset_head is None:
                raise RuntimeError(
                    "route offset residual head is not initialized"
                )
            radius = self.route_offset_context_radius
            shifts = torch.arange(
                -radius,
                radius + 1,
                device=support_index.device,
            )
            context_index = (
                support_index[:, None] + shifts[None, :]
            ).clamp(0, self.K - 1)
            route_context = grid_feature[
                support_batch[:, None], context_index
            ].flatten(1)
            route_offset_residual = (
                self.route_offset_head(route_context).squeeze(-1)
                * self.route_offset_span
            )
        if self.soft_route_power > 0.0:
            soft_shifts = torch.arange(
                -2, 3, device=support_index.device
            )
            soft_index = (
                support_index[:, None] + soft_shifts[None, :]
            ).clamp(0, self.K - 1)
            soft_probability = torch.sigmoid(
                encoded["location_logits"][
                    support_batch[:, None], soft_index
                ]
            )
            soft_weight = soft_probability.pow(self.soft_route_power)
            soft_weight = soft_weight / soft_weight.sum(
                dim=1, keepdim=True
            ).clamp_min(1.0e-8)
            soft_candidate_angle = (
                self.theta_grid[soft_index]
                + base_offset[support_batch[:, None], soft_index]
            )
            route_base_angle = (
                soft_weight * soft_candidate_angle
            ).sum(dim=1) + route_offset_residual
            grid_context = (
                soft_weight[..., None]
                * grid_feature[support_batch[:, None], soft_index]
            ).sum(dim=1)
        else:
            route_base_angle = (
                self.theta_grid[support_index]
                + base_offset[support_batch, support_index]
                + route_offset_residual
            )
        if self.interpolate_route_context and self.soft_route_power <= 0.0:
            route_coordinate = (
                (route_base_angle - self.theta_grid[0])
                / (self.theta_grid[1] - self.theta_grid[0])
            ).clamp(0.0, float(self.K - 1))
            route_left = torch.floor(route_coordinate).long()
            route_right = (route_left + 1).clamp_max(self.K - 1)
            route_fraction = (
                route_coordinate - route_left.to(route_coordinate.dtype)
            )[:, None]
            grid_context = (
                grid_feature[support_batch, route_left]
                + route_fraction
                * (
                    grid_feature[support_batch, route_right]
                    - grid_feature[support_batch, route_left]
                )
            )
        support_base = 2.0 * grid_context
        slot, counts, maximum = self._route_slots(
            support_batch, batch_size
        )
        padding_mask = (
            torch.arange(
                maximum, device=support_batch.device
            )[None, :]
            >= counts[:, None]
        )
        active_scene = counts > 0

        def build_tokens(
            centre: torch.Tensor,
            offsets: torch.Tensor,
            route_support_base: torch.Tensor,
            steer_proj: nn.Module,
            offset_embed: nn.Module,
            cross_attention: nn.Module,
        ) -> torch.Tensor:
            candidate_angle = centre[:, None] + offsets[None, :]
            steering = self._steering(candidate_angle)
            steering_feature = torch.cat(
                [steering.real, steering.imag], dim=-1
            )
            token = steer_proj(steering_feature)
            offset_scale = offsets.abs().max().clamp_min(1.0e-8)
            embedding = offset_embed(
                (offsets / offset_scale).view(1, offsets.numel(), 1)
            )
            token = (
                token + route_support_base[:, None, :] + embedding
            )
            return cross_attention(
                token, covariance_feature[support_batch]
            )

        def contextualize(
            token: torch.Tensor,
            route_support_base: torch.Tensor,
            support_transformer: nn.Module,
        ) -> torch.Tensor:
            summary = token.mean(dim=1) + route_support_base
            padded = summary.new_zeros(
                batch_size, maximum, self.d_model
            )
            padded[support_batch, slot] = summary
            transformed = padded.clone()
            transformed[active_scene] = support_transformer(
                padded[active_scene],
                src_key_padding_mask=padding_mask[active_scene],
            )
            return transformed[support_batch, slot]

        wide_logits = route_base_angle.new_empty(
            (support_index.numel(), 0)
        )
        wide_residual = torch.zeros_like(route_base_angle)
        base_angle = route_base_angle
        if self.use_wide_cascade:
            if self.wide_head is None:
                raise RuntimeError("wide cascade head is not initialized")
            wide_token = build_tokens(
                route_base_angle,
                self.wide_offsets,
                support_base,
                self.local_steer_proj,
                self.offset_embed,
                self.local_cross,
            )
            wide_context = contextualize(
                wide_token, support_base, self.support_transformer
            )
            wide_logits = self.wide_head(
                wide_token + wide_context[:, None, :]
            ).squeeze(-1)
            del wide_token, wide_context
            wide_probability = F.softmax(wide_logits, dim=-1)
            wide_residual = (
                wide_probability * self.wide_offsets[None, :]
            ).sum(dim=-1)
            del wide_probability
            base_angle = route_base_angle + wide_residual

        local_support_base = support_base
        blend = self.recenter_grid_context_blend
        if self.use_wide_cascade and blend > 0.0:
            grid_coordinate = (
                (base_angle - self.theta_grid[0])
                / (self.theta_grid[1] - self.theta_grid[0])
            ).clamp(0.0, float(self.K - 1))
            grid_left = torch.floor(grid_coordinate).long()
            grid_right = (grid_left + 1).clamp_max(self.K - 1)
            grid_fraction = (
                grid_coordinate - grid_left.to(grid_coordinate.dtype)
            )[:, None]
            recentered_context = (
                grid_feature[support_batch, grid_left]
                + grid_fraction
                * (
                    grid_feature[support_batch, grid_right]
                    - grid_feature[support_batch, grid_left]
                )
            )
            local_grid_context = (
                (1.0 - blend) * grid_context
                + blend * recentered_context
            )
            local_support_base = 2.0 * local_grid_context

        if self.use_dual_precision:
            if (
                self.precision_local_steer_proj is None
                or self.precision_offset_embed is None
                or self.precision_local_cross is None
                or self.precision_support_transformer is None
                or self.precision_local_head is None
                or self.precision_correction_gate_logit is None
            ):
                raise RuntimeError("dual precision branch is not initialized")
            local_steer_proj = self.precision_local_steer_proj
            local_offset_embed = self.precision_offset_embed
            local_cross = self.precision_local_cross
            local_support_transformer = (
                self.precision_support_transformer
            )
            local_head = self.precision_local_head
            local_correction_gate_logit = (
                self.precision_correction_gate_logit
            )
        else:
            local_steer_proj = self.local_steer_proj
            local_offset_embed = self.offset_embed
            local_cross = self.local_cross
            local_support_transformer = self.support_transformer
            local_head = self.local_head
            local_correction_gate_logit = self.correction_gate_logit

        local_token = build_tokens(
            base_angle,
            self.fine_offsets,
            local_support_base,
            local_steer_proj,
            local_offset_embed,
            local_cross,
        )
        support_context = contextualize(
            local_token,
            local_support_base,
            local_support_transformer,
        )

        local_joint = local_token + support_context[:, None, :]
        del local_token
        fine_logits = local_head(local_joint).squeeze(-1)
        del local_joint
        fine_probability = F.softmax(fine_logits, dim=-1)
        raw_residual = (
            fine_probability * self.fine_offsets[None, :]
        ).sum(dim=-1)
        correction_gate = torch.tanh(local_correction_gate_logit)
        local_residual = correction_gate * raw_residual
        route_residual = torch.zeros_like(raw_residual)
        route_blend = torch.zeros_like(raw_residual)
        if self.use_route_residual:
            if self.route_residual_head is None or self.route_blend_head is None:
                raise RuntimeError("route residual modules are not initialized")
            route_residual = (
                self.route_residual_head(support_context).squeeze(-1)
                * self.route_residual_span
            )
            route_blend = torch.sigmoid(
                self.route_blend_head(support_context).squeeze(-1)
            )
        combined_residual = (
            (1.0 - route_blend) * local_residual
            + route_blend * route_residual
        )
        conditional_angle = base_angle + combined_residual

        reconstructed = covariance.new_zeros(
            batch_size, self.N, self.N
        )
        source_power = raw_residual.new_zeros(
            support_index.numel()
        )
        noise_power = raw_residual.new_zeros(batch_size)
        if build_reconstruction:
            power_logits = self.power_head(
                support_context
            ).squeeze(-1)
            padded_power = power_logits.new_full(
                (batch_size, maximum), -1.0e4
            )
            padded_power[support_batch, slot] = power_logits
            noise_logits = self.noise_head(
                covariance_feature.mean(dim=1)
            )
            all_power = F.softmax(
                torch.cat([padded_power, noise_logits], dim=1),
                dim=1,
            )
            source_power = all_power[support_batch, slot]
            noise_power = all_power[:, -1]
            final_steering = (
                self._steering(conditional_angle)
                / math.sqrt(float(self.N))
            )
            atom = (
                final_steering[..., :, None]
                * final_steering.conj()[..., None, :]
            )
            contribution = source_power[:, None, None].to(
                atom.dtype
            ) * atom
            reconstructed.index_add_(
                0, support_batch, contribution
            )
            identity = torch.eye(
                self.N,
                dtype=reconstructed.dtype,
                device=reconstructed.device,
            )
            reconstructed = (
                reconstructed
                + noise_power[:, None, None]
                * identity.view(1, self.N, self.N)
                / float(self.N)
            )

        return {
            "fine_logits": fine_logits,
            "fine_probability": fine_probability,
            "base_angle": base_angle,
            "route_base_angle": route_base_angle,
            "route_offset_residual": route_offset_residual,
            "wide_logits": wide_logits,
            "wide_residual": wide_residual,
            "raw_residual": raw_residual,
            "conditional_angle": conditional_angle,
            "route_residual": route_residual,
            "route_blend": route_blend,
            "R_hat": reconstructed,
            "source_power": source_power,
            "noise_power": noise_power,
            "local_candidate_count": int(
                support_index.numel()
                * (self.M + (self.wide_offsets.numel() if self.use_wide_cascade else 0))
            ),
        }


def flatten_supports(
    supports: Sequence[torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    lengths = [support.numel() for support in supports]
    total = sum(lengths)
    if not total:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty
    # output_size avoids an implicit synchronization to discover repeat size.
    counts = torch.tensor(lengths, dtype=torch.long, device=device)
    scene = torch.arange(len(supports), dtype=torch.long, device=device)
    batches = torch.repeat_interleave(scene, counts, output_size=total)
    return batches, torch.cat([s for s in supports if s.numel()])


def continuous_nms(
    angles: torch.Tensor,
    scores: torch.Tensor,
    minimum_separation: float = 0.6,
) -> torch.Tensor:
    if angles.numel() == 0:
        return angles.detach().cpu()
    angle_np = angles.detach().float().cpu().numpy()
    score_np = scores.detach().float().cpu().numpy()
    order = np.argsort(score_np, kind="stable")[::-1]
    kept: List[float] = []
    for item in order:
        angle = float(angle_np[item])
        if all(
            abs(angle - previous) >= minimum_separation
            for previous in kept
        ):
            kept.append(angle)
    return torch.tensor(sorted(kept), dtype=torch.float32)
