"""Inference wrapper for covariance-conditioned feature residuals."""
from __future__ import annotations
import math
from pathlib import Path
import torch
from torch import nn
try:
    from .reliability_adapter import ReliabilityAdapter, observation_features
except ImportError:
    from reliability_adapter import ReliabilityAdapter, observation_features


class RuntimeReliabilityRefiner(nn.Module):
    """Apply bounded residual logits using covariance-derived reliability.

    The gate is clip((log(700)-log(T_equiv))/log(700/128), 0, 1).
    T_equiv is estimated from the observation covariance. Geometry tensors
    are retained and the default residual strength is one."""
    def __init__(self, checkpoint, *, theta_grid=None, strength=1.):
        super().__init__()
        if not 0. <= strength <= 1.:
            raise ValueError('strength must lie within [0,1]')
        self.strength = float(strength)
        self.adapter = ReliabilityAdapter()
        payload = torch.load(Path(checkpoint), map_location='cpu', weights_only=True)
        if payload.get('metadata', {}).get('parameters') != 9618:
            raise ValueError('Expected the 9,618-parameter observation residual checkpoint')
        self.adapter.load_state_dict(payload['state_dict'], strict=True)
        grid = torch.linspace(-70,70,281) if theta_grid is None else theta_grid.detach().float().cpu()
        if grid.shape != (281,):
            raise ValueError('Expected 281 retrieval grid positions')
        self.register_buffer('theta_grid', grid.clone())
        self.metadata = dict(payload['metadata'], checkpoint_epoch=payload.get('epoch'),
            residual_strength=self.strength, minimum_predicted_route_count=None,
            reliability_full_strength_below=128., reliability_disabled_at_or_above=700.,
            bounded_logit_residual=2., original_features_and_offsets_unchanged=True)
        self.eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, encoded, x):
        if len(x) != len(encoded['location_logits']):
            raise ValueError('Encoding/covariance scene counts must match')
        if not len(x) or self.strength == 0.:
            return encoded, {'disabled': True}
        extras, equivalent = observation_features(x, self.theta_grid)
        logits, _ = self.adapter(encoded['grid_feature'], encoded['location_logits'],
            encoded['base_offset'], extras, equivalent, self.strength)
        result = dict(encoded)
        result['location_logits'] = logits
        gate = ((math.log(700.)-equivalent.log())/math.log(700./128.)).clamp(0.,1.)
        diagnostics = dict(equivalent_snapshots=equivalent, reliability_gate=gate,
            maximum_logit_residual=(logits-encoded['location_logits']).abs().amax(1))
        return result, diagnostics
