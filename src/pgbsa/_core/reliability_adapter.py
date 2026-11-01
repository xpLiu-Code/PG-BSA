"""Covariance-conditioned residual corrections to grid logits and offsets."""
from __future__ import annotations
import math
import torch
from torch import nn


def observation_features(x, grid):
    """Compute Toeplitz reliability and normalized Bartlett/Capon features.

    x contains real/imaginary covariance channels [B,2,N,N]; grid contains
    angles in degrees. Returns five features per angle and one equivalent
    snapshot count per scene. All quantities are derived from x.
    """
    c = torch.complex(x[:, 0], x[:, 1])
    n = c.shape[-1]
    ix = torch.arange(n, device=x.device)
    lag_index = (ix[:, None]-ix[None, :]+n-1).reshape(1, -1).expand(len(c), -1)
    counts = n-torch.arange(-(n-1), n, device=x.device).abs()
    lags = c.new_zeros(len(c), 2*n-1).scatter_add_(1, lag_index, c.flatten(1))/counts
    proj = lags.gather(1, lag_index).reshape_as(c)
    residual = (c-proj).abs().square().sum((1, 2))
    # Sum all leading Toeplitz submatrix energies analytically. This removes
    # dozens of small per-lag GPU launches from every inference batch.
    m = torch.arange(1, n+1, device=x.device, dtype=x.dtype)[:, None]
    ell = torch.arange(n, device=x.device, dtype=x.dtype)[None]
    multiplier = torch.where(m == n, 1., 2.)
    weights = (multiplier*(m-ell).clamp_min(0)/m).sum(0)
    weights[1:] *= 2.
    variance = (lags[:, n-1:].abs().square()*weights).sum(1)
    tr = c.diagonal(dim1=-2, dim2=-1).real.sum(1)
    equivalent = ((tr.square()-variance).clamp_min(1e-12)/residual.clamp_min(1e-12)).clamp(8., 8192.)
    a = torch.exp(-1j*math.pi*torch.arange(n, device=x.device)[:, None]*torch.sin(torch.deg2rad(grid))[None])/math.sqrt(n)
    bartlett = torch.einsum('ng,bnm,mg->bg', a.conj(), c, a).real.clamp_min(1e-9)
    eye = torch.eye(n, device=x.device, dtype=c.dtype)
    inv_a = torch.linalg.solve(c + .005*eye[None], a.expand(len(c), -1, -1))
    capon = torch.einsum('ng,bng->bg', a.conj(), inv_a).real.clamp_min(1e-9).reciprocal()
    def standard(v):
        z = v.log()
        return (z-z.mean(1, keepdim=True))/z.std(1, keepdim=True).clamp_min(.05)
    extras = torch.stack((standard(bartlett), standard(capon),
        (equivalent.log()/math.log(1000.))[:, None].expand_as(bartlett),
        residual.sqrt()[:, None].expand_as(bartlett),
        (grid/70.)[None].expand_as(bartlett)), -1)
    return extras, equivalent


class ReliabilityAdapter(nn.Module):
    """Small shared grid head with local context and bounded logit/offset residuals."""
    def __init__(self, width=48):
        super().__init__()
        self.norm = nn.LayerNorm(128)
        self.input = nn.Linear(135, width)
        self.local = nn.Conv1d(width, width, 7, padding=3, groups=width)
        self.mix = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 2))
        nn.init.zeros_(self.mix[-1].weight)
        nn.init.zeros_(self.mix[-1].bias)

    def forward(self, features, logits, offsets, extras, equivalent, strength=1.):
        value = torch.cat((self.norm(features), logits[..., None]/8., offsets[..., None], extras), -1)
        h = torch.nn.functional.gelu(self.input(value))
        h = h+torch.nn.functional.gelu(self.local(h.transpose(1, 2)).transpose(1, 2))
        residual = self.mix(h)
        # A covariance-only gate preserves high-reliability observations exactly.
        gate = ((math.log(700.)-equivalent.log())/math.log(700./128.)).clamp(0., 1.)[:, None]
        delta_logits = 2.*torch.tanh(residual[..., 0])*gate*strength
        delta_offsets = .25*torch.tanh(residual[..., 1])*gate*strength
        return logits+delta_logits, offsets+delta_offsets
