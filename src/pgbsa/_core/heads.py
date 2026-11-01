"""Proposal, candidate scoring and offset heads, and set selection."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Tuple
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

FINE_OFFSETS = np.arange(-0.35, 0.3501, 0.05, dtype=np.float32)
WIDE_OFFSETS = np.arange(-0.7, 0.7001, 0.1, dtype=np.float32)

class GlobalCandidateGenerator(nn.Module):
    """Propose additional directions from global retrieval features.

    Proposal scores expand the candidate set independently of the retrieval
    logits. Candidate scoring and set selection determine the retained angles."""
    architecture = 'global_candidate_generator_v1'
    def __init__(self, feature_dim=128, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim + 2, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
    def forward(self, base_logits, grid_feature):
        p = torch.sigmoid(base_logits.float())
        n = grid_feature.shape[1]
        pos = torch.linspace(-1., 1., n, device=grid_feature.device).view(1,n,1).expand(grid_feature.shape[0],-1,-1)
        x = torch.cat((grid_feature.float(), p.unsqueeze(-1), pos), dim=-1)
        return self.net(x).squeeze(-1)


class CandidateScorer(nn.Module):
    architecture_version = "route_candidate_set_decoder_v1"
    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(0.05),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class DecoderBundle:
    scorer: torch.nn.Module
    scorer_mean: np.ndarray
    scorer_std: np.ndarray
    offset: torch.nn.Module
    offset_mean: np.ndarray
    offset_std: np.ndarray
    minimum_separation: float = 1.0
    topk: int = 5
    apply_under_count: int = 17
    merge_threshold: float = 1.6
    merge_probability_gate: float = 0.57


def replace_profiles(features, raw_angles, base, centre, fine, wide, *, offsets=None):
    """Resample differentiable feature values at fixed candidate positions."""
    if offsets is None:
        fine_offsets = torch.as_tensor(FINE_OFFSETS, device=base.device)
        wide_offsets = torch.as_tensor(WIDE_OFFSETS, device=base.device)
    else:
        fine_offsets, wide_offsets = offsets
    fj = (fine_offsets[None, None, :] - (raw_angles-base[:, None])[:, :, None]).abs().argmin(-1)
    wj = (wide_offsets[None, None, :] - (raw_angles-centre[:, None])[:, :, None]).abs().argmin(-1)
    def profile(p, ix):
        padded = F.pad(p, (1, 1))
        val = p.gather(1, ix)
        prom = val - torch.maximum(padded.gather(1, ix), padded.gather(1, ix+2))
        maximum = p.amax(1)[:, None].expand_as(val)
        entropy = -(p*p.clamp_min(1e-8).log()).sum(1)/math.log(15.)
        return torch.stack((val, prom, maximum, entropy[:, None].expand_as(val)), -1)
    profiles = torch.cat((profile(fine, fj), profile(wide, wj)), -1)
    return torch.cat((features[:, :, :25], profiles, features[:, :, 33:]), -1)


def decode_fixed_count(angles: np.ndarray, scores: np.ndarray, count: int, minimum_separation: float = 1.0) -> np.ndarray:
    """Maximum-score exact-cardinality interval scheduling."""
    if count <= 0 or len(angles) == 0:
        return np.empty(0, np.float32)
    order = np.argsort(angles); a = np.asarray(angles, np.float32)[order]; s = np.asarray(scores, np.float32)[order]
    # Remove near-identical copies, retaining the strongest candidate.
    keep = []
    for i in range(len(a)):
        if keep and abs(float(a[i] - a[keep[-1]])) < 0.03:
            if s[i] > s[keep[-1]]: keep[-1] = i
        else: keep.append(i)
    a = a[keep]; s = s[keep]; n = len(a); count = min(int(count), n)
    previous = np.searchsorted(a, a - float(minimum_separation) + 1e-7, side="right") - 1
    neg = -1.0e30
    dp = np.full((n + 1, count + 1), neg, np.float64); take = np.zeros((n + 1, count + 1), np.bool_); dp[:, 0] = 0.0
    for i in range(1, n + 1):
        dp[i] = dp[i-1]
        pi = int(previous[i-1]) + 1
        for k in range(1, count + 1):
            val = dp[pi, k-1] + float(s[i-1])
            if val > dp[i, k]: dp[i, k] = val; take[i, k] = True
    k = count
    while k > 0 and dp[n, k] <= neg / 2: k -= 1
    chosen = []; i = n
    while i > 0 and k > 0:
        if take[i, k]: chosen.append(i-1); i = int(previous[i-1]) + 1; k -= 1
        else: i -= 1
    return np.sort(a[np.asarray(chosen[::-1], dtype=np.int64)])


def merge_close_route_peaks(
    angles: torch.Tensor | np.ndarray | Iterable[float],
    scores: torch.Tensor | np.ndarray | Iterable[float],
    *,
    merge_threshold: float = 1.6,
    weight_power: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Merge adjacent route hypotheses using probability-weighted centroids.

    Angles are sorted before grouping. Returned angles and scores use CPU
    float32 tensors, matching continuous_nms."""
    if torch.is_tensor(angles):
        a = angles.detach().float().cpu().numpy().reshape(-1)
    else:
        a = np.asarray(list(angles) if not isinstance(angles, np.ndarray) else angles, dtype=np.float32).reshape(-1)
    if torch.is_tensor(scores):
        s = scores.detach().float().cpu().numpy().reshape(-1)
    else:
        s = np.asarray(list(scores) if not isinstance(scores, np.ndarray) else scores, dtype=np.float32).reshape(-1)
    if a.size != s.size:
        raise ValueError(f"angles/scores length mismatch: {a.size} vs {s.size}")
    if a.size == 0:
        return torch.empty(0, dtype=torch.float32), torch.empty(0, dtype=torch.float32)
    if merge_threshold <= 0:
        order = np.argsort(a)
        return torch.from_numpy(a[order].astype(np.float32)), torch.from_numpy(s[order].astype(np.float32))
    order = np.argsort(a, kind="stable")
    sa = a[order]; ss = s[order]
    groups: list[tuple[int, int]] = []
    begin = 0
    for j in range(1, sa.size):
        if float(sa[j] - sa[j - 1]) >= float(merge_threshold):
            groups.append((begin, j)); begin = j
    groups.append((begin, sa.size))
    out_a: list[float] = []; out_s: list[float] = []
    power = max(0.0, float(weight_power))
    for lo, hi in groups:
        aa = sa[lo:hi]; sc = ss[lo:hi]
        # Probabilities are in [0,1], but guard against malformed caches.
        w = np.maximum(np.nan_to_num(sc, nan=0.0), 1.0e-5) ** power
        out_a.append(float(np.sum(w * aa) / np.sum(w)))
        out_s.append(float(np.max(sc)))
    return torch.tensor(out_a, dtype=torch.float32), torch.tensor(out_s, dtype=torch.float32)


def merge_low_density_routes(
    angles: torch.Tensor,
    scores: torch.Tensor,
    *,
    support_count: int,
    max_support_count: int = 2,
    merge_threshold: float = 1.6,
    weight_power: float = 0.0,
    probability_gate: float | None = 0.57,
) -> torch.Tensor:
    """Consolidate route hypotheses in low-density scenes.

    Scenes exceeding max_support_count retain their original candidate list."""
    if int(support_count) > int(max_support_count):
        return angles.detach().float().cpu()
    # Apply the confidence gate before merging a possible close pair.
    # Passing None selects distance-only consolidation.
    if probability_gate is not None and int(support_count) == 2:
        if torch.is_tensor(scores):
            pmean = float(scores.detach().float().mean().cpu())
        else:
            pmean = float(np.asarray(list(scores), dtype=np.float32).mean())
        if pmean >= float(probability_gate):
            return angles.detach().float().cpu()
    merged, _ = merge_close_route_peaks(
        angles, scores, merge_threshold=merge_threshold, weight_power=weight_power
    )
    return merged
