"""Set recovery metrics with optimal angular matching."""

import numpy as np
from scipy.optimize import linear_sum_assignment

FIELDS = (
    "true_count",
    "predicted_count",
    "matched_count",
    "scene_f1",
    "full_recovery",
    "count_accuracy",
    "ospa2",
    "undercount",
    "overcount",
    "false_alarm",
    "precision",
    "recall",
)


def _metric_row(truth, prediction, tolerance=0.5, cutoff=5.0):
    truth, prediction = (np.sort(np.asarray(a, dtype=float)) for a in (truth, prediction))
    assert truth.ndim == prediction.ndim == 1
    assert np.isfinite(truth).all() and np.isfinite(prediction).all()
    k, p = (len(truth), len(prediction))
    hits, assigned = (0, 0.0)
    if k and p:
        distance = np.abs(truth[:, None] - prediction[None, :])
        rows, cols = linear_sum_assignment((distance > tolerance).astype(float))
        hits = int(np.sum(distance[rows, cols] <= tolerance))
        cost = np.minimum(distance, cutoff) ** 2
        rows, cols = linear_sum_assignment(cost)
        assigned = float(cost[rows, cols].sum())
    f1 = 2.0 * hits / (k + p) if k + p else 1.0
    ospa = np.sqrt((assigned + cutoff**2 * abs(k - p)) / max(k, p)) if max(k, p) else 0.0
    return np.asarray(
        [k, p, hits, f1, k == p == hits, k == p, ospa, p < k, p > k, k == 0 and p > 0], dtype=float
    )


def metric_row(truth, prediction, tolerance=0.5, cutoff=5.0):
    base = _metric_row(truth, prediction, tolerance, cutoff)
    k, p, hits = base[:3]
    precision = hits / p if p else float(k == 0)
    recall = hits / k if k else float(p == 0)
    return np.r_[base, precision, recall]
