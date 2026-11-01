"""Checkpoint conversion from a 0.5-degree grid to a 1-degree grid.

The conversion retains every second physical grid node and adjusts the
coarse offset scale and positional encoding. Wide and fine local grids are
preserved. This operation changes the network representation."""
from __future__ import annotations

import copy
from collections.abc import Mapping

import torch

_FORMAT = "pgbsa_shared_delta_v3"
_ROOTS = {"original", "localizer", "retrieval", "close_pair"}
_SOURCE_PARAMETERS = 3_986_898  # canonical pre-inference-compaction payload
_REMOVED_PARAMETERS = 9_240


def _migration_plan(payload, factor):
    if isinstance(factor, bool) or not isinstance(factor, int) or factor != 2:
        raise ValueError("Only factor=2 grid conversion is supported")
    if not isinstance(payload, Mapping) or payload.get("format") != _FORMAT:
        raise ValueError("Expected canonical pgbsa_shared_delta_v3 payload")
    if "grid_resolution" in payload or "coarse_offset_scale" in payload:
        raise ValueError("Grid resolution was already migrated")
    if any(key in payload for key in ("compression", "grid_reduction", "attention_compression")):
        raise ValueError("Migrate the unreduced canonical rank-16 architecture")
    shared = payload.get("shared_delta", {})
    if (shared.get("rank") != 16 or shared.get("owner") != "original.base"
            or shared.get("canonical_aliases_omitted") is not True
            or payload.get("shared_backbone_owner") != "original.base"):
        raise ValueError("Expected rank-16 canonical single-owner shared weights")
    configs = payload.get("configs", {})
    if not isinstance(configs, Mapping) or set(configs) != _ROOTS:
        raise ValueError("Expected the four original model configurations")
    for name, config in configs.items():
        if not isinstance(config, Mapping) or any(config.get(k) != v for k, v in {
            "sensor_count": 16, "theta_min": -70., "theta_max": 70.,
            "coarse_step": .5, "d_model": 128, "support_layers": 2,
        }.items()):
            raise ValueError(name + ": incompatible source grid/architecture")
    metadata = payload.get("metadata", {})
    count = metadata.get("unique_parameters")
    if isinstance(count, bool) or not isinstance(count, int) or count != _SOURCE_PARAMETERS:
        raise ValueError("Unexpected canonical source parameter count")
    state = payload.get("state", {})
    if not isinstance(state, Mapping) or set(state) != {"original", "dual", "generator", "scorer", "offset"}:
        raise ValueError("Expected canonical source state groups")
    if any(not isinstance(s, Mapping) for s in state.values()):
        raise ValueError("State groups must be mappings")
    if any(k.startswith("localizer.base.") or k.endswith(".shared_weight") for k in state["dual"]):
        raise ValueError("Canonical state must omit duplicate owner aliases")
    # Tuple: state group, full key, old shape, sampling dimension, parameter.
    plan = [
        ("original", "theta_grid", (281,), 0, False),
        ("original", "base_steering", (32, 281), 1, False),
        ("original", "base.pos_encoder.pe", (1, 281, 128), 1, False),
        ("original", "base.lag_to_grid.0.weight", (281, 31), 0, True),
        ("original", "base.lag_to_grid.0.bias", (281,), 0, True),
    ]
    for prefix in ("localizer.", "high_snapshot_retrieval.", "close_pair_localizer."):
        plan.extend((
            ("dual", prefix + "theta_grid", (281,), 0, False),
            ("dual", prefix + "base_steering", (32, 281), 1, False),
        ))
    plan.append(("dual", "retrieval.theta_grid", (281,), 0, False))
    for prefix in ("high_snapshot_retrieval.", "close_pair_localizer."):
        plan.extend((
            ("dual", prefix + "base.pos_encoder.pe", (1, 281, 128), 1, False),
            ("dual", prefix + "base.lag_to_grid.0.delta_left", (281, 16), 0, True),
            ("dual", prefix + "base.lag_to_grid.0.bias", (281,), 0, True),
        ))
        right = state["dual"].get(prefix + "base.lag_to_grid.0.delta_right")
        if not torch.is_tensor(right) or tuple(right.shape) != (16, 31):
            raise ValueError(prefix + "lag delta_right has an incompatible shape")
    expected_grid = torch.arange(-70., 70.0001, .5, dtype=torch.float32)
    changed = {(group, key) for group, key, *_ in plan}
    removed = 0
    for group, key, shape, dim, parameter in plan:
        value = state[group].get(key)
        if (not torch.is_tensor(value) or tuple(value.shape) != shape
                or value.dtype != torch.float32 or value.device.type != "cpu"
                or not bool(torch.isfinite(value).all())):
            raise ValueError(group + "/" + key + ": invalid shape/dtype/device/values")
        if key.endswith("theta_grid") and not torch.equal(value, expected_grid):
            raise ValueError(group + "/" + key + ": grid nodes do not match the source config")
        if parameter:
            removed += value.numel() - value.narrow(dim, 0, 1).numel() * 141
    for group, values in state.items():
        for key, value in values.items():
            if not torch.is_tensor(value):
                raise ValueError("Canonical state entries must be tensors")
            if 281 in value.shape and (group, key) not in changed:
                raise ValueError("Unrecognized grid-dependent state: " + group + "/" + key)
    if removed != _REMOVED_PARAMETERS:
        raise ValueError("Unexpected parameter count after rank-16 grid conversion")
    return plan


def coarsen_grid_payload(payload, factor=2):
    """Return a copied canonical checkpoint on a 1-degree coarse grid.

    The conversion changes grid-dependent shapes and parameter counts while
    preserving shared-weight aliases and both local refinement stages."""
    plan = _migration_plan(payload, factor)
    result = copy.deepcopy(payload)
    for group, key, _, dim, _ in plan:
        value = payload["state"][group][key]
        slices = [slice(None)] * value.ndim
        slices[dim] = slice(None, None, factor)
        result["state"][group][key] = value[tuple(slices)].clone().contiguous()
    for config in result["configs"].values():
        config["coarse_step"] = 1.
    marker = dict(factor=2, source_step=.5, target_step=1., source_tokens=281,
                  target_tokens=141, coarse_offset_scale=.5,
                  position_encoding="subsample_old_physical_nodes",
                  parameters_removed=_REMOVED_PARAMETERS,
                  migration="canonical_rank16_even_physical_nodes_v1",
                  output_equivalence_claimed=False)
    result["grid_resolution"] = marker
    result["coarse_offset_scale"] = .5
    result["metadata"] = dict(result["metadata"],
        unique_parameters=_SOURCE_PARAMETERS - _REMOVED_PARAMETERS,
        grid_resolution_parameters_removed=_REMOVED_PARAMETERS,
        grid_resolution=dict(marker), neural_weights_unchanged=False,
        distillation_used=False)
    return result
