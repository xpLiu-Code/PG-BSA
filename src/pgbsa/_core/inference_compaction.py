"""Post-load compaction of frozen inference encoders.

Apply after loading weights and selecting the final device and dtype, before
creating CUDA graphs. Compaction folds convolution and batch normalization
and removes unused heads. Use a new instance for training, checkpoint
loading, or steering-grid changes."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.fusion import fuse_conv_bn_eval


def _parameter_summary(model):
    unique = {id(p): p for root in model.roots().values() for p in root.parameters()}
    return {
        "parameters": sum(p.numel() for p in unique.values()),
        "parameter_bytes": sum(p.numel() * p.element_size() for p in unique.values()),
    }


def compact_frozen_encoders(model):
    """Compact constants, unused heads and Conv1d/BatchNorm1d eval pairs.

    Each root receives a nonpersistent _frozen_base_query buffer [1,K,d_model].
    Roots sharing a base share that tensor. base.steer_proj becomes None; forward
    must consume the buffer before touching the projection. Root
    _encoder_output_role selects which unused output head may be removed; an
    unset role preserves both heads for optional compatibility paths.

    Returns JSON-compatible parameter/structure accounting, counting Parameter
    objects once. Repeated calls before graph construction are safe. The original
    and adapted roots keep their shared base; no backbone is duplicated.
    """
    if getattr(model, "_graphs", None):
        raise RuntimeError("Compact encoders before constructing CUDA graph runners")
    roots = (
        ("original", model.engine.original),
        ("adapted", model.engine.dual.localizer),
        ("retrieval", model.retrieval),
        ("close_pair", model.engine.dual.close_pair_localizer),
    )
    for name, root in roots:
        if any(module.training for module in root.modules()):
            raise ValueError(f"{name} must be entirely in eval mode")
        if any(parameter.requires_grad for parameter in root.parameters()):
            raise ValueError(f"{name} must have frozen parameters")
        if getattr(root, "use_snapshot_conditioning", False):
            raise ValueError("Compaction requires unconditioned encoders")
        if (getattr(root, "_encoder_output_role", None) == "localization" and
                float(getattr(root, "soft_route_power", 0.0)) != 0.0):
            raise ValueError("Removing localization logits requires soft_route_power=0")

    groups = {}
    for name, root in roots:
        groups.setdefault(id(root.base), []).append((name, root))
    before = _parameter_summary(model)
    plans = []
    # Validate and prepare all replacements before mutating the live roots.
    with torch.inference_mode(False), torch.no_grad():
        for aliases in groups.values():
            _, representative = aliases[0]
            base = representative.base
            steering = representative.base_steering
            for name, root in aliases[1:]:
                if (root.base_steering.dtype != steering.dtype or
                        root.base_steering.device != steering.device or
                        not torch.equal(root.base_steering, steering)):
                    raise ValueError(f"Shared base has a different steering grid: {name}")
            roles = {getattr(root, "_encoder_output_role", None) for _, root in aliases}
            if len(roles) != 1:
                raise ValueError("Shared compact base must have one encoder output role")
            role = roles.pop()
            if role not in (None, "retrieval", "localization"):
                raise ValueError(f"Unknown encoder output role: {role}")
            projection = base.steer_proj
            if projection is None:
                query = getattr(representative, "_frozen_base_query", None)
                if query is None:
                    raise ValueError("Missing static query for an already compacted base")
                projection_parameters = 0
            else:
                query = projection(steering.T.unsqueeze(0)).detach()
                projection_parameters = sum(p.numel() for p in projection.parameters())
            if tuple(query.shape) != (1, representative.K, representative.d_model):
                raise ValueError("Frozen steering query has invalid shape")
            if not torch.isfinite(query).all():
                raise ValueError("Frozen steering query contains nonfinite values")
            folds = []
            for path in ("cov_proj", "lag_proj"):
                sequence = getattr(base, path)
                if not isinstance(sequence, nn.Sequential):
                    raise TypeError(f"Expected a Sequential at base.{path}")
                for index in range(len(sequence) - 1):
                    convolution, norm = sequence[index:index + 2]
                    if not (isinstance(convolution, nn.Conv1d) and
                            isinstance(norm, nn.BatchNorm1d)):
                        continue
                    if norm.running_mean is None or norm.running_var is None:
                        raise ValueError("BatchNorm folding requires running statistics")
                    fused = fuse_conv_bn_eval(convolution, norm).eval().requires_grad_(False)
                    removed = (sum(p.numel() for p in convolution.parameters()) +
                               sum(p.numel() for p in norm.parameters()) -
                               sum(p.numel() for p in fused.parameters()))
                    folds.append((path, sequence, index, fused, removed))
            unused_name = ("offset_head" if role == "retrieval" else
                           "loc_head" if role == "localization" else None)
            unused = getattr(base, unused_name) if unused_name is not None else None
            unused_parameters = 0 if unused is None else sum(p.numel() for p in unused.parameters())
            plans.append((aliases, role, query, projection_parameters, folds,
                          unused_name, unused_parameters))

    encoders = []
    for aliases, role, query, projection_parameters, folds, unused_name, unused_parameters in plans:
        for _, root in aliases:
            root.register_buffer("_frozen_base_query", query, persistent=False)
            if hasattr(root, "_cached_base_query"):
                root._cached_base_query = None
            root._cached_base_query_key = None
        base = aliases[0][1].base
        base.steer_proj = None
        if unused_name is not None:
            setattr(base, unused_name, None)
        for _, sequence, index, fused, _ in folds:
            sequence[index] = fused
            sequence[index + 1] = nn.Identity().eval()
        encoders.append({
            "roots": [name for name, _ in aliases],
            "role": role,
            "query_buffer": "_frozen_base_query",
            "query_shape": list(query.shape),
            "query_bytes": query.numel() * query.element_size(),
            "query_shared_by_roots": len(aliases) > 1,
            "steering_parameters_removed": projection_parameters,
            "batchnorm_parameters_removed": sum(item[4] for item in folds),
            "unused_head": "base." + unused_name if unused_name is not None else None,
            "unused_head_parameters_removed": unused_parameters,
            "folded_pairs": [f"base.{path}.{index}+{index + 1}"
                             for path, _, index, _, _ in folds],
        })
    after = _parameter_summary(model)
    return {
        "before": before,
        "after": after,
        "parameters_removed": before["parameters"] - after["parameters"],
        "parameter_bytes_removed": before["parameter_bytes"] - after["parameter_bytes"],
        "unique_encoder_bases": len(groups),
        "unique_frozen_query_bytes": sum(item["query_bytes"] for item in encoders),
        "encoders": encoders,
        "checkpoint_modified": False,
        "requires_end_to_end_numerical_validation": True,
        "query_storage_note": (
            "Static queries replace the former lazy query cache. Before that "
            "cache was populated they add storage; parameter reduction alone "
            "is not a total-memory reduction measurement."
        ),
    }
