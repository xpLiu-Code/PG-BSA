"""Feedforward channel reduction for wide and fine refinement.

The transformation preserves attention, angular adaptation, residual widths,
and shared encoder references. Channel selection changes the feedforward
function and therefore requires a compatible checkpoint."""
from __future__ import annotations
import copy
from collections.abc import Mapping
import torch
from torch import nn

FORMAT = "pgbsa_shared_delta_v3"
SUPPORT_WIDTHS = frozenset((96, 128, 192))
CROSS_WIDTHS = frozenset((64, 96, 128))
ROOT_KEYS = frozenset(("original", "localizer", "retrieval", "close_pair"))
MARKER = "refinement_bottleneck"


def normalize_refinement_bottleneck(support_ffn=128, cross_ffn=96):
    if isinstance(support_ffn, bool) or not isinstance(support_ffn, int) or support_ffn not in SUPPORT_WIDTHS:
        raise ValueError("support_ffn must be one of 96, 128, 192")
    if isinstance(cross_ffn, bool) or not isinstance(cross_ffn, int) or cross_ffn not in CROSS_WIDTHS:
        raise ValueError("cross_ffn must be one of 64, 96, 128")
    return dict(version=1, support_ffn=support_ffn, cross_ffn=cross_ffn,
                channel_selection="stable_bilateral_l2", energy_rescale=False)


def _validate_tensors(w1, b1, w2, b2, hidden, path):
    expected = ((hidden, 128), (hidden,), (128, hidden), (128,))
    for tensor, shape in zip((w1, b1, w2, b2), expected):
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
            raise ValueError("Unexpected dense refinement FFN shape: " + path)
        if not torch.is_floating_point(tensor):
            raise ValueError("Refinement FFN tensors must be floating point: " + path)
        if not torch.isfinite(tensor).all():
            raise ValueError("Nonfinite refinement FFN tensor: " + path)
    if len({(x.device, x.dtype) for x in (w1, b1, w2, b2)}) != 1:
        raise ValueError("Refinement FFN device/dtype differs: " + path)


def _channels(w1, b1, w2, width):
    # Product of input-row and output-column L2 energy ranks both sides.
    # Stable ties favor the lower original channel index. Sorting selected
    # indices restores original channel order before constructing the new FFN.
    input_energy = w1.detach().to(dtype=torch.float64).square().sum(1)
    input_energy = input_energy + b1.detach().to(dtype=torch.float64).square()
    output_energy = w2.detach().to(dtype=torch.float64).square().sum(0)
    energy = input_energy * output_energy
    indices = torch.argsort(energy, descending=True, stable=True)[:width].sort().values
    total = energy.sum()
    fraction = float(energy[indices].sum() / total) if float(total) > 0 else 1.0
    return indices, fraction


def _count(models):
    parameters = {id(p): p for root in models.values() for p in root.parameters()}
    return sum(p.numel() for p in parameters.values())


def _model_plans(models, settings):
    if not isinstance(models, Mapping) or set(models) != ROOT_KEYS:
        raise ValueError("Expected original/localizer/retrieval/close_pair roots")
    if models["original"].base is not models["localizer"].base:
        raise ValueError("Original/localizer base ownership must already be shared")
    if len({id(models[k]) for k in ("original", "localizer", "close_pair")}) != 3:
        raise ValueError("Expected three distinct refiner roots")
    plans, seen = [], set()
    for name in ("original", "localizer", "close_pair"):
        root = models[name]
        if hasattr(root, "_refinement_bottleneck_spec"):
            raise ValueError("Refinement bottleneck already applied: " + name)
        if int(root.d_model) != 128 or not root.use_wide_cascade or root.use_dual_precision:
            raise ValueError("Expected the existing D128 shared-wide/fine refiner: " + name)
        support = root.support_transformer
        support_path = "support_transformer"
        if name == "localizer":
            if not hasattr(support, "backbone"):
                raise ValueError("Expected adapted support wrapper with original backbone")
            support = support.backbone
            support_path += ".backbone"
        if not isinstance(support, nn.TransformerEncoder) or len(support.layers) != 2:
            raise ValueError("Both support attention layers must remain present: " + name)
        for index, layer in enumerate(support.layers):
            plans.append((name, support_path + ".layers." + str(index),
                          layer, "linear1", "linear2", 512, settings["support_ffn"]))
        if not isinstance(root.local_cross.mlp, nn.Sequential) or len(root.local_cross.mlp) != 3:
            raise ValueError("Expected dense local cross FFN: " + name)
        plans.append((name, "local_cross.mlp", root.local_cross.mlp, "0", "2", 256, settings["cross_ffn"]))
    for name, path, parent, first_key, second_key, old_width, width in plans:
        first, second = getattr(parent, first_key), getattr(parent, second_key)
        if type(first) is not nn.Linear or type(second) is not nn.Linear:
            raise ValueError("Refinement FFNs must be independent dense Linear modules: " + name + "." + path)
        _validate_tensors(first.weight, first.bias, second.weight, second.bias, old_width, name + "." + path)
        ids = {id(p) for p in (*first.parameters(), *second.parameters())}
        if ids & seen:
            raise ValueError("Unexpected shared refinement FFN parameters: " + name + "." + path)
        seen.update(ids)
    return plans


def _replace_pair(parent, first_key, second_key, width, initialize):
    first, second = getattr(parent, first_key), getattr(parent, second_key)
    new_first = nn.Linear(128, width, device=first.weight.device, dtype=first.weight.dtype)
    new_second = nn.Linear(width, 128, device=second.weight.device, dtype=second.weight.dtype)
    selected, fraction = None, None
    if initialize:
        selected, fraction = _channels(first.weight, first.bias, second.weight, width)
        with torch.no_grad():
            new_first.weight.copy_(first.weight[selected])
            new_first.bias.copy_(first.bias[selected])
            new_second.weight.copy_(second.weight[:, selected])
            new_second.bias.copy_(second.bias)
    for new, old in ((new_first, first), (new_second, second)):
        new.train(old.training)
        new.weight.requires_grad_(old.weight.requires_grad)
        new.bias.requires_grad_(old.bias.requires_grad)
    setattr(parent, first_key, new_first)
    setattr(parent, second_key, new_second)
    return None if selected is None else selected.cpu().tolist(), fraction


def apply_refinement_bottleneck_models(models, support_ffn=128, cross_ffn=96, *, initialize=False):
    """Apply before strict loading (False), or prune loaded dense FFNs (True).

    initialize=False keeps normal fresh Linear initialization, to be overwritten
    by the narrowed checkpoint's strict state load. Existing training/eval and
    requires_grad flags are preserved. Validation completes before mutation.
    """
    if not isinstance(initialize, bool):
        raise ValueError("initialize must be boolean")
    settings = normalize_refinement_bottleneck(support_ffn, cross_ffn)
    plans = _model_plans(models, settings)
    before = _count(models)
    rows = []
    for name, path, parent, first, second, old_width, width in plans:
        selected, fraction = _replace_pair(parent, first, second, width, initialize)
        rows.append(dict(root=name, path=path, old_hidden=old_width, hidden=width,
                         parameters_removed=(old_width-width)*257,
                         selected_channels=selected, retained_bilateral_energy=fraction))
    for name in ("original", "localizer", "close_pair"):
        models[name]._refinement_bottleneck_spec = dict(settings)
    after = _count(models)
    removed = sum(row["parameters_removed"] for row in rows)
    if before - after != removed:
        raise RuntimeError("Unique refinement parameter reduction did not match")
    return dict(spec=settings, initialize=initialize, before_unique_parameters=before,
                after_unique_parameters=after, parameters_removed=removed, matrices=rows,
                attention_layers_removed=0, fine_stage_removed=False,
                energy_rescale=False, function_equivalent=False)


def _payload_plans(payload, settings):
    if not isinstance(payload, dict) or payload.get("format") != FORMAT:
        raise ValueError("Expected canonical shared-delta v3 checkpoint")
    if MARKER in payload or MARKER in payload.get("metadata", {}):
        raise ValueError("Refinement bottleneck checkpoint already marked")
    if payload.get("shared_backbone_owner") != "original.base" or payload.get("shared_delta", {}).get("rank") != 16:
        raise ValueError("Expected the rank16 original.base checkpoint")
    state = payload.get("state")
    if not isinstance(state, dict) or not {"original", "dual"}.issubset(state):
        raise ValueError("Missing canonical state groups")
    if any(k.startswith("localizer.base.") or k.endswith(".shared_weight") for k in state["dual"]):
        raise ValueError("Canonical checkpoint must omit duplicated shared aliases")
    plans = []
    for group, prefix, adapted in (("original", "", False),
                                    ("dual", "localizer.", True),
                                    ("dual", "close_pair_localizer.", False)):
        support_prefix = prefix + "support_transformer." + ("backbone." if adapted else "")
        for index in (0, 1):
            path = support_prefix + "layers." + str(index)
            plans.append((group, path, path+".linear1", path+".linear2", 512, settings["support_ffn"]))
        path = prefix + "local_cross.mlp"
        plans.append((group, path, path+".0", path+".2", 256, settings["cross_ffn"]))
    for group, path, first, second, old_width, width in plans:
        keys = (first+".weight", first+".bias", second+".weight", second+".bias")
        try: tensors = [state[group][key] for key in keys]
        except KeyError as exc: raise ValueError("Missing refinement tensor: "+str(exc)) from exc
        _validate_tensors(*tensors, old_width, group+"."+path)
    unique = payload.get("metadata", {}).get("unique_parameters")
    if isinstance(unique, bool) or not isinstance(unique, int) or unique <= 0:
        raise ValueError("Missing positive original unique_parameters metadata")
    return plans


def prune_refinement_bottleneck_payload(payload, support_ffn=128, cross_ffn=96):
    """Return a copied v3 checkpoint with nine narrowed FFN pairs, no base edits."""
    settings = normalize_refinement_bottleneck(support_ffn, cross_ffn)
    plans = _payload_plans(payload, settings)
    result = copy.deepcopy(payload)
    rows = []
    for group, path, first, second, old_width, width in plans:
        state = result["state"][group]
        indices, fraction = _channels(state[first+".weight"], state[first+".bias"], state[second+".weight"], width)
        state[first+".weight"] = state[first+".weight"][indices].clone()
        state[first+".bias"] = state[first+".bias"][indices].clone()
        state[second+".weight"] = state[second+".weight"][:, indices].clone()
        rows.append(dict(group=group, path=path, old_hidden=old_width, hidden=width,
                         parameters_removed=(old_width-width)*257,
                         selected_channels=indices.cpu().tolist(), retained_bilateral_energy=fraction))
    removed = sum(row["parameters_removed"] for row in rows)
    result[MARKER] = dict(settings)
    result["metadata"] = dict(result["metadata"], refinement_bottleneck=dict(settings),
                              unique_parameters=result["metadata"]["unique_parameters"]-removed,
                              neural_weights_unchanged=False, distillation_used=False,
                              refinement_bottleneck_parameters_removed=removed)
    # Record initialization details for the reduced feedforward layers.
    result["refinement_bottleneck_initialization"] = dict(
        method="stable_bilateral_l2", energy_rescale=False, function_equivalent=False,
        attention_layers_removed=0, fine_stage_removed=False, matrices=rows)
    return result
