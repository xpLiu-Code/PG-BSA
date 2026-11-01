"""Shared wide-to-fine refinement with scene-level covariance projection.

The encoder emits 128-channel features. Three path adapters feed a shared
64- or 96-channel refinement core. The wide pass aggregates covariance and
support features; the fine pass resamples that representation and applies
an angular convolutional update."""
from __future__ import annotations
import copy
from collections import OrderedDict
from types import MethodType
import torch
from torch import nn
from torch.nn import functional as F

VERSION = 1
OWNER = "original.compact_core"
ROLES = ("original", "localizer", "close_pair")
DUAL_CORE_PREFIXES = ("localizer.compact_core.", "close_pair_localizer.compact_core.")
LEGACY_MODULES = (
    "local_steer_proj", "offset_embed", "local_cross", "support_transformer",
    "local_head", "wide_head", "power_head", "noise_head",
    "route_residual_head", "route_blend_head", "route_offset_head",
    "precision_local_steer_proj", "precision_offset_embed", "precision_local_cross",
    "precision_support_transformer", "precision_local_head",
)
LEGACY_PARAMETERS = ("correction_gate_logit", "precision_correction_gate_logit")
LEGACY_STATE_PREFIXES = tuple(name + "." for name in LEGACY_MODULES)


def _slots(support_batch, batch_size):
    """Pack routes even when callers supply interleaved scene order."""
    counts = torch.bincount(support_batch, minlength=batch_size)
    maximum = int(counts.max().item()) if counts.numel() else 0
    order = torch.argsort(support_batch, stable=True)
    starts = counts.cumsum(0) - counts
    sorted_slots = torch.arange(support_batch.numel(), device=support_batch.device)
    sorted_slots = sorted_slots - torch.repeat_interleave(starts, counts)
    slot = torch.empty_like(support_batch)
    slot[order] = sorted_slots
    return slot, counts, maximum


def _pack(value, support_batch, slot, batch_size, maximum):
    padded = value.new_zeros(batch_size, maximum, *value.shape[1:])
    padded[support_batch, slot] = value
    return padded


class AngleResidual(nn.Module):
    """Depthwise angular mixing plus a small channel bottleneck."""
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.depthwise = nn.Conv1d(width, width, 3, padding=1, groups=width)
        self.in_proj = nn.Linear(width, width // 2)
        self.out_proj = nn.Linear(width // 2, width)

    def forward(self, token):
        value = self.norm(token)
        # Evaluate the registered Conv1d kernel directly with zero boundaries.
        dtype = (torch.get_autocast_dtype(value.device.type)
                 if torch.is_autocast_enabled(value.device.type) else value.dtype)
        value = value.to(dtype=dtype)
        weight = self.depthwise.weight[:, 0].to(dtype=dtype)
        bias = self.depthwise.bias
        bias = bias.to(dtype=dtype) if bias is not None else None
        # Match convolution's wide accumulator for reduced-precision inputs.
        if dtype in (torch.float16, torch.bfloat16):
            value, weight = value.float(), weight.float()
            bias = bias.float() if bias is not None else None
        mixed = value * weight[:, 1]
        mixed[:, 1:].add_(value[:, :-1] * weight[:, 0])
        mixed[:, :-1].add_(value[:, 1:] * weight[:, 2])
        if bias is not None:
            mixed.add_(bias)
        mixed = mixed.to(dtype=dtype)
        return token + self.out_proj(F.gelu(self.in_proj(mixed)))


class CompactRefinementCore(nn.Module):
    def __init__(self, width=96, sensor_count=16, heads=4):
        super().__init__()
        if width not in (64, 96) or width % heads:
            raise ValueError("Compact refinement width must be 64 or 96")
        self.width = int(width)
        self.heads = int(heads)
        self.head_dim = width // heads
        self.steer_proj = nn.Sequential(nn.Linear(2 * sensor_count, width),
                                        nn.LayerNorm(width), nn.GELU())
        self.offset_embed = nn.Sequential(nn.Linear(1, width), nn.GELU(),
                                          nn.Linear(width, width))
        self.stage_embedding = nn.Parameter(torch.zeros(2, width))
        self.cross_norm = nn.LayerNorm(width)
        self.cross_q = nn.Linear(width, width)
        self.cross_k = nn.Linear(width, width)
        self.cross_v = nn.Linear(width, width)
        self.cross_out = nn.Linear(width, width)
        self.cross_ffn_norm = nn.LayerNorm(width)
        self.cross_ffn = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(),
                                       nn.Linear(2 * width, width))
        self.support_norm = nn.LayerNorm(width)
        self.support_qkv = nn.Linear(width, 3 * width)
        self.support_out = nn.Linear(width, width)
        self.support_ffn_norm = nn.LayerNorm(width)
        self.support_ffn = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(),
                                         nn.Linear(2 * width, width))
        self.support_distance_log_scale = nn.Parameter(torch.full((heads,), -2.0))
        self.fine_update = AngleResidual(width)

    def _split_heads(self, value):
        return value.reshape(*value.shape[:-1], self.heads, self.head_dim).transpose(1, 2)

    def covariance_kv(self, covariance_feature):
        # [B,N,D] is projected once before route indexing/packing.
        return (self._split_heads(self.cross_k(covariance_feature)),
                self._split_heads(self.cross_v(covariance_feature)))

    def wide_cross(self, token, key, value, support_batch, batch_size):
        slot, _, maximum = _slots(support_batch, batch_size)
        atom_count = token.shape[1]
        packed = _pack(self.cross_q(self.cross_norm(token)), support_batch, slot,
                       batch_size, maximum).reshape(batch_size, maximum * atom_count, self.width)
        query = self._split_heads(packed)
        attended = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)
        attended = attended.transpose(1, 2).reshape(batch_size, maximum, atom_count, self.width)
        result = token + self.cross_out(attended[support_batch, slot])
        return result + self.cross_ffn(self.cross_ffn_norm(result))

    def support_context(self, summary, angle, support_batch, batch_size):
        slot, counts, maximum = _slots(support_batch, batch_size)
        # Project real routes before padding the attention sequence and mask.
        packed_qkv = _pack(self.support_qkv(self.support_norm(summary)),
                           support_batch, slot, batch_size, maximum)
        packed_angle = _pack(angle, support_batch, slot, batch_size, maximum)
        q, k, v = packed_qkv.chunk(3, dim=-1)
        q, k, v = (self._split_heads(x) for x in (q, k, v))
        # Attention bias uses continuous angular distances.
        distance = ((packed_angle[:, :, None] - packed_angle[:, None, :]).abs() / 12.0).clamp_max(1.)
        bias = -F.softplus(self.support_distance_log_scale)[None, :, None, None] * distance[:, None].square()
        valid = torch.arange(maximum, device=summary.device)[None, :] < counts[:, None]
        # Empty scenes get one harmless dummy key and are never gathered.
        valid = valid.clone()
        valid[counts == 0, 0] = True
        bias = bias.masked_fill(~valid[:, None, None, :], float("-inf"))
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=0.0)
        attended = attended.transpose(1, 2).reshape(batch_size, maximum, self.width)
        attended = attended[support_batch, slot]
        transformed = summary + self.support_out(attended)
        return transformed + self.support_ffn(self.support_ffn_norm(transformed))


class CompactPathAdapter(nn.Module):
    """Private input alignment and small output heads for each physical path."""
    def __init__(self, input_width, width, use_route_residual=False, use_route_offset_residual=False):
        super().__init__()
        self.grid_proj = nn.Sequential(nn.Linear(input_width, width), nn.LayerNorm(width))
        self.cov_proj = nn.Sequential(nn.Linear(input_width, width), nn.LayerNorm(width))
        self.film_scale = nn.Parameter(torch.zeros(width))
        self.film_shift = nn.Parameter(torch.zeros(width))
        self.wide_head = nn.Sequential(nn.Linear(width, width // 2), nn.GELU(), nn.Linear(width // 2, 1, bias=False))
        self.fine_head = nn.Sequential(nn.Linear(width, width // 2), nn.GELU(), nn.Linear(width // 2, 1, bias=False))
        self.correction_gate_logit = nn.Parameter(torch.tensor(1.0))
        self.residual_head = nn.Linear(width, 1) if use_route_residual else None
        self.blend_head = nn.Linear(width, 1) if use_route_residual else None
        self.offset_head = nn.Linear(width, 1) if use_route_offset_residual else None
        for head in (self.residual_head, self.offset_head):
            if head is not None:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
        if self.blend_head is not None:
            nn.init.zeros_(self.blend_head.weight)
            nn.init.constant_(self.blend_head.bias, -3.)

    def condition(self, token):
        return token * (1. + 0.25 * torch.tanh(self.film_scale)) + 0.25 * self.film_shift


def _encoding_tensor_signature(encoded):
    values = []
    for name in ("grid_feature", "covariance_feature", "base_offset"):
        value = encoded[name]
        try:
            version = value._version
        except RuntimeError:
            version = None
        values.append((name, value.data_ptr(), tuple(value.shape), str(value.dtype), str(value.device), version))
    return tuple(values)


def _encoding_signature(root, encoded, detach_base):
    parameters = tuple(p._version for module in (root.compact_core, root.compact_adapter)
                       for p in module.parameters())
    return (id(root.compact_adapter), bool(detach_base), _encoding_tensor_signature(encoded), parameters)


def slice_compact_refinement_cache(encoded, sliced, scene_ids, index):
    """Select cached scene projections for a subset of observations.

    Adapter identity and parameter versions are retained for validation by the
    next reader. Invalid entries are discarded. Input tensors and the original
    cache remain unchanged."""
    cache = encoded.get("_compact_refinement_cache")
    if not isinstance(cache, dict):
        return None
    signature = cache.get("signature")
    if (not isinstance(signature, tuple) or len(signature) != 4
            or signature[2] != _encoding_tensor_signature(encoded)):
        return None
    batch_size, grid_size = encoded["location_logits"].shape
    # Repeated scene IDs require fresh computation to avoid ambiguous keys.
    if (len(set(scene_ids)) != len(scene_ids)
            or any(j < 0 or j >= batch_size for j in scene_ids)):
        return None
    key, value = cache.get("key"), cache.get("value")
    if (not torch.is_tensor(key) or not torch.is_tensor(value)
            or key.ndim != 4 or value.shape != key.shape
            or key.shape[0] != batch_size or key.device != index.device
            or value.device != index.device):
        return None
    result = {
        "signature": (signature[0], signature[1],
                      _encoding_tensor_signature(sliced), signature[3]),
        "key": key.index_select(0, index),
        "value": value.index_select(0, index),
    }
    if "route_key" not in cache and "wide_token" not in cache:
        return result
    route_key, wide_token = cache.get("route_key"), cache.get("wide_token")
    if (not torch.is_tensor(route_key) or not torch.is_tensor(wide_token)
            or route_key.dtype != torch.long or route_key.ndim != 1
            or wide_token.ndim != 3 or len(wide_token) != len(route_key)
            or wide_token.shape[-1] != key.shape[1] * key.shape[-1]
            or route_key.device != index.device or wide_token.device != index.device
            or wide_token.dtype != key.dtype):
        return None
    if route_key.numel() == 0:
        return result
    if (bool(((route_key < 0) | (route_key >= batch_size * grid_size)).any())
            or bool((route_key[1:] <= route_key[:-1]).any())):
        return None
    mapping = route_key.new_full((batch_size,), -1)
    mapping[index] = torch.arange(len(scene_ids), device=index.device)
    new_scene = mapping[torch.div(route_key, grid_size, rounding_mode="floor")]
    keep = torch.nonzero(new_scene >= 0, as_tuple=False).flatten()
    if keep.numel():
        remapped = new_scene[keep] * grid_size + route_key[keep].remainder(grid_size)
        order = remapped.argsort(stable=True)
        result["route_key"] = remapped[order]
        result["wide_token"] = wide_token[keep[order]]
    # An empty route cache is omitted: the normal miss path handles new routes.
    return result


def _get_eval_cache(root, encoded, detach_base):
    if root.training or torch.is_grad_enabled() or not getattr(root, "compact_cache_enabled", True):
        return None
    signature = _encoding_signature(root, encoded, detach_base)
    cache = encoded.get("_compact_refinement_cache")
    if not isinstance(cache, dict) or cache.get("signature") != signature:
        cache = {"signature": signature}
        encoded["_compact_refinement_cache"] = cache
    return cache


def _route_context(root, encoded, support_batch, support_index, detach_base):
    grid_feature = encoded["grid_feature"]
    covariance_feature = encoded["covariance_feature"]
    base_offset = encoded["base_offset"]
    if detach_base:
        grid_feature, covariance_feature, base_offset = (
            grid_feature.detach(), covariance_feature.detach(), base_offset.detach())
    grid_context = grid_feature[support_batch, support_index]
    offset_residual = grid_context.new_zeros(support_index.numel())
    if root.use_route_offset_residual:
        radius = int(root.route_offset_context_radius)
        shifts = torch.arange(-radius, radius + 1, device=support_index.device)
        index = (support_index[:, None] + shifts).clamp(0, root.K - 1)
        context = grid_feature[support_batch[:, None], index].mean(1)
        offset_residual = torch.tanh(root.compact_adapter.offset_head(
            root.compact_adapter.grid_proj(context)).squeeze(-1)) * root.route_offset_span
    if root.soft_route_power > 0.:
        shifts = torch.arange(-2, 3, device=support_index.device)
        index = (support_index[:, None] + shifts).clamp(0, root.K - 1)
        probability = torch.sigmoid(encoded["location_logits"][support_batch[:, None], index])
        weight = probability.pow(root.soft_route_power)
        weight = weight / weight.sum(1, keepdim=True).clamp_min(1.e-8)
        route_angle = (weight * (root.theta_grid[index] + base_offset[support_batch[:, None], index])).sum(1)
        grid_context = (weight[..., None] * grid_feature[support_batch[:, None], index]).sum(1)
    else:
        route_angle = root.theta_grid[support_index] + base_offset[support_batch, support_index]
    route_angle = route_angle + offset_residual
    if root.interpolate_route_context and root.soft_route_power <= 0.:
        coordinate = ((route_angle - root.theta_grid[0]) / (root.theta_grid[1] - root.theta_grid[0])).clamp(0., root.K - 1)
        left = coordinate.floor().long()
        right = (left + 1).clamp_max(root.K - 1)
        fraction = (coordinate - left.to(coordinate.dtype))[:, None]
        grid_context = grid_feature[support_batch, left] + fraction * (
            grid_feature[support_batch, right] - grid_feature[support_batch, left])
    support = root.compact_adapter.condition(2. * root.compact_adapter.grid_proj(grid_context))
    return grid_feature, covariance_feature, route_angle, offset_residual, support


def _physical_tokens(root, centre, offsets, support, stage):
    steering = root._steering(centre[:, None] + offsets[None, :])
    feature = torch.cat((steering.real, steering.imag), dim=-1)
    offset_scale = offsets.abs().max().clamp_min(1.e-8)
    embedding = root.compact_core.offset_embed((offsets / offset_scale).view(1, -1, 1))
    return (root.compact_core.steer_proj(feature) + support[:, None] + embedding
            + root.compact_core.stage_embedding[stage])


def _resample_wide(wide_token, wide_offsets, fine_offsets, wide_residual):
    coordinate = ((wide_residual[:, None] + fine_offsets[None, :] - wide_offsets[0])
                  / (wide_offsets[1] - wide_offsets[0])).clamp(0., wide_offsets.numel() - 1)
    left = coordinate.floor().long()
    right = (left + 1).clamp_max(wide_offsets.numel() - 1)
    row = torch.arange(wide_token.shape[0], device=wide_token.device)[:, None]
    fraction = (coordinate - left.to(coordinate.dtype))[..., None]
    return wide_token[row, left] + fraction * (wide_token[row, right] - wide_token[row, left])



FINE_VARIANTS = ("legacy", "remove_fine_support")


def set_compact_fine_variant(root, variant):
    """Select a parameter-free fine-input rule on one compact localization root.

    Call this for each original/adapted/close root. The checkpoint declaration
    belongs in payload metadata, since this ordinary string is not a state key.
    Wide tokens and their cache are unchanged by either fine-input rule.
    """
    if not isinstance(variant, str) or variant not in FINE_VARIANTS:
        raise ValueError("Compact fine variant must be legacy or remove_fine_support")
    if not isinstance(getattr(root, "compact_core", None), CompactRefinementCore):
        raise ValueError("Expected one compact localization root")
    if not isinstance(getattr(root, "compact_adapter", None), CompactPathAdapter):
        raise ValueError("Expected one compact localization path adapter")
    root.compact_fine_variant = variant
    return variant


def _compact_refine_routes(root, encoded, support_batch, support_index, *,
                           detach_base=True, build_reconstruction=False):
    if build_reconstruction:
        raise ValueError("The deployed compact refiner has no learned reconstruction branch")
    fine_variant = getattr(root, "compact_fine_variant", "legacy")
    if fine_variant not in FINE_VARIANTS:
        raise ValueError("Unsupported compact fine variant")
    batch_size = encoded["location_logits"].shape[0]
    if support_batch.ndim != 1 or support_index.ndim != 1 or support_batch.shape != support_index.shape:
        raise ValueError("Support batch and index must be equal-length vectors")
    if support_batch.dtype != torch.long or support_index.dtype != torch.long:
        raise ValueError("Support vectors must have dtype torch.long")
    if support_index.numel() == 0:
        fine = encoded["location_logits"].new_empty(0, root.M)
        scalar = fine.new_empty(0)
        return {"fine_logits": fine, "fine_probability": fine,
                "base_angle": scalar, "route_base_angle": scalar,
                "route_offset_residual": scalar,
                "wide_logits": fine.new_empty(0, root.wide_offsets.numel()),
                "wide_residual": scalar, "raw_residual": scalar,
                "conditional_angle": scalar, "route_residual": scalar,
                "route_blend": scalar, "source_power": scalar,
                "noise_power": fine.new_zeros(batch_size),
                "R_hat": encoded["covariance"].new_zeros(batch_size, root.N, root.N),
                "local_candidate_count": 0}
    if bool(((support_batch < 0) | (support_batch >= batch_size)).any()):
        raise ValueError("Support batch index is outside the encoded batch")
    if bool(((support_index < 0) | (support_index >= root.K)).any()):
        raise ValueError("Support index is outside the coarse grid")
    grid_feature, covariance_feature, route_angle, offset_residual, support = _route_context(
        root, encoded, support_batch, support_index, detach_base)
    cache = _get_eval_cache(root, encoded, detach_base)
    core, adapter = root.compact_core, root.compact_adapter
    if cache is not None and "key" in cache:
        key, value = cache["key"], cache["value"]
    else:
        key, value = core.covariance_kv(adapter.cov_proj(covariance_feature))
        if cache is not None:
            cache.update(key=key, value=value)
    route_key = support_batch * root.K + support_index
    if cache is not None and "route_key" in cache:
        # Sorted lookup avoids an Rnew-by-Rcached temporary on large batches.
        reuse = torch.searchsorted(cache["route_key"], route_key)
        valid = reuse < cache["route_key"].numel()
        reuse = reuse.clamp_max(cache["route_key"].numel() - 1)
        found = valid & (cache["route_key"][reuse] == route_key)
        missing = torch.nonzero(~found, as_tuple=False).flatten()
        wide_token = support.new_empty(support_index.numel(), root.wide_offsets.numel(), root.refine_width)
        wide_token[found] = cache["wide_token"][reuse[found]]
    else:
        missing = torch.arange(support_index.numel(), device=support_index.device)
        wide_token = support.new_empty(support_index.numel(), root.wide_offsets.numel(), root.refine_width)
    if missing.numel():
        token = _physical_tokens(root, route_angle[missing], root.wide_offsets, support[missing], 0)
        fresh = core.wide_cross(token, key, value, support_batch[missing], batch_size)
        wide_token[missing] = fresh
        if cache is not None:
            cache["route_key"] = (route_key[missing] if "route_key" not in cache else
                                  torch.cat((cache["route_key"], route_key[missing])))
            cache["wide_token"] = (fresh if "wide_token" not in cache else
                                   torch.cat((cache["wide_token"], fresh)))
            order = cache["route_key"].argsort(stable=True)
            keys = cache["route_key"][order]
            unique = torch.cat((keys.new_ones(1, dtype=torch.bool), keys[1:] != keys[:-1]))
            cache["route_key"] = keys[unique]
            cache["wide_token"] = cache["wide_token"][order[unique]]
    support_context = core.support_context(wide_token.mean(1) + support, route_angle,
                                           support_batch, batch_size)
    wide_joint = wide_token + support_context[:, None]
    wide_logits = adapter.wide_head(adapter.condition(wide_joint)).squeeze(-1)
    wide_residual = (F.softmax(wide_logits, dim=-1) * root.wide_offsets).sum(-1)
    base_angle = route_angle + wide_residual
    latent = _resample_wide(wide_joint, root.wide_offsets, root.fine_offsets, wide_residual)
    fine_physics = _physical_tokens(root, base_angle, root.fine_offsets, support, 1)
    fine_input = latent + fine_physics
    if fine_variant == "remove_fine_support":
        # The resampled wide latent already contains the route support. Keep
        # fresh steering/offset/stage features without adding that support again.
        fine_input = fine_input - support[:, None]
    fine_token = core.fine_update(adapter.condition(fine_input))
    fine_logits = adapter.fine_head(fine_token).squeeze(-1)
    fine_probability = F.softmax(fine_logits, dim=-1)
    raw_residual = (fine_probability * root.fine_offsets).sum(-1)
    correction = torch.tanh(adapter.correction_gate_logit) * raw_residual
    route_residual = torch.zeros_like(raw_residual)
    route_blend = torch.zeros_like(raw_residual)
    if root.use_route_residual:
        route_residual = torch.tanh(adapter.residual_head(support_context).squeeze(-1)) * root.route_residual_span
        route_blend = torch.sigmoid(adapter.blend_head(support_context).squeeze(-1))
    conditional_angle = base_angle + (1. - route_blend) * correction + route_blend * route_residual
    return {"fine_logits": fine_logits, "fine_probability": fine_probability,
            "base_angle": base_angle, "route_base_angle": route_angle,
            "route_offset_residual": offset_residual, "wide_logits": wide_logits,
            "wide_residual": wide_residual, "raw_residual": raw_residual,
            "conditional_angle": conditional_angle, "route_residual": route_residual,
            "route_blend": route_blend,
            "R_hat": encoded["covariance"].new_zeros(batch_size, root.N, root.N),
            "source_power": raw_residual.new_zeros(support_index.numel()),
            "noise_power": raw_residual.new_zeros(batch_size),
            "local_candidate_count": int(support_index.numel() * (root.M + root.wide_offsets.numel()))}


def compact_spec(width):
    if isinstance(width, bool) or width not in (64, 96):
        raise ValueError("Compact refinement width must be 64 or 96")
    return dict(version=VERSION, width=int(width), shared_core_owner=OWNER,
                wide_covariance_attention_passes=1, support_attention_passes=1,
                fine_update="wide_latent_resampling_depthwise3_channel_bottleneck",
                private_paths=list(ROLES), covariance_projection="once_per_scene",
                source_count_branch=False, initialization="fresh_refinement_preserved_encoder")


def validate_compact_spec(spec):
    if not isinstance(spec, dict):
        raise ValueError("Compact refinement marker must be a mapping")
    expected = compact_spec(spec.get("width"))
    if spec != expected:
        raise ValueError("Unsupported or inconsistent compact refinement marker")
    return expected


def apply_compact_refinement_models(models, width=96, initialize=False):
    """Attach the shared refinement topology before loading its weights.

    With initialize=False, strict checkpoint loading supplies all parameters.
    With initialize=True, the refinement layers receive new initialization.
    The encoder modules retain their shared references."""
    del initialize
    spec = compact_spec(width)
    if set(models) != {"original", "localizer", "retrieval", "close_pair"}:
        raise ValueError("Expected exactly four encoder paths")
    if any(hasattr(models[name], "compact_core") for name in ROLES):
        raise ValueError("Compact refinement was already applied")
    if any(models[name].d_model != 128 for name in ROLES):
        raise ValueError("Compact refinement expects 128-wide encoder features")
    if models["original"].base is not models["localizer"].base:
        raise ValueError("Original and localization encoder ownership must be bound first")
    for name in ROLES:
        root = models[name]
        if not root.use_wide_cascade or root.wide_offsets.numel() != 15 or root.M != 15:
            raise ValueError("Compact refinement preserves exactly 15 wide and 15 fine atoms")
        if root.use_dual_precision:
            raise ValueError("Compact refinement replaces legacy dual-precision topology")
    reference = next(models["original"].parameters())
    shared_core = CompactRefinementCore(width, models["original"].N).to(
        device=reference.device, dtype=reference.dtype)
    for name in ROLES:
        root = models[name]
        root.compact_core = shared_core
        root.compact_adapter = CompactPathAdapter(root.d_model, width, root.use_route_residual, root.use_route_offset_residual).to(
            device=reference.device, dtype=reference.dtype)
        root.refine_width = int(width)
        root.compact_cache_enabled = True
        set_compact_fine_variant(root, "legacy")
        root._compact_refinement_spec = copy.deepcopy(spec)
        for attr in LEGACY_MODULES:
            setattr(root, attr, None)
        for attr in LEGACY_PARAMETERS:
            if hasattr(root, attr):
                delattr(root, attr)
            root.register_parameter(attr, None)
        if hasattr(root, "_refinement_bottleneck_spec"):
            delattr(root, "_refinement_bottleneck_spec")
        root.refine_routes = MethodType(_compact_refine_routes, root)
    return compact_refinement_summary(models)


def compact_refinement_summary(models):
    core = models["original"].compact_core
    return dict(spec=copy.deepcopy(models["original"]._compact_refinement_spec),
                core_parameters=sum(p.numel() for p in core.parameters()),
                private_parameters={name: sum(p.numel() for p in models[name].compact_adapter.parameters())
                                    for name in ROLES},
                shared_core_identity=all(models[name].compact_core is core for name in ROLES),
                encoder_width=models["original"].d_model, refinement_width=core.width,
                legacy_refinement_removed=all(getattr(models[name], attr) is None
                                             for name in ROLES for attr in LEGACY_MODULES))


def is_compact_alias_key(key):
    return key.startswith(DUAL_CORE_PREFIXES)


def expand_compact_dual_state(state, original, dual):
    """Expand canonical shared-core tensors as references for strict loading."""
    if any(is_compact_alias_key(key) for key in state):
        raise ValueError("Canonical compact state must store the shared core only once")
    if not (original.compact_core is dual.localizer.compact_core is dual.close_pair_localizer.compact_core):
        raise ValueError("Shared refinement core identity was lost before load")
    expanded = dict(state)
    for prefix in DUAL_CORE_PREFIXES:
        expanded.update({prefix + key: value for key, value in original.compact_core.state_dict().items()})
    return expanded


def _legacy_state_key(key):
    return key in LEGACY_PARAMETERS or key.startswith(LEGACY_STATE_PREFIXES)


def convert_compact_refinement_payload(payload, width=96, seed=20261010):
    """Copy encoder and scorer tensors into a newly initialized refiner.

    The returned checkpoint requires training of the new refinement layers."""
    if payload.get("format") != "pgbsa_shared_delta_v3":
        raise ValueError("Expected canonical shared-delta v3 checkpoint")
    if "compact_refinement" in payload:
        raise ValueError("Payload already contains compact refinement")
    if "grid_resolution" in payload:
        raise ValueError("Compact conversion requires a 0.5-degree grid")
    if any(float(config["coarse_step"]) != .5 for config in payload["configs"].values()):
        raise ValueError("Compact conversion requires a 0.5-degree grid")
    for key in payload["state"]["dual"]:
        if key.startswith("localizer.base.") or key.endswith(".shared_weight"):
            raise ValueError("Payload is not in canonical shared-delta form")
    from .networks import BeamDOANetwork
    from .decoder import strip_reconstruction, strip_retrieval_refinement, InferenceNetworks
    from .shared_delta import apply_shared_delta_models, expand_canonical_dual_state
    from .heads import GlobalCandidateGenerator, CandidateScorer
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        models = {name: BeamDOANetwork(**config) for name, config in payload["configs"].items()}
        for root in models.values():
            strip_reconstruction(root)
        models["localizer"].base = models["original"].base
        strip_retrieval_refinement(models["retrieval"])
        apply_shared_delta_models(models, rank=payload["shared_delta"]["rank"], initialize=False)
        summary = apply_compact_refinement_models(models, width=width, initialize=True)
    original_state = OrderedDict((key, value) for key, value in payload["state"]["original"].items()
                                 if not _legacy_state_key(key))
    dual_state = OrderedDict()
    for key, value in payload["state"]["dual"].items():
        remainder = key
        for prefix in ("localizer.", "close_pair_localizer."):
            if key.startswith(prefix):
                remainder = key[len(prefix):]
                break
        if not _legacy_state_key(remainder):
            dual_state[key] = value
    for key, value in models["original"].state_dict().items():
        if key.startswith(("compact_core.", "compact_adapter.")):
            original_state[key] = value.detach().cpu().clone()
    for role, prefix in (("localizer", "localizer."), ("close_pair", "close_pair_localizer.")):
        for key, value in models[role].compact_adapter.state_dict().items():
            dual_state[prefix + "compact_adapter." + key] = value.detach().cpu().clone()
    result = copy.deepcopy(payload)
    result.pop("refinement_bottleneck", None)
    result["compact_refinement"] = compact_spec(width)
    result["state"]["original"] = original_state
    result["state"]["dual"] = dual_state
    dual = InferenceNetworks(models["localizer"], models["retrieval"], models["close_pair"],
                             dual_state["retrieval.theta_grid"])
    models["original"].load_state_dict(original_state, strict=True)
    expanded = expand_canonical_dual_state(dual_state, models["original"], dual)
    dual.load_state_dict(expanded, strict=True)
    auxiliary = {}
    for name, kind in (("generator", GlobalCandidateGenerator), ("scorer", CandidateScorer), ("offset", CandidateScorer)):
        module = kind(**payload["model_shapes"][name])
        module.load_state_dict(payload["state"][name], strict=True)
        auxiliary[name] = module
    unique = {id(p): p for root in [models["original"], dual, *auxiliary.values()] for p in root.parameters()}
    result["metadata"] = dict(result.get("metadata", {}), unique_parameters=sum(p.numel() for p in unique.values()),
                               neural_weights_unchanged=False, distillation_used=False,
                               compact_refinement_trained=False, compact_refinement_seed=int(seed),
                               compact_refinement_summary=summary)
    return result
