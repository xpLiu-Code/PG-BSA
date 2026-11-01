"""Dense grid queries with pooled keys and values during inference.

Pair averaging shortens the key and value sequence while retaining an output
for every query. The residual, normalization, and feedforward paths retain
their full grid resolution. Pooling approximates dense self-attention."""
from __future__ import annotations
import math
import torch
from torch.nn import functional as F
from .dense_owner_sdpa import DenseSDPAAttention
from .shared_delta import SharedDeltaAttention, ExplicitEncoderLayer

ROLES = ("original", "retrieval", "close_pair")


def validate_grid_kv_options(stride, roles, dense_layers):
    if isinstance(stride, bool) or not isinstance(stride, int) or stride not in (1, 2):
        raise ValueError("grid_kv_stride must be 1 or 2")
    if (isinstance(dense_layers, bool) or not isinstance(dense_layers, int)
            or not 0 <= dense_layers <= 4):
        raise ValueError("grid_kv_dense_layers must be an integer in [0,4]")
    if (not isinstance(roles, (list, tuple)) or not all(isinstance(role, str) for role in roles)
            or len(set(roles)) != len(roles) or not set(roles) <= set(ROLES)):
        raise ValueError("grid_kv_roles must be unique original/retrieval/close_pair roles")


def _eligible(module, query, key, value, key_padding_mask, need_weights, attn_mask, is_causal):
    return (getattr(module, "_grid_kv_stride", 1) == 2 and not module.training
            and not torch.is_grad_enabled() and query is key and key is value
            and query.ndim == 3 and torch.is_floating_point(query)
            and query.shape[1] == getattr(module, "_grid_kv_expected_tokens", -1)
            and query.shape[1] > 1 and key_padding_mask is None
            and attn_mask is None and not is_causal and not need_weights)


def _pool_pairs(value):
    length = value.shape[1]
    paired = length // 2
    pooled = (value[:, :2 * paired:2] + value[:, 1:2 * paired:2]) * 0.5
    if length % 2:
        pooled = torch.cat((pooled, value[:, -1:]), dim=1)
    return pooled


def _pooled_forward(module, query):
    width = module.embed_dim
    pooled = _pool_pairs(query)
    if isinstance(module, PooledDenseSDPAAttention):
        q = module._project_slice(query, 0, width)
        k, v = module._project_slice(pooled, width, 3 * width).chunk(2, dim=-1)
    else:
        q = module.in_projection.forward_slice(query, 0, width)
        k, v = module.in_projection.forward_slice(pooled, width, 3 * width).chunk(2, dim=-1)
    batch, length, _ = q.shape
    source = k.shape[1]
    head_dim = width // module.num_heads
    q = q.reshape(batch, length, module.num_heads, head_dim).transpose(1, 2)
    k = k.reshape(batch, source, module.num_heads, head_dim).transpose(1, 2)
    v = v.reshape(batch, source, module.num_heads, head_dim).transpose(1, 2)
    # Each average represents two keys, except the final unpaired boundary key.
    # log mass preserves duplicate-key softmax multiplicity exactly.
    bias = (torch.cat((q.new_full((source - 1,), math.log(2.0)), q.new_zeros(1)))
            if length % 2 else q.new_full((source,), math.log(2.0)))
    attended = F.scaled_dot_product_attention(q, k, v,
        attn_mask=bias.view(1, 1, 1, source), dropout_p=0.0, is_causal=False)
    attended = attended.transpose(1, 2).contiguous().reshape(batch, length, width)
    return module.out_proj(attended), None


class PooledDenseSDPAAttention(DenseSDPAAttention):
    def forward(self, query, key, value, key_padding_mask=None, need_weights=True,
                attn_mask=None, average_attn_weights=True, is_causal=False):
        if _eligible(self, query, key, value, key_padding_mask, need_weights, attn_mask, is_causal):
            return _pooled_forward(self, query)
        return super().forward(query, key, value, key_padding_mask=key_padding_mask,
            need_weights=need_weights, attn_mask=attn_mask,
            average_attn_weights=average_attn_weights, is_causal=is_causal)


class PooledSharedDeltaAttention(SharedDeltaAttention):
    def forward(self, query, key, value, key_padding_mask=None, need_weights=True,
                attn_mask=None, average_attn_weights=True, is_causal=False):
        if _eligible(self, query, key, value, key_padding_mask, need_weights, attn_mask, is_causal):
            return _pooled_forward(self, query)
        return super().forward(query, key, value, key_padding_mask=key_padding_mask,
            need_weights=need_weights, attn_mask=attn_mask,
            average_attn_weights=average_attn_weights, is_causal=is_causal)


def apply_pooled_grid_kv(roots, *, stride=1, roles=ROLES, dense_layers=0):
    """Install after SDPA configuration and before CUDA graph construction.

    Only existing DenseSDPAAttention / SharedDeltaAttention objects are adapted.
    Unsupported attention stays dense. No Parameter, buffer or state key is added.
    """
    validate_grid_kv_options(stride, roles, dense_layers)
    if set(roots) != set(ROLES):
        raise ValueError("Expected original/retrieval/close_pair roots")
    unique = {id(parameter): parameter for root in roots.values() for parameter in root.parameters()}
    ids_before = tuple(unique)
    keys_before = {name: tuple(root.state_dict()) for name, root in roots.items()}
    records = []
    for role, root in roots.items():
        encoder = root.base.grid_transformer
        active = stride == 2 and role in roles
        for index, layer in enumerate(encoder.layers):
            attn = layer.self_attn
            enabled = active and index >= dense_layers
            supported = isinstance(attn, (DenseSDPAAttention, SharedDeltaAttention))
            if enabled and supported:
                if isinstance(attn, DenseSDPAAttention):
                    attn.__class__ = PooledDenseSDPAAttention
                else:
                    attn.__class__ = PooledSharedDeltaAttention
                layer.__class__ = ExplicitEncoderLayer
                encoder.use_nested_tensor = False
            if isinstance(attn, (PooledDenseSDPAAttention, PooledSharedDeltaAttention)):
                attn._grid_kv_stride = 2 if enabled else 1
                attn._grid_kv_expected_tokens = root.K
            records.append(dict(role=role, layer=index, enabled=enabled and supported,
                                supported=supported, dense_tokens=int(root.K),
                                pooled_tokens=(int(root.K) + 1) // 2 if enabled and supported else int(root.K)))
    ids_after = tuple({id(parameter): parameter for root in roots.values() for parameter in root.parameters()})
    assert ids_before == ids_after
    assert keys_before == {name: tuple(root.state_dict()) for name, root in roots.items()}
    return dict(stride=stride, roles=list(roles), dense_layers=dense_layers,
                layers=records, parameters_added=0, state_keys_unchanged=True,
                dense_queries=True, dense_feedforward_and_norm=True,
                paired_key_mass=2, unpaired_key_mass=1,
                training_and_masks_fall_back_to_dense=True)
