"""Cached query projections for fixed steering-grid attention.

Evaluation-mode float32 and float64 queries can reuse nonpersistent projected
steering tokens. Keys and values remain observation dependent. Other inputs
use the standard attention implementation."""
from __future__ import annotations
from types import MethodType
import weakref
import torch
from torch import nn
from torch.nn import functional as F
from .shared_delta import SharedDeltaAttention


def _signature(query, parameters):
    return (query.device, query.dtype, query.data_ptr(), query._version,
            tuple((p.data_ptr(), p._version, p.device, p.dtype) for p in parameters))


def _query_parameters(attention):
    if isinstance(attention, SharedDeltaAttention):
        projection = attention.in_projection
        projection.retie()
        return tuple(projection.parameters())
    return tuple(p for p in (attention.in_proj_weight, attention.in_proj_bias) if p is not None)


def _project_query(attention, query):
    if isinstance(attention, SharedDeltaAttention):
        return attention.in_projection.forward_slice(query, 0, attention.embed_dim)
    bias = None if attention.in_proj_bias is None else attention.in_proj_bias[:attention.embed_dim]
    return F.linear(query, attention.in_proj_weight[:attention.embed_dim], bias)


def _refresh(attention, token, parameters):
    # Normal tensors can be replayed under either no_grad or inference_mode.
    with torch.inference_mode(False), torch.no_grad():
        attention._fixed_query_projected = _project_query(attention, token).detach()
    attention._fixed_query_signature = _signature(token, parameters)


def _fixed_query_forward(self, query, key, value, key_padding_mask=None,
                         need_weights=True, attn_mask=None,
                         average_attn_weights=True, is_causal=False):
    fallback = self._fixed_query_original_forward
    eligible = (not self.training and not torch.is_grad_enabled()
                and query.ndim == key.ndim == value.ndim == 3
                and query.shape[0] > 0 and key is value and query is not key
                and not need_weights and key_padding_mask is None
                and attn_mask is None and not is_causal
                and query.dtype in (torch.float32, torch.float64)
                and not torch.is_autocast_enabled(query.device.type)
                and not query.is_inference())
    token = None
    if eligible:
        # Pointer/stride/version checks never inspect scene data or synchronize
        # a CUDA device. Every installed root uses an expand of its own buffer.
        for root_ref in self._fixed_query_roots:
            root = root_ref()
            fixed = None if root is None else getattr(root, '_frozen_base_query', None)
            if (root is not None and not root.training and fixed is not None
                    and not fixed.is_inference() and fixed.device == query.device
                    and fixed.dtype == query.dtype and query.shape[1:] == fixed.shape[1:]
                    and query.data_ptr() == fixed.data_ptr()
                    and query.stride()[1:] == fixed.stride()[1:]
                    and (len(query) == 1 or query.stride(0) == 0)
                    and query._version == fixed._version):
                token = fixed
                break
        eligible = token is not None
    if eligible:
        parameters = _query_parameters(self)
        eligible = all(not p.requires_grad and not p.is_inference() for p in parameters)
    if not eligible:
        return fallback(self, query, key, value, key_padding_mask=key_padding_mask,
                        need_weights=need_weights, attn_mask=attn_mask,
                        average_attn_weights=average_attn_weights, is_causal=is_causal)
    signature = _signature(token, parameters)
    if signature != self._fixed_query_signature:
        _refresh(self, token, parameters)
    batch, target, d = query.shape
    q = self._fixed_query_projected.expand(batch, -1, -1)
    if isinstance(self, SharedDeltaAttention):
        k, v = self.in_projection.forward_slice(key, d, 3*d).chunk(2, dim=-1)
    else:
        bias = None if self.in_proj_bias is None else self.in_proj_bias[d:3*d]
        k, v = F.linear(key, self.in_proj_weight[d:3*d], bias).chunk(2, dim=-1)
    source = key.shape[1]
    head_dim = d // self.num_heads
    q = q.reshape(batch, target, self.num_heads, head_dim).transpose(1, 2)
    k = k.reshape(batch, source, self.num_heads, head_dim).transpose(1, 2)
    v = v.reshape(batch, source, self.num_heads, head_dim).transpose(1, 2)
    attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0., is_causal=False)
    attended = attended.transpose(1, 2).contiguous().reshape(batch, target, d)
    return self.out_proj(attended), None


def install_fixed_query_projection(model):
    """Install after strict checkpoint load and compaction, before graph capture."""
    if getattr(model, '_graphs', None):
        raise RuntimeError('Install fixed query projection before CUDA graphs')
    roots = (('original', model.engine.original), ('adapted', model.engine.dual.localizer),
             ('retrieval', model.retrieval), ('close_pair', model.engine.dual.close_pair_localizer))
    groups = {}
    for name, root in roots:
        attention = root.base.cov_cross.cross_attn
        groups.setdefault(id(attention), []).append((name, root, attention))
    result = []
    for aliases in groups.values():
        _, root, attention = aliases[0]
        token = getattr(root, '_frozen_base_query', None)
        if token is None:
            # Without compaction the steering query remains dynamic or lazy;
            # retain the complete original path rather than infer constancy.
            continue
        ordinary = (isinstance(attention, nn.MultiheadAttention)
                    and attention._qkv_same_embed_dim and attention.batch_first
                    and attention.bias_k is None and attention.bias_v is None
                    and not attention.add_zero_attn)
        if not (ordinary or isinstance(attention, SharedDeltaAttention)):
            raise TypeError('Fixed query projection requires ordinary batch-first attention')
        if attention.training or any(p.requires_grad for p in attention.parameters()):
            raise ValueError('Fixed query projection requires a frozen eval attention')
        parameters = _query_parameters(attention)
        if token.is_inference() or any(p.is_inference() for p in parameters):
            # Inference tensors have no version counter; retain the full path.
            continue
        keys_before = tuple(attention.state_dict())
        ids_before = tuple(id(p) for p in attention.parameters())
        if not hasattr(attention, '_fixed_query_original_forward'):
            attention._fixed_query_original_forward = type(attention).forward
            attention.register_buffer('_fixed_query_projected', None, persistent=False)
            attention.forward = MethodType(_fixed_query_forward, attention)
        attention._fixed_query_roots = tuple(weakref.ref(item[1]) for item in aliases)
        parameters = _query_parameters(attention)
        _refresh(attention, token, parameters)
        assert keys_before == tuple(attention.state_dict())
        assert ids_before == tuple(id(p) for p in attention.parameters())
        d, length = attention.embed_dim, token.shape[1]
        rank = attention.in_projection.rank if isinstance(attention, SharedDeltaAttention) else 0
        result.append(dict(roots=[name for name, _, _ in aliases], query_tokens=length,
                           query_width=d, delta_rank=rank,
                           projected_query_bytes=attention._fixed_query_projected.numel()
                                                 * attention._fixed_query_projected.element_size(),
                           saved_projection_macs_per_scene=length*(d*d+2*d*rank),
                           parameters_added=0, state_keys_unchanged=True))
    return dict(installed_attention_modules=len(result), encoders=result,
                projected_query_bytes=sum(item['projected_query_bytes'] for item in result),
                saved_projection_macs_per_scene=sum(item['saved_projection_macs_per_scene'] for item in result),
                scene_kv_cached=False, dense_weight_copy=False, checkpoint_keys_unchanged=True)
