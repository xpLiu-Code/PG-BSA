"""Scaled dot-product attention for the shared dense encoder.

The implementation evaluates explicit query, key, and value projections
while retaining parameter identities and checkpoint keys."""
from __future__ import annotations
import math
from torch import nn
from torch.nn import functional as F
import torch
from .shared_delta import SharedDeltaAttention, ExplicitEncoderLayer


class DenseSDPAAttention(nn.MultiheadAttention):
    """In-place MHA class adaptation: all original Parameters retain identity."""
    _additive = staticmethod(SharedDeltaAttention._additive)

    def _project_slice(self,x,start,stop):
        bias=None if self.in_proj_bias is None else self.in_proj_bias[start:stop]
        return F.linear(x,self.in_proj_weight[start:stop],bias)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True,
                attn_mask=None, average_attn_weights=True, is_causal=False):
        if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
            raise ValueError("Expected batched [B,L,D] attention tensors")
        d = self.embed_dim
        if query is key and key is value:
            q, k, v = F.linear(query,self.in_proj_weight,self.in_proj_bias).chunk(3, dim=-1)
        elif key is value:
            q = self._project_slice(query, 0, d)
            k, v = self._project_slice(key, d, 3*d).chunk(2, dim=-1)
        else:
            q = self._project_slice(query, 0, d)
            k = self._project_slice(key, d, 2*d)
            v = self._project_slice(value, 2*d, 3*d)
        batch, target, _ = q.shape
        source = k.shape[1]
        q = q.reshape(batch, target, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch, source, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch, source, self.num_heads, self.head_dim).transpose(1, 2)
        mask = None
        if attn_mask is not None:
            mask = self._additive(attn_mask, q.dtype, q.device)
            if mask.ndim == 2:
                if mask.shape != (target, source):
                    raise ValueError("Invalid 2D attention mask shape")
                mask = mask[None, None]
            elif mask.ndim == 3:
                if mask.shape != (batch*self.num_heads, target, source):
                    raise ValueError("Invalid 3D attention mask shape")
                mask = mask.reshape(batch, self.num_heads, target, source)
            else:
                raise ValueError("Expected a 2D or 3D attention mask")
        if key_padding_mask is not None:
            if key_padding_mask.shape != (batch, source):
                raise ValueError("Invalid key padding mask shape")
            padding = self._additive(key_padding_mask, q.dtype, q.device)[:, None, None, :]
            mask = padding if mask is None else mask + padding
        if is_causal:
            blocked = torch.ones((target, source), device=q.device, dtype=torch.bool).triu(1)
            causal = self._additive(blocked, q.dtype, q.device)[None, None]
            mask = causal if mask is None else mask + causal
        dropout = self.dropout if self.training else 0.
        weights = None
        if need_weights:
            score = (q / math.sqrt(self.head_dim)) @ k.transpose(-2, -1)
            if mask is not None:
                score = score + mask
            weights = F.softmax(score, dim=-1)
            weights = F.dropout(weights, p=dropout, training=self.training)
            attended = weights @ v
        else:
            attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                                      dropout_p=dropout, is_causal=False)
        attended = attended.transpose(1, 2).contiguous().reshape(batch, target, d)
        output = self.out_proj(attended)
        if need_weights and average_attn_weights:
            weights = weights.mean(dim=1)
        return output, weights


def apply_dense_owner_sdpa(base):
    """Configure the shared covariance and grid encoders before graph capture.

    The operation preserves support attention and expert weights and is
    idempotent."""
    encoders=(base.cov_encoder,base.grid_transformer)
    attentions=[layer.self_attn for encoder in encoders for layer in encoder.layers]
    attentions.append(base.cov_cross.cross_attn)
    for attn in attentions:
        if (not isinstance(attn,nn.MultiheadAttention) or not attn.batch_first
                or not attn._qkv_same_embed_dim or attn.bias_k is not None
                or attn.bias_v is not None or attn.add_zero_attn):
            raise ValueError('Dense owner SDPA requires ordinary batch-first QKV attention')
    parameter_ids=tuple(id(p) for p in base.parameters())
    state_keys=tuple(base.state_dict())
    changed=0
    for attn in attentions:
        if not isinstance(attn,DenseSDPAAttention):
            attn.__class__=DenseSDPAAttention
            changed+=1
    for encoder in encoders:
        encoder.use_nested_tensor=False
        for layer in encoder.layers:
            layer.__class__=ExplicitEncoderLayer
            if hasattr(layer,'_shared_delta_restore_class'):
                layer._shared_delta_restore_class=ExplicitEncoderLayer
            if not hasattr(layer,'_shared_delta_ffn_chunk'):
                layer._shared_delta_ffn_chunk=0
    assert parameter_ids==tuple(id(p) for p in base.parameters())
    assert state_keys==tuple(base.state_dict())
    return {'dense_attention_modules':len(attentions),'newly_converted':changed,
            'parameters_added':0,'parameter_identity_preserved':True,
            'state_keys_unchanged':True,'owner_transformer_fused_path':False}
