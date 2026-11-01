"""Shared encoder weights with path-specific low-rank corrections.

Each projection evaluates x W0.T + (x B.T) A.T + bias. The shared matrix W0
is registered once under original.base; experts reference that parameter.
Normalization, bias, and convolution parameters remain path specific."""
from __future__ import annotations
import copy
import math
import weakref
from collections import OrderedDict
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

FORMAT = "pgbsa_shared_delta_v3"
OWNER = "original.base"
ROOT_KEYS = {"original", "localizer", "retrieval", "close_pair"}


def _beneficial(weight, rank):
    return (weight.ndim == 2 and rank <= min(weight.shape)
            and rank * sum(weight.shape) < weight.numel())


class SharedDeltaLinear(nn.Module):
    """Linear projection with a shared dense matrix and low-rank residual."""
    def __init__(self, owner, owner_attribute, expert_weight, expert_bias, rank,
                 owner_key, *, initialize):
        super().__init__()
        shared = getattr(owner, owner_attribute)
        if shared.shape != expert_weight.shape or not _beneficial(shared, rank):
            raise ValueError("Invalid shared linear factor shape or unprofitable rank")
        self.in_features, self.out_features = shared.shape[1], shared.shape[0]
        self.rank = int(rank)
        self.owner_key = str(owner_key)
        self._owner_ref = weakref.ref(owner)
        self._owner_attribute = owner_attribute
        self.register_parameter("shared_weight", shared)
        self.delta_left = nn.Parameter(shared.new_zeros((self.out_features, rank)),
                                       requires_grad=expert_weight.requires_grad)
        self.delta_right = nn.Parameter(shared.new_zeros((rank, self.in_features)),
                                        requires_grad=expert_weight.requires_grad)
        self.register_parameter("bias", expert_bias)
        if initialize:
            with torch.no_grad():
                difference = (expert_weight.detach().float() - shared.detach().float())
                u, s, vh = torch.linalg.svd(difference, full_matrices=False)
                self.delta_left.copy_((u[:, :rank] * s[:rank]).to(shared.dtype))
                self.delta_right.copy_(vh[:rank].to(shared.dtype))

    def retie(self):
        owner = self._owner_ref()
        if owner is None:
            raise RuntimeError("Shared projection owner was destroyed")
        current = getattr(owner, self._owner_attribute)
        if self.shared_weight is not current:
            self.shared_weight = current
        return self.shared_weight

    def _apply(self, fn, recurse=True):
        # The public owner is moved before expert roots in this deployment.
        # Rebind after conversion rather than retaining an independent copy.
        result = super()._apply(fn, recurse=recurse)
        self.retie()
        return result

    def forward_slice(self, x, start=0, stop=None):
        weight = self.retie()
        stop = self.out_features if stop is None else stop
        bias = None if self.bias is None else self.bias[start:stop]
        return (F.linear(x, weight[start:stop], bias)
                + F.linear(F.linear(x, self.delta_right), self.delta_left[start:stop]))

    def forward(self, x):
        return self.forward_slice(x)


def _dense_projection(weight, bias):
    layer = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None,
                      device=weight.device, dtype=weight.dtype)
    layer.weight = weight
    layer.bias = bias
    return layer


class SharedDeltaAttention(nn.Module):
    """MHA-compatible batched projection and SDPA without fused weight access."""
    def __init__(self, owner, expert, rank, owner_path, *, initialize):
        super().__init__()
        if (not owner._qkv_same_embed_dim or not expert._qkv_same_embed_dim
                or owner.embed_dim != expert.embed_dim or owner.num_heads != expert.num_heads
                or expert.bias_k is not None or expert.bias_v is not None or expert.add_zero_attn):
            raise ValueError("Only equal-width ordinary QKV attention is supported")
        self.embed_dim = expert.embed_dim
        self.num_heads = expert.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.dropout = expert.dropout
        self.batch_first = expert.batch_first
        if not self.batch_first:
            raise ValueError("Shared-delta deployment uses batched, batch-first attention")
        self._qkv_same_embed_dim = True
        self.kdim = self.vdim = self.embed_dim
        self.in_projection = SharedDeltaLinear(
            owner, "in_proj_weight", expert.in_proj_weight, expert.in_proj_bias,
            rank, owner_path + ".in_proj_weight", initialize=initialize)
        self.out_proj = SharedDeltaLinear(
            owner.out_proj, "weight", expert.out_proj.weight, expert.out_proj.bias,
            rank, owner_path + ".out_proj.weight", initialize=initialize)
        self.train(expert.training)

    @staticmethod
    def _additive(mask, dtype, device):
        if mask.dtype == torch.bool:
            return torch.zeros(mask.shape, dtype=dtype, device=device).masked_fill(mask, -float("inf"))
        if not torch.is_floating_point(mask):
            raise TypeError("Attention masks must be boolean or floating point")
        return mask.to(device=device, dtype=dtype)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True,
                attn_mask=None, average_attn_weights=True, is_causal=False):
        if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
            raise ValueError("Expected batched [B,L,D] attention tensors")
        d = self.embed_dim
        if query is key and key is value:
            q, k, v = self.in_projection(query).chunk(3, dim=-1)
        elif key is value:
            q = self.in_projection.forward_slice(query, 0, d)
            k, v = self.in_projection.forward_slice(key, d, 3*d).chunk(2, dim=-1)
        else:
            q = self.in_projection.forward_slice(query, 0, d)
            k = self.in_projection.forward_slice(key, d, 2*d)
            v = self.in_projection.forward_slice(value, 2*d, 3*d)
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


class ExplicitEncoderLayer(nn.TransformerEncoderLayer):
    """The ordinary residual/LN equations, bypassing fused dense-weight reads."""
    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask,
                                    is_causal=is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask,
                                            is_causal=is_causal))
            x = self.norm2(x + self._ff_block(x))
        return x

    def _ff_block(self, x):
        chunk = int(getattr(self, "_shared_delta_ffn_chunk", 0))
        if chunk > 0 and not self.training and not torch.is_grad_enabled() and x.shape[1] > chunk:
            pieces = [super(ExplicitEncoderLayer, self)._ff_block(piece)
                      for piece in x.split(chunk, dim=1)]
            return torch.cat(pieces, dim=1)
        return super()._ff_block(x)


def _replace(parent, path, replacement):
    parts = path.split(".")
    target = parent
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], replacement)


def apply_shared_delta_models(models, rank=32, *, initialize=True):
    """Attach shared projections and path-specific low-rank corrections.

    Use initialize=True to factor the differences between loaded dense
    encoder weights. Use initialize=False before strict loading of a
    shared-delta checkpoint, which supplies the shared and factor tensors.
    """
    if not isinstance(models, Mapping) or set(models) != ROOT_KEYS:
        raise ValueError("Expected the four model roots")
    if isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank < 64:
        raise ValueError("Shared attention projection rank must be in [1,63]")
    if models["original"].base is not models["localizer"].base:
        raise ValueError("Share original/localizer.base first")
    if len({id(models[k].base) for k in ("original", "retrieval", "close_pair")}) != 3:
        raise ValueError("Expected three independent encoder backbones before conversion")
    if any(isinstance(m, SharedDeltaLinear) for root in models.values() for m in root.modules()):
        raise ValueError("Model already uses shared deltas")
    owner = models["original"].base
    owner_modules = dict(owner.named_modules())
    plans = []
    for name in ("retrieval", "close_pair"):
        base = models[name].base
        expert_modules = dict(base.named_modules())
        attention_paths = [p for p,m in owner_modules.items() if isinstance(m, nn.MultiheadAttention)]
        for path in attention_paths:
            original, expert = owner_modules[path], expert_modules[path]
            plans.append((name, path, SharedDeltaAttention(
                original, expert, rank, "base."+path, initialize=initialize)))
        for path, original in owner_modules.items():
            if (not isinstance(original, nn.Linear)
                    or any(path.startswith(p+".") for p in attention_paths)):
                continue
            expert = expert_modules[path]
            if _beneficial(original.weight, rank):
                replacement = SharedDeltaLinear(original, "weight", expert.weight, expert.bias,
                                                  rank, "base."+path+".weight", initialize=initialize)
                replacement.train(expert.training)
                plans.append((name, path, replacement))
    for name, path, replacement in plans:
        _replace(models[name].base, path, replacement)
    for name in ("retrieval", "close_pair"):
        for encoder in (models[name].base.cov_encoder, models[name].base.grid_transformer):
            encoder.use_nested_tensor = False
            for layer in encoder.layers:
                layer.__class__ = ExplicitEncoderLayer
                layer._shared_delta_ffn_chunk = 0
    retie_shared_weights(models)
    summary = shared_delta_summary(models)
    summary["rank"] = rank
    return summary


def retie_shared_weights(models):
    seen = set()
    for root in models.values():
        for module in root.modules():
            if id(module) in seen:
                continue
            seen.add(id(module))
            if isinstance(module, SharedDeltaLinear):
                module.retie()


def shared_delta_summary(models):
    matrices, seen = [], set()
    for name in ("retrieval", "close_pair"):
        for path, module in models[name].base.named_modules():
            if isinstance(module, SharedDeltaLinear):
                owner = module.retie()
                if id(module) in seen:
                    continue
                seen.add(id(module))
                original = module.in_features * module.out_features
                factors = module.rank * (module.in_features + module.out_features)
                matrices.append(dict(expert=name, path="base."+path, owner_key=module.owner_key,
                                     shape=[module.out_features,module.in_features], rank=module.rank,
                                     original_weight_parameters=original,delta_parameters=factors,
                                     parameters_removed=original-factors,
                                     alias_is_owner=module.shared_weight is getattr(module._owner_ref(),module._owner_attribute)))
    return dict(matrices=matrices,parameters_removed=sum(row["parameters_removed"] for row in matrices),
                shared_owner=OWNER,expert_dense_weight_cached=False)


def set_shared_delta_ffn_chunk(models, chunk=0):
    """Optional evaluation-only token chunks; default 0, training always full."""
    if isinstance(chunk, bool) or not isinstance(chunk, int) or chunk < 0:
        raise ValueError("FFN chunk must be a nonnegative integer")
    seen = set()
    for name, root in models.items():
        for encoder in (root.base.cov_encoder, root.base.grid_transformer):
            if id(encoder) in seen:
                continue
            seen.add(id(encoder))
            for layer in encoder.layers:
                if chunk and not isinstance(layer, ExplicitEncoderLayer):
                    layer._shared_delta_restore_class = layer.__class__
                    layer.__class__ = ExplicitEncoderLayer
                layer._shared_delta_ffn_chunk = chunk
                if not chunk and hasattr(layer, "_shared_delta_restore_class"):
                    layer.__class__ = layer._shared_delta_restore_class
                    del layer._shared_delta_restore_class
            if chunk:
                encoder.use_nested_tensor = False


def expand_canonical_dual_state(state, original, dual):
    """Recreate aliases as tensor references for strict loading, never copies."""
    if any(key.startswith("localizer.base.") or key.endswith(".shared_weight") for key in state):
        raise ValueError("Canonical v3 state must not contain duplicated shared weights")
    expanded = dict(state)
    owner_state = original.state_dict()
    expanded.update({"localizer.base."+key:value for key,value in original.base.state_dict().items()})
    compact_core = getattr(original, "compact_core", None)
    if compact_core is not None:
        if dual.localizer.compact_core is not compact_core or dual.close_pair_localizer.compact_core is not compact_core:
            raise ValueError("Compact refiners must retain the original core owner")
        prefixes=("localizer.compact_core.", "close_pair_localizer.compact_core.")
        if any(key.startswith(prefixes) for key in state):
            raise ValueError("Canonical checkpoint duplicates its compact refinement core")
        for prefix in prefixes:
            expanded.update({prefix+key:value for key,value in compact_core.state_dict().items()})
    for path, module in dual.named_modules():
        if isinstance(module, SharedDeltaLinear):
            if module.owner_key not in owner_state:
                raise ValueError("Missing canonical shared owner: "+module.owner_key)
            expanded[path+".shared_weight"] = owner_state[module.owner_key]
    return expanded


def export_shared_delta_payload(template, models, auxiliary):
    """Canonical CPU state with owner tensors once; valid before compaction."""
    if set(auxiliary) != {"generator","scorer","offset"}:
        raise ValueError("Expected generator/scorer/offset modules")
    from .decoder import InferenceNetworks
    for name, root in models.items():
        step = float(root.theta_grid[1] - root.theta_grid[0])
        if float(template["configs"][name]["coarse_step"]) != step:
            raise ValueError("Export template grid does not match the live model: " + name)
    spec = getattr(models["original"], "_refinement_bottleneck_spec", None)
    if spec != template.get("refinement_bottleneck"):
        raise ValueError("Export template must retain the live refinement bottleneck marker")
    compact_spec = getattr(models["original"], "_compact_refinement_spec", None)
    if compact_spec != template.get("compact_refinement"):
        raise ValueError("Export template must retain the compact refinement specification")
    if models["original"].base.steer_proj is None:
        raise ValueError("Export before frozen inference compaction")
    dual = InferenceNetworks(models["localizer"],models["retrieval"],models["close_pair"],
                             template["state"]["dual"]["retrieval.theta_grid"])
    summary = shared_delta_summary(models)
    ranks = {row["rank"] for row in summary["matrices"]}
    if len(ranks) != 1:
        raise ValueError("Expected a single shared-delta rank")
    def cpu_state(state, skip=lambda key:False):
        result = OrderedDict((key,value.detach().cpu().clone()) for key,value in state.items() if not skip(key))
        # All BN statistics are explicit; retain version metadata for safe strict loading.
        if hasattr(state,"_metadata"):
            result._metadata = copy.deepcopy(state._metadata)
        return result
    result = copy.deepcopy(template)
    result["state"] = dict(original=cpu_state(models["original"].state_dict()),
                          dual=cpu_state(dual.state_dict(),lambda key:key.startswith(("localizer.base.","localizer.compact_core.","close_pair_localizer.compact_core.")) or key.endswith(".shared_weight") or "._proposal_generator." in key))
    for name,module in auxiliary.items():
        result["state"][name] = cpu_state(module.state_dict())
    unique = {id(p):p for root in [models["original"],dual,*auxiliary.values()] for p in root.parameters()}
    result["format"] = FORMAT
    result["shared_delta"] = dict(rank=ranks.pop(),owner=OWNER,kind="independent_linear_and_packed_qkv_deltas",
                                  convolution_shared=False,canonical_aliases_omitted=True)
    result["metadata"] = dict(result.get("metadata",{}),unique_parameters=sum(p.numel() for p in unique.values()),
                               neural_weights_unchanged=False,distillation_used=False,
                               shared_delta_initialization="weight_difference_SVD",
                               shared_delta_dense_weight_cached=False)
    result["shared_delta_initialization"] = summary
    return result


def convert_checkpoint_payload(payload, rank=32):
    """Strictly load ordinary v1, initialize delta factors, return canonical v3."""
    if payload.get("format") != "pgbsa_validated_v1" or "shared_delta" in payload:
        raise ValueError("Expected an unconverted ordinary v1 checkpoint")
    if "compression" in payload or "grid_reduction" in payload:
        raise ValueError("Convert the original full backbone, not a previously pruned architecture")
    from .networks import BeamDOANetwork
    from .angular_adapter import attach_adapter
    from .decoder import strip_reconstruction,strip_retrieval_refinement,InferenceNetworks,verify_encoding_contract
    from .heads import GlobalCandidateGenerator,CandidateScorer
    models = {name:BeamDOANetwork(**cfg) for name,cfg in payload["configs"].items()}
    attach_adapter(models["localizer"],radius_deg=2.,minimum_routes=11)
    for root in models.values():
        strip_reconstruction(root)
    verify_encoding_contract(models["original"],models["localizer"])
    models["localizer"].base=models["original"].base
    strip_retrieval_refinement(models["retrieval"])
    original=models["original"]
    original.load_state_dict(payload["state"]["original"],strict=True)
    dual=InferenceNetworks(models["localizer"],models["retrieval"],models["close_pair"],
                           payload["state"]["dual"]["retrieval.theta_grid"])
    state=dict(payload["state"]["dual"])
    if any(key.startswith("localizer.base.") for key in state):
        raise ValueError("Original checkpoint duplicates its shared backbone")
    state.update({"localizer.base."+key:value for key,value in original.base.state_dict().items()})
    dual.load_state_dict(state,strict=True)
    auxiliary={}
    for name,kind in (("generator",GlobalCandidateGenerator),("scorer",CandidateScorer),("offset",CandidateScorer)):
        auxiliary[name]=kind(**payload["model_shapes"][name])
        auxiliary[name].load_state_dict(payload["state"][name],strict=True)
    apply_shared_delta_models(models,rank=rank,initialize=True)
    return export_shared_delta_payload(payload,models,auxiliary)
