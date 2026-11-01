"""CUDA graphs for single-scene wide and fine refinement.

Static route slots support graph replay. Wide tokens are recomputed during
replay. Outputs remain valid until the next call for the same route role;
the decoder consumes them within its serialized prediction operation."""
from __future__ import annotations
from dataclasses import dataclass
from collections import OrderedDict
import threading
import torch
from torch.nn import functional as F
from .compact_refinement import _route_context, _physical_tokens, _resample_wide
MIB=1024*1024


def trusted_b1_refinement(root, encoded, support_index):
    """Evaluate one scene; callers provide valid coarse-grid route indices."""
    routes=support_index.numel()
    support_batch=torch.zeros_like(support_index)
    _,covariance_feature,route_angle,offset_residual,support=_route_context(
        root,encoded,support_batch,support_index,True)
    core,adapter=root.compact_core,root.compact_adapter
    key,value=core.covariance_kv(adapter.cov_proj(covariance_feature))
    token=_physical_tokens(root,route_angle,root.wide_offsets,support,0)
    atoms=token.shape[1]
    query=core._split_heads(core.cross_q(core.cross_norm(token)).reshape(1,routes*atoms,core.width))
    attended=F.scaled_dot_product_attention(query,key,value,dropout_p=0.)
    attended=attended.transpose(1,2).reshape(routes,atoms,core.width)
    wide_token=token+core.cross_out(attended)
    wide_token=wide_token+core.cross_ffn(core.cross_ffn_norm(wide_token))
    summary=wide_token.mean(1)+support
    q,k,v=core.support_qkv(core.support_norm(summary)).unsqueeze(0).chunk(3,dim=-1)
    q,k,v=(core._split_heads(x) for x in (q,k,v))
    distance=((route_angle[None,:,None]-route_angle[None,None,:]).abs()/12.).clamp_max(1.)
    bias=-F.softplus(core.support_distance_log_scale)[None,:,None,None]*distance[:,None].square()
    attended=F.scaled_dot_product_attention(q,k,v,attn_mask=bias,dropout_p=0.)
    attended=attended.transpose(1,2).reshape(routes,core.width)
    support_context=summary+core.support_out(attended)
    support_context=support_context+core.support_ffn(core.support_ffn_norm(support_context))
    wide_joint=wide_token+support_context[:,None]
    wide_logits=adapter.wide_head(adapter.condition(wide_joint)).squeeze(-1)
    wide_residual=(F.softmax(wide_logits,dim=-1)*root.wide_offsets).sum(-1)
    base_angle=route_angle+wide_residual
    latent=_resample_wide(wide_joint,root.wide_offsets,root.fine_offsets,wide_residual)
    fine_physics=_physical_tokens(root,base_angle,root.fine_offsets,support,1)
    fine_input=latent+fine_physics
    if getattr(root,'compact_fine_variant','legacy')=='remove_fine_support':
        fine_input=fine_input-support[:,None]
    fine_token=core.fine_update(adapter.condition(fine_input))
    fine_logits=adapter.fine_head(fine_token).squeeze(-1)
    fine_probability=F.softmax(fine_logits,dim=-1)
    raw_residual=(fine_probability*root.fine_offsets).sum(-1)
    correction=torch.tanh(adapter.correction_gate_logit)*raw_residual
    route_residual=torch.zeros_like(raw_residual)
    route_blend=torch.zeros_like(raw_residual)
    if root.use_route_residual:
        route_residual=torch.tanh(adapter.residual_head(support_context).squeeze(-1))*root.route_residual_span
        route_blend=torch.sigmoid(adapter.blend_head(support_context).squeeze(-1))
    conditional_angle=base_angle+(1.-route_blend)*correction+route_blend*route_residual
    return {'fine_logits':fine_logits,'fine_probability':fine_probability,
            'base_angle':base_angle,'route_base_angle':route_angle,
            'route_offset_residual':offset_residual,'wide_logits':wide_logits,
            'wide_residual':wide_residual,'raw_residual':raw_residual,
            'conditional_angle':conditional_angle,'route_residual':route_residual,
            'route_blend':route_blend,'R_hat':encoded['covariance'].new_zeros(1,root.N,root.N),
            'source_power':raw_residual.new_zeros(routes),'noise_power':raw_residual.new_zeros(1),
            'local_candidate_count':int(routes*(root.M+root.wide_offsets.numel()))}


@dataclass
class _Entry:
    graph: object
    encoded: dict
    support_index: torch.Tensor
    outputs: dict
    bytes: int
    last_used: int = 0


class _SharedStream:
    def __init__(self):
        self.stream=None
        self.device=None
    def get(self,device):
        if self.stream is None:
            self.stream=torch.cuda.Stream(device=device)
            self.device=device
        elif self.device!=device:
            raise RuntimeError('A compact-refiner stream cannot migrate devices')
        return self.stream


class B1GraphedCompactRefiner:
    input_names=('location_logits','grid_feature','covariance_feature','base_offset','covariance')
    def __init__(self,root,stream,*,max_cached_bytes=32*MIB,workspace_allowance=16*MIB,
                 stable_requests=3,cooldown=64):
        self.root=root
        self.shared_stream=stream
        self.max_cached_bytes=int(max_cached_bytes)
        self.workspace_allowance=int(workspace_allowance)
        self.stable_requests=int(stable_requests)
        self.cooldown=int(cooldown)
        self.entries=OrderedDict();self.signature=None
        self.capacity=2;self.seen={};self.manager=None
        self.calls=self.builds=self.replays=self.evictions=0
        self.fallbacks={};self.measured_bytes={};self.retry_after={}
        self.last_requested=None;self.consecutive=0;self.last_capture=-cooldown
        self._lock=threading.RLock()
        self.peer_cached_bytes=lambda:0
    @property
    def entry(self):
        return next(reversed(self.entries.values())) if self.entries else None
    @property
    def key(self):
        return next(reversed(self.entries)) if self.entries else None
    @property
    def cached_bytes(self):
        return sum(entry.bytes for entry in self.entries.values())
    def _fallback(self,reason,encoded,sb,si):
        self.fallbacks[reason]=self.fallbacks.get(reason,0)+1
        return self.root.refine_routes(encoded,sb,si,detach_base=True,build_reconstruction=False)
    def _signature(self,encoded):
        tensors=tuple(self.root.compact_core.parameters())+tuple(self.root.compact_adapter.parameters())
        tensors+=tuple(getattr(self.root,name) for name in
                       ('theta_grid','array_position','wide_offsets','fine_offsets'))
        return (tuple((x.data_ptr(),x._version,str(x.device),str(x.dtype)) for x in tensors),
                tuple((name,tuple(encoded[name].shape),str(encoded[name].dtype),str(encoded[name].device))
                      for name in self.input_names),
                getattr(self.root,'compact_fine_variant','legacy'),
                self.root.use_route_residual,self.root.use_route_offset_residual,
                self.root.interpolate_route_context,self.root.soft_route_power)
    def __call__(self,encoded,support_batch,support_index):
        with self._lock:
            self.calls+=1
            x=encoded['grid_feature'];routes=len(support_index)
            if (x.device.type!='cuda' or len(x)!=1 or not 1<=routes<=64
                    or support_index.ndim!=1 or support_index.dtype!=torch.long
                    or support_batch.shape!=support_index.shape or support_batch.dtype!=torch.long
                    or support_index.device!=x.device or support_batch.device!=x.device):
                return self._fallback('shape_device',encoded,support_batch,support_index)
            if (self.root.training or torch.is_grad_enabled()
                    or torch.is_autocast_enabled(x.device.type)
                    or any(child.training for module in (self.root.compact_core,self.root.compact_adapter) for child in module.modules())
                    or any(p.requires_grad or p.is_inference() for module in
                           (self.root.compact_core,self.root.compact_adapter) for p in module.parameters())):
                if self.manager is not None:self.manager.clear()
                else:self.clear()
                return self._fallback('mode',encoded,support_batch,support_index)
            signature=self._signature(encoded)
            key=(routes,x.device.index,str(x.dtype))
            if self.signature is not None and signature!=self.signature:
                if self.manager is not None:self.manager.clear()
                else:self.clear(reset_admission=True)
            self.signature=signature
            if key==self.last_requested:self.consecutive+=1
            else:self.last_requested=key;self.consecutive=1
            self.seen[key]=self.seen.get(key,0)+1
            if key not in self.entries:
                recovering=self.calls<self.retry_after.get(key,0)
                if self.seen[key]<self.stable_requests or (recovering and self.consecutive<4):
                    return self._fallback('keyed_admission',encoded,support_batch,support_index)
                if not self._capture(encoded,support_index,key):
                    return self._fallback('capture_budget',encoded,support_batch,support_index)
            self.entries.move_to_end(key)
            entry=self.entries[key]
            if self.manager is not None:
                self.manager.clock+=1;entry.last_used=self.manager.clock
            else:entry.last_used=self.calls
            with torch.inference_mode(False),torch.no_grad():
                for name in self.input_names:entry.encoded[name].copy_(encoded[name])
                entry.support_index.copy_(support_index)
                entry.graph.replay()
            self.replays+=1
            return dict(entry.outputs)
    def _evict(self,key=None):
        if self.entries:
            key=next(iter(self.entries)) if key is None else key
            entry=self.entries.pop(key)
            torch.cuda.synchronize(entry.support_index.device)
            entry.outputs.clear();entry.graph.reset()
            self.retry_after[key]=self.calls+self.cooldown
            self.evictions+=1
    def _capture(self,encoded,support_index,key):
        while len(self.entries)>=self.capacity:self._evict()
        estimate=self.measured_bytes.get(str(key),4*MIB if len(support_index)>=16 else 2*MIB)
        if self.manager is not None and not self.manager.make_room(estimate):return False
        graph=None;outputs=None;complete=False
        try:
            with torch.cuda.device(support_index.device),torch.inference_mode(False),torch.no_grad():
                stream=self.shared_stream.get(support_index.device)
                static={name:torch.empty_like(encoded[name]) for name in self.input_names}
                for name in self.input_names:static[name].copy_(encoded[name])
                indices=torch.empty_like(support_index);indices.copy_(support_index)
                stream.wait_stream(torch.cuda.current_stream(support_index.device))
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        warm=trusted_b1_refinement(self.root,static,indices);del warm
                torch.cuda.current_stream(support_index.device).wait_stream(stream)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):
                    outputs=trusted_b1_refinement(self.root,static,indices)
                torch.cuda.current_stream(support_index.device).wait_stream(stream)
                pool=tuple(graph.pool())
                segments=[segment for segment in torch.cuda.memory_snapshot()
                          if segment['device']==support_index.device.index
                          and tuple(segment.get('segment_pool_id',()))==pool]
                if not segments:return False
                static_bytes=sum(value.untyped_storage().nbytes() for value in static.values())
                static_bytes+=indices.untyped_storage().nbytes()
                owned=sum(segment['total_size'] for segment in segments)+static_bytes
                self.measured_bytes[str(key)]=owned
                if self.manager is not None:
                    if not self.manager.make_room(owned):return False
                elif owned+self.cached_bytes+self.workspace_allowance>self.max_cached_bytes:
                    return False
                self.entries[key]=_Entry(graph,static,indices,outputs,owned)
                self.builds+=1;self.last_capture=self.calls;complete=True
                return True
        except torch.cuda.OutOfMemoryError:
            self.fallbacks['oom']=self.fallbacks.get('oom',0)+1
            return False
        finally:
            if not complete:
                if isinstance(outputs,dict):outputs.clear()
                if graph is not None:graph.reset()
                # Failed/budget-limited shapes are retried later; never permanently disabled.
                self.retry_after[key]=self.calls+self.cooldown
    def clear(self,*,reset_admission=True):
        with self._lock:
            while self.entries:self._evict()
            if reset_admission:
                self.last_requested=None;self.consecutive=0;self.signature=None
                self.retry_after.clear();self.seen.clear()
    def stats(self):
        return dict(calls=self.calls,builds=self.builds,replays=self.replays,evictions=self.evictions,
                    cached_graphs=len(self.entries),capacity=self.capacity,
                    cached_pool_and_static_bytes=self.cached_bytes,
                    cached_keys=[str(key) for key in self.entries],
                    keyed_seen={str(key):value for key,value in self.seen.items()},
                    max_cached_bytes=self.max_cached_bytes,workspace_allowance_bytes=self.workspace_allowance,
                    measured_pool_and_static_bytes=self.measured_bytes,fallbacks=dict(self.fallbacks),
                    peer_cached_pool_and_static_bytes=self.peer_cached_bytes(),
                    wide_route_cache_reused=False,shared_capture_stream=True,
                    budget_excludes_transient_capture_peak=True)


class RoleRefinerGraphs:
    # Initial and expanded passes have independent shape histories. They share
    # a retained pool budget so dense R12/R40 alternation can use both graphs.
    def __init__(self,root,stream,max_cached_bytes):
        self.phases={phase:B1GraphedCompactRefiner(root,stream,max_cached_bytes=max_cached_bytes)
                     for phase in ('initial','expanded')}
        self.max_cached_bytes=max_cached_bytes
        self.clock=0
        for phase,runner in self.phases.items():
            runner.manager=self
            runner.peer_cached_bytes=lambda phase=phase:sum(
                other.cached_bytes for name,other in self.phases.items() if name!=phase)
    @property
    def cached_bytes(self):
        return sum(runner.cached_bytes for runner in self.phases.values())
    def make_room(self,needed):
        available=self.max_cached_bytes-16*MIB
        if needed>available:return False
        while self.cached_bytes+needed>available:
            choices=[(entry.last_used,phase,key) for phase,runner in self.phases.items()
                     for key,entry in runner.entries.items()]
            if not choices:return False
            _,phase,key=min(choices)
            self.phases[phase]._evict(key)
        return True
    def __call__(self,encoded,sb,si,*,phase):
        if phase not in self.phases:
            raise ValueError('Unknown trusted refinement phase')
        return self.phases[phase](encoded,sb,si)
    def clear(self):
        for runner in self.phases.values():runner.clear()
    def stats(self):
        phases={name:runner.stats() for name,runner in self.phases.items()}
        total=sum(row['cached_pool_and_static_bytes'] for row in phases.values())
        return dict(phases=phases,cached_pool_and_static_bytes=total,
                    shared_workspace_allowance_bytes=16*MIB,
                    max_cached_bytes_including_workspace=self.max_cached_bytes,
                    within_retained_budget=total+16*MIB<=self.max_cached_bytes,
                    capacity_per_phase=2,maximum_graphs_per_root=4)


def install_b1_refiner_graphs(model,*,max_cached_bytes=32*MIB):
    stream=_SharedStream()
    roots={'original':model.engine.original,'adapted':model.engine.dual.localizer,
           'close_pair':model.engine.dual.close_pair_localizer}
    model._refiner_graphs={name:RoleRefinerGraphs(root,stream,max_cached_bytes)
                           for name,root in roots.items()}
    model._refiner_graph_by_root={id(roots[name]):runner for name,runner in model._refiner_graphs.items()}
