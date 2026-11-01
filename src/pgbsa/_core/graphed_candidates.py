"""CUDA graphs for expanded candidate feature computation.

Outputs use graph-owned storage until the next invocation. The decoder
serializes predictions and copies results to the host before reusing this
storage. Cache limits cover retained pools, static inputs, and stream
workspace, rather than total device memory or peak capture allocation."""
from __future__ import annotations
from collections import Counter, OrderedDict
from dataclasses import dataclass
import threading
import weakref
import torch
from .gpu_candidates import build_gpu_candidates
from .heads import FINE_OFFSETS, WIDE_OFFSETS, replace_profiles

MIB = 1024 * 1024
ROUTE_INPUTS = ('base_angle', 'route_base_angle', 'conditional_angle',
                'fine_probability', 'wide_logits', 'fine_logits')


def expanded_candidate_tensor(model, ge, supports, routes, *, prepared=None):
    """Assemble expanded candidates on the model device."""
    e = model.engine
    fine = routes['fine_logits'].softmax(-1)
    wide = routes['wide_logits'].softmax(-1)
    raw, features, _, _, _ = build_gpu_candidates(
        model.retrieval.theta_grid, ge, supports, routes, prepared=prepared)
    offsets = None if prepared is None else (prepared['fine_offsets'], prepared['wide_offsets'])
    adjusted = replace_profiles(features, features[:, :, 14]*70.,
                                routes['base_angle'], routes['route_base_angle'],
                                fine, wide, offsets=offsets)
    scores = e.scorer((adjusted-e.mean)/e.std)
    correction = e.bundle.offset((features-e.om)/e.os).clamp(-.35, .35)
    return torch.cat((torch.stack((raw+correction, scores), -1).flatten(1),
                      routes['route_base_angle'][:, None]), -1)


def initial_candidate_tensor(model, ge, supports, routes, *, prepared=None):
    """Score all independent initial candidates; compact only after one copy."""
    e = model.engine
    angles, features, keep, sb, _ = build_gpu_candidates(
        model.retrieval.theta_grid, ge, supports, routes, stable=False, prepared=prepared)
    scores = e.bundle.scorer((features-e.mean)/e.std)
    correction = e.bundle.offset((features-e.om)/e.os).clamp(-.35, .35)
    scene = sb[:, None].expand_as(keep).to(dtype=angles.dtype)
    return torch.stack((angles+correction, scores, scene, keep.to(dtype=angles.dtype)), -1)


def prepare_candidate_metadata(supports, device):
    """All CPU constants are copied before capture; counts are observed routes."""
    counts = tuple(len(s) for s in supports)
    return {
        'sb': torch.tensor([j for j,n in enumerate(counts) for _ in range(n)], dtype=torch.long, device=device),
        'si': torch.cat(supports) if sum(counts) else torch.empty(0, dtype=torch.long, device=device),
        'counts': torch.tensor(counts, dtype=torch.long, device=device),
        'ranks': torch.tensor([k for n in counts for k in range(n)], dtype=torch.long, device=device),
        'fine_offsets': torch.as_tensor(FINE_OFFSETS, device=device),
        'wide_offsets': torch.as_tensor(WIDE_OFFSETS, device=device),
    }


def _version(tensor):
    try:return tensor._version
    except RuntimeError:return None


def _tensor_signature(value):
    return (value.data_ptr(), tuple(value.shape), str(value.dtype), str(value.device), _version(value))


@dataclass
class _CandidateEntry:
    graph: object
    inputs: dict
    prepared: dict
    packed: torch.Tensor
    bytes: int
    last_used: int = 0


class ExpandedCandidateGraph:
    """Two LRU graphs, admitted after cumulative observed-shape reuse."""
    def __init__(self, model, *, max_cached_bytes=32*MIB, min_reuse=3,
                 workspace_allowance_bytes=16*MIB, phase='expanded', manager=None):
        if max_cached_bytes < MIB or not 1 <= min_reuse <= 16 or workspace_allowance_bytes < 0:
            raise ValueError('Invalid expanded candidate graph budget/admission')
        if any(m.training for m in (model.engine.scorer, model.engine.bundle.offset)):
            raise ValueError('Candidate graph requires eval-mode heads')
        if any(p.requires_grad for m in (model.engine.scorer, model.engine.bundle.offset) for p in m.parameters()):
            raise ValueError('Candidate graph requires frozen heads')
        if phase not in ('initial', 'expanded'):raise ValueError('Invalid candidate phase')
        self.phase = phase
        self._manager_ref = None if manager is None else weakref.ref(manager)
        self._model_ref = weakref.ref(model)
        self.max_cached_bytes = int(max_cached_bytes)
        self.workspace_allowance_bytes = int(workspace_allowance_bytes)
        self.min_reuse = int(min_reuse)
        self._lock = threading.RLock()
        self._stream = None
        self._device = None
        self.cache = OrderedDict()
        self.capacity = 2
        self._seen = OrderedDict()
        self._cooldown = OrderedDict()
        self._calls = 0
        self._state = None
        self._last_key = None
        self._consecutive = 0
        self._disabled = OrderedDict()
        self._fallbacks = Counter()
        self.builds = self.replays = self.evictions = self.budget_rejections = 0
        self.capture_errors = self.oom_fallbacks = self.signature_invalidations = 0
        self.last_capture_error = None

    def _model(self):
        model = self._model_ref()
        if model is None:raise RuntimeError('Candidate graph model was destroyed')
        return model

    def _state_signature(self):
        m = self._model();e=m.engine
        values = [m.retrieval.theta_grid,e.mean,e.std,e.om,e.os]
        values += list(e.scorer.parameters()) + list(e.bundle.offset.parameters())
        return tuple(_tensor_signature(value) for value in values)

    def _disable(self, key, reason):
        self._disabled[key] = reason
        self._disabled.move_to_end(key)
        while len(self._disabled) > 64:self._disabled.popitem(last=False)

    def _fallback(self, reason, ge, supports, routes):
        self._fallbacks[reason] += 1
        return self._function()(self._model(),ge,supports,routes)

    def _function(self):
        return initial_candidate_tensor if self.phase=='initial' else expanded_candidate_tensor

    def _manager(self):
        return None if self._manager_ref is None else self._manager_ref()

    @property
    def cached_bytes(self):
        return sum(entry.bytes for entry in self.cache.values())

    def _evict(self, key):
        entry=self.cache.pop(key)
        entry.graph.reset();entry.inputs.clear();entry.prepared.clear();entry.packed=None
        self.evictions+=1
        self._cooldown[key]=self._calls+64
        self._cooldown.move_to_end(key)
        while len(self._cooldown)>64:self._cooldown.popitem(last=False)

    def _release(self):
        while self.cache:self._evict(next(iter(self.cache)))

    def _invalidate_all(self):
        manager=self._manager()
        if manager is None:self.clear()
        else:manager.clear_admission()

    def _make_room(self, size):
        manager=self._manager()
        if manager is not None:return manager.make_room(size)
        available=self.max_cached_bytes-self.workspace_allowance_bytes
        if size>available:return False
        while self.cache and self.cached_bytes+size>available:self._evict(next(iter(self.cache)))
        return True

    def __call__(self, ge, supports, routes):
        model=self._model();x=ge['location_logits'];counts=tuple(len(s) for s in supports)
        if x.device.type!='cuda':return self._fallback('device',ge,supports,routes)
        if torch.is_grad_enabled():
            self._invalidate_all()
            return self._fallback('grad',ge,supports,routes)
        if torch.is_autocast_enabled(x.device.type):
            self._invalidate_all()
            return self._fallback('autocast',ge,supports,routes)
        heads=(model.engine.scorer,model.engine.bundle.offset)
        if (any(child.training for head in heads for child in head.modules())
                or any(p.requires_grad or p.is_inference() for head in heads for p in head.parameters())):
            self._invalidate_all()
            return self._fallback('unfrozen_or_training',ge,supports,routes)
        if not sum(counts):return self._fallback('empty',ge,supports,routes)
        if len(counts)!=len(x):raise ValueError('Candidate support/encoding batch mismatch')
        if any(len(routes[n])!=sum(counts) for n in ROUTE_INPUTS):raise ValueError('Candidate route batch mismatch')
        tensors=(x,)+(tuple(routes[n] for n in ROUTE_INPUTS))
        key=(counts,tuple((tuple(t.shape),str(t.dtype),str(t.device))for t in tensors))
        with self._lock:
            state=self._state_signature()
            if self._state is not None and self._state!=state:
                self._invalidate_all();self.signature_invalidations+=1
            self._state=state
            self._calls+=1
            self._consecutive=self._consecutive+1 if self._last_key==key else 1
            self._last_key=key
            self._seen[key]=self._seen.get(key,0)+1
            self._seen.move_to_end(key)
            while len(self._seen)>64:self._seen.popitem(last=False)
            if key not in self.cache:
                if key in self._disabled:return self._fallback(self._disabled[key],ge,supports,routes)
                if self._seen[key]<self.min_reuse:return self._fallback('low_cumulative_reuse',ge,supports,routes)
                if self._calls<self._cooldown.get(key,0) and self._consecutive<self.min_reuse:
                    return self._fallback('eviction_cooldown',ge,supports,routes)
                visible=sum(t.numel()*t.element_size()for t in tensors)
                estimate=2*MIB+visible+sum(counts)*31*2560
                if not self._make_room(estimate):
                    self.budget_rejections+=1;self._disable(key,'budget_estimate')
                    return self._fallback('budget_estimate',ge,supports,routes)
                while len(self.cache)>=self.capacity:self._evict(next(iter(self.cache)))
                entry=None;reason=None
                try:entry=self._capture(ge,supports,routes)
                except torch.cuda.OutOfMemoryError:reason='oom';self.oom_fallbacks+=1
                except RuntimeError as exc:
                    reason='capture_unsupported';self.capture_errors+=1;self.last_capture_error=str(exc)[:400]
                if entry is None:
                    reason=reason or 'pool_accounting_unavailable';self._disable(key,reason)
                    if reason=='oom':torch.cuda.empty_cache()
                    return self._fallback(reason,ge,supports,routes)
                if not self._make_room(entry.bytes):
                    self.budget_rejections+=1;self._disable(key,'measured_budget')
                    entry.graph.reset();entry.inputs.clear();entry.prepared.clear();entry.packed=None
                    return self._fallback('measured_budget',ge,supports,routes)
                self.cache[key]=entry;self.builds+=1
            self.cache.move_to_end(key)
            entry=self.cache[key]
            manager=self._manager()
            if manager is not None:entry.last_used=manager.touch()
            else:entry.last_used=self._calls
            with torch.inference_mode(False),torch.no_grad():
                entry.inputs['location_logits'].copy_(x)
                for name in ROUTE_INPUTS:entry.inputs[name].copy_(routes[name])
                entry.prepared['si'].copy_(torch.cat(supports))
                entry.graph.replay()
            self.replays+=1
            return entry.packed

    def _capture(self, ge, supports, routes):
        x=ge['location_logits'];graph=None;packed=None;complete=False
        try:
            with torch.cuda.device(x.device),torch.inference_mode(False),torch.no_grad():
                if self._stream is None:
                    manager=None if self._manager_ref is None else self._manager_ref()
                    self._stream=(torch.cuda.Stream(device=x.device) if manager is None else manager.stream(x.device))
                    self._device=x.device
                inputs={'location_logits':x.clone()}
                inputs.update({n:routes[n].clone()for n in ROUTE_INPUTS})
                prepared=prepare_candidate_metadata(supports,x.device)
                sg={'location_logits':inputs['location_logits']}
                sr={n:inputs[n]for n in ROUTE_INPUTS}
                self._stream.wait_stream(torch.cuda.current_stream(x.device))
                with torch.cuda.stream(self._stream):
                    for _ in range(3):
                        warm=self._function()(self._model(),sg,supports,sr,prepared=prepared)
                        del warm
                torch.cuda.current_stream(x.device).wait_stream(self._stream)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=self._stream):
                    packed=self._function()(self._model(),sg,supports,sr,prepared=prepared)
                torch.cuda.current_stream(x.device).wait_stream(self._stream)
                pool=tuple(graph.pool())
                segments=[s for s in torch.cuda.memory_snapshot()
                          if s['device']==x.device.index and tuple(s.get('segment_pool_id',()))==pool]
                if not segments:return None
                storages={}
                for t in (*inputs.values(),*prepared.values()):
                    storage=t.untyped_storage();storages[storage.data_ptr()]=storage.nbytes()
                size=sum(s['total_size']for s in segments)+sum(storages.values())
                complete=True
                return _CandidateEntry(graph,inputs,prepared,packed,size)
        finally:
            if not complete and graph is not None:graph.reset()

    def clear(self):
        with self._lock:
            if self._device is not None:torch.cuda.synchronize(self._device)
            self._release();self._last_key=None;self._consecutive=0;self._disabled.clear();self._state=None
            self._seen.clear();self._cooldown.clear()

    def stats(self):
        cached=self.cached_bytes
        return dict(capacity=self.capacity,cached_shapes=len(self.cache),graph_builds=self.builds,replays=self.replays,
                    evictions=self.evictions,cache_reserved_bytes=cached,
                    accounted_cache_bytes=cached+(self.workspace_allowance_bytes if self._stream is not None else 0),
                    max_cached_bytes=self.max_cached_bytes,workspace_allowance_bytes=self.workspace_allowance_bytes,
                    min_cumulative_reuse=self.min_reuse,graph_streams=int(self._stream is not None),
                    recent_shapes=len(self._seen),eviction_cooldown_calls=64,
                    fallback_counts=dict(self._fallbacks),budget_rejections=self.budget_rejections,
                    capture_errors=self.capture_errors,last_capture_error=self.last_capture_error,
                    oom_fallbacks=self.oom_fallbacks,signature_invalidations=self.signature_invalidations,
                    budget_is_total_device_memory_limit=False,disabled_shapes=len(self._disabled))


class DenseCandidateGraphs:
    """Share one CUDA stream and cache budget between both candidate phases.

    The default 32 MiB budget includes the stream workspace allowance and
    retained initial- and expanded-candidate graph pools.
    """
    def __init__(self,model,*,max_cached_bytes=32*MIB,min_reuse=3,workspace_allowance_bytes=16*MIB):
        self.max_cached_bytes=int(max_cached_bytes)
        self.workspace_allowance_bytes=int(workspace_allowance_bytes)
        self._stream=None
        self._clock=0
        self.runners={phase:ExpandedCandidateGraph(model,max_cached_bytes=max_cached_bytes,
                      min_reuse=min_reuse,workspace_allowance_bytes=0,phase=phase,manager=self)
                      for phase in ('initial','expanded')}
    def stream(self,device):
        if self._stream is None:self._stream=torch.cuda.Stream(device=device)
        return self._stream
    def touch(self):
        self._clock+=1
        return self._clock
    def available_for(self,requester=None):
        # Every retained entry, including the requester's other shape, counts.
        return self.max_cached_bytes-self.workspace_allowance_bytes-sum(r.cached_bytes for r in self.runners.values())
    def make_room(self,size):
        full=self.max_cached_bytes-self.workspace_allowance_bytes
        if size>full:return False
        while size>self.available_for():
            entries=[(entry.last_used,phase,key)for phase,r in self.runners.items()for key,entry in r.cache.items()]
            if not entries:return False
            _,phase,key=min(entries,key=lambda item:item[0])
            self.runners[phase]._evict(key)
        return True
    def clear_admission(self):
        for runner in self.runners.values():runner.clear()
    def __call__(self,ge,supports,routes):return self.runners['expanded'](ge,supports,routes)
    def initial(self,ge,supports,routes):return self.runners['initial'](ge,supports,routes)
    def clear(self):
        for runner in self.runners.values():runner.clear();runner._stream=None;runner._device=None
        self._stream=None
    def stats(self):
        phases={p:r.stats()for p,r in self.runners.items()}
        cached=sum(v['cache_reserved_bytes']for v in phases.values())
        return dict(phases=phases,capacity_per_phase=2,cached_shapes=sum(v['cached_shapes']for v in phases.values()),
                    cache_reserved_bytes=cached,accounted_cache_bytes=cached+(self.workspace_allowance_bytes if self._stream is not None else 0),
                    max_cached_bytes=self.max_cached_bytes,workspace_allowance_bytes=self.workspace_allowance_bytes,
                    graph_streams=int(self._stream is not None),budget_is_total_device_memory_limit=False)
