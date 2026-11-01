"""CUDA Graph inference with bounded retained-cache storage.

Outputs use graph-owned storage. The enclosing model serializes prediction
and consumes outputs before replaying a graph. The cache budget includes
retained graph pools, static inputs, and stream workspace; peak capture and
total process allocations lie outside this budget."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import threading

import torch

MIB = 1024 * 1024


@dataclass
class _Entry:
    graph: object
    static: torch.Tensor
    outputs: dict
    bytes: int


class RobustGraphedEncoder:
    def __init__(self, module, *, buckets=(1, 2, 4, 8, 16, 32), capacity=3,
                 max_cached_bytes=384 * MIB, min_reuse=2, eviction_cooldown=64,
                 max_padding_ratio=2.0, workspace_allowance_bytes=16 * MIB,
                 initial_bytes_per_batch=6 * MIB):
        if getattr(module, 'use_snapshot_conditioning', False):
            raise ValueError('Graph cache requires an unconditioned encoder')
        if module.training or any(p.requires_grad for p in module.parameters()):
            raise ValueError('Graph cache requires an eval-mode frozen encoder')
        sizes = tuple(sorted(set(int(v) for v in buckets)))
        if not sizes or sizes[0] < 1 or capacity < 1 or min_reuse < 1:
            raise ValueError('Buckets, capacity and min_reuse must be positive')
        if max_cached_bytes < 1 or workspace_allowance_bytes < 0 or initial_bytes_per_batch <= 0:
            raise ValueError('Invalid memory budget')
        if eviction_cooldown < 0 or max_padding_ratio < 1:
            raise ValueError('Invalid admission policy')
        self.module = module
        self.forward = module.forward
        self.buckets, self.capacity = sizes, int(capacity)
        self.max_cached_bytes = int(max_cached_bytes)
        self.min_reuse, self.eviction_cooldown = int(min_reuse), int(eviction_cooldown)
        self.max_padding_ratio = float(max_padding_ratio)
        self.workspace_allowance_bytes = int(workspace_allowance_bytes)
        self.cache = OrderedDict()
        self._seen, self._cooldown, self._disabled = {}, {}, {}
        self._stream = None
        self._device_index = None
        self._signature = None
        self._state = None
        self._lock = threading.RLock()
        self._initial_bytes_per_batch = float(initial_bytes_per_batch)
        self._bytes_per_batch = self._initial_bytes_per_batch
        self._fixed_pool_bytes = float(16 * MIB)
        self._observed_pool_bytes = {}
        self.over_budget_estimate_trials = 0
        self._calls = self.builds = self.replays = self.evictions = 0
        self._last_bucket, self._consecutive_requests = None, 0
        self.cooldown_early_recoveries = 0
        self._fallbacks = {}
        self.oom_fallbacks = self.budget_rejections = 0

        self.signature_invalidations = self.mode_invalidations = 0
        self._bind_guard_slots()

    def _bind_guard_slots(self):
        # Registrations are observed directly on replay. Unlike recursive
        # named_parameters(), these small dictionaries do not traverse aliases.
        self._guard_modules = tuple(self.module.modules())
        self._guard_layout = tuple(
            (child, child._parameters, child._buffers, child._modules.copy())
            for child in self._guard_modules)
        self._guard_parameter_maps = tuple(
            parameters for _, parameters, _, _ in self._guard_layout if parameters)
        self._guard_buffer_maps = tuple(
            buffers for _, _, buffers, _ in self._guard_layout if buffers)

    def _module_state(self):
        """Inspect current registration slots and normal tensor version counters."""
        if any(child.training for child in self._guard_modules):
            return None
        if any(child._parameters is not parameters or child._buffers is not buffers
               or child._modules != children
               for child, parameters, buffers, children in self._guard_layout):
            self._bind_guard_slots()
            if any(child.training for child in self._guard_modules):
                return None
        parameters = tuple(value for mapping in self._guard_parameter_maps
                           for value in mapping.values() if value is not None)
        if any(p.requires_grad for p in parameters):
            return None
        buffers = tuple(value for mapping in self._guard_buffer_maps
                        for value in mapping.values() if value is not None)
        # Normal load/in-place mutation increments _version. Assignment/device
        # migration changes the object or pointer. Avoid redundant per-call
        # dtype/device/shape conversion in the latency-sensitive replay guard.
        try:
            return tuple((id(value), value._version, value.data_ptr())
                         for value in (*parameters, *buffers))
        except RuntimeError:
            # Inference tensors cannot supply a reliable mutation version.
            return None

    def _invalidate_state(self, *, mode=False):
        self.clear(reset_admission=True)
        self._state = None
        self._signature = None
        self._observed_pool_bytes.clear()
        self._bytes_per_batch = self._initial_bytes_per_batch
        self._fixed_pool_bytes = float(16 * MIB)
        if mode:
            self.mode_invalidations += 1
        else:
            self.signature_invalidations += 1

    def _fallback(self, reason, x, snapshots):
        if reason in ('device', 'grad', 'autocast', 'empty', 'outside_buckets', 'signature'):
            self._last_bucket, self._consecutive_requests = None, 0
        self._fallbacks[reason] = self._fallbacks.get(reason, 0) + 1
        return self.forward(x, snapshots=snapshots)

    def _bucket(self, batch):
        return next((b for b in self.buckets
                     if batch <= b <= batch * self.max_padding_ratio), None)

    def __call__(self, x, snapshots=None):
        if x.device.type != 'cuda':
            return self._fallback('device', x, snapshots)
        if torch.is_grad_enabled():
            return self._fallback('grad', x, snapshots)
        if torch.is_autocast_enabled():
            return self._fallback('autocast', x, snapshots)
        if not len(x):
            return self._fallback('empty', x, snapshots)
        bucket = self._bucket(len(x))
        if bucket is None:
            return self._fallback('outside_buckets', x, snapshots)
        # One runner owns one frozen module/device/dtype and a finite key space.
        signature = (tuple(x.shape[1:]), x.dtype, x.device.index)
        with self._lock:
            state = self._module_state()
            if state is None:
                if self._state is not None or self.cache:
                    self._invalidate_state(mode=True)
                self._last_bucket, self._consecutive_requests = None, 0
                return self._fallback('unfrozen_or_training', x, snapshots)
            if self._state is not None and self._state != state:
                self._invalidate_state()
            self._state = state
            if self._device_index is not None and self._device_index != x.device.index:
                self._stream = None
                self._device_index = None
            if self._signature is None:
                self._signature = signature
            elif self._signature != signature:
                return self._fallback('signature', x, snapshots)
            self._calls += 1
            key = bucket
            self._consecutive_requests = self._consecutive_requests + 1 if key == self._last_bucket else 1
            self._last_bucket = key
            self._seen[key] = self._seen.get(key, 0) + 1
            if key not in self.cache:
                if key in self._disabled:
                    return self._fallback(self._disabled[key], x, snapshots)
                if self._seen[key] < self.min_reuse:
                    return self._fallback('low_reuse', x, snapshots)
                recovering = self._calls < self._cooldown.get(key, 0)
                # A stable request stream should recover immediately, whereas
                # alternating more buckets than capacity must remain in cooldown.
                # Even min_reuse=1 needs two consecutive calls to show reuse.
                if recovering and self._consecutive_requests < max(2, self.min_reuse):
                    return self._fallback('eviction_cooldown', x, snapshots)
                if not self._admit(x, key):
                    return self._fallback(self._disabled.get(key, 'budget'), x, snapshots)
                # Warm-up may refresh derived nonpersistent fixed-Q buffers.
                # Bind the captured state after refresh so a valid graph survives.
                self._state = self._module_state()
                if recovering:
                    self.cooldown_early_recoveries += 1
            self.cache.move_to_end(key)
            entry = self.cache[key]
            # Never create or mutate graph buffers in a caller's inference mode.
            # A single cache works for both inference_mode and ordinary no_grad.
            with torch.inference_mode(False), torch.no_grad():
                entry.static[:len(x)].copy_(x)
                if len(x) < bucket:
                    entry.static[len(x):].zero_()
                entry.graph.replay()
            self.replays += 1
            return {name: value[:len(x)] for name, value in entry.outputs.items()}

    @property
    def cached_bytes(self):
        return sum(entry.bytes for entry in self.cache.values())

    def _evict(self):
        key, entry = self.cache.popitem(last=False)
        self._cooldown[key] = self._calls + self.eviction_cooldown
        self.evictions += 1
        # These graphs have completed before the next public predict call. The
        # caller must not keep borrowed output tensors after the next invocation.
        entry.outputs.clear()
        entry.graph.reset()

    def _estimate(self, bucket):
        # Admission estimate only. The actual graph pool is measured after
        # capture and rejected if retaining it would exceed the byte budget.
        if bucket in self._observed_pool_bytes:
            return self._observed_pool_bytes[bucket]
        return self._fixed_pool_bytes + bucket * self._bytes_per_batch

    def _observe_pool(self, bucket, owned_bytes):
        """Fit fixed pool overhead separately from batch-dependent storage.

        Small graph pools have allocator/launch overhead that must not become
        a permanently inflated bytes-per-scene estimate. Measured shapes use
        their observed size directly; an affine upper envelope only guides
        eviction for unmeasured shapes and never proves they cannot fit.
        """
        self._observed_pool_bytes[bucket] = max(
            owned_bytes, self._observed_pool_bytes.get(bucket, 0))
        points = list(self._observed_pool_bytes.items())
        slope = self._initial_bytes_per_batch
        if len(points) > 1:
            mean_x = sum(x for x, _ in points) / len(points)
            mean_y = sum(y for _, y in points) / len(points)
            variance = sum((x - mean_x) ** 2 for x, _ in points)
            if variance:
                fitted = sum((x - mean_x) * (y - mean_y) for x, y in points) / variance
                slope = max(slope, fitted)
        self._bytes_per_batch = slope
        self._fixed_pool_bytes = max(
            float(16 * MIB), max(y - slope * x for x, y in points))

    def _admit(self, x, key):
        estimate = self._estimate(key)
        available = self.max_cached_bytes - self.workspace_allowance_bytes
        if available <= 0 or self._observed_pool_bytes.get(key, 0) > available:
            self.budget_rejections += 1
            self._disabled[key] = 'budget'
            return False
        if estimate > available:
            # Measure a new bucket after eviction before rejecting its cache size.
            # Peak capture allocation is handled separately by the OOM fallback.
            self.over_budget_estimate_trials += 1
        reservation = min(estimate, available)
        # Evict retained pools before allocating a new capture pool.
        while self.cache and (len(self.cache) >= self.capacity or
                              self.cached_bytes + reservation > available):
            self._evict()
        oom = False
        try:
            entry = self._capture_entry(x, key)
        except torch.cuda.OutOfMemoryError:
            oom = True
        if oom:
            # _capture_entry has exited, releasing failed-capture locals and
            # exception traceback before attempting the ordinary eager forward.
            self.oom_fallbacks += 1
            self._disabled[key] = 'oom'
            while self.cache:
                self._evict()
            torch.cuda.empty_cache()
            return False
        if entry is None:
            self._disabled[key] = 'unsupported_pool_accounting'
            return False
        self._observe_pool(key, entry.bytes)
        while self.cache and self.cached_bytes + entry.bytes > available:
            self._evict()
        if entry.bytes > available:
            self.budget_rejections += 1
            self._disabled[key] = 'budget'
            entry.outputs.clear()
            entry.graph.reset()
            return False
        self.cache[key] = entry
        self.builds += 1
        return True

    def _capture_entry(self, x, bucket):
        graph, outputs = None, None
        complete = False
        try:
            with torch.cuda.device(x.device), torch.inference_mode(False), torch.no_grad():
                if self._stream is None:
                    self._stream = torch.cuda.Stream(device=x.device)
                    self._device_index = x.device.index
                stream = self._stream
                static = torch.zeros((bucket, *x.shape[1:]), dtype=x.dtype, device=x.device)
                static[:len(x)].copy_(x)
                stream.wait_stream(torch.cuda.current_stream(x.device))
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        warm = self.forward(static)
                        del warm
                torch.cuda.current_stream(x.device).wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    outputs = self.forward(static)
                torch.cuda.current_stream(x.device).wait_stream(stream)
                if not isinstance(outputs, dict) or not all(
                    torch.is_tensor(v) and v.ndim and v.shape[0] == bucket
                    for v in outputs.values()
                ):
                    raise TypeError('Encoder graph outputs must be a dictionary of batched tensors')
                pool = tuple(graph.pool())
                segments = [s for s in torch.cuda.memory_snapshot()
                            if s['device'] == x.device.index and
                            tuple(s.get('segment_pool_id', ())) == pool]
                if not segments:
                    # Do not silently replace a true workspace budget with a
                    # count of visible output tensors on unsupported runtimes.
                    return None
                owned_bytes = sum(s['total_size'] for s in segments) + static.untyped_storage().nbytes()
                complete = True
                return _Entry(graph, static, outputs, owned_bytes)
        finally:
            if not complete:
                if isinstance(outputs, dict):
                    outputs.clear()
                if graph is not None:
                    graph.reset()

    def clear(self, *, reset_admission=True):
        """Release owned graphs; keep the fixed stream to avoid new workspaces.

        Does not clear global cuBLAS workspaces or another model's allocator
        cache. Call only when no consumer is using borrowed encoder outputs.
        """
        with self._lock:
            if self._device_index is not None:
                torch.cuda.synchronize(self._device_index)
            while self.cache:
                self._evict()
            if reset_admission:
                self._seen.clear()
                self._cooldown.clear()
                self._disabled.clear()
                self._last_bucket, self._consecutive_requests = None, 0

    def stats(self):
        return dict(cached_shapes=len(self.cache), cached_buckets=list(self.cache),
                    graph_builds=self.builds, replays=self.replays, evictions=self.evictions,
                    capacity=self.capacity, allowed_buckets=list(self.buckets),
                    cache_reserved_bytes=self.cached_bytes,
                    accounted_cache_bytes=self.cached_bytes + (
                        self.workspace_allowance_bytes if self._stream is not None else 0),
                    workspace_allowance_bytes=self.workspace_allowance_bytes,
                    max_cached_bytes=self.max_cached_bytes,
                    admission_estimate_bytes_per_scene=self._bytes_per_batch,
                    admission_estimate_fixed_bytes=self._fixed_pool_bytes,
                    observed_pool_bytes={str(k): v for k, v in self._observed_pool_bytes.items()},
                    over_budget_estimate_trials=self.over_budget_estimate_trials,
                    budget_is_process_memory_limit=False,
                    graph_streams=int(self._stream is not None),
                    min_reuse=self.min_reuse, eviction_cooldown=self.eviction_cooldown,
                    cooldown_early_recoveries=self.cooldown_early_recoveries,
                    fallback_counts=dict(self._fallbacks), oom_fallbacks=self.oom_fallbacks,
                    budget_rejections=self.budget_rejections,
                    signature_invalidations=self.signature_invalidations,
                    mode_invalidations=self.mode_invalidations,
                    disabled_buckets={str(k): v for k, v in self._disabled.items()})
