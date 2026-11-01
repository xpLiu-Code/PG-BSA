"""PG-BSA inference with covariance-based source-count and angle refinement."""
from __future__ import annotations
from collections import Counter
from dataclasses import asdict, dataclass
import numpy as np
import torch
from .decoder import GroupedDecoder
from .networks import flatten_supports, continuous_nms
from .robust_cuda_encoder import RobustGraphedEncoder
from .rank_update_search import RankUpdateSearch
from .covariance_decisions import toeplitz_evidence, fit_single, fit_pair, fit_triple
from .fast_cardinality_refinement import CompactCardinalityRefiner
from .cardinality_refinement import saturated_likelihood
from .covariance_conditioning import estimate_equivalent_snapshots
from .local_cardinality_refinement import LocalCardinalityRefiner
from .support_budget import bounded_peak_indices
from .preprocessing import adaptive_diagonal_loading
from .io import digest

@dataclass(frozen=True)
class RuntimeOptions:
    dense_owner_sdpa: bool = True
    cached_steering: bool = True
    fixed_query_projection: bool = True
    graph_compact_refiner: bool = True
    graph_refiner_cache_mib: int = 32
    vectorized_lags: bool = True
    efficient_attention: bool = True
    skip_overwritten_routes: bool = True
    compact_frozen_outputs: bool = True
    fold_frozen_layers: bool = True
    shared_delta_ffn_chunk: int = 0
    ula_score_acceleration: bool = True
    grid_kv_stride: int = 1
    grid_kv_roles: tuple = ("original", "retrieval", "close_pair")
    grid_kv_dense_layers: int = 0
    parallel_low_scene_physics: bool = False
    parallel_final_physics: bool = False
    parallel_angle_polish: bool = False
    graph_expanded_candidates: bool = False
    candidate_graph_cache_mib: int = 32
    candidate_graph_min_reuse: int = 3
    rank2_pair_gradient: bool = False
    initial_dense_nms_gap: float = 1.0
    compact_fine_variant: str = "checkpoint"
    angle_polish_steps: int = 0
    angle_polish_backend: str = 'cpu'
    angle_polish_max_count: int = 16
    angle_polish_min_span: float = 0.
    graph_buckets: tuple = (1, 2, 4, 8, 16, 32)
    graph_cache_mib_per_encoder: int = 96
    graph_cache_capacity: int = 1
    initial_capture_mib_per_scene: float = 2.

    def __post_init__(self):
        from .pooled_grid_kv import validate_grid_kv_options
        validate_grid_kv_options(self.grid_kv_stride, self.grid_kv_roles, self.grid_kv_dense_layers)
        if not isinstance(self.ula_score_acceleration, bool):
            raise ValueError("ula_score_acceleration must be bool")
        if not isinstance(self.parallel_angle_polish, bool):
            raise ValueError('parallel_angle_polish must be bool')
        gap = self.initial_dense_nms_gap
        if (isinstance(gap, bool) or not isinstance(gap, (int, float))
                or not np.isfinite(gap) or not .5 <= gap <= 1.):
            raise ValueError('initial_dense_nms_gap must be finite and lie in [.5, 1.]')

@dataclass(frozen=True)
class InferenceConfig:
    noise_evidence_threshold: float = .35
    maximum_initial_routes: int = 20
    close_pair_span: float = 6.
    single_adequacy_deviance: float = .30
    pair_likelihood_gain: float = .15
    pair_model_deviance: float = 1.
    pair_minimum_gap: float = .5
    triple_likelihood_gain: float = .002
    triple_minimum_pair_span: float = 1.8
    physical_cardinality: bool = True
    local_cardinality: bool = True
    physical_pairs: bool = True
    reject_noise: bool = True


class PGBSA(GroupedDecoder):
    def __init__(self, device='cpu', checkpoint=None, *, config=None,
                 graph_encoders=True, search_workers=4, cpu_threads=1,
                 runtime_options=None, reliability_checkpoint=None,
                 reliability_strength=1.):
        if search_workers < 1:
            raise ValueError('search_workers must be positive')
        super().__init__(device, checkpoint, cpu_threads=cpu_threads)
        self.config = config or InferenceConfig()
        self.runtime_options = (RuntimeOptions(**runtime_options) if isinstance(runtime_options, dict)
                                else runtime_options or RuntimeOptions())
        declared_variant = self.metadata.get('compact_fine_variant', 'legacy')
        requested_variant = self.runtime_options.compact_fine_variant
        variant = declared_variant if requested_variant == 'checkpoint' else requested_variant
        if variant != declared_variant:
            raise ValueError('Refinement math variant differs from checkpoint declaration')
        from .compact_refinement import set_compact_fine_variant
        for compact_root in (self.engine.original, self.engine.dual.localizer,
                             self.engine.dual.close_pair_localizer):
            set_compact_fine_variant(compact_root, variant)
        self.metadata['effective_compact_fine_variant'] = variant
        self.skip_overwritten_routes = self.runtime_options.skip_overwritten_routes
        if self._shared_delta_models is not None:
            from .shared_delta import set_shared_delta_ffn_chunk
            set_shared_delta_ffn_chunk(self._shared_delta_models, self.runtime_options.shared_delta_ffn_chunk)
        if self.runtime_options.compact_frozen_outputs and reliability_checkpoint is None:
            self.retrieval._encoder_output_role = 'retrieval'
            self.retrieval._proposal_generator = self.engine.generator
            for root in (self.engine.original, self.engine.dual.localizer,
                         self.engine.dual.close_pair_localizer):
                if root.soft_route_power != 0. or root.use_snapshot_conditioning:
                    raise ValueError('Compact localization requires unconditioned hard routing')
                root._encoder_output_role = 'localization'
        self._compaction = {}
        if self.runtime_options.fold_frozen_layers:
            from .inference_compaction import compact_frozen_encoders
            self._compaction = compact_frozen_encoders(self)
        from .networks import configure_encoder_optimizations
        for root in (self.retrieval, self.engine.original, self.engine.dual.localizer,
                     self.engine.dual.close_pair_localizer):
            configure_encoder_optimizations(
                root, cached_steering=self.runtime_options.cached_steering,
                vectorized_lags=self.runtime_options.vectorized_lags,
                efficient_attention=self.runtime_options.efficient_attention)
        self._dense_owner_sdpa = {}
        if self.runtime_options.dense_owner_sdpa:
            from .dense_owner_sdpa import apply_dense_owner_sdpa
            self._dense_owner_sdpa = apply_dense_owner_sdpa(self.engine.original.base)
        self._fixed_query_projection = {}
        if self.runtime_options.fixed_query_projection:
            from .fixed_query_cache import install_fixed_query_projection
            self._fixed_query_projection = install_fixed_query_projection(self)
        from .pooled_grid_kv import apply_pooled_grid_kv
        self._pooled_grid_kv = apply_pooled_grid_kv(
            {"original": self.engine.original, "retrieval": self.retrieval,
             "close_pair": self.engine.dual.close_pair_localizer},
            stride=self.runtime_options.grid_kv_stride,
            roles=self.runtime_options.grid_kv_roles,
            dense_layers=self.runtime_options.grid_kv_dense_layers)
        self._graphs = {}
        self._refiner_graphs = {}
        self._refiner_graph_by_root = {}
        if self.runtime_options.graph_compact_refiner and self.device.type == 'cuda':
            from .graphed_compact_refiner import install_b1_refiner_graphs
            if self.runtime_options.graph_refiner_cache_mib < 17:
                raise ValueError('Refiner graph cache needs more than its 16 MiB workspace allowance')
            install_b1_refiner_graphs(self, max_cached_bytes=int(self.runtime_options.graph_refiner_cache_mib*1024**2))
        self._closed = False
        self._events = Counter()
        self.last_diagnostics = []
        self._search = RankUpdateSearch(search_workers, 4, use_ula_score=self.runtime_options.ula_score_acceleration)
        self._cardinality = CompactCardinalityRefiner()
        self._local_cardinality = LocalCardinalityRefiner()
        self._angle_polisher = None
        if self.runtime_options.angle_polish_steps not in (0, 1, 2, 3):
            raise ValueError('angle_polish_steps must lie in [0,3]')
        if self.runtime_options.angle_polish_backend not in ('cpu', 'gpu'):
            raise ValueError('angle_polish_backend must be cpu or gpu')
        if not 5 <= self.runtime_options.angle_polish_max_count <= 16:
            raise ValueError('angle_polish_max_count must lie in [5,16]')
        if not np.isfinite(self.runtime_options.angle_polish_min_span) or not 0 <= self.runtime_options.angle_polish_min_span <= 140:
            raise ValueError('angle_polish_min_span must lie in [0,140]')
        if self.runtime_options.angle_polish_steps:
            if self.runtime_options.angle_polish_backend == 'cpu':
                from .cpu_angle_polish import CPUAnglePolisher, PolishOptions
                constructor, polish_device = CPUAnglePolisher, 'cpu'
            else:
                from .gpu_angle_polish import GPUAnglePolisher, PolishOptions
                constructor, polish_device = GPUAnglePolisher, self.device
            self._angle_polisher = constructor(polish_device, PolishOptions(
                steps=self.runtime_options.angle_polish_steps,
                maximum_count=self.runtime_options.angle_polish_max_count))
        self._reliability_refiner = None
        if reliability_checkpoint is not None:
            from .runtime_refiner import RuntimeReliabilityRefiner
            self._reliability_refiner = RuntimeReliabilityRefiner(
                reliability_checkpoint, theta_grid=self.retrieval.theta_grid,
                strength=reliability_strength).to(self.device)
            self.metadata.update(reliability_checkpoint_sha256=digest(reliability_checkpoint),
                                 reliability_refiner=self._reliability_refiner.metadata)
        if graph_encoders and self.device.type == 'cuda':
            for name, root in [('retrieval', self.retrieval), ('localization', self.engine.original),
                               ('close_pair', self.engine.dual.close_pair_localizer)]:
                runner = RobustGraphedEncoder(
                    root, buckets=self.runtime_options.graph_buckets,
                    capacity=self.runtime_options.graph_cache_capacity,
                    max_cached_bytes=int(self.runtime_options.graph_cache_mib_per_encoder * 1024**2),
                    initial_bytes_per_batch=int(self.runtime_options.initial_capture_mib_per_scene * 1024**2))
                self._graphs[name] = runner
                root.forward = runner
        self._candidate_graph = None
        if self.runtime_options.graph_expanded_candidates and graph_encoders and self.device.type == 'cuda':
            from .graphed_candidates import DenseCandidateGraphs
            self._candidate_graph = DenseCandidateGraphs(
                self, max_cached_bytes=int(self.runtime_options.candidate_graph_cache_mib * 1024**2),
                min_reuse=self.runtime_options.candidate_graph_min_reuse)
        self.metadata.update(
            input_contract='Trace-normalized covariance; source count and directions are inferred.',
            runtime_options=asdict(self.runtime_options),
            retained_graph_cache_budget_bytes=sum(r.max_cached_bytes for r in self._graphs.values()),
            graph_budget_is_total_device_memory_limit=False,
            unique_parameters=self.parameter_count(),
            inference_compaction=self._compaction,
            pooled_grid_kv=self._pooled_grid_kv,
            fixed_query_projection=self._fixed_query_projection,
            graph_compact_refiner=bool(self._refiner_graphs),
            refiner_graph_formula='Wide and fine refinement with a recomputed route cache',
            angle_polish_enabled=self._angle_polisher is not None,
            angle_polish_preserves_predicted_set_size=True,
            reliability_refiner_enabled=self._reliability_refiner is not None,
            policy=asdict(self.config), search_workers=int(search_workers),
            physical_cardinality_config=asdict(self._cardinality.config),
            local_cardinality_config=asdict(self._local_cardinality.config),
            minimum_separation=.5, sml_minimum_separation=1., maximum_output_sources=20,
            graph_encoders=bool(self._graphs), retained_neural_weights_unchanged=not self.runtime_options.fold_frozen_layers,
            checkpoint_weights_unchanged=True)

    @torch.no_grad()
    def predict(self, x, counts=None):
        with self._lock:
            if self._closed:
                raise RuntimeError('PGBSA has been closed')
            if torch.is_tensor(x) and x.ndim and not len(x):
                self.last_diagnostics = []
            return super().predict(x, counts)

    def _retrieve(self, x):
        if self._reliability_refiner is None:
            loaded, encoded, supports, sb, si = super()._retrieve(x)
        else:
            loaded, _ = adaptive_diagonal_loading(x, target_noise_floor=.020, maximum_loading=.35)
            encoded = self.retrieval(loaded)
            encoded, _ = self._reliability_refiner(encoded, x)
            supports = self.retrieval.predicted_supports(encoded, max_sources=20)
            sb, si = flatten_supports(supports, self.device)
        budget = self.config.maximum_initial_routes
        bounded = []
        changed = False
        host_logits = (encoded['location_logits'].cpu().numpy()
                       if any(len(s)>budget for s in supports) else None)
        for j, support in enumerate(supports):
            if len(support) > budget:
                indices=bounded_peak_indices(support.cpu().numpy(),host_logits[j],budget)
                support=torch.as_tensor(indices,device=self.device,dtype=torch.long)
                changed = True
                self._events['route_budget_applied'] += 1
            bounded.append(support)
        if changed:
            sb, si = flatten_supports(bounded, self.device)
        return loaded, encoded, bounded, sb, si

    def _decode_low_scene(self, angles, scores, support_count, *, covariance=None):
        decoded, events = self._decode_low_scene_with_events(
            angles, scores, support_count, covariance=covariance)
        self._events.update(events)
        return decoded

    def _decode_low_scene_job(self, job):
        angles, scores, count, covariance = job
        return self._decode_low_scene_with_events(
            angles, scores, count, covariance=covariance)

    def _decode_low_scene_jobs(self, jobs):
        result = []
        for angles, events in self._search.map_jobs(
                self._decode_low_scene_job, jobs,
                parallel=self.runtime_options.parallel_low_scene_physics):
            # Only the prediction thread commits shared diagnostics/state.
            self._events.update(events)
            result.append(angles)
        return result

    def _decode_low_scene_with_events(self, angles, scores, support_count, *, covariance=None):
        events = Counter()
        gap = self.runtime_options.initial_dense_nms_gap
        # Apply this NMS rule only to scenes with 11 to 16 predicted routes.
        if (11 <= support_count < self.engine.bundle.apply_under_count
                and gap != self.engine.bundle.minimum_separation):
            default = continuous_nms(
                torch.as_tensor(angles, dtype=torch.float32),
                torch.as_tensor(scores, dtype=torch.float32),
                minimum_separation=gap).numpy()
        else:
            default = super()._decode_low_scene(
                angles, scores, support_count, covariance=covariance)
        cfg = self.config
        if not cfg.physical_pairs or not 1 <= support_count <= 2 or covariance is None:
            return default, events
        if np.ptp(angles) > cfg.close_pair_span:
            return default, events
        r = np.asarray(covariance, np.complex128)
        center = float(np.mean(angles))
        one = fit_single(r, center)
        saturated = saturated_likelihood(r)
        if one['nll']-saturated < cfg.single_adequacy_deviance:
            events['single_covariance_accepted'] += 1
            return one['angles'], events
        two = fit_pair(r, one, max_gap=cfg.close_pair_span, starts=(1.5,),
                       rank2_gradient=self.runtime_options.rank2_pair_gradient)
        gap = float(np.diff(two['angles'])[0])
        reliability = max(float(estimate_equivalent_snapshots(r)), 16.)
        adequate = max(cfg.single_adequacy_deviance, 200./reliability)
        required_gain = max(cfg.pair_likelihood_gain, 8./reliability)
        if cfg.physical_cardinality and gap >= cfg.triple_minimum_pair_span:
            three = fit_triple(r, two)
            if (two['nll']-three['nll'] >= max(cfg.triple_likelihood_gain, 2./reliability)
                and three['nll']-saturated <= max(cfg.pair_model_deviance, adequate)
                and np.diff(three['angles']).min() >= cfg.pair_minimum_gap
                and np.max(abs(three['angles'])) <= 69.9):
                events['triple_covariance_accepted'] += 1
                return three['angles'], events
        if (one['nll']-two['nll'] >= required_gain and
            gap >= cfg.pair_minimum_gap and two['nll']-saturated <= max(cfg.pair_model_deviance, adequate) and
            np.max(abs(two['angles'])) <= 69.9):
            events['pair_covariance_accepted'] += 1
            return two['angles'], events
        if one['nll']-two['nll'] < required_gain and one['nll']-saturated <= adequate:
            events['duplicate_pair_rejected'] += 1
            return one['angles'], events
        # Keep the neural estimate when neither low-order fit is reliable.
        # Subsequent refinement can compare source counts above two.
        events['pair_physics_inconclusive'] += 1
        return default, events

    def _select_sml_jobs(self, jobs):
        return self._search(jobs, radius=2.5, strength=5., minimum_separation=1., passes=3)

    def _predict(self, x):
        if self._closed:
            raise RuntimeError('PGBSA has been closed')
        if not bool(torch.isfinite(x).all()):
            raise ValueError('Covariance contains non-finite values')
        host = (x[:, 0]+1j*x[:, 1]).cpu().numpy()
        ratios = [toeplitz_evidence(r)[0] for r in host]
        active = [j for j, ratio in enumerate(ratios)
                  if not self.config.reject_noise or ratio >= self.config.noise_evidence_threshold]
        result = [np.empty(0, dtype=float) for _ in x]
        self.last_diagnostics = [dict(noise_rejected=j not in active, toeplitz_evidence=ratios[j])
                                 for j in range(len(x))]
        self._events['scenes'] += len(x)
        self._events['noise_rejected'] += len(x)-len(active)
        if not active:
            return result
        predictions = super()._predict(x if len(active)==len(x) else x[active],
                                       covariances=host if len(active)==len(x) else host[active])
        jobs = [(host[j], prediction) for j, prediction in zip(active, predictions)]
        for j, (prediction, diagnostics, events) in zip(active, self._search.map_jobs(
                self._final_physics_job, jobs,
                parallel=self.runtime_options.parallel_final_physics)):
            self._events.update(events)
            self.last_diagnostics[j].update(diagnostics)
            result[j] = prediction
        if self._angle_polisher is not None:
            eligible = [j for j in active
                        if 5 <= len(result[j]) <= self.runtime_options.angle_polish_max_count
                        and float(np.ptp(result[j])) >= self.runtime_options.angle_polish_min_span]
            if eligible:
                if (self.runtime_options.parallel_angle_polish
                        and self.runtime_options.angle_polish_backend == 'cpu'):
                    jobs = [(host[j], result[j]) for j in eligible]
                    polished_rows = self._search.map_jobs(
                        self._angle_polish_job, jobs, parallel=True)
                else:
                    # GPU refinement uses one batch; the scheduling flag controls CPU jobs.
                    observations = (host[eligible] if self.runtime_options.angle_polish_backend == 'cpu'
                                    else x[eligible])
                    polished, diagnostic = self._angle_polisher(
                        observations, [result[j] for j in eligible], return_diagnostics=True)
                    polished_rows = zip(polished, diagnostic['scenes'])
                for j, (prediction, row) in zip(eligible, polished_rows):
                    if len(prediction) != len(result[j]):
                        raise RuntimeError('Angle polishing changed the predicted set size')
                    result[j] = prediction
                    self.last_diagnostics[j]['angle_polish'] = row
                    self._events['angle_polish_eligible'] += int(row['eligible'])
                    self._events['angle_polish_changed'] += int(row['changed'])
        return result

    def _angle_polish_job(self, job):
        # Pure worker result: only the prediction owner commits state/counters.
        covariance, prediction = job
        polished, diagnostic = self._angle_polisher(
            covariance[None], [prediction], return_diagnostics=True)
        return polished[0], diagnostic['scenes'][0]

    def _final_physics_job(self, job):
        covariance, prediction = job
        events, diagnostics = Counter(), {}
        if self.config.physical_cardinality:
            original = prediction
            prediction, diagnostic = self._cardinality.refine(covariance, prediction)
            if diagnostic['accepted'] and np.diff(prediction).min() < self.config.pair_minimum_gap:
                prediction = original
                diagnostic.update(accepted=False, reason='insufficient_physical_separation')
            events['cardinality_'+diagnostic['reason']] += 1
            diagnostics['cardinality'] = diagnostic
            if self.config.local_cardinality:
                prediction, local_diagnostic = self._local_cardinality.refine(covariance, prediction)
                events['local_cardinality_'+local_diagnostic['reason']] += 1
                diagnostics['local_cardinality'] = local_diagnostic
        prediction = np.sort(np.asarray(prediction, float)).copy()
        return prediction, diagnostics, events

    def graph_stats(self):
        result = {name: runner.stats() for name, runner in self._graphs.items()}
        result['compact_refiner'] = {name: runner.stats() for name, runner in self._refiner_graphs.items()}
        return result

    def diagnostics(self):
        return dict(events=dict(self._events), graph_stats=self.graph_stats(),
                    candidate_graph_stats={} if self._candidate_graph is None else self._candidate_graph.stats())

    def close(self):
        with self._lock:
            if not self._closed:
                self._search.close()
                if self._candidate_graph is not None:
                    self._candidate_graph.clear()
                for runner in self._graphs.values():
                    runner.clear()
                for runner in self._refiner_graphs.values():
                    runner.clear()
                self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
