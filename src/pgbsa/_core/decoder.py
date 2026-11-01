"""Checkpoint loading, shared network assembly, and density-dependent decoding."""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import threading
import numpy as np
import torch
from torch import nn
from .networks import BeamDOANetwork, flatten_supports, continuous_nms
from .angular_adapter import attach_adapter, set_context
from .heads import (GlobalCandidateGenerator, CandidateScorer, DecoderBundle,
                    merge_low_density_routes, decode_fixed_count, replace_profiles)
from .preprocessing import retrieve
from .gpu_candidates import build_gpu_candidates
from .io import digest

CHECKPOINT_FORMAT = 'pgbsa_validated_v1'
EXPECTED_PARAMETERS = 5_996_256
DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[1] / 'assets' / 'model.pth'

def strip_reconstruction(root):
    for name, _ in list(root.named_modules()):
        if name.endswith(('power_head', 'noise_head')):
            parent = root
            bits = name.split('.')
            for bit in bits[:-1]:
                parent = getattr(parent, bit)
            setattr(parent, bits[-1], nn.Identity())

UNUSED_RETRIEVAL_HEADS = (
    'local_steer_proj', 'offset_embed', 'local_cross',
    'support_transformer', 'local_head', 'wide_head',
)


def strip_retrieval_refinement(retrieval):
    """The deployed retriever only calls encode_base and predicted_supports."""
    for name in UNUSED_RETRIEVAL_HEADS:
        setattr(retrieval, name, None)
    retrieval.register_parameter('correction_gate_logit', None)


def verify_encoding_contract(original, localizer):
    # These are the parent-level inputs/flags read by BeamDOANetwork.encode_base.
    for name in ('N', 'K', 'd_model', 'use_snapshot_conditioning'):
        if getattr(original, name) != getattr(localizer, name):
            raise ValueError(f'Encoder contracts differ: {name}')
    if original.use_snapshot_conditioning:
        raise ValueError('The shared deployment requires unconditioned localization encoders')
    for name in ('base_steering', 'theta_grid'):
        if not torch.equal(getattr(original, name), getattr(localizer, name)):
            raise ValueError(f'Encoder buffers differ: {name}')


class GridReference(nn.Module):
    """Store the routing grid used by the retrieval encoder."""
    def __init__(self, grid):
        super().__init__()
        self.register_buffer('theta_grid', grid.detach().clone())

    def forward(self, *args, **kwargs):
        raise RuntimeError('Use high_snapshot_retrieval for encoding')


class InferenceNetworks(nn.Module):
    """Register retrieval, shared localization and the close-pair expert."""
    def __init__(self, localizer, retrieval, close_pair, grid):
        super().__init__()
        self.localizer = localizer
        self.high_snapshot_retrieval = retrieval
        self.close_pair_localizer = close_pair
        self.retrieval = GridReference(grid)


def close_pair_scene_mask(model, supports, span, device):
    mask = torch.zeros(len(supports), dtype=torch.bool, device=device)
    if span <= 0.:
        return mask
    eligible = [j for j, support in enumerate(supports) if support.numel() == 2]
    if eligible:
        pairs = torch.stack([supports[j] for j in eligible])
        grid = model.retrieval.theta_grid
        route_span = (grid[pairs[:, 1]]-grid[pairs[:, 0]]).abs()
        mask[torch.as_tensor(eligible, device=device)] = route_span <= float(span)+1.e-6
    return mask


class GroupedDecoder:
    def __init__(self, device='cpu', checkpoint=None, *, cpu_threads=1):
        self.device = torch.device(device)
        if self.device.type not in ('cpu', 'cuda'):
            raise ValueError('Supported devices: cpu or cuda')
        if cpu_threads < 1:
            raise ValueError('cpu_threads must be positive')
        self._lock = threading.RLock()
        torch.set_num_threads(cpu_threads)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        path = Path(checkpoint) if checkpoint is not None else DEFAULT_CHECKPOINT
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if payload.get('format') not in (CHECKPOINT_FORMAT, 'pgbsa_shared_delta_v3'):
            raise ValueError('Unsupported PG-BSA checkpoint format')
        if set(payload['configs']) != {'original', 'localizer', 'retrieval', 'close_pair'}:
            raise ValueError('Checkpoint must retain the close-pair expert configuration')
        if not any(k.startswith('close_pair_localizer.') for k in payload['state']['dual']):
            raise ValueError('Close-pair expert weights are missing')
        self._load_weights(payload)
        self._move_to_device()
        if self.engine.original.base is not self.engine.dual.localizer.base:
            raise RuntimeError('Shared localization backbone identity was lost')
        expected_parameters = (payload['metadata']['unique_parameters'] if payload.get('format') == 'pgbsa_shared_delta_v3' else EXPECTED_PARAMETERS)
        if self.parameter_count() != expected_parameters:
            raise ValueError('Unexpected parameter count for this release')
        self.metadata.update(checkpoint_sha256=digest(path),
                             unique_parameters=self.parameter_count(),
                             close_expert_removed=False, shared_backbone=True,
                             candidate_backend='torch', cpu_threads=cpu_threads)

    def _trusted_refine(self, root, encoded, sb, si, *, phase='initial'):
        # Only decoder-generated supports reach this private trusted entry.
        runner = getattr(self, '_refiner_graph_by_root', {}).get(id(root))
        if runner is not None and len(encoded['location_logits']) == 1:
            return runner(encoded, sb, si, phase=phase)
        return root.refine_routes(encoded, sb, si, detach_base=True, build_reconstruction=False)

    def _original_routes(self, fixed, loaded, supports, sb, si):
        """Reuse original covariance encoding, retaining the close-pair specialist."""
        dual = self.engine.dual
        mask = close_pair_scene_mask(dual, supports, 3.5, self.device)
        idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
        if getattr(self, 'skip_overwritten_routes', False) and idx.numel():
            return self._partition_original_routes(fixed, loaded, supports, mask)
        rr = self._trusted_refine(self.engine.original, fixed, sb, si)
        if idx.numel():
            enc = dual.close_pair_localizer(loaded[idx])
            sis = torch.cat([supports[int(j)] for j in idx])
            sbs = torch.arange(len(idx), device=self.device).repeat_interleave(2)
            special = self._trusted_refine(dual.close_pair_localizer, enc, sbs, sis)
            positions = torch.nonzero(mask[sb], as_tuple=False).squeeze(1)
            keys = ('fine_logits', 'fine_probability', 'base_angle', 'route_base_angle',
                    'route_offset_residual', 'wide_logits', 'wide_residual', 'raw_residual',
                    'conditional_angle', 'route_residual', 'route_blend', 'source_power')
            for key in keys:
                if key in rr and key in special and rr[key].shape[1:] == special[key].shape[1:]:
                    rr[key] = rr[key].clone()
                    rr[key][positions] = special[key]
        return rr

    def _partition_original_routes(self, fixed, loaded, supports, specialist_mask):
        """Partition standard and close-pair routes while preserving row order.

        The close-pair encoder consumes the loaded covariance. Standard routes use
        the raw covariance, so their observation-dependent features are distinct."""
        special_ids = torch.nonzero(specialist_mask, as_tuple=False).flatten().tolist()
        standard_ids = torch.nonzero(~specialist_mask, as_tuple=False).flatten().tolist()
        starts = np.cumsum([0] + [len(s) for s in supports])
        pieces = []
        for ids, root, expert in (
            (standard_ids, self.engine.original, False),
            (special_ids, self.engine.dual.close_pair_localizer, True),
        ):
            if not ids:
                continue
            selected = [supports[j] for j in ids]
            encoded = root(loaded[ids]) if expert else self._slice_encoding(fixed, ids)
            sb, si = flatten_supports(selected, self.device)
            routes = self._trusted_refine(root, encoded, sb, si)
            positions = torch.cat([torch.arange(starts[j], starts[j+1], device=self.device) for j in ids])
            pieces.append((positions, routes))
        result = {}
        total = int(starts[-1])
        for positions, routes in pieces:
            for name, value in routes.items():
                if name in ('R_hat', 'noise_power', 'local_candidate_count'):
                    continue
                if not torch.is_tensor(value) or value.ndim == 0 or len(value) != len(positions):
                    raise ValueError(f'Unexpected refinement output layout: {name}')
                if name not in result:
                    result[name] = value.new_empty((total, *value.shape[1:]))
                elif result[name].shape[1:] != value.shape[1:]:
                    raise ValueError(f'Expert and standard route shapes differ: {name}')
                result[name][positions] = value
        # Preserve reconstruction output fields while skipping the unused heads.
        result['R_hat'] = fixed['covariance'].new_zeros((len(supports), 16, 16))
        result['noise_power'] = fixed['base_offset'].new_zeros(len(supports))
        root = self.engine.original
        result['local_candidate_count'] = total * (root.M + (root.wide_offsets.numel() if root.use_wide_cascade else 0))
        return result

    def _load_weights(self, payload):

        if payload.get('shared_backbone_owner') != 'original.base':
            raise ValueError('Missing or incompatible backbone ownership metadata')
        states = payload['state']
        models = {name: BeamDOANetwork(**config) for name, config in payload['configs'].items()}
        compact_spec = payload.get('compact_refinement')
        if compact_spec is None:
            attach_adapter(models['localizer'], radius_deg=2., minimum_routes=11)
        for root in models.values():
            strip_reconstruction(root)
        verify_encoding_contract(models['original'], models['localizer'])
        models['localizer'].base = models['original'].base
        strip_retrieval_refinement(models['retrieval'])
        grid_spec = payload.get('grid_resolution')
        if grid_spec is not None:
            if grid_spec.get('factor') != 2 or grid_spec.get('target_step') != 1.0:
                raise ValueError('Unsupported grid-resolution metadata')
            offset_scale = float(grid_spec['coarse_offset_scale'])
            for root in models.values():
                if root.K != grid_spec['target_tokens']:
                    raise ValueError('Grid metadata/configuration mismatch')
                root.base.offset_scale = offset_scale
        refinement_spec = payload.get('refinement_bottleneck')
        if compact_spec is not None and refinement_spec is not None:
            raise ValueError('Compact refinement and feedforward bottleneck configurations are incompatible')
        if refinement_spec is not None:
            if refinement_spec.get('version') != 1:
                raise ValueError('Unsupported refinement bottleneck version')
            from .refinement_bottleneck import apply_refinement_bottleneck_models
            apply_refinement_bottleneck_models(
                models, support_ffn=refinement_spec['support_ffn'],
                cross_ffn=refinement_spec['cross_ffn'], initialize=False)
        if compact_spec is not None:
            from .compact_refinement import apply_compact_refinement_models, validate_compact_spec
            validate_compact_spec(compact_spec)
            apply_compact_refinement_models(models, width=compact_spec['width'], initialize=False)
        self._shared_delta_models = None
        if payload.get('format') == 'pgbsa_shared_delta_v3':
            from .shared_delta import apply_shared_delta_models
            settings = payload.get('shared_delta', {})
            if settings.get('owner') != 'original.base' or not settings.get('canonical_aliases_omitted'):
                raise ValueError('Invalid canonical shared-delta ownership metadata')
            apply_shared_delta_models(models, rank=settings['rank'], initialize=False)
            self._shared_delta_models = models
        dual = InferenceNetworks(models['localizer'], models['retrieval'],
                                 models['close_pair'], states['dual']['retrieval.theta_grid'])
        # Rebuild aliases as tensor references, not copies. Strict loading also
        # preserves BatchNorm counters, which PyTorch otherwise defaults to zero
        # for omitted state entries with no version metadata.
        models['original'].load_state_dict(states['original'], strict=True)
        if any(k.startswith('localizer.base.') for k in states['dual']):
            raise ValueError('Canonical state must store the shared backbone only once')
        if self._shared_delta_models is not None:
            from .shared_delta import expand_canonical_dual_state
            dual_state = expand_canonical_dual_state(states['dual'], models['original'], dual)
        else:
            dual_state = dict(states['dual'])
            dual_state.update({'localizer.base.' + k: v for k, v in models['original'].base.state_dict().items()})
        dual.load_state_dict(dual_state, strict=True)
        generator = GlobalCandidateGenerator(**payload['model_shapes']['generator'])
        generator.load_state_dict(states['generator'], strict=True)
        scorer = CandidateScorer(**payload['model_shapes']['scorer'])
        offset = CandidateScorer(**payload['model_shapes']['offset'])
        scorer.load_state_dict(states['scorer'], strict=True)
        offset.load_state_dict(states['offset'], strict=True)
        stats = {k: v.numpy() for k, v in payload['stats'].items()}
        bundle = DecoderBundle(scorer=scorer, offset=offset, **stats, **payload['decoder_settings'])
        self.engine = SimpleNamespace(dual=dual, original=models['original'], generator=generator, bundle=bundle)
        self.metadata = dict(payload['metadata'])
        for key in ('grid_resolution', 'refinement_bottleneck', 'compact_refinement'):
            if key in payload:
                self.metadata[key] = dict(payload[key])
        if self._shared_delta_models is not None:
            self.metadata['shared_delta'] = dict(payload['shared_delta'])

    def roots(self):
        e = self.engine
        roots = {'dual': e.dual, 'original': e.original, 'generator': e.generator,
                 'scorer': e.bundle.scorer, 'offset': e.bundle.offset}
        residual = getattr(self, '_reliability_refiner', None)
        if residual is not None:
            roots['reliability_refiner'] = residual
        return roots

    def parameter_count(self):
        parameters = {id(p): p for root in self.roots().values() for p in root.parameters()}
        return sum(p.numel() for p in parameters.values())

    def _move_to_device(self):
        e = self.engine
        e.device = self.device
        for root in self.roots().values():
            root.to(self.device).eval().requires_grad_(False)
        if self._shared_delta_models is not None:
            from .shared_delta import retie_shared_weights
            retie_shared_weights(self._shared_delta_models)
        e.student = e.dual.localizer
        e.scorer = e.bundle.scorer
        for name, value in [('mean', e.bundle.scorer_mean), ('std', e.bundle.scorer_std),
                            ('om', e.bundle.offset_mean), ('os', e.bundle.offset_std)]:
            setattr(e, name, torch.as_tensor(value, device=self.device))
        self.retrieval = e.dual.high_snapshot_retrieval

    def _refine(self, encoded, supports, *, phase='initial'):
        local = self.engine.dual.localizer
        sb, si = flatten_supports(supports, self.device)
        set_context(local, encoded, sb, si)
        return self._trusted_refine(local, encoded, sb, si, phase=phase)

    @torch.no_grad()
    def predict(self, x, counts=None):
        if counts is not None:
            raise ValueError('PG-BSA predicts cardinality; true source counts are not accepted')
        if not torch.is_tensor(x):
            raise TypeError('Input must be a torch.Tensor')
        if x.dtype != torch.float32:
            raise TypeError('Input must have dtype torch.float32')
        if x.ndim != 4 or tuple(x.shape[1:]) != (2, 16, 16):
            raise ValueError('Expected normalized covariance [B, 2, 16, 16]')
        if len(x) == 0:
            return []
        with self._lock:
            return self._predict(x.to(self.device))

    @staticmethod
    def _slice_encoding(encoded, scene_ids):
        # Encoding tensors must share the scene dimension.
        batch = len(encoded['location_logits'])
        if list(scene_ids) == list(range(batch)):
            return dict(encoded)
        index = torch.as_tensor(scene_ids, device=encoded['location_logits'].device)
        result = {}
        for key, value in encoded.items():
            if torch.is_tensor(value):
                if value.ndim == 0 or value.shape[0] != batch:
                    raise ValueError(f'Encoding tensor is not scene-indexed: {key}')
                result[key] = value.index_select(0, index)
            else:
                result[key] = value
        if "_compact_refinement_cache" in encoded:
            from .compact_refinement import slice_compact_refinement_cache
            cache = slice_compact_refinement_cache(encoded, result, scene_ids, index)
            if cache is None:
                result.pop("_compact_refinement_cache", None)
            else:
                result["_compact_refinement_cache"] = cache
        return result

    @staticmethod
    def _slice_routes(routes, supports, scene_ids):
        if list(scene_ids) == list(range(len(supports))):
            return dict(routes)
        device = routes['conditional_angle'].device
        starts = np.cumsum([0] + [len(s) for s in supports])
        pieces = [torch.arange(int(starts[j]), int(starts[j+1]), device=device)
                  for j in scene_ids]
        route_index = torch.cat(pieces) if pieces else torch.empty(0, dtype=torch.long, device=device)
        scene_index = torch.as_tensor(scene_ids, device=device)
        result = {}
        for key, value in routes.items():
            if key in ('R_hat', 'noise_power'):
                result[key] = value.index_select(0, scene_index)
            elif torch.is_tensor(value):
                if value.ndim == 0 or len(value) != int(starts[-1]):
                    raise ValueError(f'Unknown route tensor layout: {key}')
                result[key] = value.index_select(0, route_index)
            elif key == 'local_candidate_count':
                result[key] = int(value * len(route_index) // int(starts[-1])) if starts[-1] else 0
            else:
                result[key] = value
        return result

    def _retrieve(self, x):
        return retrieve(self.engine.dual, x)

    def _decode_low_scene(self, angles, scores, support_count, *, covariance=None):
        """Host-side policy hook; covariance is the original complex observation."""
        bundle = self.engine.bundle
        if not support_count:
            return np.empty(0, np.float32)
        angles = torch.as_tensor(angles, dtype=torch.float32)
        scores = torch.as_tensor(scores, dtype=torch.float32)
        merged = angles
        if support_count <= 2:
            merged = merge_low_density_routes(
                angles, scores, support_count=support_count, max_support_count=2,
                merge_threshold=bundle.merge_threshold, weight_power=0.,
                probability_gate=bundle.merge_probability_gate)
        score = scores if len(merged) == support_count else torch.ones_like(merged)
        return continuous_nms(
            merged, score, minimum_separation=bundle.minimum_separation).numpy()

    def _decode_low_scene_jobs(self, jobs):
        # The base decoder is serial; PGBSA can submit independent physical fits
        # to its search pool.
        return [self._decode_low_scene(a, z, count, covariance=covariance)
                for a, z, count, covariance in jobs]

    def _decode_grouped_initial(self, ge, supports, routes, covariances):
        """One scorer/offset call and one packed transfer for dense candidates."""
        e, bundle = self.engine, self.engine.bundle
        threshold = bundle.apply_under_count
        result = [None] * len(supports)
        sparse = [j for j, s in enumerate(supports) if len(s) < threshold]
        dense = [j for j, s in enumerate(supports) if len(s) >= threshold]
        starts = np.cumsum([0] + [len(s) for s in supports])
        if sparse:
            sb, si = flatten_supports(supports, self.device)
            probability = ge['location_logits'].sigmoid()[sb, si]
            host = torch.stack((routes['conditional_angle'], probability), -1).cpu().numpy()
            jobs = []
            for j in sparse:
                row = host[starts[j]:starts[j+1]]
                jobs.append((row[:, 0], row[:, 1], len(supports[j]), covariances[j]))
            for j, decoded in zip(sparse, self._decode_low_scene_jobs(jobs)):
                result[j] = decoded
        if dense:
            ds = [supports[j] for j in dense]
            dg = self._slice_encoding(ge, dense)
            dr = self._slice_routes(routes, supports, dense)
            runner = getattr(self, '_candidate_graph', None)
            if runner is None:
                angles, features, keep, sb, _ = build_gpu_candidates(
                    self.retrieval.theta_grid, dg, ds, dr, stable=False)
                features = features[keep]
                scores = bundle.scorer((features-e.mean)/e.std)
                offsets = bundle.offset((features-e.om)/e.os).clamp(-.35, .35)
                scene = sb[:, None].expand_as(keep)[keep]
                packed = torch.stack((angles[keep]+offsets, scores, scene), -1).cpu().numpy()
            else:
                packed_all = runner.initial(dg, ds, dr).cpu().numpy().reshape(-1, 4)
                packed = packed_all[packed_all[:, 3] != 0., :3]
            lengths = np.bincount(packed[:, 2].astype(np.int64), minlength=len(dense))
            split = np.cumsum(np.r_[0, lengths])
            for i, j in enumerate(dense):
                row = packed[split[i]:split[i+1]]
                result[j] = decode_fixed_count(
                    row[:, 0], row[:, 1], len(supports[j]), bundle.minimum_separation)
        return result

    def _expand_supports(self, ge, supports):
        z = (ge['proposal_scores'] if 'proposal_scores' in ge else
             self.engine.generator(ge['location_logits'], ge['grid_feature']))
        # One top-k and one compacted index transfer for the whole batch.
        # Union is a boolean grid, so output indices remain sorted and unique.
        if not supports:
            return []
        top = torch.topk(z, min(40, z.shape[1]), dim=1, sorted=False).indices
        keep = torch.zeros_like(z, dtype=torch.bool).scatter_(1, top, True)
        sb, si = flatten_supports(supports, self.device)
        keep[sb, si] = True
        pairs = torch.nonzero(keep, as_tuple=False)
        counts = keep.sum(1).cpu().tolist()
        return list(pairs[:, 1].split(counts))

    def _predict(self, x, *, covariances=None):
        e = self.engine
        loaded, ge, supports, _, si = self._retrieve(x)
        if not len(si):
            return [np.empty(0, dtype=float) for _ in x]
        if covariances is None:
            covariances = (x[:, 0]+1j*x[:, 1]).cpu().numpy()
        result = [np.empty(0, dtype=float) for _ in x]
        sparse = [j for j, s in enumerate(supports) if 0 < len(s) < 11]
        dense = [j for j, s in enumerate(supports) if len(s) >= 11]
        if getattr(self, 'skip_overwritten_routes', False):
            # An entire two-route scene is replaced by the close-pair expert.
            # Select that scene before encoding, so neither the raw-covariance
            # encoder nor its refinement heads run for discarded outputs.
            special_mask = close_pair_scene_mask(e.dual, supports, 3.5, self.device)
            specialist = torch.nonzero(special_mask, as_tuple=False).flatten().tolist()
            specialist_set = set(specialist)
            standard = [j for j in sparse if j not in specialist_set]
            raw_ids = sorted(standard + dense)
            raw_position = {scene: slot for slot, scene in enumerate(raw_ids)}
            fixed = (e.original(x if len(raw_ids) == len(x) else x[raw_ids])
                     if raw_ids else None)
            # Refinement attention is scene-local. Raw and loaded inputs must
            # stay distinct even though both encoders have frozen weights.
            for scene_ids, root, expert in (
                (standard, e.original, False),
                (specialist, e.dual.close_pair_localizer, True),
            ):
                if not scene_ids:
                    continue
                ss = [supports[j] for j in scene_ids]
                se = (root(loaded[scene_ids]) if expert else
                      self._slice_encoding(fixed, [raw_position[j] for j in scene_ids]))
                sg = self._slice_encoding(ge, scene_ids)
                sb, si = flatten_supports(ss, self.device)
                rr = self._trusted_refine(root, se, sb, si)
                decoded = self._decode_grouped_initial(sg, ss, rr, covariances[scene_ids])
                for j, angles in zip(scene_ids, decoded):
                    result[j] = np.sort(np.asarray(angles, dtype=float))
        else:
            # Use full-batch evaluation when route partitioning is disabled.
            fixed = e.original(x)
            raw_position = {scene: scene for scene in range(len(x))}
        if sparse and not getattr(self, 'skip_overwritten_routes', False):
            ss = [supports[j] for j in sparse]
            se = self._slice_encoding(fixed, sparse)
            sg = self._slice_encoding(ge, sparse)
            sb, si = flatten_supports(ss, self.device)
            rr = self._original_routes(se, loaded[sparse], ss, sb, si)
            decoded = self._decode_grouped_initial(sg, ss, rr, covariances[sparse])
            for j, angles in zip(sparse, decoded):
                result[j] = np.sort(np.asarray(angles, dtype=float))
        if not dense:
            return result
        ds = [supports[j] for j in dense]
        de = self._slice_encoding(fixed, [raw_position[j] for j in dense])
        dg = self._slice_encoding(ge, dense)
        de['base_offset'] = de['base_offset'].detach()
        initial_routes = self._refine(de, ds)
        initial = self._decode_grouped_initial(dg, ds, initial_routes, covariances[dense])
        for j, angles in zip(dense, initial):
            result[j] = np.sort(np.asarray(angles, dtype=float))
        expanded = self._expand_supports(dg, ds)
        # Use initial candidates when expanded support is too small for a
        # dense-scene search.
        search_ids = [j for j, s in enumerate(expanded) if len(s) >= 17]
        if not search_ids:
            return result
        es = [expanded[j] for j in search_ids]
        eg = self._slice_encoding(dg, search_ids)
        ee = self._slice_encoding(de, search_ids)
        routes = self._refine(ee, es, phase='expanded')
        runner = getattr(self, '_candidate_graph', None)
        if runner is None:
            from .graphed_candidates import expanded_candidate_tensor
            packed_device = expanded_candidate_tensor(self, eg, es, routes)
        else:
            packed_device = runner(eg, es, routes)
        # Include centers in the same host copy as all angle/score pairs.
        packed = packed_device.cpu().numpy()
        jobs, destinations, cursor = [], [], 0
        for subset_j, s in zip(search_ids, es):
            scene_j = dense[subset_j]
            part = packed[cursor:cursor+len(s)]
            cursor += len(s)
            values, centres = part[:, :-1].reshape(len(s), 31, 2), part[:, -1]
            ini = initial[subset_j]
            if not len(ini):
                continue
            route_id = np.argmin(abs(centres[:, None]-ini[None, :]), axis=1)
            row = dict(angles=values[:, :, 0].reshape(-1), old=values[:, :, 1].reshape(-1),
                       centres=centres, route_id=np.repeat(route_id, 31).astype(np.int64))
            jobs.append((row, covariances[scene_j], ini))
            destinations.append(scene_j)
        for j, angles in zip(destinations, self._select_sml_jobs(jobs)):
            result[j] = np.sort(np.asarray(angles, dtype=float))
        return result
