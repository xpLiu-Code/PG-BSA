"""Fixed-count likelihood search with rank-one inverse updates.

Each coordinate pass holds the fitted power fraction fixed. Poorly
conditioned inverse downdates use the dense matrix-solve implementation."""
import math
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import numpy as np
from numba import njit
from scipy.optimize import minimize_scalar
from .compiled_search import spectral_energy, score_candidates as dense_score_candidates, coordinate_pass
from .ula_search_score import score_candidates_ula
from .bounded_spectral import bounded_spectral_fraction


@njit(cache=True, fastmath=False, nogil=True)
def rank_coordinate_pass(atoms, covariance, angles, scores, chosen, pool_index,
                         pool_starts, inverse, power, strength, minimum_separation, use_ula_score=True):
    changes = 0
    removed = np.empty(16, np.complex128)
    added = np.empty(16, np.complex128)
    leave_one_out = np.empty((16,16), np.complex128)
    for q in range(len(chosen)):
        old = chosen[q]
        for i in range(16):
            value = 0.j
            for j in range(16):
                value += inverse[i,j]*atoms[old,j]
            removed[i] = value
        inner = 0.j
        for i in range(16):
            inner += atoms[old,i].conjugate()*removed[i]
        denominator = 1.-power*inner.real
        if denominator <= 1.e-9 or not math.isfinite(denominator):
            return changes, False
        scale = power/denominator
        for i in range(16):
            for j in range(16):
                leave_one_out[i,j] = inverse[i,j]+scale*removed[i]*removed[j].conjugate()
        others = np.empty(len(chosen)-1, np.int64)
        cursor = 0
        for j in range(len(chosen)):
            if j != q:
                others[cursor] = chosen[j]
                cursor += 1
        pool = pool_index[pool_starts[q]:pool_starts[q+1]]
        if use_ula_score:
            best = score_candidates_ula(atoms, covariance, leave_one_out, angles, scores,
                                        others, pool, power, strength, minimum_separation)
        else:
            best = dense_score_candidates(atoms, covariance, leave_one_out, angles, scores,
                                          others, pool, power, strength, minimum_separation)
        if best >= 0 and best != old:
            for i in range(16):
                value = 0.j
                for j in range(16):
                    value += leave_one_out[i,j]*atoms[best,j]
                added[i] = value
            inner = 0.j
            for i in range(16):
                inner += atoms[best,i].conjugate()*added[i]
            denominator = 1.+power*inner.real
            if denominator <= 1.e-9 or not math.isfinite(denominator):
                return changes, False
            scale = power/denominator
            for i in range(16):
                for j in range(16):
                    inverse[i,j] = leave_one_out[i,j]-scale*added[i]*added[j].conjugate()
            chosen[q] = best
            changes += 1
    return changes, True


def select_wide_sml_rank_updates(row, covariance, initial, *, radius=2.5,
                                 strength=5., minimum_separation=1., passes=3, use_ula_score=True):
    old_a = np.asarray(row['angles'], np.float64).reshape(-1)
    old_z = np.asarray(row['old'], np.float64).reshape(-1)
    m = len(initial)
    if not m:
        return np.empty(0, np.float32)
    route = np.asarray(row.get('route_id', np.repeat(np.arange(m), max(1, len(old_a)//m))), np.int64).reshape(-1)
    if len(route) != len(old_a):
        route = np.repeat(np.arange(m), len(old_a)//m)
    aa, zz, gg = [old_a], [old_z], [route]
    offsets = np.arange(-radius, radius+1.e-6, .25)
    sorted_a = np.sort(old_a)
    rescue = np.asarray(initial, np.float64)[:, None]+offsets[None]
    insertion = np.searchsorted(sorted_a, rescue)
    left = sorted_a[np.clip(insertion-1, 0, len(sorted_a)-1)]
    right = sorted_a[np.clip(insertion, 0, len(sorted_a)-1)]
    valid = (rescue >= -69.9) & (rescue <= 69.9) & (np.minimum(abs(rescue-left), abs(rescue-right)) > .025)
    for q in range(m):
        ix = np.flatnonzero(route == q)
        if not len(ix):
            continue
        mask = valid[q]
        if mask.any():
            aa.append(rescue[q, mask])
            zz.append(float(np.max(old_z[ix]))-.25-.02*abs(offsets[mask]))
            gg.append(np.full(int(mask.sum()), q, np.int64))
    angles, scores, groups = np.concatenate(aa), np.concatenate(zz), np.concatenate(gg)
    atoms = np.exp(-1j*np.pi*np.arange(16)[None]*np.sin(np.deg2rad(angles))[:, None])
    covariance = np.asarray(covariance, np.complex128)
    pools = [np.flatnonzero(groups == q) for q in range(m)]
    chosen = np.asarray([int(ix[np.argmin(abs(angles[ix]-float(initial[q])))]) if len(ix)
                         else int(np.argmin(abs(angles-float(initial[q])))) for q, ix in enumerate(pools)], np.int64)
    pools = [ix[abs(angles[ix]-float(initial[q])) <= radius] for q, ix in enumerate(pools)]
    pool_index = np.concatenate(pools)
    pool_starts = np.cumsum(np.asarray([0]+[len(p) for p in pools], np.int64))
    eye = np.eye(16)
    for _ in range(passes):
        a = atoms[chosen].T
        b = a@a.conj().T
        values, vectors = np.linalg.eigh(b)
        covariance_diagonal = np.diag(vectors.conj().T@covariance@vectors).real.copy()
        f = float(bounded_spectral_fraction(values, covariance_diagonal, m)[0])
        power, noise = f/(16*m), (1-f)/16.
        inverse = np.linalg.inv(power*b+noise*eye)
        original = chosen.copy()
        changes, valid_pass = rank_coordinate_pass(
            atoms, covariance, angles, scores, chosen, pool_index, pool_starts,
            inverse, power, strength, minimum_separation, use_ula_score)
        if not valid_pass:
            chosen[:] = original
            changes = coordinate_pass(atoms, covariance, angles, scores, chosen, pools,
                                      power, noise, strength, minimum_separation)
        if not changes:
            break
    return np.sort(angles[chosen].astype(np.float32))


def _run_job(job, settings):
    return select_wide_sml_rank_updates(*job, **settings)


class RankUpdateSearch:
    """Persistent bounded CPU pool; small calls retain the serial path.

    Call with a sequence of (candidate_row, covariance, initial_angles) jobs.
    ``close`` releases the worker pool; prediction owners should invoke it from
    their own close/context-manager method rather than recreating per batch.
    """
    def __init__(self, workers=4, minimum_parallel_jobs=4, *, use_ula_score=True):
        if workers < 1 or minimum_parallel_jobs < 1:
            raise ValueError('workers and minimum_parallel_jobs must be positive')
        self.workers = int(workers)
        self.use_ula_score = bool(use_ula_score)
        self.minimum_parallel_jobs = int(minimum_parallel_jobs)
        self._pool = None
        self._closed = False

    def map_jobs(self, function, jobs, *, parallel=True):
        """Execute independent covariance-search jobs with ordered results.

        Workers return results in input order. The prediction thread updates
        diagnostics and counters as it consumes those results."""
        if self._closed:
            raise RuntimeError('RankUpdateSearch has been closed')
        jobs = list(jobs)
        if not parallel or self.workers == 1 or len(jobs) < self.minimum_parallel_jobs:
            return (function(job) for job in jobs)
        if self._pool is None:
            self._pool = ThreadPoolExecutor(self.workers, thread_name_prefix='pgbsa-sml')
        return self._pool.map(function, jobs)

    def __call__(self, jobs, **settings):
        settings.setdefault("use_ula_score", self.use_ula_score)
        solve = partial(_run_job, settings=settings)
        return list(self.map_jobs(solve, jobs))

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        self._closed = True

    def __enter__(self):
        if self._closed:
            raise RuntimeError('RankUpdateSearch has been closed')
        return self

    def __exit__(self, *_):
        self.close()
