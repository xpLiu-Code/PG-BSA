"""ULA polynomial evaluation of the covariance search objective.

Dense matrix evaluation handles small candidate pools, uncertain ties,
nonfinite results, and poorly conditioned systems. The projection
H = U.conj().T @ R @ U is evaluated for the supplied covariance."""
import math
import numpy as np
from numba import njit
from .compiled_search import score_candidates as dense_score_candidates

EPS = 2.220446049250313e-16
MIN_LEGAL_POOL = 32


@njit(cache=True, fastmath=False, nogil=True)
def _legal(index, angles, others, minimum_separation):
    for other in others:
        if not abs(angles[index] - angles[other]) >= minimum_separation - 1.e-7:
            return False
    return True


@njit(cache=True, fastmath=False, nogil=True)
def _dense_value(atoms, covariance, inverse, index, scores, power, strength):
    vector = np.empty(16, np.complex128)
    for i in range(16):
        item = 0.j
        for j in range(16):
            item += inverse[i, j] * atoms[index, j]
        vector[i] = item
    inner, reduction = 0.j, 0.j
    for i in range(16):
        inner += atoms[index, i].conjugate() * vector[i]
        item = 0.j
        for j in range(16):
            item += covariance[i, j] * vector[j]
        reduction += vector[i].conjugate() * item
    denominator = 1. + power * inner.real
    red = power * reduction.real / denominator
    return scores[index] - strength * (1000. * (math.log(denominator) - red))


@njit(cache=True, fastmath=False, nogil=True)
def _lag_real_coefficients(matrix):
    # For a[n]=exp(-i*n*phi), Re(a^H M a) needs 1 + 15 pairs.
    real = np.empty(16, np.float64)
    imag = np.zeros(16, np.float64)
    diagonal = 0.
    for i in range(16):
        diagonal += matrix[i, i].real
    real[0] = diagonal
    for lag in range(1, 16):
        plus, minus = 0.j, 0.j
        for i in range(16 - lag):
            plus += matrix[i, i + lag]
            minus += matrix[i + lag, i]
        real[lag] = plus.real + minus.real
        imag[lag] = minus.imag - plus.imag
    return real, imag


@njit(cache=True, fastmath=False, nogil=True)
def score_candidates_ula(atoms, covariance, inverse, angles, scores, others, pool,
                         power, strength, minimum_separation):
    # Keep the original function's arguments and first-in-pool tie semantics.
    legal_count = 0
    for index in pool:
        if _legal(index, angles, others, minimum_separation):
            legal_count += 1
    if legal_count < MIN_LEGAL_POOL or atoms.shape[1] != 16:
        return dense_score_candidates(atoms, covariance, inverse, angles, scores,
            others, pool, power, strength, minimum_separation)
    u_abs, u_frob2, r_frob2 = 0., 0., 0.
    for i in range(16):
        for j in range(16):
            u, r = inverse[i, j], covariance[i, j]
            if not (math.isfinite(u.real) and math.isfinite(u.imag)
                    and math.isfinite(r.real) and math.isfinite(r.imag)):
                return dense_score_candidates(atoms, covariance, inverse, angles, scores,
                    others, pool, power, strength, minimum_separation)
            u_abs += abs(u)
            u_frob2 += u.real*u.real + u.imag*u.imag
            r_frob2 += r.real*r.real + r.imag*r.imag
    intermediate = np.empty((16, 16), np.complex128)
    quadratic = np.empty((16, 16), np.complex128)
    for i in range(16):
        for j in range(16):
            total = 0.j
            for k in range(16):
                total += covariance[i, k] * inverse[k, j]
            intermediate[i, j] = total
    h_abs = 0.
    for i in range(16):
        for j in range(16):
            total = 0.j
            for k in range(16):
                total += inverse[k, i].conjugate() * intermediate[k, j]
            quadratic[i, j] = total
            h_abs += abs(total)
    u_real, u_imag = _lag_real_coefficients(inverse)
    h_real, h_imag = _lag_real_coefficients(quadratic)
    # Loose absolute roundoff bounds cover matrix-association differences,
    # diagonal summation, phase products and the original direct quadratic form.
    inner_error = 512. * EPS * u_abs
    reduction_error = 2048. * EPS * (h_abs + 16. * math.sqrt(r_frob2) * u_frob2)
    denominator_error = abs(power) * inner_error
    best, best_value, second_value = -1, -np.inf, -np.inf
    max_score_error = 0.
    for index in pool:
        if not _legal(index, angles, others, minimum_separation):
            continue
        inner, reduction = u_real[0], h_real[0]
        for lag in range(1, 16):
            phase = atoms[index, lag]
            inner += u_real[lag]*phase.real + u_imag[lag]*phase.imag
            reduction += h_real[lag]*phase.real + h_imag[lag]*phase.imag
        denominator = 1. + power * inner
        lower = denominator - denominator_error
        if not (lower > 1.e-10 and math.isfinite(denominator) and math.isfinite(reduction)):
            return dense_score_candidates(atoms, covariance, inverse, angles, scores,
                others, pool, power, strength, minimum_separation)
        red = power * reduction / denominator
        value = scores[index] - strength * (1000. * (math.log(denominator) - red))
        error = abs(strength)*1000. * (denominator_error/lower
            + abs(power)*(reduction_error/lower
            + abs(reduction)*denominator_error/(lower*denominator)))
        error += 2048. * EPS * (1. + abs(value) + abs(scores[index]))
        if not (math.isfinite(value) and math.isfinite(error)):
            return dense_score_candidates(atoms, covariance, inverse, angles, scores,
                others, pool, power, strength, minimum_separation)
        max_score_error = max(max_score_error, error)
        if best < 0 or value > best_value:
            second_value = best_value
            best, best_value = index, value
        elif value > second_value:
            second_value = value
    if best < 0:
        return -1
    if best_value - second_value <= 2. * max_score_error:
        return dense_score_candidates(atoms, covariance, inverse, angles, scores,
            others, pool, power, strength, minimum_separation)
    exact_best = _dense_value(atoms, covariance, inverse, best, scores, power, strength)
    if not math.isfinite(exact_best) or abs(exact_best - best_value) > max_score_error:
        return dense_score_candidates(atoms, covariance, inverse, angles, scores,
            others, pool, power, strength, minimum_separation)
    return best
