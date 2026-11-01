"""Small-matrix likelihood and coordinate search kernels."""
import math
import numpy as np
from numba import njit

@njit(cache=True, fastmath=False)
def spectral_energy(f, eigenvalues, covariance_diagonal, source_count):
    result = 0.
    for i in range(16):
        diagonal = (f/source_count*eigenvalues[i]+(1-f))/16.
        result += math.log(diagonal)+covariance_diagonal[i]/diagonal
    return result


@njit(cache=True, fastmath=False)
def score_candidates(atoms, covariance, inverse, angles, scores, others, pool,
                     power, strength, minimum_separation):
    # Explicit 16-element loops avoid Numba's separate BLAS/OpenMP runtime.
    # Matrix inversion uses NumPy complex128 arithmetic.
    best, best_value = -1, -np.inf
    v = np.empty(16, np.complex128)
    for index in pool:
        legal = True
        for other in others:
            if not abs(angles[index]-angles[other]) >= minimum_separation-1.e-7:
                legal = False
                break
        if not legal:
            continue
        for i in range(16):
            value = 0.j
            for j in range(16):
                value += inverse[i,j]*atoms[index,j]
            v[i] = value
        inner, reduction = 0.j, 0.j
        for i in range(16):
            inner += atoms[index,i].conjugate()*v[i]
            sv = 0.j
            for j in range(16):
                sv += covariance[i,j]*v[j]
            reduction += v[i].conjugate()*sv
        den = 1. + power*inner.real
        red = power*reduction.real/den
        value = scores[index]-strength*(1000.*(math.log(den)-red))
        if best < 0 or value > best_value:
            best, best_value = index, value
    return best


def coordinate_pass(atoms, covariance, angles, scores, chosen, pools,
                    power, noise, strength, minimum_separation):
    eye, changes = np.eye(16), 0
    for q, pool in enumerate(pools):
        others = np.delete(chosen, q)
        a = atoms[others].T
        inverse = np.linalg.inv(power*(a@a.conj().T)+noise*eye)
        best = score_candidates(atoms, covariance, inverse, angles, scores, others, pool,
                                power, strength, minimum_separation)
        if best >= 0:
            changes += int(best != chosen[q])
            chosen[q] = best
    return changes
