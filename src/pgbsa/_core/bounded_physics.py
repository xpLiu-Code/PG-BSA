"""Compiled covariance fits with bounded scalar optimization.

The objectives retain their evaluation order and use the SciPy-compatible
bounded solver. Inputs follow the complex covariance convention of each fit."""
from __future__ import annotations
import numpy as np
from numba import njit
from .bounded_spectral import bounded_minimize


@njit(cache=True, fastmath=False, nogil=True)
def _sum16_pairwise(values):
    # Match NumPy's eight-lane pairwise reduction for this fixed length.
    r0 = values[0] + values[8]
    r1 = values[1] + values[9]
    r2 = values[2] + values[10]
    r3 = values[3] + values[11]
    r4 = values[4] + values[12]
    r5 = values[5] + values[13]
    r6 = values[6] + values[14]
    r7 = values[7] + values[15]
    return ((r0+r1)+(r2+r3))+((r4+r5)+(r6+r7))


@njit(cache=True, fastmath=False, nogil=True)
def fixed_fraction_objective(fraction, eigenvalues, covariance_diagonal):
    logs = np.empty(16, np.float64)
    ratios = np.empty(16, np.float64)
    for i in range(16):
        # Preserve (f*values + 1 - f), including its evaluation order.
        diagonal = (fraction*eigenvalues[i]+1.-fraction)/16.
        logs[i] = np.log(diagonal)
        ratios[i] = covariance_diagonal[i]/diagonal
    return _sum16_pairwise(logs) + _sum16_pairwise(ratios)


@njit(cache=True, fastmath=False, nogil=True)
def fit_fixed_fraction(eigenvalues, covariance_diagonal):
    return bounded_minimize(fixed_fraction_objective,
                            (eigenvalues, covariance_diagonal),
                            .001, .999999, 1.e-7, 500)


@njit(cache=True, fastmath=False, nogil=True)
def negative_single_projection(theta, covariance):
    a = np.empty(16, np.complex128)
    sine = np.sin(theta*(np.pi/180.))
    for i in range(16):
        a[i] = np.exp((-1j*np.pi*i)*sine)/4.
    value = 0.j
    for i in range(16):
        projected = 0.j
        for j in range(16):
            projected += covariance[i,j]*a[j]
        value += a[i].conjugate()*projected
    return -value.real


@njit(cache=True, fastmath=False, nogil=True)
def fit_single_projection(covariance, lower, upper):
    # Near-flat projections and tied peaks can amplify reduction roundoff.
    # Use the SciPy/NumPy path when the minimizer is numerically ambiguous.
    scale = 0.
    for i in range(16):
        for j in range(16):
            scale += abs(covariance[i,j])/16.
    error = 256.*np.finfo(np.float64).eps*max(scale, 1.e-300)
    midpoint = .5*(lower+upper)
    left = negative_single_projection(lower, covariance)
    middle = negative_single_projection(midpoint, covariance)
    right = negative_single_projection(upper, covariance)
    if max(left,middle,right)-min(left,middle,right) <= error:
        return midpoint, middle, 3, 3
    theta, value, evaluations, status = bounded_minimize(
        negative_single_projection, (covariance,), lower, upper, 2.e-4, 500)
    mirror = lower+upper-theta
    if abs(mirror-theta) > 2.e-4:
        other = negative_single_projection(mirror, covariance)
        if abs(other-value) <= error:
            return theta, value, evaluations+4, 3
    return theta, value, evaluations+3, status
