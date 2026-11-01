"""Numba implementation of SciPy bounded scalar minimization.

Adapted from scipy.optimize._optimize._minimize_scalar_bounded under the
SciPy BSD-3-Clause license. The solver preserves its stopping criteria,
evaluation limit, tie handling, and nonfinite-result status codes."""
import math
import numpy as np
from numba import njit
from .compiled_search import spectral_energy


@njit(cache=True, fastmath=False, nogil=True, inline="always")
def bounded_minimize(objective, args, lower, upper, xatol=1.e-5, maxiter=500):
    """Return (x, objective(x), evaluations, status), matching bounded SciPy.

    The caller owns bounds/input validation. A Numba objective and argument
    tuple are required. Status 0=converged, 1=max evaluations, 2=NaN.
    """
    flag = 0
    sqrt_eps = math.sqrt(2.2e-16)
    golden_mean = 0.5 * (3.0 - math.sqrt(5.0))
    a, b = lower, upper
    fulc = a + golden_mean * (b - a)
    nfc, xf = fulc, fulc
    rat = e = 0.0
    x = xf
    fx = objective(x, *args)
    num = 1
    fu = np.inf
    ffulc = fnfc = fx
    xm = 0.5 * (a + b)
    tol1 = sqrt_eps * np.abs(xf) + xatol / 3.0
    tol2 = 2.0 * tol1
    while np.abs(xf - xm) > (tol2 - 0.5 * (b - a)):
        golden = 1
        if np.abs(e) > tol1:
            golden = 0
            r = (xf - nfc) * (fx - ffulc)
            q = (xf - fulc) * (fx - fnfc)
            p = (xf - fulc) * q - (xf - nfc) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = np.abs(q)
            r = e
            e = rat
            if ((np.abs(p) < np.abs(0.5*q*r)) and (p > q*(a-xf))
                    and (p < q*(b-xf))):
                rat = (p + 0.0) / q
                x = xf + rat
                if ((x-a) < tol2) or ((b-x) < tol2):
                    si = np.sign(xm-xf) + ((xm-xf) == 0)
                    rat = tol1 * si
            else:
                golden = 1
        if golden:
            if xf >= xm:
                e = a - xf
            else:
                e = b - xf
            rat = golden_mean * e
        si = np.sign(rat) + (rat == 0)
        x = xf + si * np.maximum(np.abs(rat), tol1)
        fu = objective(x, *args)
        num += 1
        if fu <= fx:
            if x >= xf:
                a = xf
            else:
                b = xf
            fulc, ffulc = nfc, fnfc
            nfc, fnfc = xf, fx
            xf, fx = x, fu
        else:
            if x < xf:
                a = x
            else:
                b = x
            if (fu <= fnfc) or (nfc == xf):
                fulc, ffulc = nfc, fnfc
                nfc, fnfc = x, fu
            elif (fu <= ffulc) or (fulc == xf) or (fulc == nfc):
                fulc, ffulc = x, fu
        xm = 0.5 * (a + b)
        tol1 = sqrt_eps * np.abs(xf) + xatol / 3.0
        tol2 = 2.0 * tol1
        if num >= maxiter:
            flag = 1
            break
    if np.isnan(xf) or np.isnan(fx) or np.isnan(fu):
        flag = 2
    return xf, fx, num, flag


@njit(cache=True, fastmath=False, nogil=True)
def bounded_spectral_fraction(eigenvalues, covariance_diagonal, source_count,
                              lower=.05, upper=.99999, xatol=1.e-5, maxiter=500):
    return bounded_minimize(spectral_energy,
                            (eigenvalues, covariance_diagonal, source_count),
                            lower, upper, xatol, maxiter)
