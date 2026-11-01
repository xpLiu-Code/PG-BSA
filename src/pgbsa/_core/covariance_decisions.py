"""Observation-only noise and close-source decisions for the ideal ULA contract."""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize_scalar
from .direct_lbfgsb import direct_minimize as minimize
from numba import njit


@njit(cache=True, nogil=True)
def pair_energy(center, gap, fraction, r):
    sigma = (1-fraction)/16.
    power = fraction/2.
    q = power/sigma
    u = np.empty(16, np.complex128)
    v = np.empty(16, np.complex128)
    w0 = -np.pi*np.sin((center-gap/2)*np.pi/180.)
    w1 = -np.pi*np.sin((center+gap/2)*np.pi/180.)
    for i in range(16):
        u[i] = np.exp(1j*w0*i)/4.
        v[i] = np.exp(1j*w1*i)/4.
    g, h00, h11, h01 = 0.j, 0.j, 0.j, 0.j
    tr = 0.
    for i in range(16):
        g += u[i].conjugate()*v[i]
        ru, rv = 0.j, 0.j
        for j in range(16):
            ru += r[i,j]*u[j]
            rv += r[i,j]*v[j]
        h00 += u[i].conjugate()*ru
        h11 += v[i].conjugate()*rv
        h01 += u[i].conjugate()*rv
        tr += r[i,i].real
    det = (1+q)**2-q*q*abs(g)**2
    correction = ((1+q)*(h00.real+h11.real)-2*q*(g.conjugate()*h01).real)/det
    return 16*np.log(sigma)+np.log(det)+tr/sigma-power/(sigma*sigma)*correction


@njit(cache=True, nogil=True)
def pair_value_gradient(parameters, r):
    center, gap, fraction = parameters
    sigma, power = (1-fraction)/16., fraction/2.
    q = power/sigma
    aa = np.empty((16, 2), np.complex128)
    da = np.empty((16, 2), np.complex128)
    for k in range(2):
        theta = (center+(-.5 if k == 0 else .5)*gap)*np.pi/180.
        for i in range(16):
            aa[i,k] = np.exp(-1j*np.pi*i*np.sin(theta))/4.
            da[i,k] = aa[i,k]*(-1j*np.pi*i*np.cos(theta)*np.pi/180.)
    g = 0.j
    for i in range(16):
        g += aa[i,0].conjugate()*aa[i,1]
    det = (1+q)**2-q*q*abs(g)**2
    inverse = np.empty((16,16), np.complex128)
    for i in range(16):
        for j in range(16):
            reduced = ((1+q)*(aa[i,0]*aa[j,0].conjugate()+aa[i,1]*aa[j,1].conjugate())
                       -q*g*aa[i,0]*aa[j,1].conjugate()-q*g.conjugate()*aa[i,1]*aa[j,0].conjugate())/det
            inverse[i,j] = (1./sigma if i == j else 0.)-power/(sigma*sigma)*reduced
    product = np.empty((16,16), np.complex128)
    matrix = np.empty((16,16), np.complex128)
    for i in range(16):
        for j in range(16):
            value = 0.j
            for k in range(16):
                value += inverse[i,k]*r[k,j]
            product[i,j] = value
    for i in range(16):
        for j in range(16):
            value = 0.j
            for k in range(16):
                value += product[i,k]*inverse[k,j]
            matrix[i,j] = inverse[i,j]-value
    value = 16*np.log(sigma)+np.log(det)
    trace = 0.
    for i in range(16):
        value += product[i,i].real
        trace += matrix[i,i].real
    angle_gradient = np.zeros(2)
    power_gradient = -trace/16.
    for k in range(2):
        for i in range(16):
            qa = 0.j
            for j in range(16):
                qa += matrix[i,j]*aa[j,k]
            angle_gradient[k] += 2*power*(qa.conjugate()*da[i,k]).real
            power_gradient += .5*(aa[i,k].conjugate()*qa).real
    return value, np.array([angle_gradient[0]+angle_gradient[1],
                           .5*(angle_gradient[1]-angle_gradient[0]), power_gradient])


@njit(cache=True, nogil=True)
def pair_value_gradient_rank2(parameters, r):
    """Evaluate the two-source likelihood and gradient in a rank-two subspace.

    A contains two unit-norm steering vectors, and r is the observed 16x16
    covariance. Let B=(I+q A^H A)^-1, q=power/sigma, H=A^H r A, where power is
    the signal fraction per source and sigma is the noise fraction per sensor.
    C^-1 A=A B/sigma and tr(C^-1 r C^-1)=
    [tr(r)-q tr((B+B^2)H)]/sigma^2. The covariance multiplication r@A has shape
    16x2; the remaining products use 2x2 matrices without a dense inverse.
    """
    center, gap, fraction = parameters
    # At vanishing noise, trace/Q cancellation can perturb a near-zero gradient.
    # Use the dense computation in this low-noise region.
    if fraction >= .995:
        return pair_value_gradient(parameters, r)
    sigma, power = (1-fraction)/16., fraction/2.
    q = power/sigma
    aa = np.empty((16, 2), np.complex128)
    da = np.empty((16, 2), np.complex128)
    for k in range(2):
        theta = (center+(-.5 if k == 0 else .5)*gap)*np.pi/180.
        for i in range(16):
            aa[i,k] = np.exp(-1j*np.pi*i*np.sin(theta))/4.
            da[i,k] = aa[i,k]*(-1j*np.pi*i*np.cos(theta)*np.pi/180.)
    g = 0.j
    for i in range(16):
        g += aa[i,0].conjugate()*aa[i,1]
    det = (1+q)**2-q*q*abs(g)**2
    b = np.empty((2,2), np.complex128)
    b[0,0] = (1+q)/det
    b[1,1] = b[0,0]
    b[0,1] = -q*g/det
    b[1,0] = b[0,1].conjugate()
    ra = np.empty((16,2), np.complex128)
    trace_r = 0.
    for i in range(16):
        trace_r += r[i,i].real
        for k in range(2):
            value = 0.j
            for j in range(16):
                value += r[i,j]*aa[j,k]
            ra[i,k] = value
    h = np.zeros((2,2), np.complex128)
    for k in range(2):
        for l in range(2):
            for i in range(16):
                h[k,l] += aa[i,k].conjugate()*ra[i,l]
    bh = np.zeros((2,2), np.complex128)
    bhb = np.zeros((2,2), np.complex128)
    b_squared = np.zeros((2,2), np.complex128)
    for k in range(2):
        for l in range(2):
            for j in range(2):
                bh[k,l] += b[k,j]*h[j,l]
                b_squared[k,l] += b[k,j]*b[j,l]
    for k in range(2):
        for l in range(2):
            for j in range(2):
                bhb[k,l] += bh[k,j]*b[j,l]
    trace_bh = (bh[0,0]+bh[1,1]).real
    correction = 0.
    for k in range(2):
        for l in range(2):
            correction += ((b[k,l]+b_squared[k,l])*h[l,k]).real
    inv_sigma = 1./sigma
    inv_sigma2 = inv_sigma*inv_sigma
    value = 16*np.log(sigma)+np.log(det)+(trace_r-q*trace_bh)*inv_sigma
    trace_q = (14.+b[0,0].real+b[1,1].real)*inv_sigma-(trace_r-q*correction)*inv_sigma2
    angle_gradient = np.zeros(2)
    power_gradient = -trace_q/16.
    for k in range(2):
        for i in range(16):
            ab, rab, abhb = 0.j, 0.j, 0.j
            for l in range(2):
                ab += aa[i,l]*b[l,k]
                rab += ra[i,l]*b[l,k]
                abhb += aa[i,l]*bhb[l,k]
            qa = ab*inv_sigma-(rab-q*abhb)*inv_sigma2
            angle_gradient[k] += 2*power*(qa.conjugate()*da[i,k]).real
            power_gradient += .5*(aa[i,k].conjugate()*qa).real
    return value, np.array([angle_gradient[0]+angle_gradient[1],
                           .5*(angle_gradient[1]-angle_gradient[0]), power_gradient])


def toeplitz_evidence(covariance):
    r = np.asarray(covariance, np.complex128)
    r = (r+r.conj().T)*.5
    r /= max(float(np.trace(r).real), 1e-12)
    lag = np.array([np.diag(r, k).mean() for k in range(16)])
    structured = float(2*np.dot(np.arange(15, 0, -1), abs(lag[1:])**2))
    total = float(np.sum(abs(r-np.eye(16)/16)**2))
    residual = max(total-structured, 1e-12)
    return structured/residual, structured, residual


def atoms(angles):
    return np.exp(-1j*np.pi*np.arange(16)[:, None]*np.sin(np.deg2rad(angles))[None, :])/4


def covariance_nll(r, angles, fraction):
    a = atoms(np.asarray(angles))
    sigma = (1-fraction)/16
    p = fraction/len(angles)
    matrix = sigma*np.eye(16)+p*a@a.conj().T
    sign, logdet = np.linalg.slogdet(matrix)
    if sign.real <= 0:
        return 1e12
    return float(logdet+np.trace(np.linalg.solve(matrix, r)).real)


def fit_single(r, center, radius=3.):
    r = np.asarray(r, np.complex128)
    from .bounded_physics import fit_single_projection
    lower, upper = max(-69.9, center-radius), min(69.9, center+radius)
    if r.shape == (16, 16) and np.isfinite(lower) and np.isfinite(upper) and lower <= upper and np.isfinite(r).all():
        theta, objective, _, status = fit_single_projection(r, lower, upper)
        if status == 0:
            power = np.clip(-objective, 1/16+1e-8, 1-1e-8)
            fraction = (16*power-1)/15
            return dict(angles=np.array([theta]), fraction=float(fraction),
                        nll=float(np.log(power)+15*np.log((1-power)/15)+16))
    # Delegate unsupported or nonfinite inputs to SciPy.
    def projected(theta):
        a = atoms([theta])[:, 0]
        return float(np.vdot(a, r@a).real)
    fit = minimize_scalar(lambda theta: -projected(theta), bounds=(max(-69.9, center-radius), min(69.9, center+radius)),
                          method='bounded', options={'xatol': 2e-4})
    theta = float(fit.x)
    power = np.clip(projected(theta), 1/16+1e-8, 1-1e-8)
    fraction = (16*power-1)/15
    return dict(angles=np.array([theta]), fraction=float(fraction),
                nll=float(np.log(power)+15*np.log((1-power)/15)+16))


def fit_pair(r, single, *, max_gap=6., starts=(1., 3.), rank2_gradient=False):
    center = float(single['angles'][0])
    r = np.asarray(r, np.complex128)
    bounds = [(max(-69.9, center-2), min(69.9, center+2)), (.05, max_gap), (.001, .9999)]
    gradient = pair_value_gradient_rank2 if rank2_gradient else pair_value_gradient
    def objective(parameters):
        return gradient(parameters, r)
    best = None
    for gap in starts:
        result = minimize(objective, [center, min(gap, max_gap), single['fraction']], method='L-BFGS-B', jac=True,
                          bounds=bounds, options={'ftol': 1e-11, 'gtol': 1e-5, 'maxiter': 55})
        if best is None or result.fun < best.fun:
            best = result
    ctr, gap, fraction = best.x
    return dict(angles=np.array([ctr-gap/2, ctr+gap/2]), fraction=float(fraction),
                nll=float(best.fun), converged=bool(best.success))


def fit_triple(r, pair):
    """A third-source alternative initialized by the fitted two-source moment."""
    from .fast_cardinality_refinement import regular_objective
    center = float(np.mean(pair['angles']))
    span = float(np.ptp(pair['angles']))
    fit = minimize(regular_objective,
                   [center, span*np.sqrt(1.5), np.log(max(1-pair['fraction'], 1e-5))],
                   args=(np.asarray(r, np.complex128), 3), jac=True, method='L-BFGS-B',
                   bounds=[(max(-68, center-2), min(68, center+2)),
                           (max(.5, span*.6), min(12., span*2)), (np.log(1e-6), np.log(.999))],
                   options={'maxiter': 50, 'ftol': 1e-11, 'gtol': 1e-6})
    ctr, width, eta = fit.x
    return dict(angles=ctr+width*np.array([-.5, 0., .5]), fraction=float(1-np.exp(eta)),
                nll=float(fit.fun), converged=bool(fit.success))
