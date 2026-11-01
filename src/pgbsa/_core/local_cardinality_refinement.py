"""Local covariance fitting of two-, three-, and four-source hypotheses.

Covariance and predicted angles determine the eligible close group. At most
one isolated group is optimized per scene while other directions remain
fixed. The decoder applies this stage after global source-count refinement."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from numba import njit
from .direct_lbfgsb import direct_minimize as minimize
from .cardinality_refinement import covariance_from_input, fixed_angle_likelihood, saturated_likelihood
from .covariance_conditioning import estimate_equivalent_snapshots


@njit(cache=True, fastmath=False, nogil=True)
def _local_objective(parameters, covariance, count, background_lags, total_count):
    center, width, log_noise = parameters
    noise = math.exp(log_noise)
    signal = 1. - noise
    gram = np.empty((16, 16), np.complex128)
    dcenter = np.empty((16, 16), np.complex128)
    dwidth = np.empty((16, 16), np.complex128)
    dnoise = np.empty((16, 16), np.complex128)
    predicted = np.empty((16, 16), np.complex128)
    # The ULA Gram and both angular derivatives depend only on sensor lag.
    gs = np.empty(16, np.complex128)
    cs = np.empty(16, np.complex128)
    ws = np.empty(16, np.complex128)
    for lag in range(16):
        g, c, w = 0.j, 0.j, 0.j
        for q in range(count):
            fraction = -.5 + q/(count-1) if count > 1 else 0.
            angle = (center + width*fraction) * (math.pi/180.)
            phase = math.pi * lag * math.sin(angle)
            atom = complex(math.cos(phase), -math.sin(phase))
            derivative = -1j * math.pi * lag * math.cos(angle) * (math.pi/180.) * atom
            g += atom
            c += derivative
            w += derivative * fraction
        gs[lag] = (g+background_lags[lag])/total_count
        cs[lag] = signal*c/(16*total_count)
        ws[lag] = signal*w/(16*total_count)
    for i in range(16):
        for j in range(16):
            if i >= j:
                g, c, w = gs[i-j], cs[i-j], ws[i-j]
            else:
                g, c, w = gs[j-i].conjugate(), cs[j-i].conjugate(), ws[j-i].conjugate()
            gram[i,j] = g
            identity = 1. if i == j else 0.
            predicted[i,j] = (signal*g + noise*identity)/16.
            dcenter[i,j], dwidth[i,j] = c, w
            dnoise[i,j] = noise*(identity-g)/16.

    lower = np.zeros((16,16), np.complex128)
    logdet = 0.
    for i in range(16):
        for j in range(i+1):
            value = predicted[i,j]
            for k in range(j):
                value -= lower[i,k]*lower[j,k].conjugate()
            if i == j:
                if value.real <= 0.:
                    return np.inf, np.zeros(3, np.float64)
                diagonal = math.sqrt(value.real)
                lower[i,j] = diagonal
                logdet += 2.*math.log(diagonal)
            else:
                lower[i,j] = value/lower[j,j].real
    inverse = np.empty((16,16), np.complex128)
    temp = np.empty(16, np.complex128)
    for column in range(16):
        for i in range(16):
            value = complex(1. if i == column else 0., 0.)
            for k in range(i):
                value -= lower[i,k]*temp[k]
            temp[i] = value/lower[i,i].real
        for i in range(15,-1,-1):
            value = temp[i]
            for k in range(i+1,16):
                value -= lower[k,i].conjugate()*inverse[k,column]
            inverse[i,column] = value/lower[i,i].real
    product = np.empty((16,16), np.complex128)
    for i in range(16):
        for j in range(16):
            value = 0.j
            for k in range(16):
                value += inverse[i,k]*covariance[k,j]
            product[i,j] = value
    objective = logdet
    for i in range(16):
        objective += product[i,i].real
    jacobian = np.zeros(3, np.float64)
    for i in range(16):
        for j in range(16):
            value = inverse[i,j]
            for k in range(16):
                value -= product[i,k]*inverse[k,j]
            jacobian[0] += (value*dcenter[j,i]).real
            jacobian[1] += (value*dwidth[j,i]).real
            jacobian[2] += (value*dnoise[j,i]).real
    return objective, jacobian


@dataclass(frozen=True)
class LocalCardinalityConfig:
    minimum_scene_count: int = 5
    minimum_scene_span: float = 15.
    maximum_link_gap: float = 2.2
    minimum_outside_gap: float = 3.
    maximum_local_span: float = 4.5
    minimum_equivalent_snapshots: float = 500.
    minimum_gain: float = .005
    minimum_order_margin: float = .005
    maximum_model_deviance: float = 1.
    minimum_source_gap: float = .5
    maximum_output_sources: int = 20
    maximum_iterations: int = 45


def closest_isolated_group(initial, config):
    gaps = np.diff(initial)
    cuts = np.r_[0, np.flatnonzero(gaps > config.maximum_link_gap)+1, len(initial)]
    candidates = []
    for left, right in zip(cuts[:-1], cuts[1:]):
        size = int(right-left)
        if size < 2 or size > 4:
            continue
        span = float(initial[right-1]-initial[left])
        if span > config.maximum_local_span:
            continue
        if left and initial[left]-initial[left-1] < config.minimum_outside_gap:
            continue
        if right < len(initial) and initial[right]-initial[right-1] < config.minimum_outside_gap:
            continue
        candidates.append((span/(size-1), int(left), int(right)))
    return min(candidates)[1:] if candidates else None


class LocalCardinalityRefiner:
    def __init__(self, config=None):
        self.config = config or LocalCardinalityConfig()

    def refine(self, covariance, initial):
        cfg = self.config
        initial = np.sort(np.asarray(initial, float))
        diagnostic = dict(accepted=False, initial_count=len(initial))
        if len(initial) < cfg.minimum_scene_count or np.ptp(initial) < cfg.minimum_scene_span:
            return initial, dict(diagnostic, reason='scene_geometry')
        group = closest_isolated_group(initial, cfg)
        if group is None:
            return initial, dict(diagnostic, reason='no_isolated_close_group')
        covariance = covariance_from_input(covariance)
        reliability = float(estimate_equivalent_snapshots(covariance))
        if reliability < cfg.minimum_equivalent_snapshots:
            return initial, dict(diagnostic, reason='low_covariance_reliability', equivalent_snapshots=reliability)
        left, right = group
        local = initial[left:right]
        background = np.r_[initial[:left], initial[right:]]
        background_lags = np.exp(-1j*np.pi*np.arange(16)[:,None]*
                                 np.sin(np.deg2rad(background))[None,:]).sum(1)
        base, fraction = fixed_angle_likelihood(covariance, initial)
        saturated = saturated_likelihood(covariance)
        center = float((local[0]+local[-1])/2)
        span = float(local[-1]-local[0])
        bounds = [(max(-69., center-.8), min(69., center+.8)),
                  (.3, min(6., span+2.)), (math.log(1.e-6), math.log(.999))]
        # Continuous route offsets can move an edge group beyond the domain
        # supported by these local fits. Do not pass reversed bounds to SciPy
        # or silently project an out-of-domain angular starting point.
        if any(not np.isfinite(value) or lower > upper or
               not lower <= value <= upper
               for value, (lower, upper) in zip((center, span), bounds[:2])):
            return initial, dict(diagnostic, reason='outside_angle_domain')
        hypotheses = []
        for count in (2,3,4):
            total = len(background)+count
            fit = minimize(_local_objective, [center, span, math.log(max(1-fraction,1.e-5))],
                           args=(covariance,count,background_lags,total), method='L-BFGS-B', jac=True,
                           bounds=bounds, options={'maxiter':cfg.maximum_iterations,'ftol':1.e-10,'gtol':1.e-6})
            c,w,eta = fit.x
            angles = c+w*np.linspace(-.5,.5,count)
            complete = np.sort(np.r_[background,angles])
            legal = (len(complete) <= cfg.maximum_output_sources and
                     np.max(abs(complete)) <= 69.9 and
                     np.diff(complete).min() >= cfg.minimum_source_gap)
            hypotheses.append(dict(count=count,angles=complete.tolist(),local_angles=angles.tolist(),
                                   nll=float(fit.fun),converged=bool(fit.success),legal=bool(legal)))
        hypotheses.sort(key=lambda h:h['nll'])
        best = hypotheses[0]
        diagnostic.update(reason='insufficient_evidence',group=[left,right],local_input=local.tolist(),
                          equivalent_snapshots=reliability,base_nll=base,saturated_nll=saturated,
                          gain=base-best['nll'],order_margin=hypotheses[1]['nll']-best['nll'],
                          model_deviance=best['nll']-saturated,hypotheses=hypotheses)
        if (best['legal'] and best['converged'] and diagnostic['gain']>=cfg.minimum_gain and
            diagnostic['order_margin']>=cfg.minimum_order_margin and
            diagnostic['model_deviance']<=cfg.maximum_model_deviance):
            return np.asarray(best['angles']), dict(diagnostic,accepted=True,reason='local_covariance_evidence')
        return initial, diagnostic
