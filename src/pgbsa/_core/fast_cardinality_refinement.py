"""Compact-cluster hypothesis fitting with Numba covariance kernels.

Complex Cholesky decomposition, triangular solves, and matrix products use
explicit loops to avoid loading a separate BLAS or OpenMP runtime."""
from __future__ import annotations

import math
import numpy as np
from numba import njit
from .direct_lbfgsb import direct_minimize as minimize

from .cardinality_refinement import (
    CardinalityConfig, covariance_from_input, fixed_angle_likelihood,
    saturated_likelihood, steering,
)


@njit(cache=True, fastmath=False, nogil=True)
def _objective(parameters, covariance, count):
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
        gs[lag] = g/count
        cs[lag] = signal*c/(16*count)
        ws[lag] = signal*w/(16*count)
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


def regular_objective(parameters, covariance, count, gradient=True):
    value, jacobian = _objective(parameters, covariance, count)
    return (value, jacobian) if gradient else value


def compact_hypotheses(covariance, initial, *, counts=range(2,21), starts=1, maxiter=80,
                       _base_fit=None):
    covariance = covariance_from_input(covariance)
    initial = np.sort(np.asarray(initial,float))
    if len(initial) < 2:
        return []
    center = float((initial[0]+initial[-1])/2)
    span = float(initial[-1]-initial[0])
    mean_gap = span/(len(initial)-1)
    nll, fraction = fixed_angle_likelihood(covariance, initial) if _base_fit is None else _base_fit
    hypotheses = []
    for count in counts:
        guesses = [span+mean_gap*.5]
        if starts > 1:
            guesses.append(span)
        best = None
        for width in guesses:
            x0 = np.array([center,max(width,.1),np.log(max(1-fraction,1.e-5))])
            result = minimize(regular_objective,x0,args=(covariance,count),jac=True,method='L-BFGS-B',
                bounds=[(max(-68,center-6),min(68,center+6)),
                        (max(.2,span*.5),min(120,span+max(8,span*.5))),
                        (np.log(1.e-6),np.log(.999))],
                options={'maxiter':maxiter,'ftol':1.e-11,'gtol':1.e-6,'maxls':25})
            if best is None or result.fun < best.fun:
                best = result
        c,w,eta = best.x
        angles = c+w*np.linspace(-.5,.5,count)
        hypotheses.append(dict(count=int(count),angles=angles.tolist(),nll=float(best.fun),
            gain=float(nll-best.fun),center=float(c),width=float(w),
            signal_fraction=float(1-np.exp(eta)),converged=bool(best.success),iterations=int(best.nit)))
    return sorted(hypotheses,key=lambda h:h['nll'])


class CompactCardinalityRefiner:
    """Fit compact-cluster hypotheses using compiled likelihood evaluation."""
    def __init__(self, config=None):
        self.config = config or CardinalityConfig()

    def refine(self, covariance, initial):
        cfg = self.config
        initial = np.sort(np.asarray(initial,float))
        diagnostic = dict(accepted=False,initial_count=len(initial))
        if not cfg.minimum_input_count <= len(initial) <= cfg.maximum_input_count:
            return initial,dict(diagnostic,reason='count_outside_trigger')
        if np.ptp(initial) >= cfg.maximum_span_deg:
            return initial,dict(diagnostic,reason='not_compact')
        covariance = covariance_from_input(covariance)
        base,fraction = fixed_angle_likelihood(covariance,initial)
        saturated = saturated_likelihood(covariance)
        diagnostic['base_deviance'] = base-saturated
        if base-saturated < cfg.minimum_base_deviance:
            return initial,dict(diagnostic,reason='existing_fit_adequate')
        hypotheses = compact_hypotheses(covariance,initial,
            counts=range(2,cfg.maximum_candidate_count+1),starts=1,
            maxiter=cfg.maximum_optimizer_iterations,_base_fit=(base,fraction))
        nll = np.array([h['nll'] for h in hypotheses])
        counts = np.array([h['count'] for h in hypotheses])
        weights = np.exp(-cfg.likelihood_temperature*(nll-nll.min()))
        weights /= weights.sum()
        if hypotheses[0]['count'] <= 4:
            index = 0
        else:
            target = int(round(float(weights@counts)))
            index = int(np.argmin(np.abs(counts-target)))
        selected = hypotheses[index]
        order = np.argsort(counts)
        cdf = np.cumsum(weights[order])
        interval = [int(counts[order[min(np.searchsorted(cdf,q),len(order)-1)]]) for q in (.05,.95)]
        diagnostic.update(candidate_count=selected['count'],gain=selected['gain'],
            model_deviance=selected['nll']-saturated,likelihood_weight=float(weights[index]),
            count_interval90=interval,low_count_confidence=bool(weights[index]<.8),
            candidate_angles=selected['angles'])
        if selected['count'] == len(initial):
            return initial,dict(diagnostic,reason='same_count')
        if selected['gain'] < cfg.minimum_likelihood_gain:
            return initial,dict(diagnostic,reason='insufficient_improvement')
        if selected['nll']-saturated > cfg.maximum_model_deviance:
            return initial,dict(diagnostic,reason='model_mismatch_or_noisy_covariance')
        if np.max(np.abs(selected['angles'])) > 70:
            return initial,dict(diagnostic,reason='outside_angle_domain')
        return np.array(selected['angles'],dtype=float),dict(diagnostic,accepted=True,reason='compact_covariance_evidence')
