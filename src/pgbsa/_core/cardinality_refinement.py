"""Shared covariance likelihoods and compact-cluster configuration."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize_scalar

def covariance_from_input(x):
    x = np.asarray(x)
    if x.shape == (2, 16, 16):
        return x[0].astype(np.complex128) + 1j*x[1]
    if x.shape == (16, 16):
        return x.astype(np.complex128)
    raise ValueError('Expected normalized 16-sensor covariance or two channels')


def steering(angles):
    return np.exp(-1j*np.pi*np.arange(16)[:, None]*np.sin(np.deg2rad(angles))[None, :])


def fixed_angle_likelihood(covariance, angles):
    """Return the fitted covariance negative log-likelihood and signal fraction.

    Angles are in degrees and covariance has unit trace. For k trial angles,
    C = f A A^H/(16 k) + (1-f) I/16, where f is the total signal-power fraction.
    The empty angle set uses the noise-only covariance I/16 and returns f=0.
    """
    angles = np.asarray(angles, float)
    if not len(angles):
        return float(-16*np.log(16)+16*np.trace(covariance).real), 0.
    a = steering(angles)
    values, vectors = np.linalg.eigh(a@a.conj().T/len(angles))
    diagonal = np.diag(vectors.conj().T@covariance@vectors).real
    from .bounded_physics import fit_fixed_fraction
    if (values.shape == (16,) and diagonal.shape == (16,)
            and np.isfinite(values).all() and np.isfinite(diagonal).all()
            and np.max(abs(values-1.)) > 256.*np.finfo(np.float64).eps*max(1., np.max(abs(values)))):
        fraction, objective, _, status = fit_fixed_fraction(values, diagonal)
        if status == 0:
            return float(objective), float(fraction)
    # Use SciPy when the compiled solver cannot handle the supplied input.
    def objective(f):
        d = (f*values+1-f)/16
        return float(np.log(d).sum()+(diagonal/d).sum())
    fit = minimize_scalar(objective, bounds=(.001, .999999), method='bounded',
                          options={'xatol': 1.e-7})
    return float(fit.fun), float(fit.x)


def saturated_likelihood(covariance):
    covariance=covariance_from_input(covariance)
    eigenvalues=np.linalg.eigvalsh(covariance)
    return float(np.log(np.maximum(eigenvalues,1.e-12)).sum()+16)


@dataclass(frozen=True)
class CardinalityConfig:
    """Thresholds and bounds for compact-cluster covariance fitting."""
    minimum_input_count: int = 2
    maximum_input_count: int = 10
    maximum_span_deg: float = 40.
    minimum_base_deviance: float = .15
    minimum_likelihood_gain: float = .02
    maximum_model_deviance: float = 1.
    likelihood_temperature: float = 1000.
    maximum_candidate_count: int = 20
    maximum_optimizer_iterations: int = 60
