"""Observation-derived covariance reliability proxy."""
from __future__ import annotations
import numpy as np

def estimate_equivalent_snapshots(covariance):
    """Estimate an equivalent snapshot count from Toeplitz covariance residuals.

    Accepts complex [N,N]/[B,N,N] or real/imag [2,N,N]/[B,2,N,N].
    Returns a scalar for one covariance matrix or an array for a batch. The
    estimate assumes an ideal ULA, uncorrelated sources, and stationary white
    noise. Array or signal-model mismatch can reduce this reliability proxy
    even when the actual number of snapshots is large.
    """
    x=np.asarray(covariance)
    if np.iscomplexobj(x):
        single=x.ndim==2
        c=(x[None] if single else x).astype(np.complex128)
    else:
        single=x.ndim==3
        if single:x=x[None]
        c=x[:,0].astype(np.complex128)+1j*x[:,1].astype(np.complex128)
    n=c.shape[-1];projection=np.zeros_like(c,dtype=np.complex128)
    for lag in range(n):
        index=np.arange(n-lag);mean=c[:,index+lag,index].mean(1)
        projection[:,index+lag,index]=mean[:,None]
        if lag:projection[:,index,index+lag]=mean.conj()[:,None]
    residual=(abs(c-projection)**2).sum((1,2))
    trace=c.diagonal(axis1=1,axis2=2).real.sum(1)
    variance=np.zeros(len(c))
    for lag in range(n):
        length=n-lag
        variance+=(1 if lag==0 else 2)*(abs(projection[:,:length,:length])**2).sum((1,2))/length
    estimate=np.maximum(trace**2-variance,1.e-12)/np.maximum(residual,1.e-12)
    return float(estimate[0]) if single else estimate
