"""CPU Fisher scoring for covariance-based angle refinement.

The float64 objective, noise initialization, constraints, step scales, and
acceptance criterion match the GPU implementation. Each call operates on
predicted angles and the observed covariance with local workspaces."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from numba import njit


@dataclass(frozen=True)
class PolishOptions:
    steps: int = 3
    trust_degrees: float = .35
    minimum_separation: float = .5
    angle_limit: float = 69.9
    minimum_count: int = 5
    maximum_count: int = 16
    maximum_noise_fraction: float = .5
    minimum_noise_fraction: float = 1.e-5
    noise_grid_size: int = 32
    fisher_ridge: float = .05
    improvement_tolerance: float = 1.e-9
    precision: str = 'float64'
    def __post_init__(self):
        if self.steps not in (1,2,3):raise ValueError('steps must be 1, 2 or 3')
        if self.precision!='float64':raise ValueError('CPU angle refinement requires float64 precision')
        if not(0<self.minimum_count<=self.maximum_count<=16):raise ValueError('count range must lie in [1,16]')
        if not(0<self.minimum_noise_fraction<self.maximum_noise_fraction<1):raise ValueError('invalid noise limits')
        if self.noise_grid_size<4 or self.trust_degrees<=0 or self.minimum_separation<0:raise ValueError('invalid geometric/grid bounds')


@njit(cache=True,nogil=True)
def _steering(theta):
    k=len(theta);a=np.empty((16,k),np.complex128);d=np.empty((16,k),np.complex128)
    for j in range(k):
        angle=theta[j]*(np.pi/180.);s=np.sin(angle);c=np.cos(angle)
        for i in range(16):
            a[i,j]=.25*np.exp(-1j*np.pi*i*s)
            d[i,j]=a[i,j]*(-1j*np.pi*i*c*(np.pi/180.))
    return a,d


@njit(cache=True,nogil=True)
def _inverse_cholesky(c):
    """Small Hermitian Cholesky and triangular solves, without LAPACK dispatch."""
    n=c.shape[0];lower=np.zeros((n,n),np.complex128);logdet=0.
    for i in range(n):
        for j in range(i+1):
            value=c[i,j]
            for h in range(j):value-=lower[i,h]*lower[j,h].conjugate()
            if i==j:
                if not np.isfinite(value.real)or value.real<=0.:
                    return False,logdet,np.eye(n,dtype=np.complex128)
                lower[i,j]=np.sqrt(value.real);logdet+=2.*np.log(lower[i,j].real)
            else:lower[i,j]=value/lower[j,j].real
    inverse_lower=np.zeros((n,n),np.complex128)
    for column in range(n):
        for row in range(column,n):
            value=1.+0j if row==column else 0.+0j
            for h in range(column,row):value-=lower[row,h]*inverse_lower[h,column]
            inverse_lower[row,column]=value/lower[row,row].real
    inverse=np.empty((n,n),np.complex128)
    for i in range(n):
        for j in range(n):
            value=0.+0j
            for h in range(max(i,j),n):value+=inverse_lower[h,i].conjugate()*inverse_lower[h,j]
            inverse[i,j]=value
    return True,logdet,inverse


@njit(cache=True,nogil=True)
def _value(theta,eta,r):
    a,_=_steering(theta);k=len(theta);noise=np.exp(eta);power=(1.-noise)/max(1,k)
    c=np.empty((16,16),np.complex128)
    for i in range(16):
        for j in range(16):
            value=0.+0j
            for h in range(k):value+=a[i,h]*a[j,h].conjugate()
            c[i,j]=power*value+(noise/16. if i==j else 0.)
    ok,logdet,inverse=_inverse_cholesky(c)
    if not ok:return np.inf,inverse,a
    trace=0.
    for i in range(16):
        for j in range(16):trace+=(inverse[i,j]*r[j,i]).real
    return logdet+trace,inverse,a


@njit(cache=True,nogil=True)
def _mm(left,right):
    result=np.zeros((left.shape[0],right.shape[1]),np.complex128)
    for i in range(left.shape[0]):
        for h in range(left.shape[1]):
            for j in range(right.shape[1]):result[i,j]+=left[i,h]*right[h,j]
    return result


@njit(cache=True,nogil=True)
def _adjoint(value):
    result=np.empty((value.shape[1],value.shape[0]),np.complex128)
    for i in range(value.shape[0]):
        for j in range(value.shape[1]):result[j,i]=value[i,j].conjugate()
    return result


@njit(cache=True,nogil=True)
def _value_gradient_fisher(theta,eta,r):
    """Equivalent rank-one derivative contractions of tr(V dCp V dCq)."""
    value,v,a=_value(theta,eta,r);_,d=_steering(theta)
    k=len(theta);noise=np.exp(eta);p=(1.-noise)/k
    va=_mm(v,a);vd=_mm(v,d);rv=_mm(r,v);q=v-_mm(v,rv);qa=_mm(q,a)
    ah=_adjoint(a);dh=_adjoint(d)
    m=_mm(ah,va);u=_mm(ah,vd);l=_mm(dh,vd)
    gradient=np.empty(k+1,np.float64);fisher=np.empty((k+1,k+1),np.float64)
    traceq=0.;aqa=0.;vnorm=0.;vanorm=0.;trace_mm=0.
    for i in range(16):
        traceq+=q[i,i].real
        for j in range(16):vnorm+=(v[i,j]*v[i,j].conjugate()).real
        for j in range(k):
            aqa+=(a[i,j].conjugate()*qa[i,j]).real
            vanorm+=(va[i,j]*va[i,j].conjugate()).real
    for i in range(k):
        g=0.+0j;mixed=0.+0j;vv=0.+0j
        for h in range(16):
            g+=d[h,i].conjugate()*qa[h,i]
            vv+=va[h,i].conjugate()*vd[h,i]
        gradient[i]=2.*p*g.real
        for j in range(k):
            fisher[i,j]=2.*p*p*(u[i,j]*u[j,i]+m[i,j]*l[j,i]).real
            mixed+=m[i,j]*u[j,i]
            trace_mm+=(m[i,j]*m[j,i]).real
        fisher[i,k]=2.*p*noise*(vv/16.-mixed/k).real
        fisher[k,i]=fisher[i,k]
    gradient[k]=noise*(traceq/16.-aqa/k)
    fisher[k,k]=noise*noise*(vnorm/(16.*16.)-2.*vanorm/(16.*k)+trace_mm/(k*k))
    for i in range(k+1):
        for j in range(i):
            average=.5*(fisher[i,j]+fisher[j,i]);fisher[i,j]=average;fisher[j,i]=average
    return value,gradient,fisher


@njit(cache=True,nogil=True)
def _real_solve_cholesky(fisher,gradient):
    n=len(gradient);lower=np.zeros((n,n),np.float64)
    for i in range(n):
        if not np.isfinite(gradient[i]):return False,np.zeros(n)
        for j in range(i+1):
            value=fisher[i,j]
            for h in range(j):value-=lower[i,h]*lower[j,h]
            if not np.isfinite(value):return False,np.zeros(n)
            if i==j:
                if value<=0.:return False,np.zeros(n)
                lower[i,j]=np.sqrt(value)
            else:lower[i,j]=value/lower[j,j]
    forward=np.empty(n);step=np.empty(n)
    for i in range(n):
        value=-gradient[i]
        for h in range(i):value-=lower[i,h]*forward[h]
        forward[i]=value/lower[i,i]
    for i in range(n-1,-1,-1):
        value=forward[i]
        for h in range(i+1,n):value-=lower[h,i]*step[h]
        step[i]=value/lower[i,i]
    return True,step


def _initial_noise(theta,r,grid):
    a,_=_steering(theta);eigen,vectors=np.linalg.eigh(a@np.ascontiguousarray(a.conjugate().T))
    diagonal=(vectors.conjugate().T@r@vectors).diagonal().real
    spectra=grid[:,None]/16.+(1.-grid[:,None])/len(theta)*np.maximum(eigen,0.)[None,:]
    objective=(np.log(spectra)+diagonal[None,:]/spectra).sum(-1)
    return np.log(grid[objective.argmin()])


@njit(cache=True,nogil=True)
def _polish_valid_scene(r,theta0,grid,initial_eta,steps,trust,minsep,angle_limit,max_noise,ridge,tolerance,eligible):
    k=len(theta0);lower=np.maximum(theta0-trust,-angle_limit);upper=np.minimum(theta0+trust,angle_limit)
    for i in range(k-1):
        midpoint=.5*(theta0[i]+theta0[i+1]);lower[i+1]=max(lower[i+1],midpoint+minsep*.5);upper[i]=min(upper[i],midpoint-minsep*.5)
    geometry=True
    for i in range(k):geometry=geometry and lower[i]<=theta0[i]<=upper[i]
    eta=initial_eta;initial_noise=np.exp(eta);high_noise=initial_noise>max_noise
    eligible=eligible and geometry and not high_noise
    base,_,_=_value(theta0,eta,r);current=base;theta=theta0.copy();accepted=0;failed=False
    if eligible:
        for iteration in range(steps):
            _,gradient,fisher=_value_gradient_fisher(theta,eta,r)
            for i in range(k+1):fisher[i,i]+=ridge*max(fisher[i,i],1.e-8)
            ok,step=_real_solve_cholesky(fisher,gradient)
            if not ok:failed=True;continue
            step[k]=min(max(step[k],-1.),1.);best_value=np.inf;best_theta=theta.copy();best_eta=eta
            for scale in (1.,.5,.25):
                proposal=np.minimum(np.maximum(theta+scale*step[:k],lower),upper)
                proposed_eta=min(max(eta+scale*step[k],np.log(grid[0])),np.log(.999))
                proposed_value,_,_=_value(proposal,proposed_eta,r)
                if np.isfinite(proposed_value)and proposed_value<best_value:
                    best_value=proposed_value;best_theta=proposal;best_eta=proposed_eta
            if best_value<current-tolerance:
                theta=best_theta;eta=best_eta;current=best_value;accepted+=1
    diagnostic=np.array([base,current,initial_noise,np.exp(eta),1. if eligible else 0.,float(accepted),1. if failed else 0.,1. if geometry else 0.,1. if high_noise else 0.])
    return theta,diagnostic


class CPUAnglePolisher:
    """CPU angle refinement for batches with variable predicted source counts."""
    def __init__(self,device='cpu',options=None):
        if str(device)!='cpu':raise ValueError('CPUAnglePolisher requires device=cpu')
        self.options=options or PolishOptions()
        if self.options.precision!='float64':raise ValueError('CPU angle refinement requires float64 precision')
        # The only retained numeric array is the 32-value initial noise grid.
        self._grid=np.logspace(math.log10(self.options.minimum_noise_fraction),math.log10(.999),self.options.noise_grid_size,dtype=np.float64)
    def __call__(self,covariance,predictions,*,return_diagnostics=False):
        return self.polish(covariance,predictions,return_diagnostics=return_diagnostics)
    def polish(self,covariance,predictions,*,return_diagnostics=False):
        original=[np.asarray(a,dtype=np.float64).reshape(-1).copy()for a in predictions]
        if not original:return ([],{'scenes':[]})if return_diagnostics else[]
        if hasattr(covariance,'detach'):covariance=covariance.detach().cpu().numpy()
        raw=np.asarray(covariance)
        if raw.ndim==4 and raw.shape[1:]==(2,16,16):raw=raw[:,0].astype(np.float64)+1j*raw[:,1].astype(np.float64)
        elif raw.ndim==3 and raw.shape[1:]==(16,16):raw=raw.astype(np.complex128)
        else:raise ValueError('covariance must be [B,2,16,16] or [B,16,16]')
        if len(raw)!=len(original):raise ValueError('covariance and prediction batch sizes differ')
        opt=self.options;output=[];scenes=[];eye=np.eye(16,dtype=np.complex128)/16.
        for covariance,old in zip(raw,original):
            n=len(old);host_valid=opt.minimum_count<=n<=opt.maximum_count and np.isfinite(old).all()
            finite=np.isfinite(covariance).all();r=(covariance+covariance.conjugate().T)*.5 if finite else eye.copy()
            trace=np.trace(r).real;valid_trace=trace>np.finfo(np.float64).eps;r=r/max(trace,np.finfo(np.float64).eps)if valid_trace else eye.copy()
            psd=bool(np.linalg.eigvalsh(r)[0]>=-1.e-8)
            if host_valid:
                order=np.argsort(old,kind='stable');theta0=np.ascontiguousarray(old[order])
                theta,d=_polish_valid_scene(np.ascontiguousarray(r),theta0,self._grid,float(_initial_noise(theta0,r,self._grid)),opt.steps,opt.trust_degrees,opt.minimum_separation,opt.angle_limit,opt.maximum_noise_fraction,opt.fisher_ridge,opt.improvement_tolerance,bool(finite and valid_trace and psd))
            else:
                order=np.arange(n);theta=old.copy();noise=.999
                base=16.*np.log(noise/16.)+16.*np.trace(r).real/noise
                d=np.array([base,base,noise,noise,0.,0.,0.,1.,float(noise>opt.maximum_noise_fraction)])
            updated=old.copy();changed=bool(d[4])and int(d[5])>0
            if changed:updated[order]=theta
            output.append(updated)
            if return_diagnostics:
                reasons=[]
                if not host_valid:reasons.append('unsupported_count_or_nonfinite_angles')
                if not finite:reasons.append('nonfinite_covariance')
                if not psd:reasons.append('indefinite_covariance')
                if not d[7]:reasons.append('invalid_initial_geometry')
                if d[8]:reasons.append('high_observed_noise')
                if not valid_trace:reasons.append('nonpositive_covariance_trace')
                if d[6]:reasons.append('fisher_factorization_failed')
                scenes.append(dict(predicted_count=n,eligible=bool(d[4]),accepted_steps=int(d[5]),changed=changed,nll_before=float(d[0]),nll_after=float(d[1]),noise_before=float(d[2]),noise_after=float(d[3]),maximum_movement=float(np.max(np.abs(updated-old),initial=0))if np.isfinite(old).all()else 0.,fallback_reasons=reasons))
        return (output,dict(scenes=scenes,parameters_added=0,steps=opt.steps,precision='float64',backend='numba_cpu'))if return_diagnostics else output
