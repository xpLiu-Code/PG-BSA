"""GPU Fisher scoring for covariance-based angle refinement.

The optimizer preserves the predicted source count. Covariance and predicted
angles are its inputs; workspaces are allocated within each call."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
import torch

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
        if self.steps not in (1,2,3): raise ValueError('steps must be 1, 2 or 3')
        if self.precision not in ('float32','float64'): raise ValueError('unsupported precision')
        if not (0<self.minimum_count<=self.maximum_count<=16): raise ValueError('count range must lie in [1,16]')
        if not (0<self.minimum_noise_fraction<self.maximum_noise_fraction<1): raise ValueError('invalid noise limits')
        if self.noise_grid_size<4 or self.trust_degrees<=0 or self.minimum_separation<0: raise ValueError('invalid geometric/grid bounds')

class GPUAnglePolisher:
    """Refine batches whose scenes have different predicted source counts.

    Scenes outside the count, covariance, geometry, or noise limits retain
    their input angle estimates. The same implementation can run on CPU.
    """
    def __init__(self,device='cuda:0',options=None):
        self.device=torch.device(device); self.options=options or PolishOptions()
        self.real_dtype=torch.float64 if self.options.precision=='float64' else torch.float32
        self.complex_dtype=torch.complex128 if self.options.precision=='float64' else torch.complex64
    def __call__(self,covariance,predictions,*,return_diagnostics=False):
        return self.polish(covariance,predictions,return_diagnostics=return_diagnostics)
    @torch.inference_mode()
    def polish(self,covariance,predictions,*,return_diagnostics=False):
        original=[np.asarray(a,dtype=np.float64).reshape(-1).copy() for a in predictions]
        batch=len(original); opt=self.options
        if not batch: return ([],{'scenes':[]}) if return_diagnostics else []
        raw=torch.as_tensor(covariance,device=self.device)
        if raw.ndim==4 and raw.shape[1:]==(2,16,16):
            raw=torch.complex(raw[:,0].to(self.real_dtype),raw[:,1].to(self.real_dtype))
        elif raw.ndim==3 and raw.shape[1:]==(16,16): raw=raw.to(self.complex_dtype)
        else: raise ValueError('covariance must be [B,2,16,16] or [B,16,16]')
        if raw.shape[0]!=batch: raise ValueError('covariance and prediction batch sizes differ')
        counts_np=np.asarray([len(a) for a in original],dtype=np.int64)
        width=max(1,min(opt.maximum_count,int(counts_np.max(initial=0))))
        angles_np=np.zeros((batch,width));host_valid=np.ones(batch,dtype=bool)
        for b,a in enumerate(original):
            host_valid[b]=opt.minimum_count<=len(a)<=opt.maximum_count and np.isfinite(a).all()
            if host_valid[b]: angles_np[b,:len(a)]=np.sort(a)
        theta0=torch.as_tensor(angles_np,device=self.device,dtype=self.real_dtype)
        count=torch.as_tensor(counts_np,device=self.device)
        mask=(torch.arange(width,device=self.device)[None,:]<count[:,None])&torch.as_tensor(host_valid,device=self.device)[:,None]
        eye=torch.eye(16,dtype=self.complex_dtype,device=self.device)
        finite=torch.isfinite(raw).all(dim=(-1,-2))
        r=torch.where(finite[:,None,None],raw,eye[None]/16);r=(r+r.mH)*.5
        trace=r.diagonal(dim1=-2,dim2=-1).real.sum(-1)
        valid_trace=trace>torch.finfo(self.real_dtype).eps
        r=r/trace.clamp_min(torch.finfo(self.real_dtype).eps)[:,None,None]
        r=torch.where(valid_trace[:,None,None],r,eye[None]/16)
        eig_r=torch.linalg.eigvalsh(r)
        psd=eig_r[:,0]>=(-1.e-8 if self.real_dtype==torch.float64 else -2.e-6)
        lower=torch.maximum(theta0-opt.trust_degrees,theta0.new_tensor(-opt.angle_limit))
        upper=torch.minimum(theta0+opt.trust_degrees,theta0.new_tensor(opt.angle_limit))
        if width>1:
            mid=(theta0[:,:-1]+theta0[:,1:])*.5;neighbors=mask[:,:-1]&mask[:,1:]
            lower[:,1:]=torch.where(neighbors,torch.maximum(lower[:,1:],mid+opt.minimum_separation*.5),lower[:,1:])
            upper[:,:-1]=torch.where(neighbors,torch.minimum(upper[:,:-1],mid-opt.minimum_separation*.5),upper[:,:-1])
        geometry=(((lower<=theta0)&(theta0<=upper))|~mask).all(-1)
        eligible=torch.as_tensor(host_valid,device=self.device)&finite&valid_trace&psd&geometry
        safe_count=count.clamp(min=1,max=16).to(self.real_dtype)
        a0,_=self._steering(theta0,mask)
        eigen,vectors=torch.linalg.eigh(a0@a0.mH)
        diagonal_r=(vectors.mH@r@vectors).diagonal(dim1=-2,dim2=-1).real
        grid=torch.logspace(math.log10(opt.minimum_noise_fraction),math.log10(.999),opt.noise_grid_size,device=self.device,dtype=self.real_dtype)
        spectra=grid[None,:,None]/16+(1-grid[None,:,None])/safe_count[:,None,None]*eigen.clamp_min(0)[:,None,:]
        objective=(spectra.log()+diagonal_r[:,None,:]/spectra).sum(-1)
        eta=grid[objective.argmin(-1)].log();high_noise=eta.exp()>opt.maximum_noise_fraction
        eligible&=~high_noise;theta=theta0.clone()
        base,_,_=self._value(theta,eta,mask,safe_count,r,eye);current=base.clone()
        accepted_steps=torch.zeros(batch,dtype=torch.int64,device=self.device)
        failed_system=torch.zeros(batch,dtype=torch.bool,device=self.device)
        scales=theta.new_tensor([1.,.5,.25]);initial_eta=eta.clone()
        for _ in range(opt.steps):
            _,inverse,a=self._value(theta,eta,mask,safe_count,r,eye)
            noise=eta.exp();power=(1-noise)/safe_count;_,derivative=self._steering(theta,mask)
            aa,dd=a.transpose(1,2),derivative.transpose(1,2)
            dtheta=power[:,None,None,None]*(dd[...,:,None]*aa.conj()[...,None,:]+aa[...,:,None]*dd.conj()[...,None,:])
            deta=noise[:,None,None]*(eye[None]/16-(a@a.mH)/safe_count[:,None,None])
            dc=torch.cat((dtheta,deta[:,None]),dim=1);q=inverse-inverse@r@inverse
            gradient=torch.einsum('bij,bpji->bp',q,dc).real
            transformed=inverse[:,None]@dc
            fisher=torch.einsum('bpij,bqji->bpq',transformed,transformed).real
            fisher=(fisher+fisher.transpose(-1,-2))*.5
            active=torch.cat((mask,torch.ones((batch,1),dtype=torch.bool,device=self.device)),-1)
            diagonal=fisher.diagonal(dim1=-2,dim2=-1).clamp_min(1.e-8)
            fisher=fisher+torch.diag_embed(torch.where(active,opt.fisher_ridge*diagonal,torch.ones_like(diagonal)))
            chol,info=torch.linalg.cholesky_ex(fisher)
            system_ok=(info==0)&torch.isfinite(gradient).all(-1)&torch.isfinite(fisher).all(dim=(-1,-2))
            failed_system|=eligible&~system_ok
            chol=torch.where(system_ok[:,None,None],chol,torch.eye(width+1,device=self.device,dtype=self.real_dtype)[None])
            step=-torch.cholesky_solve(gradient[:,:,None],chol).squeeze(-1)
            step=torch.where((eligible&system_ok)[:,None],step,torch.zeros_like(step));step[:,-1]=step[:,-1].clamp(-1.,1.)
            proposal=(theta[:,None,:]+scales[None,:,None]*step[:,None,:-1]).clamp(lower[:,None,:],upper[:,None,:])
            eta_proposal=(eta[:,None]+scales[None,:]*step[:,None,-1]).clamp(math.log(opt.minimum_noise_fraction),math.log(.999))
            values,_,_=self._value(proposal.flatten(0,1),eta_proposal.flatten(),mask[:,None].expand(-1,3,-1).flatten(0,1),safe_count[:,None].expand(-1,3).flatten(),r[:,None].expand(-1,3,-1,-1).flatten(0,1),eye)
            values=values.reshape(batch,3);values=torch.where(torch.isfinite(values),values,torch.full_like(values,float('inf')))
            best_value,which=values.min(-1);accept=eligible&system_ok&(best_value<current-opt.improvement_tolerance)
            rows=torch.arange(batch,device=self.device)
            theta=torch.where(accept[:,None],proposal[rows,which],theta)
            eta=torch.where(accept,eta_proposal[rows,which],eta);current=torch.where(accept,best_value,current)
            accepted_steps+=accept.to(torch.int64)
        diagnostic=torch.stack((base,current,initial_eta.exp(),eta.exp(),eligible.to(self.real_dtype),accepted_steps.to(self.real_dtype),failed_system.to(self.real_dtype),finite.to(self.real_dtype),psd.to(self.real_dtype),geometry.to(self.real_dtype),high_noise.to(self.real_dtype),valid_trace.to(self.real_dtype)),-1)
        packed=torch.cat((theta,diagnostic),-1).cpu().numpy();output=[];scenes=[]
        for b,old in enumerate(original):
            d=packed[b,width:];changed=bool(d[4]) and int(d[5])>0;updated=old.copy()
            if changed: updated[np.argsort(old,kind='stable')]=packed[b,:len(old)]
            output.append(updated)
            if return_diagnostics:
                reasons=[]
                if not host_valid[b]: reasons.append('unsupported_count_or_nonfinite_angles')
                if not d[7]: reasons.append('nonfinite_covariance')
                if not d[8]: reasons.append('indefinite_covariance')
                if not d[9]: reasons.append('invalid_initial_geometry')
                if d[10]: reasons.append('high_observed_noise')
                if not d[11]: reasons.append('nonpositive_covariance_trace')
                if d[6]: reasons.append('fisher_factorization_failed')
                scenes.append(dict(predicted_count=int(counts_np[b]),eligible=bool(d[4]),accepted_steps=int(d[5]),changed=changed,nll_before=float(d[0]),nll_after=float(d[1]),noise_before=float(d[2]),noise_after=float(d[3]),maximum_movement=float(np.max(np.abs(updated-old),initial=0)) if np.isfinite(old).all() else 0.,fallback_reasons=reasons))
        return (output,{'scenes':scenes,'parameters_added':0,'steps':opt.steps,'precision':opt.precision}) if return_diagnostics else output
    def _steering(self,theta,mask):
        index=torch.arange(16,dtype=self.real_dtype,device=self.device)[None,:,None];radians=theta[:,None]*(math.pi/180)
        a=torch.exp(-1j*math.pi*index*radians.sin())*.25*mask[:,None]
        derivative=a*(-1j*math.pi*index*radians.cos()*(math.pi/180));return a,derivative
    def _value(self,theta,eta,mask,count,r,eye):
        a,_=self._steering(theta,mask);noise=eta.exp()
        c=noise[:,None,None]/16*eye[None]+((1-noise)/count)[:,None,None]*(a@a.mH)
        chol,info=torch.linalg.cholesky_ex(c);ok=(info==0)&torch.isfinite(chol).all(dim=(-1,-2))
        chol=torch.where(ok[:,None,None],chol,eye[None]);inverse=torch.cholesky_inverse(chol)
        value=2*chol.diagonal(dim1=-2,dim2=-1).real.log().sum(-1)+torch.einsum('bij,bji->b',inverse,r).real
        return torch.where(ok,value,torch.full_like(value,float('inf'))),inverse,a
