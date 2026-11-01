"""Specialized dispatch to the SciPy 1.13.1 L-BFGS-B kernel.

The direct path accepts float64 three-variable problems with finite bounds
and analytic gradients. Other inputs use scipy.optimize.minimize. It retains
the kernel stopping conditions, evaluation limits, and objective cache.

Adapted from scipy.optimize._lbfgsb_py and _differentiable_functions.
SciPy BSD-3-Clause license:
Copyright (c) 2001-2002 Enthought, Inc. 2003-2024, SciPy Developers.
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

Original Python-wrapper license:
Copyright (c) 2004 David M. Cooke <cookedm@physics.mcmaster.ca>
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.
Modifications by Travis Oliphant and Enthought, Inc. for inclusion in SciPy.
"""
from __future__ import annotations
import numpy as np
import scipy
from scipy.optimize import OptimizeResult, minimize as scipy_minimize
from scipy.optimize import _lbfgsb

_OPTIONS=frozenset(('maxcor','ftol','gtol','maxfun','maxiter','maxls'))
_SUPPORTED_ABI=(scipy.__version__=='1.13.1' and hasattr(_lbfgsb,'types')
                and hasattr(_lbfgsb.types,'intvar'))
_EPS=np.finfo(float).eps


def direct_minimize(fun,x0,args=(),method=None,jac=None,hess=None,hessp=None,
                    bounds=None,constraints=(),tol=None,callback=None,options=None):
    """Optimize three-variable objectives with the SciPy L-BFGS-B kernel.

    Unsupported arguments are delegated to scipy.optimize.minimize. The result
    contains objective, solution, termination status, iteration counts, and
    gradient evaluations. No inverse-Hessian approximation is returned."""
    def fallback():
        return scipy_minimize(fun,x0,args=args,method=method,jac=jac,hess=hess,
          hessp=hessp,bounds=bounds,constraints=constraints,tol=tol,
          callback=callback,options=options)
    if (not _SUPPORTED_ABI or method not in(None,'L-BFGS-B','l-bfgs-b')
        or jac is not True or callback is not None or tol is not None
        or hess is not None or hessp is not None
        or not isinstance(constraints,(list,tuple)) or len(constraints)
        or not isinstance(bounds,(list,tuple,np.ndarray))
        or (options is not None and (not isinstance(options,dict) or not set(options)<=_OPTIONS))):
        return fallback()
    try:
        initial=np.atleast_1d(np.asarray(x0))
        if initial.dtype.kind in np.typecodes['AllInteger']:initial=initial.astype(float)
        limits=np.asarray(bounds,dtype=np.float64)
        if (initial.shape!=(3,) or initial.dtype!=np.dtype('float64')
            or limits.shape!=(3,2) or not np.isfinite(initial).all()
            or not np.isfinite(limits).all() or not np.all(limits[:,0]<limits[:,1])):
            return fallback()
        opts={} if options is None else options
        m=opts.get('maxcor',10);maxiter=opts.get('maxiter',15000)
        maxfun=opts.get('maxfun',15000);maxls=opts.get('maxls',20)
        ftol=opts.get('ftol',2.2204460492503131e-9);gtol=opts.get('gtol',1e-5)
        if (not all(isinstance(v,(int,np.integer))for v in(m,maxiter,maxfun,maxls))
            or m<=0 or maxls<=0 or not np.isfinite(ftol)or not np.isfinite(gtol)
            or ftol<0 or gtol<0):return fallback()
    except (TypeError,ValueError):return fallback()
    if not isinstance(args,tuple):args=(args,)
    # ScalarFunction evaluates this clipped point BEFORE the first setulb call.
    x=np.clip(initial,limits[:,0],limits[:,1]).astype(np.float64,copy=True)
    cached_x=x.copy();nfev=1
    def evaluate(point):
        value,gradient=fun(point.copy(),*args)
        if not np.isscalar(value):
            try:value=np.asarray(value).item()
            except (TypeError,ValueError)as error:
                raise ValueError('The user-provided objective function must return a scalar value.')from error
        # MemoizeJac invalidates equality at NaN coordinates. Its derivative
        # therefore invokes the objective again; keep that edge case identical.
        if np.isnan(point).any():_,gradient=fun(point.copy(),*args)
        return value,np.atleast_1d(gradient)
    cached_f,cached_g=evaluate(x)
    integer=_lbfgsb.types.intvar.dtype
    nbd=np.full(3,2,dtype=integer)
    low=limits[:,0].copy();high=limits[:,1].copy()
    f=np.array(0.,dtype=np.float64);g=np.zeros(3,dtype=np.float64)
    wa=np.zeros(6*m+15+11*m*m+8*m,dtype=np.float64)
    iwa=np.zeros(9,dtype=integer);task=np.zeros(1,dtype='S60')
    csave=np.zeros(1,dtype='S60');lsave=np.zeros(4,dtype=integer)
    isave=np.zeros(44,dtype=integer);dsave=np.zeros(29,dtype=np.float64)
    task[:]='START';nit=0;factr=ftol/_EPS
    while True:
        # Match SciPy's protective conversion even when the objective gives f32 g.
        g=g.astype(np.float64)
        _lbfgsb.setulb(m,x,low,high,nbd,f,g,factr,gtol,wa,iwa,task,-1,
                      csave,lsave,isave,dsave,maxls)
        event=task.tobytes()
        if event.startswith(b'FG'):
            if not np.array_equal(x,cached_x):
                cached_x=x.copy();cached_f,cached_g=evaluate(x);nfev+=1
            f,g=cached_f,cached_g
        elif event.startswith(b'NEW_X'):
            nit+=1
            if nit>=maxiter:task[:]='STOP: TOTAL NO. of ITERATIONS REACHED LIMIT'
            elif nfev>maxfun:task[:]='STOP: TOTAL NO. of f AND g EVALUATIONS EXCEEDS LIMIT'
        else:break
    message=task.tobytes().strip(b'\x00').strip().decode()
    status=0 if message.startswith('CONV')else(1 if nfev>maxfun or nit>=maxiter else 2)
    return OptimizeResult(x=x,fun=f,success=status==0,nit=nit,status=status,
                          message=message,jac=g,nfev=nfev,njev=nfev)
