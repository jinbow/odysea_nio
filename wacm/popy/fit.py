'''
Created on Jan 28, 2013

@author: Jinbo Wang <jinbow@gmail.com>
@organization: Scripps Institute of Oceanography
'''

import numpy as np
import sys
import pylab as plt
sys.path.append('/u/jwang23/projects/software/popy/')
import popy

def fit2Dsurf(x,y,p,kind='linear'):
    """
      given y0=f(t0), find the best fit
      p = a + bx + cy + dx**2 + ey**2 + fxy
      and return a,b,c,d,e,f
    """
    from scipy.optimize import leastsq
    import numpy as np

    def err(c,x0,y0,p):
        if kind=='linear':
            a,b,c=c
            return p - (a + b*x0 + c*y0 )
        if kind=='quadratic':
            a,b,c,d,e,f=c
            return p - (a + b*x0 + c*y0 + d*x0**2 + e*y0**2 + f*x0*y0)

    def surface(c,x0,y0):
        if kind=='linear':
            a,b,c=c
            return a + b*x0 + c*y0
        if kind=='quadratic':
            a,b,c,d,e,f=c
            return a + b*x0 + c*y0 + d*x0**2 + e*y0**2 + f*x0*y0

    dpdx=(p.max()-p.min())/(x.max()-x.min())
    dpdy=(p.max()-p.min())/(y.max()-y.min())
    xf=x.flatten()
    yf=y.flatten()
    pf=p.flatten()

    if kind=='linear':
        c = [pf.mean(),dpdx,dpdy]
    if kind=='quadratic':
        c = [pf.mean(),dpdx,dpdy,1e-22,1e-22,1e-22]

    coef = leastsq(err,c,args=(xf,yf,pf))[0]
    vm = surface(coef,x,y) #mean surface
    va = p - vm #anomaly
    return va,vm

