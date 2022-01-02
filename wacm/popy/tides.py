# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 15:09:54 2016

@author: Jinbo Wang (@jpl.nasa.gov) 

"""

def tide_reconstruct(coef,t,nc,periods=[]):
    import numpy as np
    """coef contains amplitudes, phase, is of size 2*nc
    t: the time coordinate in a unit of second
    nc:  the number of tidal constituents
    """
    
    amp=np.array(coef[:coef.size/2]).ravel()
    phase=np.mod(np.array(coef[coef.size/2:]).ravel(),np.pi*2)
    t=t.ravel()
    n=phase.size
    #speed rad/hour
    #cons=np.r_[28.984104,30.0,28.43973,15.041069,
    #           57.96821,13.943035,86.95232,44.025173].reshape(-1,1)
    #tidal periods:

    cons=3600.0*np.r_[12.421,    12.0,      12.658,    11.967,   23.934,
                      25.819,    24.066,    23.804,    23.869,   24.0,
                      13.661*24, 27.555*24, 180.0*24,  360.0*24]#14 total


    if np.isscalar(nc):
        cons=cons[:nc] 
    else:
        cons=cons[nc]

    if len(periods)>0:
        cons=periods
    
    omega=2*np.pi/cons.ravel()
   
    nt=t.size 

    ser=np.zeros((nt))

    for i in range(n):
        ser += amp[i]*np.cos(omega[i]*t + phase[i])

    return ser

def func(coef,d,t,nc,weight,periods=[]):
    #if periods is specified, use it to override default
    d=d.flatten()
    n=coef.size/2
    ser=tide_reconstruct(coef,t,nc,periods)
    err=(d-ser.flatten())*weight
    return err

def get_tidal_coef(d,t,nc,weight=1,periods=[]):
    from scipy import optimize
    import numpy as np

    if len(periods)>0: #if periods is specified, use it to override the default tidal periods
        nc=len(periods)
    try:
        p0=np.ones((nc*2,))*0.1
    except:
        p0=np.ones((len(nc)*2,))*0.1

    a=optimize.leastsq(func,p0, args=(d,t,nc,weight,periods))

    return a[0]

def detides(d,t,nc,weight=1,periods=[]):
    a=get_tidal_coef(d,t,nc=nc,weight=weight,periods=periods)
    dr=tide_reconstruct(a,t,nc,periods=periods)
    res=d-dr
    return res,dr,a

if __name__=="__main__":
    import numpy as np
    import pylab as plt
    """demonstrate using a synthetic time series"""
    coef=np.zeros(26)
    coef[4]=1.0
    coef[5]=0.5
    t=np.arange(10*24)*3600.0

    a=tide_reconstruct(coef,t,nc=13)
    b=a+np.random.normal(0,1,a.size)

    res,dr=detides(b,t,nc=13)
    plt.plot(t/3600,b,'o',markerfacecolor='None',label='with noise')
    plt.plot(t/3600,a,'k-',lw=2,label='truth')
    plt.plot(t/3600,dr,'r-',lw=2,label='fitted')
    plt.legend()
    plt.xlabel('hours')
    plt.savefig('../calval.figures/c_detides_01.png') 
    plt.show()
