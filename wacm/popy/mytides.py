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

def remove_tides(d,tt,lat=35):
    """
    Parameter:
    ---------
    d: array like
       (npoints, nt)

    Return
    ---------
    a: array (3)
       tidal variance, m2 amplitude, m2 phase
    """
    import utide
    import numpy as np


    ni,nt=d.shape
    dout=np.zeros((ni,nt),dtype=np.float32,order='C')

    for i in range(ni):
        ddl=d[i,:]
        if ddl[0]==0:
            dout[i,:]=0
        else:
            cc=utide.solve(tt,ddl,lat=lat,verbose=False)
            dout[i,:]=utide.reconstruct(tt,cc,verbose=False)['h']

    return np.ascontiguousarray(dout)

def detides(d,t,weight=1,periods=[],method='utide',usempi=True):
    """
    Parameter:
    ---------
    d: array like
       (npoints, nt)
    t: time records for the data array 
       array_like (nt,)

    Return
    ---------
    a: (npoints, nt)
       detided signal
 
    unfinished ---
    """
    import numpy as np
    if usempi:
  
        import parallel

        rank,size,comm=parallel.start_mpi()
        
 
        ni,nt=d.shape

        dnew=np.zeros(( np.int(np.ceil(1.0*ni/size)*size),nt),dtype=np.float32,order='C')
        dnew[:ni,:]=d.copy()
        del d

        nn=np.ceil(1.0*ni/size).astype('i')
 
        dlocal=np.empty(shape=(nn,nt),dtype=np.float32,order='C')
        comm.Scatter(dnew,dlocal,root=0)

        bb=remove_tides(dlocal,t) #a list of (nt) time series

        dout=None

        if rank==0:
            dout=np.empty(shape=(size,nn,nt),dtype=np.float32,order='C')

        comm.Gather(bb,dout,root=0)

        if rank==0:
            return dout
    else:
        dout=remove_tides(dlocal,t)

        return dout

def tidal_freq():
    import utide
    a={}
    for i, n in enumerate(utide.harmonics.const.name):
        a[n]=utide.harmonics.const.freq[i]
    return a

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
