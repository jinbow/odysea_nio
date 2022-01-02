def KG_interp_dask(din,target_time,freq=1,
                   pas={'model':'gaussian',
                        'params':{'sill': 0.5,
                                  'range': 3600*2,
                                  'nugget': 0.05}}):
    """
    using dask to calculate the KG_interp_delayed

    din: xarray or pandas.Series with "time" dimension


    target_time: array of np.datetime64
    freq: the time interval for splitting the tasks

    Return:
    =======

    zz: interpolated data on target_time index
    ss: uncertainties on the same index

    """
    import dask
    import numpy as np
    import xarray as xr
    import pandas as pd
    import sys

    ccd=[]

    #remove spikes
    din=din.where(np.abs(din-np.nanmedian(din)) < np.nanstd(din)*3,np.nan)

    days=pd.date_range(target_time[0],target_time[-1],freq='%iD'%freq)

    if target_time[0]<din.time[0]:
        print('warning, the interpolated time is earlier than the first data point.')
        print('Shrink the interpolation range to fit the data')
        print('Define target_time later than',din.time[0].values)
        return

    if target_time[-1]>din.time[-1].values:
        print('warning, the interpolated time is later than the last data point.')
        print('Shrink the interpolation range to fit the data')
        print('Define target_time earlier than',din.time[-1].values)
        return


    data=[]
    time=[]

    for day in days:
        dd=din.sel(time=slice(day-np.timedelta64(2,'h'),day+np.timedelta64(24*freq+2,'h')))
        if dd.size>6:
            data.append(dd)
            time.append(target_time[(target_time>=day)&(target_time<day+np.timedelta64(freq,'D'))])

    for  i in range(len(data)):
        da,ti = data[i],time[i]
        #print(i,da,ti)
        #KG_interp(da,ti,pas)
        ccd.append(dask.delayed(KG_interp)(da,ti,pas))

    nseg=len(data)
    cc=dask.compute(*ccd)
    zz=xr.concat([cc[i][0].to_xarray() for i in range(nseg) ],'index')
    ss=xr.concat([cc[i][1].to_xarray() for i in range(nseg) ],'index')
    #tt=cc[:][2]

    return zz,ss


def coherence_test(dof,alpha=0.05):

    """

    dof: degree of freedom, dof=n*2 where n is usually the number of segments,
    e.g., in Welch method. 
    alpha: the significance level, 0.05 is 95%

    Return
    =======
    C_2: significant coherence squared at 1-alpha level 

    Reference:
    ==========
    This code is copied from https://dsp.stackexchange.com/questions/16558/statistical-significance-of-coherence-values

    Theory:  Priestley 1981 page 706

    """

    import numpy as np
    import scipy.stats as st

    p = 1 - alpha

    n = dof / 2.

    fval = st.f.ppf(p, 2, dof - 2)
    C_2 = fval / (n - 1. + fval)

    return C_2



def KG_interp(din,target_time,
              pas={'model':'gaussian',
                   'params':{'sill': 0.5,
                             'range': 3600*1.5,
                             'nugget': 0.05}},
              reference_time='1900-01-01'):


    import numpy as np
    import pandas as pd

    dtt=((target_time-np.datetime64(reference_time))/np.timedelta64(1,'s')).values

    din=din.sel(time=slice(target_time.min()-np.timedelta64(2,'h'),target_time.max()+np.timedelta64(2,'h')))

    #remove spikes
    din=din.where(np.abs(din-np.median(din)) < din.std()*3,np.nan)

    for i in range(4):
        b=din.diff('time')
        din[1:]=din[1:].where(np.abs(b)<b.std()*3,np.nan)
        #b.where(np.abs(b)>=b.std()*3,np.nan).plot(marker='.',ax=ax[0])

    from pykrige.ok import OrdinaryKriging as OK

    dclean=din.where(~np.isnan(din),drop=True)
    time=((dclean.time-np.datetime64(reference_time))/np.timedelta64(1,'s')).values
    d=dclean.values

    okh=OK(time,np.zeros_like(time),d-d.mean(),variogram_model=pas['model'],variogram_parameters=pas['params'],
                verbose=False,enable_plotting=False)

    z,s=okh.execute('grid',dtt,np.r_[0])

    zz=pd.Series(data=z.squeeze()+d.mean(),index=target_time.values)
    ss=pd.Series(data=s.squeeze(),index=target_time.values)

    return zz,ss

def denoise(d,t=1):
    import numpy as np

    """remove spikes based on three sigma rule"""

    tt=np.arange(d.size)
    msk=np.ma.masked_invalid(d).mask

    if msk.sum()>0:
        d=np.interp(tt,tt[~msk],d[~msk])

    if not np.isscalar(t):
        dt=np.diff(t)
        dt=np.r_[dt[0],dt]
    else:
        dt=1

    for i in range(8):
        dss=np.diff(np.r_[d[0],d])/dt
        msk=np.abs(dss)<dss.std()*3
        d=np.interp(tt,tt[msk],d[msk])

    return d

def filter_butter(data,cutoff,fs, btype,filter_order=4,axis=0):
    """filter signal data using butter filter.

    Parameters
    ==================
    data: N-D array
    cutoff: scalar
        the critical frequency
    fs: scalar
        the sampling frequency
    btype: string
        'low' for lowpass, 'high' for highpass, 'bandpass' for bandpass
    filter_order: scalar
        The order for the filter
    axis: scalar
        The axis of data to which the filter is applied

    Output
    ===============
    N-D array of the filtered data

    """

    import numpy as np
    from scipy import signal

    if btype=='bandpass':
            normal_cutoff=[cutoff[0]/0.5/fs,cutoff[1]/0.5/fs]
    else:
        normal_cutoff=cutoff/(0.5*fs)  #normalize cutoff frequency
    b,a=signal.butter(filter_order, normal_cutoff,btype)
    y = signal.filtfilt(b, a, data, axis=axis)

    return y

def spectrum_semi3D(dd,dx=2.0,dt=1/24.0,detrend='linear',windowing=False,plot=True):
    from scipy import signal, fftpack

    """

    dd is 3D (time, y, x)

       detrend:
           'linear': remove linear trend in all three dimensions
           'constant': remove a domain constant

    return
    ======
    fx: wavenumber
    spec: spectrum energy 
    c: one-dimensional \omega-integrated wavenumber spectrum 
    
    """

    import numpy as np

    sd=signal.detrend

    print('original data variance',dd.var())

    if detrend=='linear':
        dd=sd(sd(sd(dd,0),1),2)
    elif detrend=='constant':
        dd=dd-dd.flatten().mean()

    print('after detrend, data variance',dd.var())
    nt,ny,nx=dd.shape

    if windowing:
        win=np.hanning(nx)
        win=win.reshape(nx,1)*win.reshape(1,nx)
        win=np.hanning(nt).reshape(-1,1,1)*win.reshape(1,nx,nx)
        dd=dd*win
        scale=dx*dx*dt/nt/nx/ny/(win**2).mean()
    else:
        scale=dx*dx*dt/nt/nx/ny

    ddhat=np.abs(fftpack.fftn(dd))**2

    a=ddhat*scale

    

    fx=fy=fftpack.fftfreq(dd.shape[-1],d=dx)
    fxx,fyy=np.meshgrid(fx,fy)
    f2=(fxx**2+fyy**2).ravel()

    
    if nt//2==nt/2:
        b=a[1:nt//2+1,...]
        b[:-1,...]+=a[nt//2+1:,...][::-1,...] #if nt is even, the nt//2 corresponds to the nyquist, not in nt//2+1
    else:
        b=a[1:nt//2+1,...]
        b+=a[nt//2+1:,...][::-1,...]

    b=b.reshape(b.shape[0],-1)
    c=np.zeros((b.shape[0],nx//2))

    for i in range(nx//2):
        msk=(f2>=fx[i]**2)&(f2<fx[i+1]**2)
        c[:,i]=b[:,msk].sum(axis=-1)

    ft=fftpack.fftfreq(nt,d=dt)[1:nt//2+1]
    
    print('variance in spectram',a.sum()*(fx[1]-fx[0])*(ft[1]-ft[0]))
    print('ratio of the variance', (a.sum()*(fx[1]-fx[0])*(ft[1]-ft[0]))/ dd.var())
    return fx[:nx//2],np.abs(ft),c


def spectrum_2D(dd,dx,dy):
    import scipy as sp
    from scipy import signal as spsignal
    import numpy as np

    ny,nx=dd.shape

    z=spsignal.detrend(spsignal.detrend(dd,0,'linear'),1,'linear')

    win=np.hanning(nx).reshape(nx,1)*np.hanning(ny).reshape(1,ny)
    win=win/win.max()
    z=z*win
    #scale=win.sum()/dx/dy
    scale=nx*ny/dx/dy

    a=4*abs(np.fft.fft2(z.T))**2/scale

    b=(a[1:ny//2,1:nx//2]+a[1:ny//2,nx//2+1:][:,::-1]+
       a[ny//2+1:,1:nx//2][::-1,:]+a[ny//2+1:,nx//2+1:][::-1,::-1])

    fx=np.fft.fftfreq(nx,dx)[1:nx//2]
    fy=np.fft.fftfreq(ny,dy)[1:ny//2]

    return fx,fy, b
