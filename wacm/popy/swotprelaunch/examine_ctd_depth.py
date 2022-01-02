"""
optimize the mooring CTD spacing
"""
from pylab import *
import popy,gsw
import scipy as sp
import numpy as np
from scipy import signal,interpolate,optimize

def through_leastsq():

    def func(dzs):
        orf=np.r_[35,dzs].cumsum()
        orc=(orf[1:]+orf[:-1])/2.0
        print dzs,orc
        pden1=itp(orc)
        dyn1=((pden1-1027.5)/1027.5*(dzs).reshape(1,-1)).sum(axis=1)
        dyn1=dyn1-dyn1.mean()
        err=np.array(dyn1)-np.array(dyn0)
        return err

    g=popy.pleiades.load_llc_vertical_grid()

    salt=popy.io.loadh5('stations/125.4W35.7N.h5','salt')
    theta=popy.io.loadh5('stations/125.4W35.7N.h5','theta')
    pden=popy.io.loadh5('stations/125.4W35.7N.h5','pden')
    hfacc=popy.io.loadh5('stations/125.4W35.7N.h5','hfacc')[:,0]
    rc=popy.pleiades.load_llc_vertical_grid()['RC'][:]
    rf=popy.pleiades.load_llc_vertical_grid()['RF'][:]
    drf=popy.pleiades.load_llc_vertical_grid()['DRF'][:]

    pden=pden[5600:5600+24*90,:]
    #theta=theta[5600:5600+24*90,:]
    #salt=salt[5600:5600+24*90,:]
    itp=interpolate.interp1d(rc,pden,axis=1)
    dyn0=(1/1027.5*(pden-1027.5)*drf.reshape(1,-1)).sum(axis=1)
    dyn0=dyn0-dyn0.mean()
    print dyn0



    #['39', '52', '68', '124', '171', '189', '209', '229', '274', '380', '441', '576', '689', '812', '992', '1246', '1480', '1753', '2180', '3214', '4019', '4265', '4828']
    zobs=np.r_[34, 63, 103, 143, 183, 253, 454, 704, 938, 1439, 1947, 2448, 2949, 3443, 3944, 4437]
    zobsf=(zobs[1:]+zobs[:-1])/2.0
    zobsf=np.r_[16,zobsf,4700]

    dz=np.ones((21))*10
    a=optimize.leastsq(func,dz)
    print a

    stop()

    dyn=np.array(dyn)
    dyn=dyn-dyn.mean()
    dyn_obs=np.array(dyn_obs)
    dyn_obs=dyn_obs-dyn_obs.mean()
    dyn_obs_yi=np.array(dyn_obs_yi)
    dyn_obs_yi=dyn_obs_yi-dyn_obs_yi.mean()
    #plot(dyn-dyn.mean())
    #plot(dyn_obs-dyn_obs.mean())
    figure(figsize=(4,6))
    subplot(211)
    a,b=signal.coherence(dyn,dyn_obs,noverlap=0,nfft=480,nperseg=480,detrend='linear',window='hann',fs=24.0)
    semilogx(a,b,'b',label='truth<->optimal')
    a,b=signal.coherence(dyn,dyn_obs_yi,noverlap=0,nfft=480,nperseg=480,detrend='linear',window='hann',fs=24.0)
    semilogx(a,b,'r',label='truth<->original')
    legend()
    xlabel('frequency (cpd)')
    ylabel('coherence')
    subplot(212)
    a,b=signal.welch(dyn*1e2,noverlap=0,nfft=480,nperseg=480,detrend='linear',window='hann',fs=24.0)
    loglog(a,b,'k',label='truth')
    a,b=signal.welch(dyn_obs*1e2,noverlap=0,nfft=480,nperseg=480,detrend='linear',window='hann',fs=24.0)
    loglog(a,b,'b',label='optimal')
    a,b=signal.welch(dyn_obs_yi*1e2,noverlap=0,nfft=480,nperseg=480,detrend='linear',window='hann',fs=24.0)
    loglog(a,b,'r',label='original')
    legend()
    xlabel('frequency (cpd)')
    ylabel('cm$^2$/cpd')
    tight_layout()
    show()
    savefig('../prelaunch.figures/through_correlation.2.pdf')
    savefig('../prelaunch.figures/through_correlation.2.png',dpi=300)
    return

def through_correlation():

    g=popy.pleiades.load_llc_vertical_grid()

    salt=popy.io.loadh5('stations/125.4W35.7N.h5','salt')
    theta=popy.io.loadh5('stations/125.4W35.7N.h5','theta')
    pden=popy.io.loadh5('stations/125.4W35.7N.h5','pden')
    hfacc=popy.io.loadh5('stations/125.4W35.7N.h5','hfacc')[:,0]
    rc=popy.pleiades.load_llc_vertical_grid()['RC'][:]
    rf=popy.pleiades.load_llc_vertical_grid()['RF'][:]
    drf=popy.pleiades.load_llc_vertical_grid()['DRF'][:]

    pden=pden[5600:5600+24*90,:]
    theta=theta[5600:5600+24*90,:]
    salt=salt[5600:5600+24*90,:]

    dz=1.0
    newz=np.arange(rc[0],rc[83],dz)
    newzf=np.r_[rf[0],newz,newz[-1]+dz/2.0]

    pdeni=interpolate.interp1d(rc,pden,axis=1)(newz)
    thetai=interpolate.interp1d(rc,theta,axis=1)(newz)
    salti=interpolate.interp1d(rc,salt,axis=1)(newz)

    corr=np.corrcoef(thetai.T,)
    contourf(newz,newz,corr,)
    colorbar()
    ylim(0,rc[83])
    xlim(0,rc[83])
    xlabel('depth')
    ylabel('depth')
    tight_layout()
    savefig('../prelaunch.figures/through_correlation.1.png',dpi=300)

    print newz
    i=np.arange(newz.size)[newz<=36][-1];ii=[i];a=[newz[i]]
    print i, newz[i]

    threshold=0.71
    ni=corr.shape[0]
    print corr.shape
    while (i<ni-1):
        j=0
        while (corr[i,i+j]>threshold) and i+j<ni-1:
            j+=1
        i+=j
        print i,j
        a.append(newz[i])

    print len(a)
    print ['%i'%c for c in a]
    a=np.array(a)
    b=(a[1:]+a[:-1])/2.0

    dyn=[];dyn_obs=[];dyn_obs_yi=[]
    #['39', '52', '68', '124', '171', '189', '209', '229', '274', '380', '441', '576', '689', '812', '992', '1246', '1480', '1753', '2180', '3214', '4019', '4265', '4828']
    zobs=np.r_[34, 63, 103, 143, 183, 253, 454, 704, 938, 1439, 1947, 2448, 2949, 3443, 3944, 4437]
    zobsf=(zobs[1:]+zobs[:-1])/2.0
    zobsf=np.r_[16,zobsf,4700]

    for i in range(theta.shape[0]):
        d0,d1=popy.utils.dyn_from_model(salt[i,:],theta[i,:],rc,rf,35.7,drf,
                                        z_obs_rc=b,z_obs_rf=a)
        d0,d2=popy.utils.dyn_from_model(salt[i,:],theta[i,:],rc,rf,35.7,drf,
                                        z_obs_rc=zobs,z_obs_rf=zobsf)
        dyn.append(d0)
        dyn_obs.append(d1)
        dyn_obs_yi.append(d2)

    dyn=np.array(dyn)
    dyn=dyn-dyn.mean()
    dyn_obs=np.array(dyn_obs)
    dyn_obs=dyn_obs-dyn_obs.mean()
    dyn_obs_yi=np.array(dyn_obs_yi)
    dyn_obs_yi=dyn_obs_yi-dyn_obs_yi.mean()
    #plot(dyn-dyn.mean())
    #plot(dyn_obs-dyn_obs.mean())
    figure(figsize=(4,6))
    subplot(211)
    a,b=signal.coherence(dyn,dyn_obs,noverlap=0,nfft=480,nperseg=480,detrend='linear',window='hann',fs=24.0)
    semilogx(a,b,'b',label='truth<->optimal')
    a,b=signal.coherence(dyn,dyn_obs_yi,noverlap=0,nfft=480,nperseg=480,detrend='linear',window='hann',fs=24.0)
    semilogx(a,b,'r',label='truth<->original')
    legend()
    xlabel('frequency (cpd)')
    ylabel('coherence')
    subplot(212)
    a,b=signal.welch(dyn*1e2,noverlap=0,nfft=480,nperseg=480,detrend='linear',window='hann',fs=24.0)
    loglog(a,b,'k',label='truth')
    a,b=signal.welch(dyn_obs*1e2,noverlap=0,nfft=480,nperseg=480,detrend='linear',window='hann',fs=24.0)
    loglog(a,b,'b',label='optimal')
    a,b=signal.welch(dyn_obs_yi*1e2,noverlap=0,nfft=480,nperseg=480,detrend='linear',window='hann',fs=24.0)
    loglog(a,b,'r',label='original')
    legend()
    xlabel('frequency (cpd)')
    ylabel('cm$^2$/cpd')
    tight_layout()
    show()
    savefig('../prelaunch.figures/through_correlation.2.pdf')
    savefig('../prelaunch.figures/through_correlation.2.png',dpi=300)
    return

def through_bc_modes():
    g=popy.pleiades.load_llc_vertical_grid()

    salt=popy.io.loadh5('stations/125.4W35.7N.h5','salt')
    theta=popy.io.loadh5('stations/125.4W35.7N.h5','theta')
    pden=popy.io.loadh5('stations/125.4W35.7N.h5','pden')
    hfacc=popy.io.loadh5('stations/125.4W35.7N.h5','hfacc')[:,0]
    rc=popy.pleiades.load_llc_vertical_grid()['RC'][:]
    rf=popy.pleiades.load_llc_vertical_grid()['RF'][:]
    drf=popy.pleiades.load_llc_vertical_grid()['DRF'][:]

    lat1=35.7
    pressure=np.abs(gsw.p_from_z(-rc,lat1))

    newpden=np.zeros_like(pden)
    for i in range(theta.shape[0]):
        newpden[i,:]=popy.jmd95.densjmd95(salt[i,:],theta[i,:],pressure)

    rhomean=popy.jmd95.densjmd95(salt.mean(axis=0),theta.mean(axis=0),pressure)
    #rhomean=pden.mean(axis=0)
    rhomean[hfacc==0]=rhomean.max()
    nz=(hfacc>0).sum()
    drhodz=np.diff(rhomean[:nz])/drf[:nz-1]

    drhodzn=popy.utils.smooth(drhodz,5)

    #spl=interpolate.splrep(rf[1:nz],drhodz,s=3)
    #newrho=interpolate.splev(rf[1:nz],spl,)
    #plot(rhomean)
    #plot(np.arange(nz),newrho)
    #plot(drhodz,-rc[:nz-1])
    #plot(drhodzn,-rc[:nz-1])
    n2=9.81/1027.5*drhodz
    print n2.shape

    w,vi=popy.utils.qgdecomp(-rc[:nz],-rf[:nz+1],n2,gsw.f(35.7))
    print w.shape,vi.shape

    plot(vi[:,5],-rc[:nz],'-o')

    show()

if __name__=="__main__":
    through_correlation()
    #through_leastsq()
    #through_bc_modes()
