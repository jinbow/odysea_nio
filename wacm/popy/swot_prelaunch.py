"""
some commonly used routines for analyzing the prelaunch field campaign data

"""

import numpy as np
import sys
sys.path.append('/u/jwang23/popy/')
import myio,ctds,gsw
import xarray as xr
import matplotlib
import pylab as plt

font = {'family' : 'normal',
        'size'   : 12}

matplotlib.rc('font', **font)

data_path='/home1/jwang23/projects/swot/prelaunch.data/'
fig_path='/home1/jwang23/projects/swot/prelaunch.figures/'

def qc_rbr():
    """
    check the consistency between rbr and microcat on the wirewalker
    """
    pth=data_path+'sio/swot1_01/csv/'
    tmp=np.genfromtxt(pth+'press-rbr-pass.csv',skip_header=4,delimiter=',')
    rbr_time=tmp[:,0]/1e3
    rbr_press=tmp[:,1]
    rbr_salt=np.genfromtxt(pth+'sal-rbr-pass.csv',skip_header=4,delimiter=',')[:,1]
    rbr_temp=np.genfromtxt(pth+'temp-rbr-pass.csv',skip_header=4,delimiter=',')[:,1]

    tmp=np.genfromtxt(pth+'press-ww.csv',skip_header=4,delimiter=',')
    ww_time=tmp[:,0]/1e3
    ww_press=tmp[:,1]
    ww_salt=np.genfromtxt(pth+'sal-ww.csv',skip_header=4,delimiter=',')[:,1]
    ww_temp=np.genfromtxt(pth+'temp-ww.csv',skip_header=4,delimiter=',')[:,1]

    plt.plot(rbr_time,np.ones_like(rbr_time),'r.')
    plt.plot(ww_time,np.ones_like(ww_time)*2,'b.')
    plt.show()

def process_prawler(t0=None,t1=None):
    """
    t0,t1: numpy.datetime64 define the beginning and the end of the returned data
    """
    fn='/nobackup/jwang23/projects/prelaunch.data/Prawler/Prawler.20200106.nc'
    dd=xr.open_dataset(fn)
    print(dd.keys())
    temp=np.ma.masked_outside(dd['SB_Temp'].values,3,34)
    time=dd['time']
    if t0!=None and t1!=None:
        msk=(time>t0)&(time<t1)
    else:
        msk=np.ones_like(temp).astype('bool')

    temp=temp[msk]
    cond=dd['SB_Conductivity'].values[msk]*10
    time=time[msk]
    depth=dd['SB_Depth'].values[msk]
    lat=dd['latitude'].values[msk]
    lon=dd['longitude'].values[msk]
    press=gsw.z_from_p(-np.abs(depth),lat.mean())
    sp=np.ma.masked_outside(gsw.SP_from_C(cond,temp, press), 10,37)

    rowsize=dd['rowSize'].values

    d=xr.Dataset.from_dict({'temp':{'dims':('t'),'data':temp},
        'ps':{'dims':('t'),'data':sp},
        'depth':{'dims':('t'),'data':depth},
        'time':{'dims':('t'),'data':time},
        'lat':{'dims':('t'),'data':lat},
        'lon':{'dims':('t'),'data':lon},
        'press':{'dims':('t'),'data':press},
        'rowsize':{'dims':('profile'),'data':rowsize}} )


    del dd

    return d

def spectrum():
    import pylab as plt
    from scipy import signal
    from matplotlib.gridspec import GridSpec as GS

    dd=xr.open_dataset('sio_combine.h5')['dyn'].values[:,:]*100
    nt,nn=dd.shape
    print(nt)
    ntt=nt//10 * 10
    dd=dd[-ntt:,:]
    print(dd.shape)
    men=np.nanmean(dd,axis=0)
    dd=dd-men.reshape(1,-1)

    msk=np.isnan(dd[:,-1])
    dd[msk,-1]=dd[msk,-2]
    tt=np.arange(dd.shape[0])/24.*0.25
    idep=int(sys.argv[1])

    gs=GS(2,3)

    deps=[500,700,1000,1500,2000,4400]
    plt.figure(figsize=(12,5))
    ax=plt.subplot(gs[0,:-1])
    for iz in [idep,5]:
        ax.plot(tt,dd[:,iz],label='%im'%deps[iz])
    ax.legend()
    ax.set_title('dynamic height')
    ax.set_ylabel('cm')
    ax.set_ylim(-6,6)
    #plt.savefig(fig_path+'dynamic_height_%im_full.png'%deps[idep],dpi=300)

    ax=plt.subplot(gs[1,:-1])
    ax.plot(tt,dd[:,5]-dd[:,idep],c='g',label='difference')
    ax.legend()
    ax.set_ylabel('cm')
    ax.set_xlabel('time (day)')
    ax.set_ylim(-6,6)
    #plt.savefig(fig_path+'dynamic_height_%im_full_difference.png'%deps[idep],dpi=300)

    ax=plt.subplot(gs[:,-1])
    ax.scatter(dd[:,5],dd[:,idep],s=16,marker='+',linewidth=1)
    ax.plot(np.arange(-6,6.1,0.1),np.arange(-6,6.1,0.1),'r-',lw=1)
    ax.set_xlim(-6,6)
    ax.set_ylim(-6,6)
    ax.set_title('dynamic height')
    ax.set_ylabel('%im'%deps[idep])
    ax.set_xlabel('full depth')

    plt.tight_layout()
    plt.savefig(fig_path+'dynamic_height_%im_full_scatter.png'%deps[idep],dpi=300)

    print(deps)
    print((dd[:,-1:]-dd[:,:-1]).std(axis=0)/dd[:,-1].std())
    print(((dd[:,-1:]-dd[:,:-1]).std(axis=0)/dd[:,-1].std())**2)
    print('error to signal ratio',(dd[:,-1:]-dd[:,:-1]).var(axis=0)/dd[:,-1].var())
    print((dd[:,-1:]-dd[:,:-1]).std(axis=0))
    print('full depth dynamic height variance ',dd[:,-1].std())

    def avg(a,b):
        n=a.size
        a=a[:n//10*10].reshape(-1,10).mean(axis=-1)
        b=b[:n//10*10].reshape(-1,10).mean(axis=-1)
        return a,b
    fig=plt.figure()
    ax=plt.subplot(111)
    a,b=signal.welch(dd[:,idep],nfft=nt,noverlap=0,nperseg=10*24*4,detrend='linear',window='hann',fs=24/0.25)
    print(a.shape)
    a,b=avg(a,b)
    msk=a<24
    ax.loglog(a[msk],b[msk],label='%im'%deps[idep],)


    a,b=signal.welch(dd[:,-1],nfft=nt,nperseg=10*24*4,noverlap=0,detrend='linear',window='hann',fs=24/0.25)
    a,b=avg(a,b)
    ax.loglog(a[msk],b[msk],label='4400m',)

    ax.legend(loc='lower left')

    a,b=signal.coherence(dd[:,-1],dd[:,idep],nfft=nt,noverlap=0,nperseg=10*24*4,detrend='linear',window='hann',fs=24/0.25)
    a,b=avg(a,b)
    ax1=ax.twinx()

    ax1.semilogx(a[msk],b[msk],c='r')
    ax1.set_ylim(0,1.1)
    ax.set_xlabel('frequency (cpd)')
    ax.set_ylabel(r'cm$^2$/cpd')
    ax1.set_ylabel('coherence',color='r')
    ax1.tick_params(axis='y', colors='red')

    plt.savefig(fig_path+'spectrum_cohere_%im_full.png'%deps[idep],dpi=300)

    plt.show()


def despike_sio_wirewalker():
    """ctime: time center in hours
    dtime: the time range to subtract, unit:hours"""

    import ctds,gsw
    import scipy as sp
    import pyhht

    pth=data_path+'sio/swot1_01/csv/'
    pressure=np.genfromtxt(pth+'press-ww.csv',skip_header=4,delimiter=',')
    salt=np.genfromtxt(pth+'sal-ww.csv',skip_header=4,delimiter=',')[:,1]
    temp=np.genfromtxt(pth+'temp-ww.csv',skip_header=4,delimiter=',')[:,1]

    print('data size for pressure, salinity, and temperature:')
    print(pressure.shape,salt.shape,temp.shape)
    time=pressure[:,0]
    time=(time-time[0])/1e3/3600 #minutes
    press=pressure[:,1]
    salt=denoise(salt)
    temp=denoise(temp)
    z=gsw.z_from_p(np.abs(press), lat=35.85)
    myio.saveh5s('sio_wirewalker_despike.h5',{'z':z,'time':time,'temp':temp,'salt':salt,'press':press})


    return time,z,salt,temp

def plotit():
    emd=pyhht.EMD(salt,time,n_imfs=5)
    aa=emd.decompose()
    aa=aa-aa.mean(axis=-1)[:,np.newaxis]
    plt.plot(time,aa[-1,:])
    plt.show()


    nn=time.size//5*5
    time=time[:nn].reshape(-1,5).mean(axis=-1)
    z=z[:nn].reshape(-1,5).mean(axis=-1)
    den=den[:nn].reshape(-1,5).mean(axis=-1)

    rbf=sp.interpolate.Rbf(time,z,den)

    newtime,newz=np.meshgrid(np.arange(ctime-dtime+1,ctime+dtime-1,0.5),np.arange(0,-500,-5))
    newden=rbf(newtime,newz)
    print(newden.shape)

    print(newtime.shape,newz.shape,newden.shape)
    plt.pcolor(newtime[:,2:],newz[:,2:],np.diff(newden,axis=-1)[:,1:],vmin=-0.01,vmax=0.01,cmap=plt.cm.bwr)
    plt.colorbar()

    plt.scatter(time,z,c=den,marker='o',edgecolor='w',s=20,vmin=1023,vmax=1027,cmap=plt.cm.jet)
    plt.show()


    return time,z,salt,temp

def denoise(d,t=1):
    """remove spikes based on three sigma rule"""

    tt=np.arange(d.size)
    msk=np.isnan(d)

    if msk.sum()>0:
        print('nan value',msk.sum())
        d=np.interp(tt,tt[~msk],d[~msk])

    if not np.isscalar(t):
        dt=np.diff(t)
        dt=np.r_[dt[0],dt]
    else:
        dt=1

    for i in range(5):
        dss=np.diff(np.r_[d[0],d])/dt
        msk=np.abs(dss)<dss.std()*3
        print(msk.sum(),d.size)
        d=np.interp(tt,tt[msk],d[msk])

    return d

def load_wirewalker(t0=None,t1=None,do_denoise=False,instrument='RBR',binned=False):
    """t0,t1 numpy.datetime64 define the beginning and the end of the returned data"""

    gd=xr.open_dataset('SIO_CTDPROF_L2_20190906_20191202_VER001.nc')

    vn=instrument

    time=gd['TIME_WW_'+vn].values
    print('time range',time.min(),time.max())
    print(time[:10])

    if t0==None:
        t0=np.datetime64('2019-09-10 15:12:00')
    if t1==None:
        t1=np.datetime64('2019-11-30 15:00:00')

    msk=(time>t0)&(time<t1)&(~np.isnan(gd['depth'].values))

    dd={}


    time=time[msk].astype('f8')/1e9/86400
    time=time-time[0]
    if do_denoise:
        salt=denoise(gd['PSAL_WW_'+vn][msk].values,time)
        temp=denoise(gd['TEMP_WW_'+vn][msk].values,time)
    else:
        salt=gd['PSAL_WW_'+vn][msk].values
        temp=gd['TEMP_WW_'+vn][msk].values

    if binned:
        time=time-time[0]
        print(time.min(),time.max(),depth.min(),depth.max())
        pts=np.c_[time[msk].flatten(),depth[msk].flatten()/100]
        t,z=np.meshgrid(np.arange(time[0],time[-1],1/24./2),np.arange(10,500,5)/100)
        ptst=np.c_[t.flatten(),z.flatten()]
        aa=griddata(pts,temp.flatten(),ptst)
        bb=griddata(pts,salt.flatten(),ptst)
        dd['time']={'dims':('z','t'),'data':t}
        dd['depth']={'dims':('z','t'),'data':z}
        dd['temp']={'dims':('z','t'),'data':aa}
        dd['salt']={'dims':('z','t'),'data':bb}
        del aa,bb
    else:
        dd['time']={'dims':('t'),'data':time[msk]}
        dd['press']={'dims':('t'),'data':gd['PRES_WW_'+vn][msk].values}
        dd['depth']={'dims':('t'),'data':gd['DEPTH_WW_'+vn][msk].values}

        dd['ps']={'dims':('t'),'data':salt}
        dd['temp']={'dims':('t'),'data':temp}

        dd['flag']={'dims':('t'),'data':gd['QC_FLAG_WW_'+vn][msk].values}
    dout=xr.Dataset.from_dict(dd)

    return dout

def load_slocum_glider(t0=None,t1=None):
    """t0,t1 numpy.datetime64 define the beginning and the end of the returned data"""

    pth='/home1/jwang23/projects/swot/prelaunch.data/analyzed/'
    gld=xr.open_dataset(pth+'all.slocum.glider.nc')

    time=gld['time'].values
    print(time)

    if t0!=None and t1!=None:
        t0=(t0-np.datetime64('1970-01-01 00:00:00')).astype('f')
        t1=(t1-np.datetime64('1970-01-01 00:00:00')).astype('f')
        msk=(time>t0)&(time<t1)&(time<1e30)&(gld['temp'].values<40)&(gld['sp'].values<37)
        time=time[time<1e30]
    else:
        msk=np.ones((time.size)).astype('bool')

    dd=xr.Dataset.from_dict({'ps':{'dims':('t'),'data':gld['sp'][msk].values},
        'temp':{'dims':('t'),'data':gld['temp'][msk].values},
        'depth':{'dims':('t'),'data':gld['depth'][msk].values},
        'press':{'dims':('t'),'data':gld['press'][msk].values}} )
    return dd

def load_sio_microcat1(t0=None,t1=None):
    """
    copy of load_sio_microcat() but load the SIO QC'ed version
    """
    pth=data_path+'sio.mooring/'
    dd=xr.open_dataset(pth+'SIO_CTDFIXED_L2_20190906_20191027_VER009.nc')
    pressure=dd['PRES']

    time=dd['TIME'][:].values #days since 1950 01 01 0000
    print(time[:10])


    if t0==None:
        t0=np.datetime64('2019-09-10 15:12:00')
    if t1==None:
        t1=np.datetime64('2019-11-30 15:00:00')

    msk=(time>t0)&(time<t1)

    dd=xr.Dataset.from_dict({'time':{'dims':('t'),'data':time[msk]},
                             'ps':{'dims':('z','t'),'data':dd['PSAL'][:,msk].values},
        'temp':{'dims':('z','t'),'data':dd['TEMP'][:,msk].values},
        'depth':{'dims':('z','t'),'data':dd['DEPTH'][:,msk].values},
        'press':{'dims':('z','t'),'data':dd['PRES'][:,msk].values}} )
    return dd

def load_sio_microcat(ctime=None,dtime=None):
    pth=data_path+'sio/swot1_01/csv/'
    pressure=np.genfromtxt(pth+'press.csv',skip_header=4,delimiter=',')

    time=pressure[:,0]
    time=(time-time[0])/1e3/3600
    press=-pressure[:,1:]

    salt=np.genfromtxt(pth+'sal.csv',skip_header=4,delimiter=',')[:,1:]
    temp=np.genfromtxt(pth+'temp.csv',skip_header=4,delimiter=',')[:,1:]
    for i in range(salt.shape[-1]):
        print(i)
        print(salt[:,i].min())
        print(temp[:,i].min())
        salt[:,i]=denoise(salt[:,i])
        temp[:,i]=denoise(temp[:,i])


    #fig,ax=plt.subplots(3,1,sharex=True)
    #ax[0].plot(time,salt[:,1])
    #dss=np.diff(np.r_[salt[0,1],salt[:,1]])
    #ax[1].plot(time,dss)
    #msk=np.abs(dss)>3*dss.std()
    #ax[1].plot(time[msk],dss[msk],'o')

    #sal=denoise(salt[:,1])
    #ax[0].plot(time,sal)
    #plt.show()
    #exit()

    if ctime!=None:
        msk=(time>ctime-dtime)&(time<ctime+dtime)
        time=time[msk]
        press=-pressure[msk,1:]
        salt=salt[msk,1:]
        temp=temp[msk,1:]

    print('data size for pressure, salinity, and temperature:')
    print(pressure.shape,salt.shape,temp.shape)

    den=ctds.c_rho(salt,temp,np.abs(press),lat=35.85,lon=-125.05)
    z=gsw.z_from_p(np.abs(press), lat=35.85)

    plt.contourf(np.ones_like(z)*time.reshape(-1,1),z,den)
    plt.show()

    fig,ax=plt.subplots(3,1,sharex=True)
    print(time)
    ax[0].plot(time,salt[:,1:],)
    ax[0].invert_yaxis()
    ax[0].set_ylabel('salinity')

    ax[1].plot(time,temp[:,1:],)
    ax[1].set_ylabel('temperature')

    ax[2].plot(time,pressure[:,1:],)
    plt.xlabel('days')


    fig1,ax1=plt.subplots(2,1)
    cs=ax1[0].scatter(salt[:,1],temp[:,1],s=10,c=pressure[:,1])
    plt.colorbar(cs,)

    fig1,ax2=plt.subplots(2,1)
    tmp=np.diff(temp[:,2],1,axis=0)
    ss=tmp.std()
    print(ss)
    tt=time[1:]
    ax2[0].plot(tt,tmp)
    msk=np.abs(tmp)>ss*3
    ax2[0].plot(tt[msk],tmp[msk],'ro')

    plt.show()



    return time,z,salt,temp,den
def combine_fixedctd_wirewalker():
    """copied from combine(), now use SIO L2 data"""

    from scipy.interpolate import interp1d
    dd=load_wirewalker()
    wwtime=dd['time'].values.astype('f8')/1e9/3600 #hours since 1950/1/1
    print(wwtime[:10])
    wwz=dd['depth'].values
    wwsalt=dd['ps'].values
    wwtemp=dd['temp'].values


    dd= load_sio_microcat1()
    mctime=dd['time'].values.astype('f8')/1e9/3600 #hours since 1950/1/1
    print(mctime[:10])
    mcz=dd['depth'].values
    mcsalt=dd['ps'].values
    mctemp=dd['temp'].values

    ctime=np.arange(mctime[1],mctime[-2],1)
    zz=wwz
    for i,tt in enumerate(ctime):
        print(i,ctime.size)
        a=[]
        for zz in range(-5, -500, -10):
            msk=(wwz>zz-15)&(wwz<zz+15)&(wwtime>tt-1)&(wwtime<tt+1)
            z=wwz[msk].mean()
            s=wwsalt[msk].mean()
            t=wwtemp[msk].mean()
            den=ctds.c_rho(s,t,z,ispressure=False,lat=35.85,lon=-125.05)
            a.append([tt,z,s,t,den])

        a=np.array(a)


        msk=(mctime>tt-1)&(mctime<tt+1)

        mct=mctemp[:,msk].mean(axis=1)
        mcs=mcsalt[:,msk].mean(axis=1)
        mczz=mcz[:,msk].mean(axis=1)
        #mcd=mcden[:,msk].mean(axis=1)

        newz=np.r_[a[:,1], mczz].ravel()

        z=np.abs(newz)
        z[0]=0
        z[-1]=4350
        dz=10.0
        newz=np.arange(5,4400,dz)


        #pden=interp1d(z,np.r_[a[:,4], mcd].ravel(),bounds_error=False,fill_value='extrapolate')(newz)
        temp=interp1d(z,np.r_[a[:,3], mct].ravel(),bounds_error=False)(newz)
        salt=interp1d(z,np.r_[a[:,2], mcs].ravel(),bounds_error=False)(newz)

        #dd={'pden':{'dims':('var','t'),'data':pden},
        #dd={'temp':{'dims':('z','t'),'data':temp},
        #'depth':{'dims':('z'),'data':newz},
        #'time':{'dims':('t'),'data':ctime},
        #'ps':{'dims':('z','t'),'data':salt}}

        #dout=xr.Dataset.from_dict(dd)
        #dout.to_netcdf('sio_combined.nc')
        myio.saveh5('sio_combine.nc','temp',temp,nrec=ctime.size,irec=i)
        myio.saveh5('sio_combine.nc','ps',salt,nrec=ctime.size,irec=i)
    myio.saveh5('sio_combine.nc','depth',newz)
    myio.saveh5('sio_combine.nc','time',ctime)

    return

def combine():
    from scipy.interpolate import interp1d

    wwtime=myio.loadh5('sio_wirewalker_despike.h5','time')
    wwz=myio.loadh5('sio_wirewalker_despike.h5','z')
    wwsalt=myio.loadh5('sio_wirewalker_despike.h5','salt')
    wwtemp=myio.loadh5('sio_wirewalker_despike.h5','temp')

    mctime,mcz,mcsalt,mctemp,mcden=load_sio_microcat()

    ctime=np.arange(mctime[1],mctime[-2],0.25)

    for i,tt in enumerate(ctime):
        a=[]
        for zz in range(-5, -500, -10):
            msk=(wwz>zz-15)&(wwz<zz+15)&(wwtime>tt-0.5)&(wwtime<tt+0.5)
            z=wwz[msk].mean()
            s=wwsalt[msk].mean()
            t=wwtemp[msk].mean()
            den=ctds.c_rho(s,t,z,ispressure=False,lat=35.85,lon=-125.05)
            a.append([tt,z,s,t,den])

        a=np.array(a)

        msk=(mctime>tt-0.5)&(mctime<tt+0.5)

        mct=mctemp[msk,:].mean(axis=0)
        mcs=mcsalt[msk,:].mean(axis=0)
        mczz=mcz[msk,:].mean(axis=0)
        mcd=mcden[msk,:].mean(axis=0)

        newz=np.r_[a[:,1], mczz].ravel()
        newd=np.r_[a[:,-1], mcd].ravel()

        z=np.abs(newz)
        z[0]=0
        z[-1]=4350
        dz=10.0
        newz=np.arange(5,4400,dz)
        pden=interp1d(z,newd,bounds_error=False,fill_value='extrapolate')(newz)

        rho0=1027
        msk1=newz<=500
        msk2=newz<=700
        msk3=newz<=1000
        msk4=newz<=1500
        msk5=newz<=2000
        dyn=np.r_[(pden[msk1]-rho0).sum()*dz/rho0,
                   (pden[msk2]-rho0).sum()*dz/rho0,
                   (pden[msk3]-rho0).sum()*dz/rho0,
                   (pden[msk4]-rho0).sum()*dz/rho0,
                   (pden[msk5]-rho0).sum()*dz/rho0,
                   (pden-rho0).sum()*dz/rho0]
        print(dyn)

        myio.saveh5('sio_combine.h5','z',newz,irec=i,nrec=ctime.size)
        myio.saveh5('sio_combine.h5','rho',newd,irec=i,nrec=ctime.size)
        myio.saveh5('sio_combine.h5','dyn',dyn,irec=i,nrec=ctime.size)
        print(i,ctime.size)


        #fig,ax=plt.subplots(1,3,sharey=True)
        #ax[0].plot(mct,mczz,'ro')
        #ax[0].plot(a[:,3],a[:,1],'bo')
        #ax[0].set_title('in-situ temperature')
#
#        ax[1].plot(mcs,mczz,'ro')
#        ax[1].plot(a[:,2],a[:,1],'bo')
#        ax[1].set_title('salinity')
#
#        ax[2].plot(mcd,mczz,'ro')
#        ax[2].plot(a[:,4],a[:,1],'bo')
#        ax[2].set_title('potential density')

#        plt.show()





if __name__=='__main__':
    import pylab as plt
    #load_sio_microcat()
    #despike_sio_wirewalker()
    #combine()
    spectrum()
    #qc_rbr()
