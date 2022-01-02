"""
Calculate dynamic height from CTD T/S profiles.
Prepared for Bruce Haines by Jinbo Wang.
02/02/2020 @JPL

This program needs numpy, gsw, scipy python package and has been tested using python3.

"""


def c_rho(s,t,p,ispressure=True,lat=36.7,lon=-122.0335,pr=0):
    """
    Calculate potential density from practical
    salinity, in-situ temperature and pressure.

    Parameters
    ==========
    s: array
       practical salinity
    t: array
       in-situ temperature
    p: array
       pressure or depth
    pr float
       reference level
    lat: float
       latitude of the measurement

    """

    import gsw
    import numpy as np
    p=np.abs(p)
    if not ispressure:
        p=gsw.p_from_z(-p,lat)
    sa=gsw.SA_from_SP(s,p,lon,lat)
    rho=gsw.pot_rho_t_exact(sa,t,p,pr)

    return rho

def c_dyn(s,t,p,ispressure=True,lat=35,lon=-124,rho0=1024.0,pr=0):
    """
    Calculate the upper ocean dynamic height from T/S.
    the T/S profiles are interpolated onto 2m-grid before the calculation.
    """
    from scipy.interpolate import interp1d
    import numpy as np
    z=np.abs(p)
    dz=2.0
    newz=np.arange(5,z.max(),dz)
    news=interp1d(z,s,bounds_error=False,fill_value='extrapolate')(newz)
    newt=interp1d(z,t,bounds_error=False,fill_value='extrapolate')(newz)
    pden=c_rho(news,newt,newz,ispressure=ispressure,pr=pr)
    dyn=(pden-rho0).sum()*dz/rho0
    return dyn

def calc(fn):
    da=np.loadtxt(fn,comments='#')
    print('Data %s has %i rows and %i columns'%(fn,da.shape[0],da.shape[1]))
    idmax=da[:,0].max().astype('i')
    dyn=np.zeros((idmax))
    tout=np.zeros((idmax,))
    idd=da[:,0]
    #loop through all profiles
    for i in range(idmax):
        print('calculating dynamic height for profile %i out of total %i'%(i,idmax))
        msk=idd==i
        time=da[msk,1]
        press=da[msk,2]
        salt=da[msk,6]
        temp=da[msk,5]
        dyn[i]=c_dyn(salt,temp, press)
        tout[i]=time.mean()

    fnout=fn+'.dynamicHeight'
    np.savetxt(fnout,dyn,fmt='%10.6f',comments='#',header='dynamic height calculated from data in %s'%fn)
    print('saved dynamic height to %s'%fnout)
    return tout,dyn

if __name__=="__main__":
    import sys
    import numpy as np
    import pylab as plt
    time,dyn=calc(sys.argv[1])
    plt.plot(time,dyn)
    plt.xlabel('time')
    plt.ylabel('dynamic height (meter)')
    plt.show()



