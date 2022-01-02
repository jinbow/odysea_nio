"""
functions dealing with CTD measurements
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

def dyn(s,t,p,ispressure=True,lat=35,lon=-124,rho0=1024.0,pr=0):
    """
    Calculate the upper ocean dynamic height from T/S.
    the T/S profiles are interpolated onto 2m-grid before the calculation.
    """
    from scipy.interpolate import interp1d
    import numpy as np
    z=np.abs(z)
    dz=2.0
    newz=np.arange(0.1,z.max(),dz)
    news=interp1d(z,s,bounds_error=False,fill_value=nan)(newz)
    newt=interp1d(z,t,bounds_error=False,fill_value=nan)(newz)
    pden=c_rho(news,newt,newz,ispressure=ispressure,pr=pr)
    dyn=(pden-rho0).sum()*dz/rho0
    return dyn

