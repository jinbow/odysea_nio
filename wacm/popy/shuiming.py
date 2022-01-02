"""
routines related to manipulate data provided by shuiming.
"""

def intp_regional_shuimingsdh(time,lon,lat,pindex=[]):
    """ interpolate the dynamic height field provided by Shuiming to the time,lon,lat grids

    Paramters
    ========
    time: datetime 
          The available time is between 11/01/2011 00:00 and 10/31/2012 23:00
          load the annual mean if time=0

    lon (N): array_like 
          longitudes of the interpolated points
    lat (N): array_like 
          latitudes of the interpolated points
    
    Return
    ======
    D(N) : array_like
          the dynamic height interpolated onto the (time,lon,lat)
    """

    from scipy.io import loadmat
    from scipy.interpolate import RegularGridInterpolator as rgi
    import datetime
    import numpy as np
    import pylab as plt
 
    p=[lon.min(),lon.max(),lat.min(),lat.max()]

    
    lon_dh,lat_dh,dh=load_shuimingsdh(time,p,pindex=pindex)

    if type(lon_dh)==type([]):
        return np.zeros_like(lat)

    dhp=rgi((lat_dh,lon_dh),dh,method='linear',bounds_error=False)((lat,lon))

    #unique_time=np.unique(np.round(time))
    #print "there are %i unique time steps in the altimeter data"%len(unique_time)
    #tim=(datetime.datetime(1,1,1)+datetime.deltatime(hours=time)).strftime('%Y_%m_%d_%H')

    #plt.contourf(lon_dh,lat_dh,dh,30,vmin=-50,vmax=50)
    #plt.scatter(lon,lat,c=dhp,s=30,marker='o',edgecolors='k',vmin=-50,vmax=50)
    #plt.show()

    return dhp


def load_shuimingsdh(time,p,pindex=[]):
    """ Find and load the dynamic height field calculated by Shuiming.
  
    Parameter
    ========
    time: datetime.datetime
          if 0, load the mean field

    p: list of 4
       [lonmin,lonmax,latmin,latmax]

    Returns
    ========
    dh: array_like
       The dynamic height at the time stamp "time" for regions defined by p
    lat: array_like
       The latitude associated with dh
    lon: array_like
       The longitude associated with dh

    """
    from scipy.io import loadmat
    import datetime
    import popy
    import numpy as np

    pth='/nobackup/jwang23/llc4320_stripe/global.dynamic.height.shuiming/dh/'
    if time==np.nan: #missing time stamp
        print "time stamp is ", time, "abort in popy.shuiming.load-shuimingdh"
        return [],[],[]
    if type(time)==type(''):
        fn=pth+'/%s.h5'%time
        print "load data from ",fn
    elif type(time)==type(1):
        if time==0:
            fn=pth[:-3]+'annual_mean_dynamic_height.h5'
    else: 
        fn=pth+'%s.h5'%time.strftime('%Y_%m_%d_%H')
        print "load data from ",fn

    try:
        dd=popy.io.loadh5(fn,'dyn')
    except:
        print "Reading error, could not load ",fn #there's no dynamic heigh file associated with this time stamp
        return [],[],[]

    dh_grid=loadmat('/nobackupp2/schen16/dh_grids_fname.mat',squeeze_me=True)
    dh_x=dh_grid['lon_eta'][:]
    dh_y=dh_grid['lat_eta'][:]

    if pindex==[]:
        print p,dh_y.min(),dh_y.max()
        i0=np.where(dh_x>p[0]-2)[0][0]
        i1=np.where(dh_x<p[1]+2)[0][-1]
        j0=np.where(dh_y>p[2]-2)[0][0]
        j1=np.where(dh_y<p[3]+2)[0][-1]
    else:
        i0,i1,j0,j1=p

    print "The domain for interpolation is ",dh_x[i0],dh_x[i1],dh_y[j0],dh_y[j1]
    print "The index for the domain coordinate is ", i0,i1,j0,j1

    dhx=dh_x[i0:i1]
    dhy=dh_y[j0:j1]
    dh=dd[j0:j1,i0:i1]
    del dd
    return dhx,dhy,dh

