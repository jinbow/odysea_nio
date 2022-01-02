'''
Mapping tools built on Basemap
@author: Jinbo Wang <jinbow@gmail.com>
@Institution: Scripps Institution of Oceanography
Created on Mar 12, 2013
'''

def distance_between_points(lon0, lons, lat0, lats):
    import numpy as np
    # Convert latitude and longitude to 
    # spherical coordinates in radians.
    degrees_to_radians = np.pi/180.0
        
    # phi = 90 - latitude
    phi1 = lat0*degrees_to_radians
    phi2 = lats*degrees_to_radians
    dphi = phi1-phi2
    
    # theta = longitude
    theta1 = lon0*degrees_to_radians
    theta2 = lons*degrees_to_radians
    dtheta=theta1-theta2
    # Compute spherical distance from spherical coordinates.
        
    # For two locations in spherical coordinates 
    # (1, theta, phi) and (1, theta, phi)
    # cosine( arc length ) = 
    #    sin phi sin phi' cos(theta-theta') + cos phi cos phi'
    # distance = rho * arc length
    
    #co = (np.sin(phi1)*np.sin(phi2) + 
    #       np.cos(phi1)*np.cos(phi2)*np.cos(theta1-theta2))
    #arc = np.arccos( co )
    
    #The haversine formula
    co = np.sqrt(np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dtheta/2.0)**2)
    arc = 2* np.arcsin(co)
    dist = arc*6371.0e3
    
    return dist

def setmap(lon=[],lat=[],p=[0,360,-90,90],
           proj='cyl',fix_aspect=False,
           fillcontinents=True,continentcolor='k',
           dlat=10,dlon=30,withtopo=False,
           resolution='l',boundinglat=-10,lon_0=0,latlabels=True,lonlabels=True):
    """ set up a map and return a handle.
    """
    from mpl_toolkits.basemap import Basemap
    import numpy as np
#     if crop !=[]:
#         p = crop
#     elif len(lon) !=0:
#         p[0],p[1],p[2],p[3]=lon.min(),lon.max(),lat.min(),lat.max()
#         
    def mint(p):
        return 5*int(p[1]-p[0])/6/5
   
    if proj == 'sp' or proj == 'np':
        m = Basemap(projection=proj+'stere',boundinglat=boundinglat,lon_0=lon_0,resolution='l')
    else:
        m = Basemap(projection=proj,llcrnrlon=p[0],urcrnrlon=p[1],
                llcrnrlat=p[2],urcrnrlat=p[3],resolution=resolution,fix_aspect=fix_aspect)
    if latlabels and lonlabels:
        labels=[1,0,0,1]
    elif latlabels:
        labels=[1,0,0,0]
    elif lonlabels:
        labels=[0,0,0,1]
    elif not (lonlabels or latlabels):
        labels=[0,0,0,0]
    m.drawparallels(np.arange(-90.,90.,dlat),labels=labels,linewidth=1)
    m.drawmeridians(np.arange(0,361.,dlon),labels=labels,linewidth=1)
    m.drawcoastlines()
    if fillcontinents or continentcolor==None:
        m.fillcontinents(color=continentcolor)
    m.drawmapboundary(fill_color='w')
    return m

def load_bath(maskdepth=-6000,binsize=3,p=None):
    from numpy import ma,roll,argmin
    from netCDF4 import Dataset
    import popy
    ice = '/Users/wangjinb/Project/Eclipse/popy/data//ETOPO1_Ice_c_gmt4.grd'
    ff= Dataset(ice).variables
    x=ff['x'][...]
    y=ff['y'][...]
    z=ff['z'][...]
    z[z<maskdepth]=maskdepth
    nx,ny=x.size,y.size
    nn=binsize
    x=x.reshape(-1,nn).mean(axis=-1)
    y=y.reshape(-1,nn).mean(axis=-1)
    
    z[z>0]=0
    z=ma.masked_equal(z,0)
    z = z.reshape(ny/nn,nn,nx/nn,nn)
    zmask = z.mask.sum(axis=1).sum(axis=-1)
    z=z.mean(axis=1).mean(axis=-1)

    ix = argmin(abs(x))
    x=roll(x,ix)
    x[x<0]=x[x<0]+360.
    z=roll(z,ix,axis=-1)
    
    if p!=None:
        x,y,z = popy.utils.subtractsubdomain(x,y,p,z)
        
    return x,y,z.data 

def bathymetry():
    x,y,z=load_bath(p=[50,110,-70,-30])
    
    return 

def mapwithtopo(p,ax=[],cutdepth=[],aspect=2.5,
                cmap=None,dlon=30,dlat=10,smooth=False,
                lightsource=False,binsize=3,contourlevel=None):
    import popy,os
    from netCDF4 import Dataset
    from mpl_toolkits.basemap import shiftgrid
    from matplotlib.colors import LightSource
    import pylab as plt
    import numpy as np
    
    etopofn='../data/ETOPO1_Bed_c_gmt4.grd'
    etopo = Dataset(etopofn,'r').variables
    x,y,z=etopo['x'][:],etopo['y'][:],etopo['z'][:]
    dx,dy=binsize,binsize
    x=x.reshape(-1,dx).mean(axis=-1)
    y=y.reshape(-1,dx).mean(axis=-1)
    z=z.reshape(y.size,dx,x.size,dx).mean(axis=-1).mean(axis=1)
    if smooth:
        z=popy.utils.smooth2d(z,window_len=3)
    
    if cutdepth!=[]:
        z[z<cutdepth]=cutdepth
        
    if ax==[]:
        fig=plt.figure()
        ax=fig.add_subplot()
    if cmap==None:
        cmap=plt.cm.gist_earth
        
    #z,x = shiftgrid(p[0],z,x,start=True)
    lon,lat,z = popy.utils.subtractsubdomain(x,y,p,z)
    z[z>0]=0
    
    if lightsource:
        m = setmap(p=p,dlon=dlon,dlat=dlat,fillcontinents=False,resolution='i')
        x, y = m(*np.meshgrid(lon, lat))
        ls = LightSource(azdeg=90, altdeg=45)
        rgb = ls.shade(z, cmap=cmap)
        m.imshow(rgb, aspect=aspect)
    else:
        m = setmap(p=p,dlon=dlon,dlat=dlat,fillcontinents=False,resolution='i')
        x, y = m(*np.meshgrid(lon, lat))
        m.imshow(z,cmap=cmap)
        
    m.colorbar(ticks=np.arange(-5000,1,1000))
    if contourlevel!=None:
        m.contour(x,y,z,levels=contourlevel)
    plt.savefig('/tmp/tmp.png',dpi=200)
    os.popen('eog /tmp/tmp.png')
    return m

def polygon(m,coor,facecolor='gray',alpha=0.4):
    """draw a patch on a map m using coordinate in coor
    coor is a [n,2] array, [:,0] contains longitude and [:,1] latitude.
    coor also can be [lon0,lon1,lat0,lat1] for a rectangle box
    popy.map.polygon(m,coor)
    """
    from numpy import array, c_
    from matplotlib.patches import Polygon
    import matplotlib.pylab as plt
    coor = array(coor) # in case array is a list
    if coor.ndim ==1 and coor.size ==4:
        p=coor
        coor = array([[p[0],p[2]],
                      [p[1],p[2]],
                      [p[1],p[3]],
                      [p[0],p[3]] ])
    x,y = m(coor[:,0],coor[:,1])
    poly=Polygon(c_[x,y],facecolor=facecolor,alpha=alpha,closed=True)
    plt.gca().add_patch(poly)
    return

def contourf(data, m=None, lon=[], lat=[], shift=0,
             colorbarticks=[], levels=[], p=[], dlon=30,dlat=10,
             isindex=False,
             extend='both', cmap=[],
             cborientation='horizontal',
             continentcolor='gray'):
    import popy
    from mpl_toolkits.basemap import shiftgrid
    import numpy as np
    import sys
    import pylab as plt
    
    if (lon == [] or lat==[]):
        print('please provide coordinates for the map projection')
        sys.exit()
    
    if shift != 0:
        data, lon = shiftgrid(lon[shift], data, lon, start=False)
    if p==[]:
        p=[0.00,360,-80,80]
    if cmap==[]:
        cmap=plt.cm.jet
    #else:
    #    lon, lat, data = popy.utils.subtractsubdomain(lon, lat, p, data, index=isindex)
    #    print lon.shape, lat.shape, data.shape
    m = popy.map.setmap(lon, lat, dlat=dlat, dlon=dlon, continentcolor=continentcolor,p=p)
    x, y = m(*np.meshgrid(lon, lat))
    if levels == []:
        cs0 = m.contourf(x, y, data, extend=extend, cmap=cmap)
    else:
        cs0 = m.contourf(x, y, data, levels=levels, extend=extend, cmap=cmap)
    if cborientation != None:
        if colorbarticks == []:
            plt.colorbar(cs0, orientation=cborientation, shrink=0.8, pad=.12, aspect=40)
        else:
            plt.colorbar(cs0, orientation=cborientation, ticks=colorbarticks, shrink=0.8, pad=.12, aspect=40)
    return x, y, m
