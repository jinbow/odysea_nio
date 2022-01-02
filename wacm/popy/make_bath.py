from numpy import *

from netCDF4 import Dataset

from pylab import *
import popy
import os

def load_bath():
    ice = '/Users/jinbo/Project/Collection/OBS/ETOPO1/ETOPO1_Ice_c_gmt4.grd'
    ice = '/net/mazdata2/jinbo/mdata5-jinbo/obs/ETOPO/ETOPO1_Ice_c_gmt4.grd'
    print os.path.exists(ice)
    ff= Dataset(ice).variables
    print ff.keys()
    x=ff['x'][...]
    y=ff['y'][...]
    z=ff['z'][...]
    z[z<-5700]=-5700
    #ix = argmin(abs(x))
    #x=roll(x,ix)
    #x[x<0]=x[x<0]+360.
    #z=roll(z,ix,axis=-1)
    return x,y,z

def bin_bath():
    x,y,z=load_bath()
    x=x.reshape(360,60).mean(axis=-1)
    y=y.reshape(180,60).mean(axis=-1)
    
    z[z>0]=0
    z=ma.masked_equal(z,0)
    z = z.reshape(180,60,360,60)
    zmask = z.mask.sum(axis=1).sum(axis=-1)
    z=ma.median(ma.median(z,axis=1),axis=-1)
    z[zmask>1800]=0
    
    return x,y,z.data

def gaussian_smooth(smd=1):
    
    x,y,z=load_bath()
    
    dd=100
    
    z = c_[z[:,-dd*smd:],z,z[:,:dd*smd]]
    dx=x[1]-x[0]
    x = r_[arange(-smd*dd,0,1)*dx,x,arange(1,smd*dd+1)*dx+x[-1]]
    
    print x.shape,z.shape
    
    print x
    
    xs = arange(0.5,360,1)
    ys = arange(-79.5,80,1)
    
    zout = zeros((len(ys),len(xs)))
    ny,nx=zout.shape
    mask = zeros_like(zout)
    
    xx,yy = meshgrid(arange(-dd*smd,dd*smd)/float(dd*smd),arange(-dd*smd,dd*smd)/float(dd*smd))
    print xx.min(),xx.max()
    weight = exp(-(xx**2+yy**2)/0.5)
    #contourf(weight)
    #show()
    print weight.shape, 
    for i in range(len(xs)):
        ix = argmin(abs(x-xs[i]))
        for j in range(len(ys)):
            iy = argmin(abs(y-ys[j]))
            zz=z[s_[iy-dd*smd:iy+dd*smd],s_[ix-dd*smd:ix+dd*smd]]
            if (zz>0).sum()>size(zz)/3:
                zout[j,i]=0
            else: 
                zout[j,i]=(zz[zz<0]*weight[zz<0]).sum()/weight[zz<0].sum()
                mask[j,i]=1
    
    return xs,ys,zout

def smooth_griddata():
    from scipy.interpolate import griddata
    
    x,y,z=bin_bath()
    ip=(z==0)
    x,y=np.meshgrid(x,y)
    z[z>-50]=-50
    newz=griddata((x.flatten(),y.flatten()),z.flatten(),(x,y),method='cubic')
    
    plt.pcolor(x,y,newz)
    plt.show()
    return

def load_smoothed():
    import os
    if os.path.exists('z_180x360.bin'):
        
        zout=fromfile('z_180x360.bin',dtype='>f4').reshape(-1,360)
        xs=fromfile('x_180x360.bin',dtype='>f4')
        ys=fromfile('y_180x360.bin',dtype='>f4')
    else:
        xs,ys,zout = gaussian_smooth()
        
    return xs,ys,zout

if __name__=='__main__':
    smooth_griddata()
    #x,y,z=gaussian_smooth()
    #plt.pcolor(z)
    
#     
#     x,y,z = load_smoothed()
#     mask = 1-popy.plt.image2mask('landmask_1x1.tif')
#     z = array(imread('landmask_1x1_smooth.tif')).astype('>f4')
#     z=(z/z.max()-1)*5650 -50
#     
#     print z.min(),z.max()
#     print "save data to bathymetry"
#     (z*mask).astype('>f4').tofile('bathymetry_etopo1_smooth_1x1.bin')
#     z
#     #popy.plt.image4mask(z,'landmask_1x1_smooth.tif')
#     subplot(211)
#     pcolor(x,y,z)
#     colorbar()
#     subplot(212)
#     pcolor(x,y,ma.masked_equal(z*mask,0))
#     colorbar()
#     savefig('bathymetry_etopo1_smooth_1x1.png')
#     #pcolor(x,y,ma.masked_equal(z,0))
# 
#     #colorbar()

    #show()
