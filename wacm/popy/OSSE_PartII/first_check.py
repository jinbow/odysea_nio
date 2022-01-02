import xarray as xr
from pylab import *
import datetime
from matplotlib.gridspec import GridSpec
import sys

a=sys.argv[1]
yy,mm,dd=int(a[:4]),int(a[4:6]),int(a[6:])

t0=datetime.datetime(2012,6,6)
t1=datetime.datetime(2012,6,28)
t1=datetime.datetime(yy,mm,dd)

dt=(t1-t0).days
print(dt)

fn0='result/avg_ms/%s03_avg.nc'%t1.strftime('%Y%m%d')
fn1='result/remote_obs/ssh_swot/%s03_swot_grd.nc'%t1.strftime('%Y%m%d')
fn2='/home1/jwang23/9km_domain/dyn.1600m.bias.free/dyn1600m_bias_free_constant.nc_on_swot_grd.nc.1.nc'


d0=xr.open_dataset(fn0)['zeta'].values.squeeze()
d1=xr.open_dataset(fn1)['ssh'].values.squeeze()
d2=xr.open_dataset(fn2)['zeta'].values[dt,...]

lev=linspace(0.3,0.7,30)
fig, axes = plt.subplots(nrows=2, ncols=3)

ax=axes[0,0]
ax.contourf(d0,levels=lev,extend='both',cmap=cm.jet)
ax=axes[0,1]
ax.contourf(d1,levels=lev,extend='both',cmap=cm.jet)
ax=axes[0,2]
im=ax.contourf(d2,levels=lev,extend='both',cmap=cm.jet)

fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.95, 0.55, 0.02, 0.4])
fig.colorbar(im, cax=cbar_ax)

fn1='result/remote_obs/sst_gridded/%s03.swot_grd.nc'%t1.strftime('%Y%m%d')
fn2='/home1/jwang23/9km_domain/daily.v008/%s.nc'%t1.strftime('%Y%m%d')

d0=xr.open_dataset(fn0)['temp'].values[0,-1,...].squeeze()
d1=xr.open_dataset(fn1)['sst'].values.squeeze()
d2=xr.open_dataset(fn2)['Theta'][0,0,...].values

lev=linspace(9,19,40)
ax=axes[1,0]
d0[d0==0]=np.nan
ax.imshow(d0[::-1,:],cmap=cm.jet)
ax=axes[1,1]
d1[np.isnan(d0)]=np.nan
im=ax.imshow(d1[::-1,:],cmap=cm.jet)
ax=axes[1,2]
im=ax.imshow(d2[::-1,:],cmap=cm.jet)

fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.95, 0.05, 0.02, 0.4])
fig.colorbar(im, cax=cbar_ax)

show()
