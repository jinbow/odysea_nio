"""
routines to process roms data

"""

import os
import warnings
import xarray as xr
import numpy as np
from scipy import interpolate as itp
import utils

def load_roms_grid(grd_fn='',domain='9km'):
    """
    load ROMS grid file.

    domain='9km', the 9km domain

    """
    if grd_fn=='':
        if domain=='9km':
            grd_fn='/nobackup/jwang23/projects/MSDA/grid/swot_grd.nc'
        if domain=='3km':
            grd_fn='/nobackup/jwang23/projects/MSDA/grid/swot_grd.nc.1'
        else:
            print('Error, specific grid filename or domain size (9km, 3km)')
            exit()

    data=xr.open_dataset(grd_fn)

    return data

class roms_data:
    """
    roms data class
    Need initial his data filename and grid filename

    Usage:

        d=roms_data(data_fn, grd_fn)

        #interpolate temperature to z-levels:
        d.interpz('temp',np.r_[100,1000])

        #load SSH:
        ssh=d.variable('zeta')
        d.data is identical to ssh
        d.lat, d.lon, d.mask has all necessary information about zeta

    """
    def __init__(self,data_fn,grd_fn='',theta_b=0,theta_s=6,hc=10,N=66,vtrans=1,calc_z=True):
        self.data_fn=data_fn
        if grd_fn=='':
            grd_fn='/nobackup/jwang23/projects/MSDA/grid/swot_grd.nc'
        self.grd_fn=grd_fn
        self.theta_b=theta_b
        self.theta_s=theta_s
        self.hc=hc
        self.N=N
        self.vtrans=vtrans
        self.calc_z=calc_z

        if calc_z:
            aa=gen_zgrid(grd_fn,theta_b,theta_s,hc,N)
            self.z_w,self.z_r,self.z_w_u,self.z_r_u,self.z_w_v,self.z_r_v=aa
            self.calc_z=True

        return

    def variable(self,varn,only_coords=False):
        """load varn data and associated cooridnates"""
        data=xr.open_dataset(self.data_fn)
        grd=xr.open_dataset(self.grd_fn)
        if not self.calc_z:
            self.z_w,self.z_r=gen_zgrid(grd_fn,theta_b,theta_s,hc,N)
            self.calc_z=True

        dic={'temp':['mask_rho','lon_rho','lat_rho','z_r','x_rho','y_rho'],
            'salt':['mask_rho','lon_rho','lat_rho','z_r','x_rho','y_rho'],
            'u':['mask_u','lon_u','lat_u','z_r','x_u','y_u'],
            'ubar':['mask_u','lon_u','lat_u','z_r','x_u','y_u'],
            'v':['mask_v','lon_v','lat_v','z_r','x_v','y_v'],
            'vbar':['mask_v','lon_v','lat_v','z_r','x_v','y_v'],
            'zeta':['mask_rho','lon_rho','lat_rho','z_r','x_rho','y_rho']}

        mskn,lonn,latn,zn,xn,yn=dic[varn][:]

        if not only_coords:
            self.data =data[varn].values
            self.mask=grd[mskn].values
        self.lat =grd[latn].values
        self.lon =grd[lonn].values
        self.lon_name=lonn
        self.lat_name=latn
        self.mask_name=mskn
        self.x=grd[xn].values/1e3
        self.y=grd[yn].values/1e3

        print('x,y shape',self.x.shape,self.y.shape)

        if zn=='z_r':
            self.z   =self.z_r
        else:
            self.z   =self.z_w
        del data, grd
        return


    def interpz(self,data,varn,target_z):
        print("Loading %s"%varn)
        self.variable(varn)
        self.data=data
        print("Interpolating %s to z-levels"%varn)
        data_z=interp_z(self,target_z)
        self.data_z=data_z
        return data_z

    def interp2s_from_z(self,datain,source_z,grid_type=['rho','r']):

        """ interpolate datain onto s-coordinate
            datain: (nt,nz,ny,nx) on z-coordinates
            grid_type: [horizontal, vertical],
                       [rho] for T/S/W grid, 'u' for u grid 'v' for v grid
                       vertical ['r'] for rho point, 'w' for w points

        """

        if grid_type[0]=='rho':
            if grid_type[1]=='r':
                target_z=self.z_r
            else:
                target_z=self.z_w
        elif grid_type[0]=='u':
            if grid_type[1]=='r':
                target_z=self.z_r_u
            else:
                target_z=self.z_w_u
        elif grid_type[0]=='v':
            if grid_type[1]=='r':
                target_z=self.z_r_v
            else:
                target_z=self.z_w_v

        dout=interp2s(datain,source_z,target_z)

        return dout

    def vertical_integrate(self,ddd):
        for aa in [self.z_w,self.z_w_u,self.z_w_v]:
            if ddd.shape[-2:]==aa.shape[-2:]:
                z_w=aa

        drf=np.diff(z_w,axis=0)
        ddd=(ddd*drf).sum(axis=0)/drf.sum(axis=0)
        return ddd
    def load_grid(self,grid='xy'):
        if grid=='xy':
            with xr.open_dataset(self.grd_fn) as ff:
                self.x=ff['x_rho'].values/1e3
                self.y=ff['y_rho'].values/1e3
        else:
            with xr.open_dataset(self.grd_fn) as ff:
                self.lon=(ff['lon_rho'].values+360)%360
                self.lat=ff['y_rho'].values

    def interp2xy(self,dd,dx,varn='zeta',method='linear'):
        """interpret data (dd) onto new uniform grid with grid spacing given in dx

        dd: 2D numpy array

        return
            interpolated 2D field
        """
        if varn=='Eta':varn='zeta'
        if varn=='Theta':varn='temp'
        if varn=='Salt':varn='salt'
        if varn=='U':varn='u'
        if varn=='V':varn='v'
        self.variable('u',only_coords=True)
        xmax=self.x.max()
        self.variable('v',only_coords=True)
        ymax=self.y.max()
        self.variable(varn,only_coords=True)
        self.variable(varn,only_coords=True)
        x=self.x
        y=self.y
        tx=np.arange(0,xmax,dx)
        ty=np.arange(0,ymax,dx)
        target_grid={'unit':'km','grid':'xy','y':ty,'x':tx}

        dd=np.ma.masked_equal(dd,0)
        dd=np.ma.masked_invalid(dd)
        dd.x=x;dd.y=y
        print('data to be interpolated has shape',dd.shape)
        dout=utils.interp2d(dd,target_grid,coarsen_grid_n=0,method=method)
        return dout
    def interp3d(self,varn,target_grid,method='linear'):
        """interpolate a variable onto 3D field, with uniform horizonal grid for later spectrum analysis.
        varn: string
           temp,salt,u,v,w,zeta
        target_grid: dic
           {'z':z,'dx':5km}
        method: 'linear' or 'nearest'
        """

    #first interpolate to z if not zeta
        if varn !='zeta':
            tmp=self.interpz(varn,target_grid['z'])
            tmp=self.interp2xy(tmp,target_grid['dx'],varn,method)
        else:
            tmp=self.variable(varn)
            tmp=self.interp2xy(self.data,target_grid['dx'],varn,method)
        return tmp
    def roms2roms(self,newgridfn,varns):
        """
        interpolate variables in varns to new roms grid.
        used to interpolate nested domain from larger to smaller to generate initial conditions.
        newgridfn: the filename of the new roms grid
        varns: list, variable names

        """
        ngrd=xr.open_dataset(newgridfn)
        dout={}
        for varn in varns:
            self.variable(varn)
            dd=np.ma.masked_less(self.data,-100)
            dd=np.ma.masked_invalid(dd)
            dd=np.ma.masked_equal(dd,0)
            dd.lon=self.lon
            dd.lat=self.lat
            target={'unit':'degree','lon':ngrd[self.lon_name].values.squeeze(),'lat':ngrd[self.lat_name].values.squeeze()}
            dout[varn]=utils.interp2d(dd,target,coarsen_grid_n=0,method='nearest')
        dout['lon']=target['lon'][:]
        dout['lat']=target['lat'][:]
        dout['lat_name']=self.lat_name
        dout['lon_name']=self.lon_name
        return dout

def gen_zgrid(grd_fn,theta_b=0,theta_s=6,hc=10,N=66):
    """ calculate the depth for the s-coordinate
    grd_fn: grid file that contains h the bottom depth
    theta_b, theta_s,hc, ROMS s-coordinate parameters
    N: the number of z levels
    """

    if type(grd_fn)==str:
        dd=xr.open_dataset(grd_fn)
    elif type(grd_fn)==np.ndarray:
        dd=grd_fn
    else:
        print(100*'#')
        print('Error in the input file. grd_fn is either the grid filename or an ndarry of bottom depth')
        exit()

    hh=dd['h'].values.squeeze()
    ss=s_coordinate(hh,theta_b,theta_s,hc,N,)
    ssu=s_coordinate((hh[:,1:]+hh[:,:-1])/2.0,theta_b,theta_s,hc,N,)
    ssv=s_coordinate((hh[1:,:]+hh[:-1,:])/2.0,theta_b,theta_s,hc,N,)

    #z_w_index=np.arange(ss.z_w.shape[1])
    #z_r_index=np.arange(ss.z_r.shape[1])+0.5

    #z_w=xr.DataArray(ss.z_w.squeeze(), name='z_w',coords=[z_w_index,dd['eta_rho'].values,dd['xi_rho'].values],dims=['s_w','eta_rho','xi_rho'])
    #z_r=xr.DataArray(ss.z_r.squeeze(), name='z_r',coords=[z_r_index,dd['eta_rho'].values,dd['xi_rho'].values],dims=['s_r','eta_rho','xi_rho'])

    #dout=xr.merge([z_w,z_r])

    #dout.to_netcdf(grd_fn+'.z')
    z_w,z_r=ss.z_w.squeeze(), ss.z_r.squeeze()
    return z_w, z_r, ssu.z_w.squeeze(),ssu.z_r.squeeze(),ssv.z_w.squeeze(),ssv.z_r.squeeze()

def fill_corner(d,mask):

    """d is a 2D array"""

    ##fill the corner values
    mean=d[~mask].mean()
    for i in [0,-1]:
        for j in [0,-1]:
            if mask[j,i]:
                d[j,i]=mean #fill the corner values
                mask[j,i]=False
    return d,mask

def interp2s(datain,source_z,target_z):
    """
    interpolate data on z levels to s-coordinate

    datain, (nt,nz,ny,nx) the data on z levels to be interpolated
    source_z, (nz) or (nz,ny,nx) the z levels for datain
    target_z, (nz0,ny,nx) the z levels for the interpolated field, usually the depth info for s-coordinate
    """

    if datain.ndim==3:
        data=datain[np.newaxis,...]
        print("expand data into four dimensions. New data has size:",data.shape)
    else:
        data=datain

    print(datain)

    nt,nz,ny,nx=datain.shape

    source_z=np.abs(source_z)
    target_z=np.abs(target_z)
    if source_z.ndim==1:
        source_z = np.ones((source_z.size,ny,nx))*source_z.reshape(-1,1,1)

    nz0,ny,nx=target_z.shape

    dout=np.zeros((nt,nz0,ny,nx))
    print("Loop through all vertical profiles. It may take a while")
    for t in range(nt):
        for j in range(ny):
            for i in range(nx):
                   msk=~datain.mask[t,:,j,i]
                   if msk.sum()>2:
                       dout[t,:,j,i]=itp.interp1d(np.abs(source_z[msk,j,i]),datain[t,msk,j,i],fill_value='extrapolate',bounds_error=False)(target_z[:,j,i])

    print("Interpolation is done")
    return dout

def interp_z(droms,target_z=False):
    if type(target_z)==bool:
        print("Load default vertical grid from llc4320")
        z=np.abs(np.fromfile('/u/dmenemen/llc_4320/grid/RC.data','>f4'))
    else:
        z=np.abs(target_z)

    if droms.data.ndim==3:
        data=droms.data[np.newaxis,...]
        print("expand data into four dimensions. New data has size:",data.shape)
    else:
        data=droms.data

    nt,nz,ny,nx=data.shape

    zroms=np.abs(droms.z)
    zroms[-1,...]=0

    dout=np.zeros((nt,z.size,ny,nx))
    print("Loop through all vertical profiles. It may take a while")
    print(zroms.min(),zroms.max(),z.min(),z.max())
    for t in range(nt):
        for j in range(ny):
            for i in range(nx):
                   dout[t,:,j,i]=itp.interp1d(np.abs(zroms[:,j,i]),data[t,:,j,i],fill_value=np.nan,bounds_error=False)(z)

    print("Interpolation is done")
    return dout



def interp_roms(data_fn,varns=['temp']):
    data=load_his(data_fn,'u')
    dd=interp_z(data,)


class s_coordinate(object):
    """
    Song and Haidvogel (1994) vertical coordinate transformation (Vtransform=1) and
    stretching functions (Vstretching=1).

    return an object that can be indexed to return depths

    s = s_coordinate(h, theta_b, theta_s, Tcline, N)
    """

    def __init__(self, h, theta_b, theta_s, Tcline, N, hraw=None, zeta=None):
        self.hraw = hraw
        self.h = np.asarray(h)
        self.hmin = h.min()
        self.theta_b = theta_b
        self.theta_s = theta_s
        self.Tcline = Tcline
        self.N = int(N)
        self.Np = self.N+1

        self.hc = min(self.hmin, self.Tcline)

        self.Vtrans = 1

        if (self.Tcline > self.hmin):
            warnings.warn('Vertical transformation parameters are not defined correctly in either gridid.txt or in the history files: \n Tcline = %d and hmin = %d. \n You need to make sure that Tcline <= hmin when using transformation 1.' %(self.Tcline,self.hmin))

        self.c1 = 1.0
        self.c2 = 2.0
        self.p5 = 0.5

        if zeta is None:
            self.zeta = np.zeros(h.shape)
        else:
            self.zeta = zeta

        self._get_s_rho()
        self._get_s_w()
        self._get_Cs_r()
        self._get_Cs_w()

        self.z_r = get_z_r(self.h, self.hc, self.N, self.s_rho, self.Cs_r, self.zeta, self.Vtrans)
        self.z_w = get_z_w(self.h, self.hc, self.Np, self.s_w, self.Cs_w, self.zeta, self.Vtrans)


    def _get_s_rho(self):
        lev = np.arange(1,self.N+1,1)
        ds = 1.0 / self.N
        self.s_rho = -self.c1 + (lev - self.p5) * ds

    def _get_s_w(self):
        lev = np.arange(0,self.Np,1)
        ds = 1.0 / (self.Np-1)
        self.s_w = -self.c1 + lev * ds

    def _get_Cs_r(self):
        if (self.theta_s >= 0):
            Ptheta = np.sinh(self.theta_s * self.s_rho) / np.sinh(self.theta_s)
            Rtheta = np.tanh(self.theta_s * (self.s_rho + self.p5)) / \
                      (self.c2 * np.tanh(self.p5 * self.theta_s)) - self.p5
            self.Cs_r = (self.c1 - self.theta_b) * Ptheta + self.theta_b * Rtheta
        else:
            self.Cs_r = self.s_rho

    def _get_Cs_w(self):
        if (self.theta_s >= 0):
            Ptheta = np.sinh(self.theta_s * self.s_w) / np.sinh(self.theta_s)
            Rtheta = np.tanh(self.theta_s * (self.s_w + self.p5)) / \
                      (self.c2 * np.tanh(self.p5 * self.theta_s)) - self.p5
            self.Cs_w = (self.c1 - self.theta_b) * Ptheta + self.theta_b * Rtheta
        else:
            self.Cs_w = self.s_w



class s_coordinate_2(s_coordinate):
    """
    A. Shchepetkin (2005) UCLA-ROMS vertical coordinate transformation (Vtransform=2) and
    stretching functions (Vstretching=2).

    return an object that can be indexed to return depths

    s = s_coordinate_2(h, theta_b, theta_s, Tcline, N)
    """

    def __init__(self, h, theta_b, theta_s, Tcline, N, hraw=None, zeta=None):
        self.hraw = hraw
        self.h = np.asarray(h)
        self.hmin = h.min()
        self.theta_b = theta_b
        self.theta_s = theta_s
        self.Tcline = Tcline
        self.N = int(N)
        self.Np = self.N+1

        self.hc = self.Tcline

        self.Vtrans = 2

        self.Aweight = 1.0
        self.Bweight = 1.0

        self.c1 = 1.0
        self.c2 = 2.0
        self.p5 = 0.5

        if zeta is None:
            self.zeta = np.zeros(h.shape)
        else:
            self.zeta = zeta

        self._get_s_rho()
        self._get_s_w()
        self._get_Cs_r()
        self._get_Cs_w()

        self.z_r = get_z_r(self.h, self.hc, self.N, self.s_rho, self.Cs_r, self.zeta, self.Vtrans)
        self.z_w = get_z_w(self.h, self.hc, self.Np, self.s_w, self.Cs_w, self.zeta, self.Vtrans)


    def _get_s_rho(self):
        super(s_coordinate_2, self)._get_s_rho()

    def _get_s_w(self):
        super(s_coordinate_2, self)._get_s_w()

    def _get_Cs_r(self):
        if (self.theta_s >= 0):
            Csur = (self.c1 - np.cosh(self.theta_s * self.s_rho)) / \
                     (np.cosh(self.theta_s) - self.c1)
            if (self.theta_b >= 0):
                Cbot = np.sinh(self.theta_b * (self.s_rho + self.c1)) / \
                       np.sinh(self.theta_b) - self.c1
                Cweight = (self.s_rho + self.c1)**self.Aweight * \
                          (self.c1 + (self.Aweight / self.Bweight) * \
                          (self.c1 - (self.s_rho + self.c1)**self.Bweight))
                self.Cs_r = Cweight * Csur + (self.c1 - Cweight) * Cbot
            else:
                self.Cs_r = Csur
        else:
            self.Cs_r = self.s_rho

    def _get_Cs_w(self):
        if (self.theta_s >= 0):
            Csur = (self.c1 - np.cosh(self.theta_s * self.s_w)) / \
                     (np.cosh(self.theta_s) - self.c1)
            if (self.theta_b >= 0):
                Cbot = np.sinh(self.theta_b * (self.s_w + self.c1)) / \
                       np.sinh(self.theta_b) - self.c1
                Cweight = (self.s_w + self.c1)**self.Aweight * \
                          (self.c1 + (self.Aweight / self.Bweight) * \
                          (self.c1 - (self.s_w + self.c1)**self.Bweight))
                self.Cs_w = Cweight * Csur + (self.c1 - Cweight) * Cbot
            else:
                self.Cs_w = Csur
        else:
            self.Cs_w = self.s_w


class s_coordinate_4(s_coordinate):
    """
    A. Shchepetkin (2005) UCLA-ROMS vertical coordinate transformation (Vtransform=2) and
    stretching functions (Vstretching=4).

    return an object that can be indexed to return depths

    s = s_coordinate_4(h, theta_b, theta_s, Tcline, N)
    """

    def __init__(self, h, theta_b, theta_s, Tcline, N, hraw=None, zeta=None):
        self.hraw = hraw
        self.h = np.asarray(h)
        self.hmin = h.min()
        self.theta_b = theta_b
        self.theta_s = theta_s
        self.Tcline = Tcline
        self.N = int(N)
        self.Np = self.N+1

        self.hc = self.Tcline

        self.Vtrans = 4

        self.c1 = 1.0
        self.c2 = 2.0
        self.p5 = 0.5

        if zeta is None:
            self.zeta = np.zeros(h.shape)
        else:
            self.zeta = zeta

        self._get_s_rho()
        self._get_s_w()
        self._get_Cs_r()
        self._get_Cs_w()

        self.z_r = get_z_r(self.h, self.hc, self.N, self.s_rho, self.Cs_r, self.zeta, self.Vtrans)
        self.z_w = get_z_w(self.h, self.hc, self.Np, self.s_w, self.Cs_w, self.zeta, self.Vtrans)


    def _get_s_rho(self):
        super(s_coordinate_4, self)._get_s_rho()

    def _get_s_w(self):
        super(s_coordinate_4, self)._get_s_w()

    def _get_Cs_r(self):
        if (self.theta_s > 0):
            Csur = (self.c1 - np.cosh(self.theta_s * self.s_rho)) / \
                     (np.cosh(self.theta_s) - self.c1)
        else:
            Csur = -self.s_rho**2
        if (self.theta_b > 0):
            Cbot = (np.exp(self.theta_b * Csur) - self.c1 ) / \
                   (self.c1 - np.exp(-self.theta_b))
            self.Cs_r = Cbot
        else:
            self.Cs_r = Csur

    def _get_Cs_w(self):
        if (self.theta_s > 0):
            Csur = (self.c1 - np.cosh(self.theta_s * self.s_w)) / \
                     (np.cosh(self.theta_s) - self.c1)
        else:
            Csur = -self.s_w**2
        if (self.theta_b > 0):
            Cbot = (np.exp(self.theta_b * Csur) - self.c1 ) / \
                   ( self.c1 - np.exp(-self.theta_b) )
            self.Cs_w = Cbot
        else:
            self.Cs_w = Csur


class s_coordinate_5(s_coordinate):
    """
    A. Shchepetkin (2005) UCLA-ROMS vertical coordinate transformation (Vtransform=2) and
    stretching functions (Vstretching=5).

    return an object that can be indexed to return depths

    s = s_coordinate_5(h, theta_b, theta_s, Tcline, N)

    Brian Powell's surface stretching.
    """

    def __init__(self, h, theta_b, theta_s, Tcline, N, hraw=None, zeta=None):
        self.hraw = hraw
        self.h = np.asarray(h)
        self.hmin = h.min()
        self.theta_b = theta_b
        self.theta_s = theta_s
        self.Tcline = Tcline
        self.N = int(N)
        self.Np = self.N+1

        self.hc = self.Tcline

        self.Vtrans = 5

        self.c1 = 1.0
        self.c2 = 2.0
        self.p5 = 0.5

        if zeta is None:
            self.zeta = np.zeros(h.shape)
        else:
            self.zeta = zeta

        self._get_s_rho()
        self._get_s_w()
        self._get_Cs_r()
        self._get_Cs_w()

        self.z_r = get_z_r(self.h, self.hc, self.N, self.s_rho, self.Cs_r, self.zeta, self.Vtrans)
        self.z_w = get_z_w(self.h, self.hc, self.Np, self.s_w, self.Cs_w, self.zeta, self.Vtrans)

    def _get_s_rho(self):
        lev = np.arange(1, self.N+1) - .5
        self.s_rho = -(lev * lev - 2 * lev * self.N + lev + self.N * self.N - self.N) / \
            (1.0 * self.N * self.N - self.N) - \
            0.01 * (lev * lev - lev * self.N) / (1.0 - self.N)

    def _get_s_w(self):
        lev = np.arange(0,self.Np,1)
        s = -(lev * lev - 2 * lev * self.N + lev + self.N * self.N - self.N) / \
            (self.N * self.N - self.N) - \
            0.01 * (lev * lev - lev * self.N) / (self.c1 - self.N)
        self.s_w = s

    def _get_Cs_r(self):
        if self.theta_s > 0:
            csur = (self.c1 - np.cosh(self.theta_s * self.s_rho)) / \
                (np.cosh(self.theta_s) - self.c1)
        else:
            csur = -(self.s_rho * self.s_rho)
        if self.theta_b > 0:
            self.Cs_r = (np.exp(self.theta_b * (csur + self.c1)) - self.c1) / \
                (np.exp(self.theta_b) - self.c1) - self.c1
        else:
            self.Cs_r = csur

    def _get_Cs_w(self):
        if self.theta_s > 0:
            csur = (self.c1 - np.cosh(self.theta_s * self.s_w)) / \
                (np.cosh(self.theta_s) - self.c1)
        else:
            csur = -(self.s_w * self.s_w)
        if self.theta_b > 0:
            self.Cs_w = (np.exp(self.theta_b * (csur + self.c1)) - self.c1) / \
                (np.exp(self.theta_b) - self.c1) - self.c1
        else:
            self.Cs_w = csur


def get_z_r( h, hc, N, s_rho, Cs_r, zeta, Vtrans):
    """
    return an object that can be indexed to return depths of rho point

    z_r = get_z_r(h, hc, N, s_rho, Cs_r, zeta, Vtrans)
    """
    if True:
        if h.ndim == zeta.ndim:       # Assure a time-dimension exists
            zeta = zeta[np.newaxis, :]

        ti = zeta.shape[0]
        z_r = np.empty((ti, N) + h.shape, 'd')
        if Vtrans == 1:
            for n in range(ti):
                for  k in range(N):
                    z0 = hc * s_rho[k] + (h - hc) * Cs_r[k]
                    z_r[n,k,:] = z0 + zeta[n,:] * (1.0 + z0 / h)
        elif Vtrans == 2 or Vtrans == 4 or Vtrans == 5:
            for n in range(ti):
                for  k in range(N):
                    z0 = (hc * s_rho[k] + h * Cs_r[k]) / \
                          (hc + h)
                    z_r[n,k,:] = zeta[n,:] + (zeta[n,:] + h) * z0

        return z_r


def get_z_w(h, hc, Np, s_w, Cs_w, zeta, Vtrans):
    """
    return an object that can be indexed to return depths of w point

    z_w = get_z_w(h, hc, Np, s_w, Cs_w, zeta, Vtrans)
    """
    if True:
        if h.ndim == zeta.ndim:       # Assure a time-dimension exists
            zeta = zeta[np.newaxis, :]

        ti = zeta.shape[0]
        z_w = np.empty((ti, Np) + h.shape, 'd')
        if Vtrans == 1:
            for n in range(ti):
                for  k in range(Np):
                    z0 = hc * s_w[k] + (h - hc) * Cs_w[k]
                    z_w[n,k,:] = z0 + zeta[n,:] * (1.0 + z0 / h)
        elif Vtrans == 2 or Vtrans == 4 or Vtrans == 5:
            for n in range(ti):
                for  k in range(Np):
                    z0 = (hc * s_w[k] + h * Cs_w[k]) / \
                          (hc + h)
                    z_w[n,k,:] = zeta[n,:] + (zeta[n,:] + h) * z0

        return z_w



class z_coordinate(object):
    """
    return an object that can be indexed to return depths

    z = z_coordinate(h, depth, N)
    """

    def __init__(self, h, depth, N):
        self.h = np.asarray(h)
        self.N = int(N)

        ndim = len(h.shape)
#       print(h.shape, ndim)

        if ndim == 2:
            Mm, Lm = h.shape
            self.z = np.zeros((N, Mm, Lm))
        elif ndim == 1:
            Sm = h.shape[0]
            self.z = np.zeros((N, Sm))

        for k in range(N):
            self.z[k,:] = depth[k]

if __name__=="__main__":

    pth='/home1/jwang23/MSDA/foreward-level0/result/rst_noda/'
    fn=pth+'2012060103_rst.nc'
    grd_fn='/home1/jwang23/MSDA/foreward-level0/grid/swot_grd.nc'
    dd=roms_data(fn,grd_fn)
    target_z=np.fromfile('/u/dmenemen/llc_4320/grid/RC.data','>f4')[:87]
    dd.interpz('temp',target_z)

