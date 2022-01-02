# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 14:26:46 2016

@author: Jinbo Wang
Some routines to manipulate the llc outputs on Plateaus computer.
"""
import pylab as plt
import numpy as np
import popy,sys


def missing_PhiBot():
    """the following files has zeros in PhiBot"""
    a=[]

    b=open('/home1/jwang23/local/popy/popy/phibot_zeros.asc').readlines()
    for c in b:
        a.append(int(c.split('.')[1]))

    return a


def load_compressed_2d(varn,k=0,time=None,t_index=None,remapping=False):
    """load compressed file from /u/dmenemen/llc_4320/compressed.
    Parameters:
    ===========
    time: datetime.datetime
    varn: the variable name
    remapping: bool, False (default) to return compressed unstructured 1D array; True to return 2D array mapped to lat-lon grid with shape 4320*3 by 4320*4
    k: the level to load

    Return:
    =======
    dout: 1d numpy array

    """

    import datetime
    import numpy as np

    missing=missing_PhiBot()

    if time!=None:
        t0=datetime.datetime(2011,9,13)
        dt=(time-t0).total_seconds()/3600. * 144 + 10368 #get the time step for reading the file
    elif t_index!=None:
        dt=10368+t_index*144
    else:
        raise ValueError('specify time record either through datetime or t_index')

    if varn=='PhiBot' and dt in missing:
        fn='/nobackup/jwang23/llc4320_stripe/missing.PhiBot.compressed/PhiBot.%010i.data'%dt
        print("load data from /nobackup/jwang23/llc4320_stripe/missing.PhiBot.compressed for zero PhiBot")
    else:
        fn='/u/dmenemen/llc_4320/compressed/%010i/%s.%010i.data.shrunk'%(dt,varn,dt)

    """
    read a layer from compressed file.
    First find the offset, and the shape.
    Then map them to 13*4320**2

    """
    def get_offset(k):
        pth='/u/jwang23/popy/llc4320/'
        if varn=='U':
            fn=pth+'/data_index.hFacW.txt'
        elif varn=='V':
            fn=pth+'/data_index.hFacS.txt'
        else:
            fn=pth+'/data_index.hFacC.txt'
        d=np.loadtxt(fn)
        tt,offset,shape=d[k,:]
        return int(offset),int(shape)

    offset,shape=get_offset(k)
    dout=np.memmap(fn,dtype='>f4',offset=offset*4,shape=shape,mode='r')

    if remapping:
        loc='C'
        if varn=='U':loc='W'
        if varn=='V':loc='S'
        msk =load_hfac(k,loc)
        print((msk==1).sum(),dout.size)
        msk[msk==1]=dout
        del dout
        dout=msk

    return dout

def load_hfac(k,loc='C'):
    import numpy as np
    fn0='/u/dmenemen/llc_4320/compressed/hFac%s.data'%loc
    nn=4320**2
    offset=(k*nn*13)*4
    shape=(nn*13)
    df=np.memmap(fn0,dtype='>f4',offset=offset,mode='r',shape=shape)
    dd=df.copy()
    del df
    return dd
    d=np.fromfile(fn,dtype='>f4')
    if remapping:
        mask=np.memmap('/home1/jwang23/nobackup/llc4320_stripe/grid/hFacC.mask.k0.dtype_i1.data',dtype='>i1',mode='r')
        dout=np.zeros((mask.size))
        dout[mask==1]=d
        del mask,d
        return dout

    return d


def load_llc_vertical_grid():
    rc=np.fromfile('/u/dmenemen/llc_4320/grid/RC.data','>f4')
    drf=np.fromfile('/u/dmenemen/llc_4320/grid/DRF.data','>f4')
    rf=np.fromfile('/u/dmenemen/llc_4320/grid/RF.data','>f4')
    d={'RF':np.abs(rf),'DRF':drf,'RC':np.abs(rc)}
    return d

class paths:
    def __init__(self,nx,tt=0,isdaily=False,run_name=''):
        if nx==4320:
            self.data_dir0='/u/dmenemen/llc_4320/MITgcm/run/'
            self.data_dir1='/u/dmenemen/llc_4320/MITgcm/run_485568/'
            if run_name!='':
                self.data_dir='/u/dmenemen/llc_4320/MITgcm/%s/'%run_name
            self.grid_dir='/u/dmenemen/llc_4320/grid/'
            self.output_dir='/u/jwang23/llc4320_stripe/'
            self.tstart=10368
            self.dt=144
            self.nt=9416
            self.nx=nx
        elif nx==2160:
            if isdaily:
                self.data_dir='/u/dmenemen/llc_2160/regions/global/'
            else:
                if tt>=(1198080-92160)/80:
                    self.data_dir='/u/dmenemen/llc_2160/MITgcm/run/'
                else:
                    self.data_dir='/u/dmenemen/llc_2160/MITgcm/run_day49_624/'
            self.grid_dir='/u/dmenemen/llc_2160/grid/'
            self.output_dir='/nobackup/jwang23/llc2160_striped/'
            self.tstart=92160
            self.dt=80
            self.nt_daily=750
            self.nt=(1586400-92160)/self.dt +1
        self.nx=nx
        self.tt=tt

    def get_fn(self,varn,tt=0,folder=0):
        if folder==0:
            self.data_dir=self.data_dir0
        else:
            self.data_dir=self.data_dir1

        if tt>self.nt-1:
            sys.exit('error: tt is larger than max')
        else:
            if self.nx==4320:
                if varn in ['Theta','Salt','U','V','W','PhiBot','Eta']:
                    fn=self.data_dir+'%s.%010i.data'%(varn,self.tstart+self.dt*tt)
                else:
                    fn=self.grid_dir+'%s.data'%(varn)
            else:
                if varn in ['Theta','Salt','U','V','W','PhiBot','Eta']:
                    fn=self.data_dir+'%s.%010i.data'%(varn,self.tstart+self.dt*tt)
                else:
                    fn=self.grid_dir+'%s.data'%(varn)
        return fn

def load_llc_grid(varn):
    import popy
    d=paths(4320)
    fn=d.output_dir+'grids.h5'
    return popy.io.loadh5(fn,varn)

def load_llc(varn,step,tile_number=11,p=[0,4320,0,4320],
             nx=4320,k=0,return_grid=True,
             offset=0,shape=None):
    """

    """


    pth=paths(nx,tt=step)
    nxx=nx**2

    if tile_number in range(8,14):
        if tile_number<11:
            offset=k*nx*nx*13+7*nx*nx
        else:
            offset=k*nx*nx*13+10*nx*nx
        try:
            d=np.memmap(pth.get_fn(varn,step,0),'>f4',mode='r',offset=offset*4,shape=(nx,nx*3))
        except:
            d=np.memmap(pth.get_fn(varn,step,1),'>f4',mode='r',offset=offset*4,shape=(nx,nx*3))

        ii=np.mod(tile_number-8,3)
        i0,i1=ii+p[0],ii+p[1]
        j0,j1=p[2:]
        dd=d[j0:j1,i0:i1]
        del d
        xx,yy=0,0
        if return_grid==True:
            x=np.memmap(pth.get_fn('XC',0),'>f4',mode='r',offset=offset*4,shape=(nx,nx*3))
            y=np.memmap(pth.get_fn('YC',0),'>f4',mode='r',offset=offset*4,shape=(nx,nx*3))
            yy=y[j0:j1,i0:i1]
            xx=x[j0:j1,i0:i1]
    elif tile_number in range(1,7):
        offset=k*nx*nx*13+(tile_number-1)*nxx
        try:
            d=np.memmap(pth.get_fn(varn,step,0),'>f4',mode='r',offset=offset*4,shape=(nx,nx))
        except:
            d=np.memmap(pth.get_fn(varn,step,1),'>f4',mode='r',offset=offset*4,shape=(nx,nx))

        i0,i1=p[0],p[1]
        j0,j1=p[2:]
        dd=d[j0:j1,i0:i1]
        del d
        xx,yy=0,0
        if return_grid==True:
            x=np.memmap(pth.get_fn('XC',0),'>f4',mode='r',offset=offset*4,shape=(nx,nx))
            y=np.memmap(pth.get_fn('YC',0),'>f4',mode='r',offset=offset*4,shape=(nx,nx))
            yy=y[j0:j1,i0:i1]
            xx=x[j0:j1,i0:i1]
    elif tile_number == '1-7':
        offset=k*nx*nx*13
        try:
            d=np.memmap(pth.get_fn(varn,step,0),'>f4',mode='r',offset=offset*4,shape=(nx*7,nx))
        except:
            d=np.memmap(pth.get_fn(varn,step,1),'>f4',mode='r',offset=offset*4,shape=(nx*7,nx))

        xx,yy=0,0
        if return_grid==True:
            x=np.memmap(pth.get_fn('XC',0),'>f4',mode='r',offset=offset*4,shape=(nx,nx))
            y=np.memmap(pth.get_fn('YC',0),'>f4',mode='r',offset=offset*4,shape=(nx,nx))
            yy=y[j0:j1,i0:i1]
            xx=x[j0:j1,i0:i1]
        return d,xx,yy

    else:
        offset0=k*nx*nx*13+offset
        try:
            dd=np.memmap(pth.get_fn(varn,step,0),'>f4',mode='r',offset=offset0*4)
        except:
            dd=np.memmap(pth.get_fn(varn,step,1),'>f4',mode='r',offset=offset0*4)

        xx=np.memmap(pth.get_fn('XC'),'>f4',mode='r',offset=offset*4)
        yy=np.memmap(pth.get_fn('YC'),'>f4',mode='r',offset=offset*4)
        dd=mds2d(dd)
        xx=mds2d(xx)
        yy=mds2d(yy)


    return dd,xx,yy


def check_llc_grid(nx=4320):
    '''check the orientation of XC,YC,DXC,DYC for face 5
    XC represent longitude pointing upward along row-axis
    YC represent latitude pointing to right from North to South
    DYC measures XC grid size, which is the conventional DXC
    DXC measures YC grid size, which is the conventional DYC.'''

    pth=paths(nx)

    k=0
    offset=k*nx*nx*13+10*nx*nx
    fn=pth.get_fn('YC')
    y=np.memmap(fn,'>f4',mode='r',offset=offset*4,shape=(nx,nx*3))
    x=np.memmap(pth.get_fn('XC'),'>f4',mode='r',offset=offset*4,shape=(nx,nx*3))
    dxc=np.memmap(pth.get_fn('DYC'),'>f4',mode='r',offset=offset*4,shape=(nx,nx*3))
    dis=popy.map.distance_between_points(x[3000,:],x[3001,:],y[3000,:],y[3000,:])
    dxcc=dxc[3000,:].copy()
    dxcc[dxcc==0]=np.nan
    plt.plot(dxcc,'r-')
    plt.plot(dis,'b--')
    del x,y,dxc
#    plt.imshow(np.ma.masked_equal(d[::10,::10],0))
#    plt.colorbar()
    plt.show()
    del d

def load_simple(nx=4320):

    pth=paths(nx)
    k=0
    offset=k*nx*nx*13+10*nx*nx

    d=np.memmap(pth.get_fn('U',3311),'>f4',mode='r',offset=offset*4,shape=(nx,nx*3))
    x=np.memmap(pth.get_fn('XG'),'>f4',mode='r',offset=offset*4,shape=(nx,nx*3))
    y=np.memmap(pth.get_fn('YG'),'>f4',mode='r',offset=offset*4,shape=(nx,nx*3))
    plt.imshow(np.ma.masked_equal(-d[::10,::10].T,0))
    plt.colorbar()
    plt.show()
    del d,x,y

def load_mooring(varn,step,tile,k,j,i,nx=4320):
    fn=paths(nx,step).get_fn(varn)
    nxx=nx**2
    if type(k) == type([0]):
        dout=[]
        for kk in k:
            offset=kk*nxx*13+tile*nxx+j*nx+i
            d=np.memmap(fn,'>f4',mode='r',offset=offset*4,shape=(1))
            dout.append(d)
            del d
        dout=np.array(dout)
    else:
        offset=k*nxx*13+tile*nxx+j*nx+i
        d=np.memmap(fn,'>f4',mode='r',offset=offset*4,shape=(1))
        dout =d[:]
        del d
    return dout

def re_tile(d,n=4320,shift_U=False):
    import numpy as np
    de=d[:n**2*6].reshape(-1,n)
    de=np.c_[de[:n*3,:],de[n*3:,:]]
    dw=d[n**2*7:].reshape(n*2,-1)
    dw=np.r_[dw[:n,:],dw[n:,:]]
    if shift_U: #shift DYG, U grid on tiles 8-13
        dw=np.roll(dw,-1,axis=1)

    dw=dw.T[::-1,:]
    xc=np.c_[de,dw]
    return xc


def mds2d(dd,nx=4320):
    '''reshape the llc grid into east and west hemispheres
       tiles 1-6 for east and 8-13 for the west
       NO rotation is performed (u, v are mixed)


       Parameters
       ============
       dd: numpy array of size 13*nx**2 or a list of numpy array with size 13*nx**2
       nx: llc grid size, Default is 4320 for llc4320.

       Return
       ======
       deast: numpy array (4320x3, 4320x2) or a list of those arrays.
       dwest: numpy array (4320x2, 4320x3) non-rotated (x is lat, y is long)

    '''

    def rearrange(d):
        deast=np.c_[d[:nx*nx*3].reshape(3*nx,nx),
                    d[nx*nx*3:nx*nx*6].reshape(3*nx,nx)]
        dwest=d[nx*nx*7:].reshape(nx*2,nx*3)
        return deast,dwest

    if type(dd)==type([1]):
        dout=[]
        for d in dd:
            dout.append(rearrange(d))
        return dout
    else:
        return rearrange(dd)


def global_vorticity(u,v,dx,dy,nx,llc_original=True):
    from popy import pleiades
    if llc_original:
        dd=pleiades.mds2d([u,v,dx,dy],nx)

        #eastern hemisphere
        u,v,dx,dy=dd[0][0],dd[1][0],dd[2][0],dd[3][0]

        u=np.r_[u[:1,:]*0.0,u]
        v=np.c_[v[:,:1]*0.0,v]

        dy=np.r_[dy[:1,:],dy];dy=(dy[1:,:]+dy[:-1,:])/2.0
        dx=np.c_[dx[:,:1],dx];dx=(dx[:,1:]+dx[:,:-1])/2.0

        dudy=np.diff(u,n=1,axis=0)/dy
        dvdx=np.diff(v,n=1,axis=1)/dx
        vore=dvdx-dudy

        ve=v[::-1,-1]

        #western hemisphere
        u,v,dx,dy=dd[0][1],dd[1][1],dd[2][1],dd[3][1]
        u=np.r_[u[:1,:]*0.0,u]
        u[0,1:]=-ve[:-1]
        v=np.c_[v[:,:1]*0.0,v]

        dy=np.r_[dy[:1,:],dy];dy=(dy[1:,:]+dy[:-1,:])/2.0
        dx=np.c_[dx[:,:1],dx];dx=(dx[:,1:]+dx[:,:-1])/2.0

        dudy=np.diff(u,n=1,axis=0)/dy
        dvdx=np.diff(v,n=1,axis=1)/dx
        vorw=dvdx-dudy

        #piece together east and west
        vor=np.c_[vore,vorw.T[::-1,:]]
    else:
        u=np.r_[u[:1,:]*0.0,u]
        v=np.c_[v[:,:1]*0.0,v]

        dy=np.r_[dy[:1,:],dy];dy=(dy[1:,:]+dy[:-1,:])/2.0
        dx=np.c_[dx[:,:1],dx];dx=(dx[:,1:]+dx[:,:-1])/2.0

        dudy=np.diff(u,n=1,axis=0)/dy
        dvdx=np.diff(v,n=1,axis=1)/dx
        vor=dvdx-dudy

    #dtmp=u[:,2*nx:]
    #u[:,2*nx:]=v[:,2*nx:]
    #v[1:,2*nx:]=-dtmp[:-1,:]
    #dy[1:,2*nx:]=dy[:-1,2*nx:]
    #uu=np.r_[u,u[-1:,:]]
    #vv=np.c_[v,v[:,:1]]
    #vor=(-np.diff(uu,n=1,axis=0)/dy+np.diff(vv,n=1,axis=1)/dx)
    #vor[(u==0)|(v==0)]=0
    vor[:,nx*2-1]=vor[:,nx*2]
    del u,v,dx,dy,dudy,dvdx
    return vor


def global_from_segments(fns,varn,tindex):
    """
    1. get the data from nobackup/llc4320_stripe/global.square.segments/
    2. piece them together to produce a global lat-lon matrix
    pth='/nobackup/jwang23/llc4320_stripe/global.square.segments/'
    le=popy.io.loadh5(pth+'mask.h5','east_index')
    lw=popy.io.loadh5(pth+'mask.h5','west_index')

    Paramter:
    --------
    fns: the file names that contain the subsection data

    varn: string
         the variable name in the fns

    tindex: scalar, the time index of the data

    Return
    ------
    dout: nd array
         global data without tile 7


    """
    import popy
    import numpy as np

    deast=np.zeros((36,360,24,360))
    dwest=np.zeros((24,360,36,360))


    for fn in fns:
        tmp=fn.split('/')[-1]
        loc=tmp[:4]
        j=int(tmp.split('.')[1][1:])
        i=int(tmp.split('.')[2][1:])
        dd=popy.io.loadh5(fn)[varn][:,:,tindex]
        if loc=='East':
            deast[j,:,i,:]=dd
        else:
            dwest[j,:,i,:]=dd

        print("reading ",fn)

    return deast, dwest





if __name__=='__main__':
    load_simple()
