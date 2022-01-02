# $Header: /u/gcmpack/MITgcm/utils/python/MITgcmutils/MITgcmutils/llc.py,v 1.9 2015/11/17 13:21:22 mlosch Exp $
# $Name:  $
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

#Jinbo Wang started to edit 2/10/2016

class surface_forcing:
    
    def get_coor(self):
    
        forcing='/nobackup/dmenemen/forcing/ECMWF_operational/'
        #create the coordinate array for ECMWF surface forcing field
        nlat,nlon=1280,2560
        lat0,lon0=-89.8924,0.0
        dlon=0.140625
        dlat=np.r_[0,.1394, .14018, .14039, .1404695, .140496, .1405145, .1405275, .1405375,
                   .1405455, .140552, .1405575, .140562, .1405655, .140568, .1405695]
        dlat=np.r_[dlat,np.ones((1249))*0.14057]
        dlat=np.r_[dlat,.1405695, .140568, .1405655, .140562, .1405575, .140552, .1405455,
                   .1405375, .1405275, .1405145, .140496, .1404695, .14039, .14018,.1394] 
        lons=lon0+np.arange((nlon))*dlon
        lats=lat0+dlat.cumsum()
        self.nlon,self.nlat=nlon,nlat
        self.lons,self.lats=lons,lats
        self.path=forcing
        return lons,lats
    
    def get_index(self,lon0,lat0):
        """get the index for a point at lon0,lat0"""
        lon0=(lon0+360)%360
        i=np.argmin(abs(self.lons-lon0))
        j=np.argmin(abs(self.lats-lat0))
        return i,j

    def save_sample1d(self,dfn,lon0,lat0):
        """
        sample and save the time series of a station at lon0,lat0
        """
        dfn=self.path+dfn
        dd=np.memmap(dfn,'>f4','r').reshape(-1,self.nlat,self.nlon)
        nt=dd.shape[0]
        del dd
        
        fnout='../data/samples/mooring_lon%.3f_lat%.3f_%04ix001.%s.data'%(lon0,lat0,nt,dfn.split('/')[-1].split('.')[0])
        
        i,j=self.get_index(lon0,lat0)
        print("the original and located coordinates are ", self.lons[i],lon0, self.lats[j], lat0)
        if os.path.exists(fnout):
            dout = np.fromfile(fnout,'>f4')
        else:
            dout = np.zeros((nt))
            for t in range(nt):
                dd=np.memmap(dfn,'>f4','r').reshape(-1,self.nlat,self.nlon)
                dout[t]=dd[t,j,i]
                del dd
            dout.astype('>f4').tofile(fnout)
        return dout

    def get_data(self,t0,pres_only=False):
        """ 
        t0: datetime object of the time of the interpolation
        pres_only: True: only atm pressure, False: atm pressure + tides

        """

        import datetime,popy,os
        from scipy import interpolate 

        self.get_coor()

        if pres_only:
            dfn=self.path+'EOG_pres_%4i'%(t0.year)
            deltat=3600*6
            varn='eog_pres'
        else:
            dfn=self.path+'EOG_pres_tide_%4i'%(t0.year)
            deltat=3600.0 #output frequency in the data
            varn='eog_pres_tide'

        t0_forcing=datetime.datetime(t0.year,1,1,0,0)
        dt=t0-t0_forcing
        nt=int(dt.total_seconds()/deltat)

        print("read record ", nt)
        dd=np.memmap(dfn,'>f4',mode='r',offset=nt*self.nlon*self.nlat*4,shape=(self.nlat,self.nlon))

        return dd


    def interp2d(self,lons,lats,t0_llc,pres_only=False,save_format='None'):
        """
        interpolate the forcing field to llc grid given by lons and lats
        lons: targeted interpolation longitude
        lats: targeted interpolation latitude
        t0_llc: datetime object of the time of the interpolation
        """
        import datetime,popy,os
        from scipy import interpolate

        self.get_coor()
        dd=self.get_data(t0_llc,pres_only=pres_only)


        lons=(lons+360.0)%360.0

        xmin,xmax,ymin,ymax=lons.min(),lons.max(),lats.min(),lats.max()
        print("range of the subdomain is ", [xmin,xmax,ymin,ymax])
        #find index for the subdomain
        i0,i1=np.where(self.lons<xmin)[0][-1]-1,np.where(self.lons>xmax)[0][0]+1
        j0,j1=np.where(self.lats<ymin)[0][-1]-1,np.where(self.lats>ymax)[0][0]+1

        if lons.ndim==1:
            lons,lats=np.meshgrid(lons,lats)

        target_points=np.c_[lons.ravel(),lats.ravel()]

        olons,olats=np.meshgrid(self.lons[i0:i1],self.lats[j0:j1])
        points=np.c_[olons.ravel(),olats.ravel()]

        dout=interpolate.LinearNDInterpolator(points,dd[j0:j1,i0:i1].ravel())(target_points).reshape(lons.shape)

        del dd
        return dout

    
    def interp2d_all(self,lons,lats,pres_only=False):
        """ 
        interpolate the forcing field to llc grid given by lons and lats
        lons(ny,nx) 
        lats(ny,nx)

        Return:
          
        save the pressure or pressure+tide data into self.fn_output

        """
        import datetime,popy,os
        from scipy.interpolate import RectBivariateSpline as rbs
        self.get_coor()
        lons=(lons+360.0)%360.0
        xmin,xmax,ymin,ymax=lons[0,:].min(),lons[0,:].max(),lats[:,0].min(),lats[:,0].max()
        print("range of the subdomain is ", [xmin,xmax,ymin,ymax])
        #find index for the subdomain
        i0,i1=np.where(self.lons<xmin)[0][-1]-2,np.where(self.lons>xmax)[0][0]+2
        j0,j1=np.where(self.lats<ymin)[0][-1]-2,np.where(self.lats>ymax)[0][0]+2
        #
        fac=1
        if pres_only:
            dfn1=self.path+'EOG_pres_2011'
            dfn2=self.path+'EOG_pres_2012'
            fac=6
        else:
            dfn1=self.path+'EOG_pres_tide_2011'
            dfn2=self.path+'EOG_pres_tide_2012'
            deltat=3600.0 #output frequency in the data
        deltat=3600*fac
        nt=int(os.stat(dfn1).st_size/4/self.nlon/self.nlat)

        t0_forcing=datetime.datetime(2011,1,1,0,0)
        t0_llc = datetime.datetime(2011,9,13,0,0)
        dt=t0_llc-t0_forcing
        nt0=int(dt.total_seconds()/deltat)

        print("starting record at ",nt0)
        #dd=np.memmap(dfn1,'>f4','r')
        #nt=int(114819072000/4/self.nlon/self.nlat)
        print("The number of records in file is ",nt)
        #del dd
        #
        fntmp=self.fn_output
        def cn(fnin,t0,t1,toffset,total_steps):
            for i in range(t0,t1,1):
                print("read record ",i)
                dd=np.memmap(fnin,'>f4',mode='r',offset=i*self.nlon*self.nlat*4,shape=(self.nlat,self.nlon))
                dtmp=dd[j0:j1,i0:i1].copy()
                itp=rbs(self.lons[i0:i1],self.lats[j0:j1],dtmp.T)
                newd=itp.ev(lons.flatten(),lats.flatten()).reshape(lons.shape)
                newd[lons==0]=0
                popy.io.saveh5(fntmp,'/forcing',newd.astype('>f4'),nrec=total_steps,irec=(i-t0)/fac+toffset)
                print("save subdomain data from"+fnin+" to temporary file "+fntmp+" at time step", (i-t0)/fac+toffset)
                del dd, newd, itp, dtmp
        total_steps=9415
        nt_llc=total_steps/fac #total forcing steps, fac is the forcing time interval in a unit of hours
        cn(dfn1,nt0,nt,0,total_steps) #2011
        cn(dfn2,0,nt_llc+nt0-nt,(nt-nt0)/fac,total_steps) #2012

        return
        
        
    def interp1(self,lon0,lat0):
        #glue tide pressure for 2011 and 2012 and interpolate onto llc_4320 time records
        import datetime
        from scipy.interpolate import interp1d
        
        d1=self.save_sample1d('EOG_pres_tide_2011',lon0,lat0)
        d2=self.save_sample1d('EOG_pres_tide_2012',lon0,lat0)
        dd=np.r_[d1,d2]
        del d1,d2
        t0_forcing=datetime.datetime(2011,1,1,0,0)
        t0_llc = datetime.datetime(2011,9,13,0,0)
        dt=t0_llc-t0_forcing
    
        t_forcing=np.arange(dd.size)
        t_llc=np.arange(3301)+dt.total_seconds()/3600.0
    
        print("the first value of t_llc cooresponds ",datetime.datetime(2011,1,1,0,0)+datetime.timedelta(hours=t_llc[0]))
        print("the first value of t_forcing cooresponds ",datetime.datetime(2011,1,1,0,0))
    
        dintp=interp1d(t_forcing, dd.flatten())
        dout=dintp(t_llc)
        return t_llc,dout,t_forcing,dd


def barotropic_velocity(u,hfac):
    """
    u: velocity
    hfac: hfacW for U hfacS for V
    
    return

    ubar: barotropic velocity

    up: baroclinic velocity

    """

    nz,ny,nx=u.shape
    drf=np.fromfile('/u/dmenemen/llc_4320/grid/DRF.data','>f4')
    drf=hfac*drf[:nz,np.newaxis,np.newaxis]
   
    hm=drf.sum(axis=0)
    hm[hm==0]=np.inf
    ubar=(u*drf).sum(axis=0)/hm

    up=u-ubar[np.newaxis,:,:]

    return ubar, up

