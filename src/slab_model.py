from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import numpy as np
from utils import filter_butter, lat2f
import slab_model


class mld:
    
    def __init__(self):

        data = xr.open_dataset('/u/bura0/wineteer/WACM/jinbo_odysea/Argo_mixedlayers_monthlyclim_04142022.nc')
        data = data.set_coords(("lat", "lon", "month"))
        
        self.dataset = xr.Dataset() 

        self.dataset = self.dataset.assign_coords(coords={'lon': (['lon'], data['lon'].values),
                                                'lat': (['lat'], data['lat'].values),
                                                'month': (['month'], data['month'].values)})   
        
        self.dataset = self.dataset.assign({'mld_da_mean': (['lat', 'lon', 'month'], data['mld_da_mean'].values),
                                           'mld_da_std': (['lat', 'lon', 'month'], data['mld_da_std'].values)})
        
    
    def getmld(self,lat,lon,month):
        
        interp_ds = self.dataset.interp(lat=lat,lon=lon,month=month,method='linear',kwargs={"fill_value": "extrapolate"})
        
        return interp_ds.mld_da_mean.item(),interp_ds.mld_da_std.item()
    
    
    
class synthetic_wacm:
    def __init__(self,din,
                 lat0,lon0,cutoff_truth=1/3,
                 cutoff_wacm=1/30,
                 remove_super_high=False,
                 orbit='700_1800'):
        
        
        """
        din: list of [sea surface velocity u, v, and wind vector]
        lat0: the latitude of the point
        lon0: the longitude of a point
        cutoff_truth: the frequency (/day) threshold for filtering the true time series
        cutoff_wacm: the frequency threshold for filtering wacm time series
        remove_super_high: bool, whether discard super-high frequency
        orbit: the orbit identifier, it was written for earlier development when orbit is not fixed. It is now default '700_1800'.
        """
        u_truth,v_truth,uw_truth,vw_truth=din
        
        self.lat0=lat0
        self.wacm_sampling_period=load_wacm_period(lat0,orbit)
        self.f0=lat2f(lat0)
        self.inertial_period=2*np.pi/self.f0 / 3600 
        self.u_truth_total=u_truth*1
        self.v_truth_total=v_truth*1
        self.uw_truth=uw_truth*1
        self.vw_truth=vw_truth*1
        
        #display(u_truth)
        
        print("seperate hourly truth into low and high frequencies")
        uobs_low0,uobs= filter_seperation(u_truth,cutoff=cutoff_truth)
        vobs_low0,vobs= filter_seperation(v_truth,cutoff=cutoff_truth)
        
       # display(uobs_low0)
       # display(uobs)
        
        self.u_truth_low=uobs_low0*1
        self.v_truth_low=vobs_low0*1
        self.u_truth_high=uobs*1
        self.v_truth_high=vobs*1
        
        #display(self.u_truth_high)
        
        if remove_super_high:
            uobs_low0,uobs= filter_seperation(u_truth,cutoff=24/self.inertial_period * 2)
            vobs_low0,vobs= filter_seperation(v_truth,cutoff=24/self.inertial_period * 2)
            #self.u_truth_low=uobs_low0*1
            #self.v_truth_low=vobs_low0*1
            self.u_truth_high=self.u_truth_high-uobs
            self.v_truth_high=self.v_truth_high-vobs
            self.u_truth_super_high=uobs*1
            self.v_truth_super_high=vobs*1
            
        print("Remove tides from high-frequency truth")
        u,v, ut, vt=remove_tides([self.u_truth_high.copy(),self.v_truth_high.copy()],lat0)
       
       # display(self.u_truth_high)

        self.u_truth_high_notide=u*1
        self.v_truth_high_notide=v*1
        self.u_truth_high_tide=ut*1
        self.v_truth_high_tide=vt*1
        print("bandpass to get true NIO")
        self.u_truth_high_notide_nio=NIO_bandpass(self.u_truth_high.copy(),lat0)
        self.v_truth_high_notide_nio=NIO_bandpass(self.v_truth_high.copy(),lat0)
        
        print("generate synthetic wacm")
        wacm_u,wacm_v,wacm_uw,wacm_vw=interp_wacm([u_truth,v_truth, uw_truth,vw_truth],
                                                  lat0,lon0,
                                                  t_range=[u_truth.time[0].values,u_truth.time[-1].values],
                                                  orbit=orbit)
        
        #display(self.u_truth_high)

        self.u_wacm_total=wacm_u*1
        self.v_wacm_total=wacm_v*1
        self.uw_wacm=wacm_uw*1
        self.vw_wacm=wacm_vw*1
        
        u_low0,u=filter_seperation(self.u_wacm_total,cutoff=cutoff_wacm)
        v_low0,v=filter_seperation(self.v_wacm_total,cutoff=cutoff_wacm)
        self.u_wacm_low=u_low0*1
        self.v_wacm_low=v_low0*1
        self.u_wacm_high=u*1
        self.v_wacm_high=v*1
        
        #display(self.u_truth_high)

       # display(u)
        
        u,v, ut, vt=remove_tides([u.copy(),v.copy()],lat0)
       # display(u)
        self.u_wacm_high_notide=u*1
        self.v_wacm_high_notide=v*1
        self.u_wacm_high_tide=ut*1
        self.v_wacm_high_tide=vt*1
        
        #display(self.u_truth_high)

        return
    
    def get_wacm_nio(self,t_out,
                     t_out_short=[],
                     uv_noise=0,
                     wind_noise=0,
                     has_tides=False,
                     use_hourly_wind=False,
                    fitting_method='Slab',
                    periods=[1.07581, 0.99727, 0.517525, 0.5, 0.2587625, 0.2587625],
                    include_nio=True,
                    is_windstress=False,
                    c_clim=60,c_min=0,c_max=2000):
        #Include extra points in winds to avoid extrapolation problem
        """
        t_out: numpy.datetime64
        include_nio: bool, only used for Harmonics method. 
        
        """
                
        t0=t_out.min()-np.timedelta64(int(np.ceil(self.wacm_sampling_period) ),'h')
        t1=t_out.max()+np.timedelta64(int(np.ceil(self.wacm_sampling_period) ),'h')
        
        u=self.u_wacm_high.sel(time=slice(t0,t1))
        t0,t1=u.time.values[0],u.time.values[-1]
        
        u=self.u_wacm_high.sel(time=slice(t0,t1))
        v=self.v_wacm_high.sel(time=slice(t0,t1))
        
        u+=np.random.normal(0,uv_noise,u.size)
        v+=np.random.normal(0,uv_noise,u.size)

        if fitting_method=='Harmonics':
            up,vp,param=optimize_harmonics(u,v,lat2f(self.lat0),t_out=t_out,periods=periods,include_nio=include_nio)
            return up,vp,None,None,param
        else:
            if 'time' in str(type(t0)):
                t00,t11=t0-np.timedelta64(2,'D'),t1+np.timedelta64(2,'D')
            else:
                t00,t11=np.datetime64(t0)-np.timedelta64(2,'D'),np.datetime64(t1)+np.timedelta64(2,'D')

            if use_hourly_wind:
                uw=self.uw_truth.sel(time=slice(t00,t11)) #longer-period for wind to avoid extrapolation
                vw=self.vw_truth.sel(time=slice(t00,t11))
            else:
                uw=self.uw_wacm.sel(time=slice(t00,t11))
                vw=self.vw_wacm.sel(time=slice(t00,t11))

            uw+=np.random.normal(0,wind_noise,uw.size)
            vw+=np.random.normal(0,wind_noise,uw.size)
            
            if has_tides:
                utid,vtid,param=optimize_harmonics(u.copy(),v.copy(),self.f0,
                                              periods=[1.07581, 0.99727, 0.517525, 0.5, 0.2587625, 0.2587625],
                                              include_nio=False)
                u=u-utid.interp(time=u.time.values)
                v=v-vtid.interp(time=u.time.values)
            else:
                utid=0
                vtid=0
                
            up,vp,_,_,param=reconstruct_NIO_short_segment(u,v,
                                                          uw,vw,self.lat0,
                                                          t_out=t_out,
                                                          uv_noise=uv_noise,
                                                          wind_noise=wind_noise,
                                                          has_tides=False,
                                                          is_windstress=is_windstress,
                                                         c_clim=c_clim,c_min=c_min,c_max=c_max)
       
            if len(t_out_short)>0:
                up=up.interp(time=t_out_short)
                vp=vp.interp(time=t_out_short)
                utid=utid.interp(time=t_out_short)
                vtid=vtid.interp(time=t_out_short)
                
            return up,vp,utid,vtid,param
        
import utide
import numpy as np    
    
def remove_tides(d,lat0,constit=('O1', 'K1', 'M2', 'S2', 'M4', 'M6')):
    
    """
    Parameter:
    ---------
    d: [u,v] xarray with 'time' axis
    
    Return
    ------
    d, xarray
       detided time series on the original time axis
    
    """

    u=d[0]
    v=d[1]
    
    # why was Jinbo using this? It seems to break the tide reconstruction. 
    # Used to have tt where u.time.values is now in .solve and .reconstruct
    #tt=(u.time.values-u.time.values[0])/np.timedelta64(1,'s')/86400 #convert to days
    
    if len(constit)>0:
        cc=utide.solve(u.time.values.copy(),u.values.copy(),v.values.copy(),
                       constit=constit, 
                       lat=lat0,verbose=False)
    else:
        cc=utide.solve(u.time.values.copy(),u.values.copy(),v.values.copy(), 
                       lat=lat0,verbose=False)

    tide=utide.reconstruct(u.time.values.copy(),cc,verbose=False)
    
    #display(u.values)
    #display(tide.u)
    
    u.values=u.values-tide.u
    v.values=v.values-tide.v
    #display(u.values)

    ut=d[0]*0
    vt=d[1]*0
    ut.values[:]=tide.u*1
    vt.values[:]=tide.v*1
    #dtide.values=utide.reconstruct(tt,cc,verbose=False)['h']
    
    return u, v, ut,vt

import xarray as xr

def NIO_bandpass(uobs,f0,band=[0.96,1.04]):
    """
    subtract NIO from high-frequency velocities
    
    uobs: xarry with time axis
    
    """
    
    if abs(f0)>0.1: #latitude
        f0=np.abs(lat2f(f0))
    
    tt=uobs.time.values
    dtt=np.diff(tt)
    if (dtt.min()-dtt.max())!=0:
        tt_new=pd.date_range(tt.min()-np.timedelta64(1,'h'),tt.max()+np.timedelta64(1,'h'),freq='1h')
        newd=uobs.interp(time=tt_new)
        period=3600
    else:
        period=dtt[100]/np.timedelta64(1,'s')
        newd=uobs
        
    d_filtered=filter_butter(newd.values,[band[0]*f0,band[1]*f0],2*np.pi/period,'bandpass')
    
    if (dtt.min()-dtt.max())!=0: #need interpolation on to uniform grid
        dd=xr.DataArray(d_filtered,dims=('time'),coords={'time':tt_new} )
        dout=dd.interp(time=tt)
    else:
        dout=xr.DataArray(d_filtered,dims=('time'),coords={'time':tt} )
    return dout

import pandas as pd
    
def generate_wacm(uobs,vobs,uwobs,vwobs,lat0,lon0=140,wacm_error_v=0,wacm_error_wind=[0,0],orbit='700_1800'):
    """
    wacm_error_v: velocity errors in m/s, std
    wacm_error_wind: wind erros, [speed percetage, wind direction in degrees]
    
    orbit: string, '500_1000', '700_1800' or 'hourly'
    """
    import odysea_class as ody
    
    
    tt=uobs.time.values
    if orbit=='hourly':
        return uobs,vobs,uwobs,vwobs
    elif oribt[-1]=='h':
        wacm_sampling=int(orbit[:-1])*3600
    else:
        wacm_sampling=load_wacm_period(lat0,orbit)*3600 #convert to seconds

    #print('The averaged wacm sampling period at lat=%4.1f is %5.2f hours.'%(lat0,wacm_sampling/3600))
    tt_wacm=ody.wacmLatLon().getSamplingTimes([lat0],[lon0],tt.min(),tt.max())
    #tt_wacm=pd.date_range(tt.min(),tt.max(),freq='%is'%wacm_sampling)

    #interpolate hourly observations to wacm time to generate synthetic wacm surface velocity
    wacm_u=uobs.interp(time=tt_wacm)+np.random.normal(0,wacm_error_v,tt_wacm.size)
    wacm_v=vobs.interp(time=tt_wacm)+np.random.normal(0,wacm_error_v,tt_wacm.size)
    
    #generate wacm winds
    
    #direction_error=np.random.normal(0,10,uwobs.size) #directional uncertainty in degrees
    #speed_error=np.random.normal(0,0.05,uwobs.size)*np.abs(uwobs.data**2+vwobs.data**2)**0.5
    ww=uwobs.data+1j*vwobs.data
    speed=np.abs(ww)
    direction=np.angle(ww)
    
    speed=np.random.normal(1,wacm_error_wind[0],speed.size)*speed
    direction=direction+np.random.normal(0,wacm_error_wind[1]/180*np.pi,speed.size)
    wind=speed*np.exp(1j*direction)
    
    uwobs.data=np.real(wind)
    vwobs.data=np.imag(wind)
    
    wacm_uw=uwobs.interp(time=tt_wacm) #+uw_error #np.random.normal(0,wacm_error_wind,tt_wacm.size)
    wacm_vw=vwobs.interp(time=tt_wacm) #+vw_error #np.random.normal(0,wacm_error_wind,tt_wacm.size)
    
    return wacm_u, wacm_v, wacm_uw, wacm_vw
    

def fitting_error(uobs,vobs,uwobs,vwobs,t0,t1,lat0,wacm_error_v=0,wacm_error_wind=0,has_tides=False,is_windstress=False):
    """
    predict NIO using slab model prediction
    calculate the fitting error
    uobs,vobs: xarray on 'time' axis, high-frequency velocities from mooring or model, serving as the truth (cm/s)
    uwobs,vwobs: xarray on 'time' axis, zonal and meridonal winds (m/s)
    
    t0,t1: the starting and end time for the analysis period
    
    lat0: latitude
    
    """
    import pandas as pd
    
    tt=uobs.time.values
    
    wacm_sampling=load_wacm_period(lat0)*3600 #convert to seconds

    #print('The averaged wacm sampling period at lat=%4.1f is %5.2f hours.'%(lat0,wacm_sampling/3600))

    tt_wacm=pd.date_range(tt.min(),tt.max(),freq='%is'%wacm_sampling)

    #interpolate hourly observations to wacm time to generate synthetic wacm surface velocity
    wacm_u=uobs.interp(time=tt_wacm)
    wacm_v=vobs.interp(time=tt_wacm)

    
    #generate wacm winds
    wacm_uw=uwobs.interp(time=tt_wacm)
    wacm_vw=vwobs.interp(time=tt_wacm)
 

    #subset to a smaller time window between t0,t1
    u=wacm_u.sel(time=slice(t0,t1))
    v=wacm_v.sel(time=slice(t0,t1))
    t_uv=u.time.values
    noise_u=np.random.normal(0,wacm_error_v,u.size)
    noise_v=np.random.normal(0,wacm_error_v,v.size)
    
    uw=wacm_uw.sel(time=slice(t0,t1))
    vw=wacm_vw.sel(time=slice(t0,t1))
    t_tau=uw.time.values
   
    noise_uw=np.random.normal(0,wacm_error_wind,uw.size)
    noise_vw=np.random.normal(0,wacm_error_wind,uw.size)
    
    v_sub=vobs.sel(time=slice(t_uv.min(),t_uv.max()))
    u_sub=uobs.sel(time=slice(t_uv.min(),t_uv.max()))
    t_output=v_sub.time.values

    f0=lat2f(lat0)
    #print('The inertial period = %5.2f hours'%(2*np.pi/f0/86400*24))

    u_pred,v_pred=optimize_slab_noshear_withtide(t_uv,u.data+noise_u,v.data+noise_v,
                                                 t_tau,uw.data+noise_uw,vw.data+noise_vw,
                                                 f0,t_output,has_tides=has_tides,is_windstress=is_windstress)

    error=(((u_pred-u_sub.values)**2+(v_pred-v_sub.values)**2).mean()/2)**0.5
    #print('The prediction error for the current speed is %6.3f cm'%(error))
   
    std_truth=((u_sub.values**2+v_sub.values**2).mean()/2)**0.5
    
    #error=(((u_pred-NIO_bandpass(u_sub,f0).values)**2+(v_pred-NIO_bandpass(v_sub,f0).values)**2).mean()/2)**0.5
    #print('The prediction error for the current speed is %6.3f cm'%(error))
   
    #std_truth=((NIO_bandpass(u_sub,f0).values**2+NIO_bandpass(v_sub,f0).values**2).mean()/2)**0.5
    
    
    return error, std_truth

def reconstruct_NIO_short_segment(uobs,vobs,uwobs,vwobs,lat0,
                                  t_out=[],
                                  uv_noise=0,
                                  wind_noise=0,
                                has_tides=False,
                                is_windstress=False,
                                 c_clim=60,c_min=0,c_max=2000):
    """
    Take high-frequency observations
    subsample according to WaCM scenario
    
    Use the synthetic WaCM observations, predict NIO using slab model prediction

    uobs,vobs: xarray on 'time' axis, high-frequency velocities from mooring or model, serving as the truth (cm/s)
    uwobs,vwobs: xarray on 'time' axis, zonal and meridonal winds (m/s)
        
    lat0: latitude
    
    fitting_window: the time window for the fitting (days), default None
    
    t_out, time axis for the output, default None
    
    wacm_error_v: wacm velocity error, default 0
    
    wacm_error_wind: wacm wind error, default 0
    
    has_tides: whether use tides in the fitting, default False
    
    Return
    ======
    
    fitted NIO on the same time axis of the input or t_out if t_out!=None
    
    """
    import pandas as pd
    import xarray as xr
    
    f0=lat2f(lat0)
    wacm_u,wacm_v,wacm_uw,wacm_vw=uobs,vobs,uwobs,vwobs
    if len(t_out)==0:
        t_out=pd.date_range(wacm_u.time.values[0],wacm_u.time.values[-1],freq='1h')
    if has_tides:
        wacm_u, wacm_v, ut, vt=remove_tides([wacm_u, wacm_v],lat0)
        ut=ut.interp(time=t_out)
        vt=vt.interp(time=t_out)
    else:
        vt=0;ut=0
        
    u_pred,v_pred,param=optimize_slab_noshear_withtide(wacm_u.time.values,wacm_u.data,wacm_v.data,
                                         wacm_uw.time.values,wacm_uw.data,wacm_vw.data,
                                         f0,has_tides=has_tides,is_windstress=is_windstress,
                                                      c_clim=c_clim,c_min=c_min,c_max=c_max)
    
    u_pred=u_pred.interp(time=t_out) 
    v_pred=v_pred.interp(time=t_out)
    
    return u_pred, v_pred, ut, vt, param

def get_tide(tt,T,param):
    """
    tt: numpy Array, time in days
    T: list of periods of the tidal constituents to be considered
    param: list with size of N x 4, where N is the lenght of T.
           For each constituent, the param corresponds to [U_amp, U_phase, V_amp, V_phase]
    
    """
    n=len(T)
    pam=np.array(param).reshape(n,4)
    
    #if 'time' in str(type(tt[0])):
    #    tt=(tt-np.datetime64('1979-01-01 00:00:00'))/np.timedelta64(1,'D')
        
    u=0
    v=0
    
    for i, period in enumerate(T):
        omg=2*np.pi/period
        u+=pam[i,0]*np.cos(omg*tt+pam[i,1])
        v+=pam[i,2]*np.cos(omg*tt+pam[i,3])
        
    return u, v    

def reconstruct_NIO(uobs,vobs,uwobs,vwobs,lat0,
                    fitting_window=None,
                    t_out=None,wacm_error_v=0,
                    wacm_error_wind=0,
                    has_tides=False,
                   is_windstress=False,
                   orbit='700_1800',
                   c_clim=60,c_min=0,c_max=2000):
    """
    Take high-frequency observations
    subsample according to WaCM scenario
    
    Use the synthetic WaCM observations, predict NIO using slab model prediction

    uobs,vobs: xarray on 'time' axis, high-frequency velocities from mooring or model, serving as the truth (cm/s)
    uwobs,vwobs: xarray on 'time' axis, zonal and meridonal winds (m/s)
        
    lat0: latitude
    
    fitting_window: the time window for the fitting (days), default None
    
    t_out, time axis for the output, default None
    
    wacm_error_v: wacm velocity error, default 0
    
    wacm_error_wind: wacm wind error, default 0
    
    has_tides: whether use tides in the fitting, default False
    
    Return
    ======
    
    fitted NIO on the same time axis of the input or t_out if t_out!=None
    
    """
    import pandas as pd
    import xarray as xr
    
    f0=lat2f(lat0)
    if input_wacm_data:
        wacm_u,wacm_v,wacm_uw,wacm_vw=uobs,vobs,uwobs,vwobs
    else:
        wacm_u,wacm_v,wacm_uw,wacm_vw=generate_wacm(uobs,vobs,uwobs,vwobs,lat0,
                                                wacm_error_v=wacm_error_v,
                                                wacm_error_wind=wacm_error_wind,
                                               orbit=orbit)
 
    wacm_sampling=load_wacm_period(lat0,orbit) #sampling period in hours
    
    inertial_period=2*np.pi/f0/86400*24 #inertial period in hours
    
    #the time window for the fitting (days). choose enough data points, either 8 ineria periods or 4 days, which ever the maximum
    if type(fitting_window)==type(None):
        dtt=int(max(np.ceil(inertial_period*8/24), 4) )
    else:
        dtt=fitting_window
    print("fitting window = %i days with %i data points"%(dtt,dtt*24//wacm_sampling))
    
    #loop through by running through smaller window 
    #make sure to have at least 15 observations or 
    t0s=pd.date_range(vobs.time[0].values,vobs.time[-1].values,freq='%iD'%dtt)
    
    nn=int(dtt*24//wacm_sampling)
    print("split total %i points into %i segments"%(wacm_u.size,np.arange(0,wacm_u.size//nn*nn,nn).size))
    
    for ii in np.arange(0,wacm_u.size//nn*nn,nn):
    
        u=wacm_u[ii:ii+nn]
        v=wacm_v[ii:ii+nn]
        t_uv=u.time.values
  
        uw=wacm_uw[ii:ii+nn]
        vw=wacm_vw[ii:ii+nn]
        t_tau=uw.time.values
        
        #print(wacm_sampling)
        nend=min(ii+nn,wacm_u.size-1)
        t_output=pd.date_range(t_uv[0],wacm_u[ii+nn].time.values,freq='1h')
        
        if ii==0:
            u_pred,v_pred,_=optimize_slab_noshear_withtide(t_uv,u.data,v.data,
                                                 t_tau,uw.data,vw.data,
                                                 f0,t_output,has_tides=has_tides,is_windstress=is_windstress,
                                                           c_clim=c_clim,c_min=c_min,c_max=c_max)

            
            
            u_pred=xr.DataArray(u_pred,dims=('time'),coords={'time':t_output})
            v_pred=xr.DataArray(v_pred,dims=('time'),coords={'time':t_output})
            
        else:
            u_pred0,v_pred0,_=optimize_slab_noshear_withtide(t_uv,u.data,v.data,
                                                 t_tau,uw.data,vw.data,
                                                 f0,t_output,has_tides=has_tides,is_windstress=is_windstress,
                                                  c_clim=c_clim,c_min=c_min,c_max=c_max)

            u_pred0=xr.DataArray(u_pred0,dims=('time'),coords={'time':t_output})
            v_pred0=xr.DataArray(v_pred0,dims=('time'),coords={'time':t_output})
            
            u_pred=xr.concat([u_pred,u_pred0],dim='time')
            v_pred=xr.concat([v_pred,v_pred0],dim='time')
            del u_pred0,v_pred0
            
    return u_pred, v_pred


import odysea_class

def interp_wacm(data,lat0,lon0,t_range=[],orbit='700_1800'):
                    
    """
    interpolate data onto wacm sampling
    
    data: xarray with 'time' axis
    
    lat0;lon0 provide the coordinates of the station
    
    orbit: default '700_1800'
    
    """

    import pandas as pd
    
#    if orbit=='hourly': #provide option to return hourly data
#        return data
#    else orbit[-1]=='h': #this option specify a uniform sampling interval with a unit of hour
#        wacm_sampling=int(orbit[:-1])*3600
        
    #print('The averaged wacm sampling period at lat=%4.1f is %5.2f hours.'%(lat0,wacm_sampling/3600))
    if type(data) != type([]):
        if len(t_range)==0:
           # tt=data.time.values
           # tt_wacm=pd.date_range(tt.min(),tt.max(),freq='%is'%wacm_sampling)
            tt_wacm = odysea_class.wacmLatLon().getSamplingTimes([lat0],[lon0],data.time.values[0],data.time.values[-1])[0]

        else:
            #tt_wacm=pd.date_range(t_range[0],t_range[1],freq='%is'%wacm_sampling)
            tt_wacm = odysea_class.wacmLatLon().getSamplingTimes([lat0],[lon0],t_range[0],t_range[1])[0]

            
        return data.interp(time=tt_wacm,method='linear')
    
    else:
        dout=[]
        #tt_wacm=pd.date_range(t_range[0],t_range[1],freq='%is'%wacm_sampling)
        tt_wacm = odysea_class.wacmLatLon().getSamplingTimes([lat0],[lon0],t_range[0],t_range[1])[0]
        for dd in data:
            dout.append(dd.interp(time=tt_wacm,method='linear'))
        return dout
    
    return 


def load_wacm_period(lat,orbit='700_1800'):
    """
    This is obsolete as the orbit is fixed now. 
    """
    
    if orbit=='hourly':
        return 1
    if orbit[-1]=='h':
        return int(orbit[:-1])
    
    fn='../src/wacm_sampling_period_orbit_%s.txt'%orbit
    d=np.loadtxt(fn)
    
    period=interp1d(d[:,0],d[:,1])(np.array(lat))
    return period


import pandas as pd

def filter_seperation(uv,cutoff=1/10):
    """
    uv: xarray with time axis
    cutoff: the filter's cutoff frequency (1/day)
    """
    
    dtt=np.diff(uv.time.values)
    
    if dtt.min()!=dtt.max(): #interpolate to hourly
        newt=pd.date_range(uv[0].time.values,uv[-1].time.values,freq='1h')
        uv_low=uv.interp(time=newt)
        dtt=1.0 # 1 hour
    else:
        uv_low=uv*1
        dtt=dtt[0]/np.timedelta64(1,'s')/3600 # hours
        
    #low-pass filter at cutoff period of 10 days
    uv_low.data=filter_butter(uv_low.data,cutoff=cutoff,fs=24/dtt,btype='low')
    
    uv_low=uv_low.interp(time=uv.time)
    uv_high=uv-uv_low
    
    return uv_low,uv_high


def wind2stress(uw,vw):
    """
    Input:
    
    uw numpy array for zonal wind velocity
    vw numpy array for meridional wind velocity
    
    Output:
    
    uws, vws
    wind stress in zonal and meridional directions
    
    """

    Cd=1.2e-3
    rho_air=1.22
    rho_water=1027.5

    uws=Cd*rho_air*np.abs(uw)*uw/rho_water
    vws=Cd*rho_air*np.abs(vw)*vw/rho_water

    return uws, vws

def tauxy_func(t,taux, tauy):
    """
    produce interpolation function for wind stress
    
    t: time array_like(N)
    taux: zonal windstress array_like (N)
    tauy: meridional windstress array_like (N)
    
    Return:
        taux_func
        tauy_func
    """
    
    taux_func=interp1d(t,taux,fill_value="extrapolate")
    tauy_func=interp1d(t,tauy,fill_value="extrapolate")
    
    return taux_func, tauy_func

def slab(t, y, f,H,c,taux,tauy,U_x,U_y,V_x,V_y): 
    """
    A slab mixed layer model for simulating inertial oscillation.
    
    t: time in days, the time is shifted and already reference to the first point as t=0
    y: array_like or list with two elements representing u and v
    
    param: list
        
        [f, H, c, taux, tauy, U_x, U_y, V_x, V_y]
        
        f: coriolis frequency
        H: mixed layer depth
        c: damping coefficient
        taux: zonal wind stress (N/m2), func taux(t)
        tauy: meridional wind stress (N/m2), func tauy(t)
        
        U_x,U_y,V_x,V_y: velocity shears
        
    """
     
    tau_x=taux(t)
    tau_y=tauy(t)
    
    
    yy=np.zeros_like(y)
    
    u=y[0]
    v=y[1]
    
    factor=86400
    dudt= (f*v - c*u + tau_x/H - U_x*u - U_y*v) * factor
    dvdt= (-f*u - c*v + tau_y/H - V_x*u - V_y*v) * factor
    
        
    yy=np.r_[dudt,dvdt]
    
    return yy



def slab_noshear(t, y, f,H,c,taux,tauy): 
    """
    A slab mixed layer model for simulating inertial oscillation. No background shear is included. 
    
    t: time in days, the time is shifted and already reference to 1970-01-01 t=0
    y: array_like or list with two elements representing u and v
    
    param: list
        
        [f, H, c, taux, tauy]
        
        f: coriolis frequency
        H: mixed layer depth
        c: damping coefficient
        taux: zonal wind stress (N/m2), func taux(t)
        tauy: meridional wind stress (N/m2), func tauy(t)
                
    """
     
    tau_x=taux(t)
    tau_y=tauy(t)
    
    
    yy=np.zeros_like(y)
    
    u=y[0]
    v=y[1]
    
    factor=86400
    dudt= (f*v - c*u + tau_x/H ) * factor
    dvdt= (-f*u - c*v + tau_y/H ) * factor
        
    yy=np.r_[dudt,dvdt]
    
    return yy


def predict_slab_noshear(x,tt,taux,tauy,f): 
    """
        
    x: the parameter space [u0,v0,H,c,u_bias,v_bias]
    
    tt: array of time in days referenced to 1979-01-01 00:00:00, or array of datetime64 values
    
    use the parameters in x, integrate a slab model forward, compare to the truth giving in y
    
    taux,tauy : wind stress function, with time axis referened to 1979-01-01 00:00:00
    
    f: coriolis param
    
    """
    
    #u0,v0,H,c,U_x,U_y,V_x,V_y,a,b,c,d=x
    #u0,v0,H,c=x
    
    #if 'time' in str(type(tt[0])):
    #    tt=(tt-np.datetime64('1979-01-01 00:00:00'))/np.timedelta64(1,'D')
        
    u0,v0,H,c,u_bias,v_bias=x[:6]
    #H=50
    #c=1./(6*86400) #1e-8
    #u_bias=0
    #v_bias=0
    #t_eval=np.arange(tt.min(),tt.max()+1/24,1/24) #integrate using dt=1 hr

    
    sol = solve_ivp(slab_noshear, [tt.min(), tt.max()], [u0,v0], t_eval=tt, args=(f,H,c,taux,tauy)) #(1e-4, 10.0, 1e-5, taux,tauy,0.,0.,0.,0.) )
    
    # add a linear trend to represent background velocity
    
    u=sol.y[0] - u_bias #+ a*t_eval + b
    v=sol.y[1] - v_bias #+ c*t_eval + d
    
    #u=interp1d(t_eval,u,fill_value="extrapolate")(tt)
    #v=interp1d(t_eval,v,fill_value="extrapolate")(tt)
    
    
    return u,v

def loss_slab_noshear(x,y,tt,taux,tauy,f,weight,T=[],has_tides=False): 
    """
    the loss function
    
    x: the parameter space [u0,v0,H,c,U_x=0,U_y=0,V_x=0,V_y=0,a,b,c,d, amp_u, phase_u, amp_v, phase_v]
    
    use the parameters in x, integrate a slab model forward, compare to the truth giving in y
    
    y: observations, y[:n/2]=u and y[n/2:]=v
    
    tt: time axis in days (referenced to 1979-01-01 00:00:00) or array of datetime64
    
    taux, tauy : wind stress function, the time axis is already shifted to referenece to 1979-01-01
    
    f: coriolis param

    T: list of periods of the tidal constituents to be considered
    
    """
    #if 'time' in str(type(tt[0])):
    #    tt=(tt-np.datetime64('1979-01-01 00:00:00'))/np.timedelta64(1,'D')
    import sys
    
    u,v=predict_slab_noshear(x,tt,taux,tauy,f)
    
    if has_tides:
        u_tide,v_tide=get_tide(tt,T,x[-len(T)*4:])

        u-=u_tide
        v-=v_tide
    
    

    loss=(np.r_[u, v].flatten() - y) * weight
    
    return loss



def optimize_slab_noshear_withtide(t_uv,u,v,t_tau,taux,tauy,f0,
                                   has_tides=False,
                                   t_out=[],
                                   T_tide=[1.07581, 0.99727, 0.517525, 0.5, 0.2587625, 0.2587625],
                                   is_windstress=False,c_clim=40,c_max=700,c_min=10,use_weight=False):
    
    """
    t_uv: time axis for input u and v, np.datetime64 or pandas.date_range
    
    u,v: Array (N), zonal and meridonal velocities on t_uv axis
    
    t_tau: time axis for taux, tauy, can be different from t_uv
    
    taux,tauy: Array, windstress on t_tau on t_tau axis,
    
    f0: coriolis parameter
    
    t_out: the time axis of the output, array of np.datetime64/pd.date_range
    
    T_tide: periods of the tidal constituents (unit: day)
            default values correspond to [O1, K1, M2, S2, M4, M6]
    """
    import pandas as pd
    from scipy.optimize import least_squares
    import xarray as xr

    #######################################
    ## The following has tides
    #######################################
    #without shear, with bias, with tides The last four numbers are for the tidal amplitude and phase (u,v)

    T_tide=np.array(T_tide)
    
    omg=2*np.pi/(T_tide*86400)
    msk=(omg>0.8*f0)&(omg<1.2*f0)
    
    if msk.sum()>0 and has_tides:
        T_tide=T_tide[msk]
        print("include tides with periods of ",T_tide)
        
    if has_tides:
        #T_tide=T_tide[1:4]
        x0=[u[0],v[0],c_clim,2e-6,0,0,]+[0,0,0,0]*len(T_tide)
        #print(x0)

          
        bounds = [[-3,-3,c_min,2e-7,-3,-3] + [-np.inf]*4*len(T_tide),
                  [3,3,c_max,2e-5,3,3] + [np.inf]*4*len(T_tide)]
        
    else:
        x0=[u[0],v[0],c_clim,2e-6,0,0]
        bounds = [[-3,-3,c_min,1e-6,-3,-3],
                  [3,3,c_max,1e-4,3,3]]
        
        # bounds are:
        # -3 to 3 m/s currents (pretty wide)
        # 5 to 2000m mixed layer; fairly reasonable, not including 0 incase of numerical problems (Jinbo changed min,max to 10,700m)
        # 2e-7 to 2e-5; 1 order of magnitude around orginal c value; no basis (Jinbo changed to 1e-8,1e-4)
        # -3 to 3 m/s current bias (very wide)
        
   # print(bounds)
   # print(np.shape(bounds))
    uv_truth=np.r_[u,v].flatten()
    
    #win=np.hanning(u.size)
    #weight=np.r_[win,win].flatten()
    x=np.linspace(-np.pi/2+np.pi/40,np.pi/2-np.pi/40,u.size)
    weight=1
    if use_weight:
        weight=np.r_[np.cos(x),np.cos(x)].flatten()
    
    
    if not is_windstress:
        #convert wind speed to windstress
        Cd=1.2e-3
        rho_air=1.22
        rho_water=1027.5
        taux=Cd*rho_air*np.abs(taux)*taux/rho_water
        tauy=Cd*rho_air*np.abs(tauy)*tauy/rho_water
        
    #re-align the time axis
    
    t_tau_days=(t_tau-np.datetime64('1979-01-01 00:00:00'))/np.timedelta64(1,'D') 
    t_uv_days=(t_uv-np.datetime64('1979-01-01 00:00:00'))/np.timedelta64(1,'D') 
    t_out0=pd.date_range(t_uv.min(),t_uv.max(),freq='1h')
    t_out_days=(t_out0-np.datetime64('1979-01-01 00:00:00'))/np.timedelta64(1,'D') 
    
    
    #interpolate onto arbitrary time axis
    taux_func=interp1d(t_tau_days,taux,fill_value="extrapolate")
    tauy_func=interp1d(t_tau_days,tauy,fill_value="extrapolate")
    
    res_lsq = least_squares(loss_slab_noshear, x0, bounds=bounds,loss='cauchy',
                        args=(uv_truth, t_uv_days, taux_func, tauy_func, f0, weight, T_tide,has_tides))
    
        
    #for i,vn in enumerate(['u0','v0','H','c','bias_u','bias_v','amp_u','phase_u','amp_v','phase_v']):
    #    print(vn,res_lsq.x[i])

    #HH[ii] = res_lsq.x[2]
    #CC[ii] = res_lsq.x[3]        

    u_pred,v_pred = predict_slab_noshear(res_lsq.x,t_out_days,taux_func,tauy_func,f0)
    
    if has_tides:
        
        u_tide,v_tide=get_tide(t_out_days,T_tide,res_lsq.x[-4*len(T_tide):])

        u_pred-=u_tide
        v_pred-=v_tide

    u_pred=xr.DataArray(u_pred,dims=('time'),coords={'time':t_out0})
    v_pred=xr.DataArray(v_pred,dims=('time'),coords={'time':t_out0})
    
    if len(t_out)>0:
        u_pred=u_pred.interp(time=t_out)
        v_pred=v_pred.interp(time=t_out)
    #print('cost=',(((u.data-u_pred)**2/2+(v.data-v_pred)**2)/2).mean()**0.5)

    dout=xr.Dataset({'nio_u_predicted':u_pred,'nio_v_predicted':v_pred,
                     'H':res_lsq.x[2],'c':res_lsq.x[3],'cost':res_lsq.cost,'success':res_lsq.success})
    
    return dout
    

def harmonics(x,tt,periods):
    """
    tt: numpy Array, time in days
    periods: list of periods (in days) of the tidal constituents to be considered
    x: A list with size of N x 4, where N is the lenght of T.
           For each constituent, the param corresponds to [amp, phase], the last two correspond to a linear trend. 
    
    """
    import numpy
    from scipy.interpolate import CubicSpline

    n=len(periods)
    xx=np.array(x).reshape(-1,4)[:n+1,:]
    
    u=xx[-1,0]+xx[-1,1]*tt 
    v=xx[-1,2]+xx[-1,3]*tt
    
    for i, period in enumerate(periods):
        omg=2*numpy.pi/period
        u=u+xx[i,0]*numpy.cos(omg*tt+xx[i,1])
        v=v+xx[i,2]*numpy.cos(omg*tt+xx[i,3])

    #t0=tt-tt[0]
    #t0=t0/t0.max()
    
    #modulation=CubicSpline(np.linspace(0,1,8),x[-8:])(t0)
    #modulation=1
    
    #u=u*modulation
    #v=v*modulation
    
    return np.c_[u,v]

def loss_harmonics(x,y,tt,periods,weight): 
    """
    the loss function
    
    x: The parameter space [u_amplitude, u_phase, v_amplitude, v_phase] * (N+1) where N is the number of harmonics considered.
        
    y: complex, observations, 
    
    tt: time axis in days (referenced to 1979-01-01 00:00:00) or array of datetime64
        
    f: coriolis param
    
    """
    
    uv=harmonics(x,tt,periods)
    
    loss=(uv - y) #* weight
    
    return loss.flatten()

def optimize_harmonics(u,v,f0,t_out=[],
                      periods=[1.07581, 0.99727, 0.517525, 0.5, 0.2587625, 0.2587625],
                      include_nio=True):
    
    """
    t_uv: time axis for input u and v, np.datetime64 or pandas.date_range
    
    u,v: Array (N), zonal and meridonal velocities on t_uv axis
    
    t_tau: time axis for taux, tauy, can be different from t_uv
    
    taux,tauy: Array, windstress on t_tau on t_tau axis,
    
    f0: coriolis parameter
    
    t_out: the time axis of the output, array of np.datetime64/pd.date_range
    
    T_tide: periods of the tidal constituents (unit: day)
            default values correspond to [O1, K1, M2, S2, M4, M6]
    """
    import pandas as pd
    from scipy.optimize import least_squares
    import xarray as xr

    #######################################
    ## The following has tides
    #######################################
    #without shear, with bias, with tides The last four numbers are for the tidal amplitude and phase (u,v)
    if include_nio:
        periods=np.r_[periods,2*np.pi/f0/86400]
        
    omg=2*np.pi/(np.array(periods)*86400)
    
    x0=np.zeros( ( len(periods)*4 + 4 + 8,) ).flatten() #the first additional 4 points are for linear trend, the last 8 points are for the envelop
    x0[-8]=1.0
    uv=np.c_[u.values,v.values]
    
    weight=np.hanning(u.size)
    weight=np.c_[weight,weight]/weight.max()
    
    #x=np.linspace(-np.pi/2+np.pi/40,np.pi/2-np.pi/40,u.size)
    #weight=np.r_[np.cos(x),np.cos(x)]
    
    t_uv=u.time.values
    t_uv_days=(t_uv-np.datetime64('1979-01-01 00:00:00'))/np.timedelta64(1,'D') 
  
    res_lsq = least_squares(loss_harmonics, x0, loss='cauchy',
                        args=(uv,t_uv_days,periods,weight)) 
    #print(res_lsq)
    if len(t_out)==0:
        t_out=pd.date_range(t_uv[0],t_uv[-1]+np.timedelta64(1,'D'),freq='1h')
    t_out_days=(t_out-np.datetime64('1979-01-01 00:00:00'))/np.timedelta64(1,'D') 
        
    uv = harmonics(res_lsq.x,t_out_days,periods)
    
    u_pred=xr.DataArray(uv[:,0],dims=('time'),coords={'time':t_out})
    v_pred=xr.DataArray(uv[:,1],dims=('time'),coords={'time':t_out})
    
    return u_pred,v_pred, res_lsq.x
    

def create_wacm_from_mooring(ss,cutoff_truth=1/4,
                             cutoff_wacm=1/4,
                             orbit='700_1800',
                             remove_super_high=False,
                             velocity_type='obs'):
    """
    ss is the station name chosen from ss=['PaPa2','KEO2','NTAS2','Stratus2','WHOTS2']
    velocity_type: string, one from ['obs','mod5','modx','modt']
    """
    import xarray as xr
    dd=xr.open_dataset('data/from_Hong/%s.h5'%ss)
    din=[]
    vt=velocity_type
    for key in ['UU_%s'%vt,'VV_%s'%vt,'Uwind_o','Vwind_o']:
        din.append(dd[key])
    
    d700=synthetic_wacm(din,dd.lat0,dd.lon0,
                        cutoff_truth=cutoff_truth,
                        cutoff_wacm=cutoff_wacm,
                        remove_super_high=remove_super_high,
                        orbit=orbit)
    del dd
    
    return d700

import xarray as xr

def create_wacm_from_model(model_ds,cutoff_truth=1/4,
                             cutoff_wacm=1/4,
                             orbit='700_1800',
                             remove_super_high=False,
                             lat_i=0,lon_i=0,times=None):

    
    # din = u_truth,v_truth,uw_truth,vw_truth


    din = model_ds.getModelLatLon(lat_i,lon_i) #[model_ds.U,model_ds.V,model_ds.TX,model_ds.TY]
    
    
    #model_ds.colocateModelPoints(lat_is,lon_is,times) # returns u,v,tx,ty
    
    d700=synthetic_wacm(din,lat_i,lon_i,
                        cutoff_truth=cutoff_truth,
                        cutoff_wacm=cutoff_wacm,
                        remove_super_high=remove_super_high,
                        orbit=orbit)
    
    return d700

