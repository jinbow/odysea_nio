import xarray as xr
import numpy as np
import datetime
import warnings
import copy
import itertools
import glob
import matplotlib.pyplot as plt
import os
import pandas as pd

warnings.simplefilter(action='ignore')


class wacmLatLon:
    
    
    def __init__(self,config_fname=None):
                 
        self.loadSampling(fn = '../src/wacm_simple_orbit_s2012-01-01 21:11:04.301710e2012-01-05 20:54:17.444398.nc')
        
    def loadSampling(self,fn):
        
        self.sampling = xr.open_dataset(fn,decode_times=True)
        
        self.min_time = np.nanmin(self.sampling['sample_time'].values)
        self.max_time = np.nanmax(self.sampling['sample_time'].values)
        self.dt = self.max_time-self.min_time
    
    def getSamplingTimes(self,lats,lons,start_time,end_time):
                   
        times = []

        ds = self.sampling.sel(lat=xr.DataArray(lats, dims='z'),
                               lon=xr.DataArray(lons, dims='z'),
                               method='nearest')
        
        
        for idx in range(len(lats)):
            t = ds['sample_time'].values[:,idx]
            t = t[np.isfinite(t)]
            times.append(t)

        
        n_repeats = np.ceil((end_time - start_time)/np.timedelta64(4,'D'))
                           
        n_points = len(lats)
        repeated_times = copy.copy(times)

        for r in np.arange(1,n_repeats):
            offset = np.timedelta64(4,'D') * r
            for pt_idx in range(n_points):
                
                add_times = times[pt_idx] + offset
                repeated_times[pt_idx] = np.append(repeated_times[pt_idx],add_times)

        
        orbit_start_offset = start_time - self.min_time
        for pt_idx in range(n_points):
            repeated_times[pt_idx] = repeated_times[pt_idx] + orbit_start_offset 
            repeated_times[pt_idx] = repeated_times[pt_idx][repeated_times[pt_idx] < end_time]

        return repeated_times
    

    
    def getErrors(self,size=1,resolution=5000,etype='baseline'):
        
        if etype=='baseline':
            base_std = .30
            
        elif etype=='low':
            base_std = .20
            
        elif etype=='threshold':
            base_std= .50
        
        n_samples = (resolution/5000)**2
        std = base_std/np.sqrt(n_samples)
        
        errors = np.random.normal(scale=std,size=size)
        
        return errors