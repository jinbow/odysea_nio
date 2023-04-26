"""
Routines related to COAS model output on Bura

The routines were originally written by Alex Wineteer
"""

import glob 
import os

import sys
sys.path.append('../src')
sys.path.append('../')

import slab_model
import pylab as plt
import numpy as np
import xarray as xr

from scipy.io import loadmat
import pandas as pd
import pickle

xrod=xr.open_dataset

class COAS:
    

    def __init__(self,days=20):
        # note this is only appropriate for the coupled model as written on JPL Bura server.
        u_folder = '/u/bura-m0/hectorg/COAS/llc2160/coarse/Coarse/U/'
        v_folder =     '/u/bura-m0/hectorg/COAS/llc2160/coarse/Coarse/V/'
        tau_y_folder = '/u/bura-m0/hectorg/COAS/llc2160/coarse/Coarse/oceTAUY/'
        tau_x_folder = '/u/bura-m0/hectorg/COAS/llc2160/coarse/Coarse/oceTAUX/'

        u_files = np.sort(glob.glob(u_folder + '/*.nc'))[1:20*24]
        v_files = np.sort(glob.glob(v_folder + '/*.nc'))[1:20*24]
        tau_x_files = np.sort(glob.glob(tau_x_folder + '/*.nc'))[1:20*24]
        tau_y_files = np.sort(glob.glob(tau_y_folder + '/*.nc'))[1:20*24]

        self.U = xr.open_mfdataset(u_files,parallel=True,preprocess=addTimeDimCoarse)
        self.V = xr.open_mfdataset(v_files,parallel=True,preprocess=addTimeDimCoarse)
        self.TX = xr.open_mfdataset(tau_x_files,parallel=True,preprocess=addTimeDimCoarse)
        self.TY = xr.open_mfdataset(tau_y_files,parallel=True,preprocess=addTimeDimCoarse)

        self.U = self.U.rename({'U':'U'})
        self.V = self.V.rename({'V':'V'})
        self.TX = self.TX.rename({'oceTAUX':'TX'})
        self.TY = self.TY.rename({'oceTAUY':'TY'})

            
    
    def colocateModelPoints(self,lats,lons,times):
        
        
        if len(times) == 0:
            return [],[]
            
        ds_u =  self.U.interp(time=xr.DataArray(times.flatten(), dims='z'),
                            lat=xr.DataArray(lats.flatten(), dims='z'),
                            lon=xr.DataArray(lons.flatten(), dims='z'),
                            method='linear')

        ds_v =  self.V.interp(time=xr.DataArray(times.flatten(), dims='z'),
                            lat=xr.DataArray(lats.flatten(), dims='z'),
                            lon=xr.DataArray(lons.flatten(), dims='z'),
                            method='linear')
        
        ds_tx =  self.TX.interp(time=xr.DataArray(times.flatten(), dims='z'),
                            lat=xr.DataArray(lats.flatten(), dims='z'),
                            lon=xr.DataArray(lons.flatten(), dims='z'),
                            method='linear')

        ds_ty =  self.TY.interp(time=xr.DataArray(times.flatten(), dims='z'),
                            lat=xr.DataArray(lats.flatten(), dims='z'),
                            lon=xr.DataArray(lons.flatten(), dims='z'),
                            method='linear')
        

        u=np.reshape(ds_u['U'].values,np.shape(lats))
        v=np.reshape(ds_v['V'].values,np.shape(lats))
        tx=np.reshape(ds_tx['oceTAUX'].values,np.shape(lats))
        ty=np.reshape(ds_ty['oceTAUY'].values,np.shape(lats))

        return u,v,tx,ty
    
    def getModelLatLon(self,lats,lons):
        

        ds_u =  self.U.sel(lat=lats,method='nearest').sel(lon=lons,method='nearest').compute()

        ds_v =  self.V.sel(lat=lats,method='nearest').sel(lon=lons,method='nearest').compute()
        
        ds_tx =  self.TX.sel(lat=lats,method='nearest').sel(lon=lons,method='nearest').compute()
        
        ds_ty =  self.TY.sel(lat=lats,method='nearest').sel(lon=lons,method='nearest').compute()
        
        return ds_u.U,ds_v.V,ds_tx.TX,ds_ty.TY

def addTimeDim(ds):
    # helper for COAS model format with mfdataset
    ds = ds.isel(time=0)
    fn = os.path.basename(ds.encoding["source"])
    time_str = fn.split('_')[-1].split('.')[0]
    time = pd.to_datetime(time_str, format='%Y%m%d%H')
    ds = ds.expand_dims(time=[time])

    #display(ds)
    return ds


def addTimeDimCoarse(ds):
    # helper for COAS model format with mfdataset
    fn = os.path.basename(ds.encoding["source"])
    time_str = fn.split('.')[0].split('_')[-1]

    time = pd.to_datetime(time_str, format='%Y%m%d%H')
    ds = ds.expand_dims(time=[time])

    return ds

def normalizeTo180(angle):
    # note this strange logic was originally to use numba JIT
    for idx,ang in np.ndenumerate(angle):
    
        ang =  ang % 360
        ang = (ang + 360) % 360
        if ang > 180:
            ang -= 360
        angle[idx] = ang
        
    return angle
            
def fixLon(ds):

    ds['lon'].values = normalizeTo180(ds['lon'].values)

    return ds    

  



