"""
routines related to MSDA ROMS version
"""


import pylab as plt
import sys,os
import numpy as np
import xarray as xr
import roms,utils

def llc2roms_horizontal(fns):
    """horizontal interpolation of llc4320 onto roms grid"""

    pth='/nobackup/jwang23/llc4320_stripe/calcoast4zhijin/9km_domain/'
    romsgrd=roms.load_roms_grid()

    for fn in fns:
        d=xr.open_dataset(fn)
        mitgrd=d
        lat,lon=mitgrd['lat'].values,mitgrd['lon'].values+360
        latv,lonu=mitgrd['lat_v'].values,mitgrd['lon_u'].values+360
        dout=[]
        for varn in ['theta','salt','U','V','SSH']:
            if varn in ['theta','salt','SSH']:
                lonn,latn='lon_rho','lat_rho'
                lon0,lat0=lon,lat
            elif varn == 'U':
                lonn,latn='lon_u','lat_rho'
                lon0,lat0=lonu,lat
            else:
                lonn,latn='lon_rho','lat_v'
                lon0,lat0=lon,latv
 

            lont,latt=romsgrd[lonn].values[0,:],romsgrd[latn].values[:,0]

            dd=np.ma.masked_equal(d[varn].values,0)
            dd.lat,dd.lon=lat0,lon0

            if varn=='SSH':
                dims=['t','z0','latn','lonn']
            else:
                dims=['t','z',latn,lonn]
    
            d2=xr.DataArray(utils.interp2d(dd,{'lat':latt,'lon':lont}),dims=dims,name=varn)
            print(d2.shape)
    
            dout.append(d2)

        fnout=fn.replace('.nc','_on_roms_xygrid.nc')

        xr.merge(dout).to_netcdf(fnout)

        print('Saved file to ',fnout)
        del d, mitgrd 

