"""
quantify dynamic height variance from different integartion depth
"""


import xarray as xr
import utils
import numpy as np

def dyn(zcut=500):
    pth='/home1/jwang23/nobackup/llc4320_stripe/calval_california/diamond/'
    df=xr.open_dataset(pth+'diamond_california_swath_i13.h5')
    print(df['YC'][150])
    print(df['XC'][150])
    theta=df['Theta'][-90*24:-30*24,:84,150]
    salt=df['Salt'][-90*24:-30*24,:84,150]
    rc=xr.open_dataset('/home1/jwang23/nobackup/llc4320_stripe/grid/grids.h5')['RC'][:84].squeeze()
    rf=xr.open_dataset('/home1/jwang23/nobackup/llc4320_stripe/grid/grids.h5')['RF'][:85].squeeze()

    a=np.zeros((720,6))

    for i in range(720):
        print(i)
        z_obs_rc=np.arange(5,500,10)
        dyn, part=utils.dyn_from_model(salt[i,:],theta[i,:],rc=rc,rf=rf,lat0=35.5,z_obs_rc=z_obs_rc)
        a[i,0]=dyn*100
        a[i,1]=100*part

        z_obs_rc=np.arange(5,700,10)
        dyn, part=utils.dyn_from_model(salt[i,:],theta[i,:],rc=rc,rf=rf,lat0=35.5,z_obs_rc=z_obs_rc)
        a[i,2]=100*part

        z_obs_rc=np.arange(5,1000,10)
        dyn, part=utils.dyn_from_model(salt[i,:],theta[i,:],rc=rc,rf=rf,lat0=35.5,z_obs_rc=z_obs_rc)
        a[i,3]=100*part

        z_obs_rc=np.arange(5,1500,10)
        dyn, part=utils.dyn_from_model(salt[i,:],theta[i,:],rc=rc,rf=rf,lat0=35.5,z_obs_rc=z_obs_rc)
        a[i,4]=100*part

        z_obs_rc=np.arange(5,2000,10)
        dyn, part=utils.dyn_from_model(salt[i,:],theta[i,:],rc=rc,rf=rf,lat0=35.5,z_obs_rc=z_obs_rc)
        a[i,5]=100*part


    print(a.std(axis=0))
    print((a-a[:,:1]).std(axis=0))
    print((a-a[:,:1]).var(axis=0))
    print((a-a[:,:1]).var(axis=0)/a[:,0].var())

    return


dyn()
