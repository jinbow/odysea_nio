"""
assemble 2d field in history files into one netcdf file

Usage:

python assemble.history.py result
"""

import xarray as xr
import os,glob
import numpy as np


fns=sorted(glob.glob('result/his_ms/*nc'))

time=[]
for fn in fns:
    size=os.stat(fn).st_size/1024**3
    if size>3: #larger than 3G
        d=xr.open_dataset(fn)
        try:
            dout=np.concatenate((dout,d['v'][:,-1,...].values),axis=0)
        except:
            dout=d['v'][:,-1,...].values
        time.append(d['time'].values)
        del d
        print(dout.shape,len(time),fn)
    else:
        os.popen('rm -f %s'%fn)
        print('delete %s'%fn)

time=np.array(time).flatten()
dd=xr.DataArray(dout,name='v',dims=('time','lat','lon'),coords={'time':time})
dd.to_netcdf('result/analyzed/v_his.nc')
