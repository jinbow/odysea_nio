"""utilities for making model bathymetry, initional conditions etc. """

def fill_holes(lat,lon,datain,mask_value=0):
    
    import numpy as np
    import popy
    import scipy.interpolate as itp
    
    print '============= datain',datain.sum()
    
    if not np.ma.is_masked(datain):
        datain = np.ma.masked_equal(datain, mask_value)
    
    if lat.ndim==1:
        lon,lat = np.meshgrid(lon,lat)
    
    lon1d,lat1d=lon.flatten(),lat.flatten()
    
    if datain.ndim==2:
        datain=datain[np.newaxis,...]
    
    
    
    data_new=np.zeros_like(datain)
    
    for k in range(data_new.shape[0]):
        data1d=datain[k,...].flatten()
        ip = ~data1d.mask
        if ip.sum() !=0:
            data_new[k,...] = itp.griddata((lon1d[ip],lat1d[ip]), 
                                       data1d.data[ip], (lon,lat), 
                                       method='nearest', fill_value=0)
    del data1d, datain, lon,lat,lon1d,lat1d
    print data_new    
    return data_new