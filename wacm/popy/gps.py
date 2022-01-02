"""
Routines to deal with GPS data

Jinbo Wang jinbo.wang@jpl.nasa.gov

"""

class gps:

    def __init__(self,time,height,ib=None,tides=None,bp=None):
        self.ssh=height
        self.time=time
        self.ib=ib
        self.tides=tides
        self.bp=bp
        return


    def bin(self,dt,window0,window1,nmode=3,fnout=''):
        """
        use pyhht to remove high frequency surface waves first
        then use simply boxcar to get low frequency ssh
        
        dt: time steps for the final time series
        window0: window size for the averaging
        window1: window size for EMS analysis.
        nmode: the index of EMD mode from which the final averaging is conducted.
        
        """
        import pyhht
        import xarray as xr
        import numpy as np
        from popy import myio
        
        newt=np.arange(self.time[0]+window1, self.time[-1], dt)

        nn=newt.size

        nvalues=[]
        binned=[]
        for i, tt in enumerate(newt):
            print(i,nn,i/nn)
            msk=(self.time>tt-window1)&(self.time<=tt+window1)
            
            nvalues.append(msk[window1-window0//2:window1+window0//2].sum()/window0)
            try:
                ff=pyhht.EMD(np.array(self.ssh)[msk]).decompose()
                binned.append(ff[nmode:,window1-window0//2:window1+window0//2].sum(axis=0).mean(axis=-1))
            except:
                binned.append(np.nan)
            del msk,ff
        self.bin_time=newt
        self.bin_nvalues=np.array(nvalues)
        self.bin_ssh=np.array(binned)

        if fnout!='':
            myio.saveh5s(fnout,{'time':newt.astype('>f8'),
                                'gps':np.array(binned).astype('>f8'),
                                'missing':np.array(nvalues).astype('>f8')})
            
        return 

    
def c_swh_noise(dd):
    from scipy import signal
    """calculate the swh and gps noise from gps 1Hz data
    dd: array type, 1Hz GPS data
    
    return:
    =======
    
    swh: wave height
    va: gps noise assuming 0.4cpm<fs<0.6cpm frequency band is the noise floor. 
    
    """
    
    swh=signal.detrend(dd.reshape(-1,360),axis=-1).std(axis=-1).mean()*4./100.0
    a,b=signal.welch(dd,noverlap=0,nfft=3600,nperseg=3600,window='hanning',detrend='linear',fs=60.0)
    va=b[(a>0.3)&(a<0.9)].mean()/60.
    
    return swh, va 





