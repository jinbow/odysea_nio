'''
Created on Sep 3, 2013

@author: Jinbo Wang
@contact: <jinbow@gmail.com>
@organization: Scripps Institution of Oceanography

'''

class sst:
    import pickle
    from numpy import loadtxt, arange
    from datetime import datetime, timedelta
    
    def __init__(self):
        import numpy as np
        self.root = '/net/mazdata2/jinbo/Project/OBS/oisst2/OI-daily-v2/NetCDF/'
        f1 = open(self.root + 'filelist.data')
        self.FileList = {'oisst2':[line.rstrip() for line in f1]}
        f1.close()
        self.lon = np.arange(1440) * 0.25 + 0.125
        self.lat = np.arange(720) * 0.25 - 89.875
        self.date = self.LoadDate()
        return
    

    def LoadDate(self, varn='oisst2'):
        days = []
        for fn in self.FileList[varn]:
            days.append(DateFromFile(fn))
        return days
    
    def FileName(self, YMD, vname='oisst2'):
        """timestampe in format :YYYYMMDD"""
        import popy
        if YMD=='mean':
            return self.root + vname + '_mean.pkl'
        
        YMD = popy.utils.Time2Str(YMD)
        fn = self.root + 'linkall/avhrr-only-v2.%s.nc' % YMD
        return fn

    def LoadGrid(self):
        import os, pickle
        from netCDF4 import Dataset
        fnpkl = self.root + 'grid.pkl'
        if os.path.exists(fnpkl):
            d = pickle.load(open(fnpkl, 'r'))
            return d
        else:
            fn = self.root + self.FileList['ssh'][10].strip()
            f = Dataset(fn, 'r').variables
            lat, lon = f['NbLatitudes'][:], f['NbLongitudes'][:]
            date = self.timestamps()
            g = {'lat':lat, 'lon':lon, 'date':date}
            pickle.dump(g, open(fnpkl, 'wb'))
            return g
        
    def LoadData(self, YMD='19921223', subdomain=[],index=True):
        from numpy import ma, c_, array
        from netCDF4 import Dataset
        import popy,os,pickle,glob
        
        fn = self.FileName(YMD)
        
        def load(fn):
            ff = Dataset(fn, 'r')
            f = ff.variables
            data = f['sst'][:].squeeze()
            data = c_[data[:, -1], data]
            data = (data[:, :-1] + data[:, 1:]) / 2. 
            data = ma.masked_outside(data, 1e-10, 40).squeeze()
            ff.close()
            return data
        
        if YMD=='mean':
            fnall=self.root+'sst_allinone_1993-2012.pkl'
            fnmean=self.root+'sst_mean1993-2012.pkl'
            if os.path.exists(fn):
                data=pickle.load(open(fnmean,'r'))
            else:
                mean=[]
                fnlist=sorted(glob.glob(self.root+'seven_day_mean/????????.pkl'))
                for fns in fnlist:
                    print fns
                    f=open(fns,'r')
                    mean.append(pickle.load(f))
                    f.close()
                mean=array(mean)
                pickle.dump(mean, open(fnall,'wb'))
                mean = mean.mean(axis=0)
                pickle.dump(mean, open(fnmean,'wb'))
                return mean
        else:
            data = load(fn)
            
        lon, lat = self.lon, self.lat
        
        if subdomain != []:
            lon, lat, data = popy.utils.subtractsubdomain(self.lon, self.lat, subdomain, data, index=index)
        
        return lon, lat, data
    
    def LoadDataMean(self, YMD='19921223', deltadays=3, subdomain=[]):
        from numpy import ma, c_, arange
        from netCDF4 import Dataset
        import popy,pickle,os
        from datetime import timedelta
        data=0
        YMDs=[]
        days = arange(2*deltadays+1)-deltadays
        pklfn = self.root+'seven_day_mean/%s.pkl'%popy.utils.Time2Str(YMD)
        if os.path.exists(pklfn):
            f=open(pklfn,'r')
            data = pickle.load(f)
            f.close()
        else:
            for d in days:
                fn = self.FileName(YMD+timedelta(days=int(d)))
                ff = Dataset(fn, 'r')
                f = ff.variables
                data += f['sst'][:].squeeze()/(2*deltadays+1.)
                ff.close()
                
            data = c_[data[:, -1], data]
            data = (data[:, :-1] + data[:, 1:]) / 2. 
            data = ma.masked_outside(data, 1e-10, 40).squeeze()
            f=open(pklfn,'w')
            pickle.dump(data, f)
            f.close()
        lon, lat = self.lon, self.lat
        if subdomain != []:
            lon, lat, data = popy.utils.subtractsubdomain(self.lon, self.lat, subdomain, data, index=True)
        
        return lon, lat, data
    
def DateFromFile(fn):
    """subtract time information from a filename"""
    import datetime
    fn = fn.split('.')[-2]
    i = 0
    return datetime.date(int(fn[i:i + 4]), int(fn[i + 4:i + 6]), int(fn[i + 6:i + 8]))

    
if __name__ == '__main__':
    from pylab import contourf, savefig, colorbar
    a = sst()
    
    print "test a.timestamps, there are %i files" % len(a.date)
    
    ssh = a.LoadData()
    print ssh.shape, ssh.max(), ssh.min()
    contourf(ssh);colorbar()
    savefig('tmp.png')
    
