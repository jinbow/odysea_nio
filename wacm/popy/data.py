'''
various data type

@author: Jinbo Wang <jinbo@gmail.com>
@organization: Scripps Institution of Oceanography
@since: April 17, 2013
'''


def EnsembleSSHEddy():
    import sqlite3
    from numpy import array, nan, genfromtxt,arange
    from datetime import datetime
    from netCDF4 import Dataset
    import os
    root_path = '/net/mazdata2/jinbo/Project/OBS/Chelton/'
    fn = root_path+'tracks.20130125.nc'
    fndb = fn.replace('.nc','.db')
    if os.path.exists(fndb):
        os.remove(fndb)
        
    f = Dataset(fn, 'r')
    fv = f.variables
    nrecords = len(fv['track'][:])    
    conn = sqlite3.connect(fn.replace('.nc','.db'))
    
    c= conn.cursor()
    names = ['track', 'n', 'j1', 'cyc', 'lon', 'lat', 'A', 'L', 'U']
    
    c.execute("""CREATE TABLE eddy (track real, n real, j1 real, cyc real,
                 lon real, lat real, A real, L real, U real)""")
    for i in range(nrecords):
        a=[]
        for name in names:
            a.append(fv[name][i].astype('f8'))
        c.execute('INSERT INTO eddy VALUES (?,?,?,?,?,?,?,?,?)', a)
        print i
    conn.commit()
    conn.close()
    return

def SSHEddy():
    return

def RossbyDeformation(p):
    import sqlite3
    from numpy import array, nan
    conn=sqlite3.connect('/net/mazdata2/jinbo/Project/Eclipse/popy/data/RossbyRadius_Chelton.db')
    c=conn.cursor()
    c.execute('SELECT r1 FROM rossby WHERE lon>? AND lon<? AND lat>? AND lat<?', p)
    d = array(c.fetchall())
    if len(d)==0:
        d=nan
    else:
        d= d.mean()
    conn.close()
    return d

def EnsembleArgo():
    import sqlite3
    from numpy import array, nan, genfromtxt,arange
    from datetime import datetime
    root_path = '/net/mazdata2/jinbo/Project/OBS/argo/dac/'
    fn = root_path+'ar_index_global_prof.txt'
    d = genfromtxt(fn, dtype='S64,S16,f8,f8,S8,S8,S8,S16', comments='#', delimiter=',',skip_header=9)
    
    conn = sqlite3.connect('/net/mazdata2/jinbo/Project/Eclipse/popy/data/Argo_meta.db')
    c= conn.cursor()
    c.execute("""CREATE TABLE argo (filename text, dates text, 
                 lat real, lon real, ocean text, 
                 profiler_type text,institution text,
                 date_update text)""")
    for a in d:
        c.execute('INSERT INTO argo VALUES (?,?,?,?,?,?,?,?)', a)
    conn.commit()
    conn.close()
    return 

def ConnectArgo():
    import sqlite3
    conn=sqlite3.connect('/net/mazdata2/jinbo/Project/OBS/argo/dac/Argo_meta.db')
    return conn

def QueryArgo(dates, p):
    """
    d,mld=popy.data.QueryArgo(['200211010000','200212300000'],[170,180,50,60])
    the mixed layer depth data is from Holte and Talley 2009.
    """
    import numpy as np
    from netCDF4 import Dataset
    if p[0]>180:
        p[0]=p[0]-360
    if p[1]>180:
        p[1]=p[1]-360
        
    conn=ConnectArgo()
    c=conn.cursor()
    timeregion=[p[0],p[1],p[2],p[3],dates[0],dates[-1]]
    
    c.execute('select filename from argo where lon<1000 and lon>? and lon<? and lat>? and lat<? and dates>? and dates<?', timeregion)
    a=c.fetchall()
    data=[]
    root_path = '/net/mazdata2/jinbo/Project/OBS/argo/dac/'
    for filename in a:
        f=Dataset(root_path+filename[0],'r') 
        temp=f.variables['TEMP'][:].flatten()
        pres=f.variables['PRES'][:].flatten()
        data.append({'temp':temp,'pres':pres})
        f.close()
    conn.close()
    
    conn=MixedLayerDepth()
    c=conn.cursor()
    c.execute('select mld from mld where lon>? and lon<? and lat>? and lat<? and dates>? and dates<?', timeregion)
    mld=np.array(c.fetchall())
    conn.close()
    return data, mld

def QueryMLD(dates, p):
    """
    d,mld=popy.data.QueryArgo(['200211010000','200212300000'],[170,180,50,60])
    the mixed layer depth data is from Holte and Talley 2009.
    """
    import numpy as np
    if p[0]>180:
        p[0]=p[0]-360
    if p[1]>180:
        p[1]=p[1]-360
    timeregion=[p[0],p[1],p[2],p[3],dates[0],dates[-1]]
    conn=MixedLayerDepth()
    c=conn.cursor()
    c.execute('select mld from mld where lon>? and lon<? and lat>? and lat<? and dates>? and dates<?', timeregion)
    mld=np.array(c.fetchall())
    conn.close()
    return mld

def ArgoExample(dates, p):
    from pylab import savefig,subplot,diff
    import os
    data = QueryArgo(dates, p)
    print 'number of records found,',len(data)
    ax1=subplot(121)
    ax2=subplot(122)
    for d in data:
        ax1.plot(d['temp'],-1*d['pres'])
        ax2.plot(-1*diff(d['temp'])/diff(d['pres']), -1*d['pres'][:-1])
    ax1.set_ylim(-2000,0)
    ax1.set_ylim(-2000,0)
    savefig('/tmp/tmp.png')
    os.popen('eog /tmp/tmp.png')
    return

def MixedLayerDepth():
    import sqlite3,os
    from netCDF4 import Dataset
    import datetime
    rootpath = '/net/mazdata2/jinbo/Project/OBS/argo/HolteAndTalley2009/'
    fndb = rootpath+'mldinfo_varDT.db'
    if not os.path.exists(fndb):
        fn_org = rootpath+'mldinfo_varDT.nc'
        f = Dataset(fn_org,'r'); fv=f.variables
        lons,lats,mlds,dates=fv['longitude'][:].flatten(),fv['latitude'][:].flatten(),\
                             fv['da_mld'][:].flatten(),fv['date'][:].flatten()
        d0 = datetime.datetime(2000,01,01,00,00,00)
        dnames=[]
        for i in range(len(dates)):
            dstr = (d0+datetime.timedelta(days=(dates[i]-730486.)) ).strftime('%Y%m%d%H%M%S')
            dnames.append(dstr)
            print dstr
        del dates
        conn = sqlite3.connect(fndb)
        c= conn.cursor()
        c.execute("""CREATE TABLE mld (dates text, lat real, lon real, mld real)""")
        for i in range(len(lons)):
            c.execute('INSERT INTO mld VALUES (?,?,?,?)', [dnames[i],lats[i],lons[i],mlds[i]])
        conn.commit()
        conn.close()
    conn = sqlite3.connect(fndb)
    return conn
    

class DIMES_ctd:
    def __init__(self,cruise):
        
        self.cruise=cruise
        return
    
    