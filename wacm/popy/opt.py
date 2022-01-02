'''
Created on Oct 18, 2014

@author: jinbo Wang @ Scripps Institution of Oceanography

'''


def loadopt(fn,metafn='default',varns=['x','y'],usememmap=True):
    """load offline particles given data file name and metafilename"""
    import numpy as np
    import popy
    
    ds={}
    if np.isscalar(fn):
        fn=[fn]
    
    for varn in varns:
        for i,fnn in enumerate(fn):
            if metafn=='default':
                metafn=fnn+'.meta'
            meta=popy.mds.parsemeta(metafn)['dimList']
            shape=tuple(meta[::3][::-1])
            fnnn=fnn+'.%s.bin'%varn
            if usememmap:
                dd = np.memmap(fnnn,dtype='>f4',mode='r',shape=shape)
            else:
                dd = np.fromfile(fnnn, '>f4').reshape(shape)
            if i>0:
                print ds.keys()
                ds[varn]=np.r_[ds[varn][:],dd]
            else:
                ds[varn]=dd
            
    print "load particle file %s, data shape is "%fn, shape
    return ds
    
def bin_particles(x,y,parti_mass=1,weight=True,domain=[],dx=1,dy=1,ws=9,fout=None):
    """
    calculate particle-tracer density
    x,y contain particle locations of longitude and latitude,
    parti_mass is the mass per particle, default is 1
    domain=[xmin,xmax,ymin,ymax]
    dx,dy is the bin resolution
    ws is the window size for smoothing, default 9
    return: x, y, concentration
    """
    import popy
    import numpy as np

    if domain==[]:
        xmin,xmax=np.floor(x.min()),np.ceil(x.max())
        ymin,ymax=np.floor(y.min()),np.ceil(y.max())
    else:
        xmin,xmax,ymin,ymax=domain
        
    nx,ny=(xmax-xmin)/dx,(ymax-ymin)/dy
    
    ab,xe,ye=np.histogram2d(x,y,range=[[xmin,xmax],[ymin,ymax]],\
                            bins=[nx,ny])
  
    window=np.exp(-np.linspace(-4,4,ws)**2/2.0)
    window=window/window.sum()
    ab=ab.T
    nj,ni=ab.shape
    for i in range(ni):
        ab[:,i]=np.convolve(ab[:,i],window,mode='same')
    for j in range(nj):
        ab[j,:]=np.convolve(ab[j,:],window,mode='same')
    if weight:
        dxx,dyy=popy.map.distance_grids(xe,ye)
        ab=ab*parti_mass/dxx/dyy # mod/particle = 3.8765e-2
    xe=popy.utils.twopave(xe)
    ye=popy.utils.twopave(ye)
    
    if fout!=None:
        popy.io.saveh5(fout,'d',ab)
        popy.io.saveh5(fout,'xe',xe)
        popy.io.saveh5(fout,'ye',ye)

    return xe,ye,ab

def glue_opt(datapath,casename):
    import glob
    import numpy as np
    if datapath[-1] != '/':
        datapath=datapath+'/'

    fns=sorted(glob.glob(datapath+casename+'.XYZ.*.data'))

    print "total %i files"%len(fns)

    f=open('filenames_%s'%casename,'w')
    nrec=len(fns)
    d=np.fromfile(fns[0],'>f4').reshape(3,-1)
    nxy,nopt=d.shape
    del d
    t0=fns[0].split('.')[-2]
    t1=fns[-1].split('.')[-2]
    dds=np.memmap(casename+'.%s.%s.data'%(t0,t1),dtype='>f4',shape=(nrec,3,nopt),mode='write')
    
    for i in range(nrec):
        print i,'out of ',nrec
        d=np.fromfile(fns[i],'>f4').reshape(3,-1)
        f.writelines(fns[i]+'\n')
	dds[i,...]=d
	del d
    f.close()
    return

def interp_opt_2d(x,y,psi,xnew,ynew):
    from scipy.interpolate import RectBivariateSpline
   
    itf=RectBivariateSpline(x,y,psi.T)
    a=itf.ev(xnew.flatten(),ynew.flatten())
    a=a.reshape(xnew.shape)
    return a 

if __name__ == '__main__':
    pass
