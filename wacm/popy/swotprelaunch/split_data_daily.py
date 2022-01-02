"""
some commonly used routines for analyzing the prelaunch field campaign data

"""

import numpy as np
import gsw
import datetime
import sys
import myio

data_path='/home1/jwang23/projects/swot/prelaunch.data/'

def load_sio_wirewalker():
    """ctime: time center in hours
    dtime: the time range to subtract, unit:hours"""

    import scipy as sp

    pth=data_path+'sio/swot1_01/csv/'
    pressure=np.genfromtxt(pth+'press-ww.csv',skip_header=4,delimiter=',')
    salt=np.genfromtxt(pth+'sal-ww.csv',skip_header=4,delimiter=',')
    temp=np.genfromtxt(pth+'temp-ww.csv',skip_header=4,delimiter=',')
    time=pressure[:,0]/1e3

    t0=datetime.datetime(1970,1,1)

    tend=t0+datetime.timedelta(seconds=time[-1])
    print(tend)

    t1=datetime.datetime(2019,9,7)

    while t1+datetime.timedelta(days=1)<tend:
        print(t1)
        center=(t1-t0).total_seconds()
        print(center)
        msk = (time>center - 43200 - 3600)&(time<center + 43200 + 3600)
        pre=pressure[msk,1]
        tem=temp[msk,1]
        sal=salt[msk,1]
        myio.saveh5('wirewalker_%s.h5'%t1.strftime('%Y%m%d%H'),'p', pre)
        myio.saveh5('wirewalker_%s.h5'%t1.strftime('%Y%m%d%H'),'temp', tem)
        myio.saveh5('wirewalker_%s.h5'%t1.strftime('%Y%m%d%H'),'salt', sal)
        myio.saveh5('wirewalker_%s.h5'%t1.strftime('%Y%m%d%H'),'time', time[msk])
        t1=t1+datetime.timedelta(days=1)
        print(t1)




def load_sio_microcat():
    pth=data_path+'sio/swot1_01/csv/'
    pressure=np.genfromtxt(pth+'press.csv',skip_header=4,delimiter=',')
    time=pressure[:,0]/1e3

    salt=np.genfromtxt(pth+'sal.csv',skip_header=4,delimiter=',')[:,1:]
    temp=np.genfromtxt(pth+'temp.csv',skip_header=4,delimiter=',')[:,1:]
    press=pressure[:,1:]

    t0=datetime.datetime(1970,1,1)

    tend=t0+datetime.timedelta(seconds=time[-1])
    print(tend)

    t1=datetime.datetime(2019,9,7)

    while t1+datetime.timedelta(days=1)<tend:
        print(t1)
        center=(t1-t0).total_seconds()
        print(center)
        msk = (time>center - 43200 - 3600)&(time<center + 43200 + 3600)
        pre=pressure[msk,1:]
        tem=temp[msk,:]
        sal=salt[msk,:]
        myio.saveh5('SIO_MC_%s.h5'%t1.strftime('%Y%m%d%H'),'p', pre)
        myio.saveh5('SIO_MC_%s.h5'%t1.strftime('%Y%m%d%H'),'temp', tem)
        myio.saveh5('SIO_MC_%s.h5'%t1.strftime('%Y%m%d%H'),'salt', sal)
        myio.saveh5('SIO_MC_%s.h5'%t1.strftime('%Y%m%d%H'),'time', time[msk])

        t1=t1+datetime.timedelta(days=1)
        print(t1)


    return


if __name__=='__main__':
    import pylab as plt
    load_sio_microcat()
    #load_sio_wirewalker()
