import pylab as plt
import numpy as np


def load_wirewalker():
    tz=np.loadtxt('press-rbr-pass.csv',skiprows=2,delimiter=',')
    ts=np.loadtxt('sal-rbr-pass.csv',skiprows=2,delimiter=',')
    tt=np.loadtxt('temp-rbr-pass.csv',skiprows=2,delimiter=',')
    time=(tz[:,0]-tz[0,0])/86400000.
    plt.scatter(time,tz[:,1],c=tt[:,1],s=5,vmin=5,vmax=20,cmap=plt.cm.jet)
    plt.colorbar()
    plt.ylabel('depth (m)')
    plt.title('wirewalker temperature')
    plt.xlabel('time (day)')
    plt.show()



load_wirewalker()
