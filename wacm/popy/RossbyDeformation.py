'''
Created on Sep 3, 2013

@author: jbw
'''
from tables import *

class RossbyDeformation(IsDescription):
    '''
       save rossby deformation by Chelton et al. 1997 into pytables
       "Rossby Deformation radius calculated by Chelton D. Downloaded from
        http://www-po.coas.oregonstate.edu/research/po/research/rossby_radius/"
    '''
    lon, lat = Float64Col(),Float64Col()
    c1, r1 = Float64Col(),Float64Col() 
    
    
if __name__=='__main__':
    from numpy import genfromtxt
    import tables as tbl
    import numpy as np
    
    fn = '/net/mazdata2/jinbo/Project/OBS/Chelton/'
    h5file = tbl.openFile("rossrad.h5", mode = "w", title = "Rossby deformation radius")
    group = h5file.createGroup("/", 'Chelton', 'data from Chelton')
    table = h5file.createTable(group, 'DeformationRadius', RossbyDeformation, "RossbyDeformation")
    rad = table.row
    
    fn = 'rossrad.dat'
    d = genfromtxt(fn, dtype='>f8')
    nj,ni=d.shape
    
    import sqlite3
    conn = sqlite3.connect('RossbyRadius_Chelton.db')
    c= conn.cursor()
    c.execute("""CREATE TABLE rossby (lat real, lon real, c1 real, r1 real)""")
    
    for j in np.arange(nj):
        print (d[j,:])
        c.execute('INSERT INTO rossby VALUES (?,?,?,?)', (d[j,:]) )
    conn.commit()
    conn.close()
        