"""
work with zhongxiang zhao
"""

def load_zhongxiang_tracks():
    """
       Zhongxiang provided track information for conventional altimeter in the california current region
       Load all of them using this routine. 

    Returns:
    =========
       time (N): list
             datetime instance
       lon (N): array_like
             The longitude of the conventional altimeter Nadir points 
       lat (N): array_like
             The latitude of the conventional altimeter Nadir points 
    """
    import popy
    import numpy as np
    import datetime
 
    pth='/nobackup/jwang23/projects/conventional.altimeter/data/from.zhongxiang/Altimeter-tracks/'
    dd=popy.io.loadh5(pth+'altimeter.tracks.all.h5','tracks')
    time,lon,lat=dd[:,0],dd[:,1],dd[:,2]
    #times=[datetime.datetime(1,1,1)+datetime.timedelta(hours=tim) for tim in time]

    return time,lon,lat

