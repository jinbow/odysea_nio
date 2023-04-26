"""
Some utility routines
"""


def lat2f(d):
    """ Calculate Coriolis parameter from latitude
    d: latitudes, 1- or 2-D
    """
    from numpy import pi, sin
    return 2.0*0.729e-4*sin(d*pi/180.0)


def filter_butter(data,cutoff,fs, btype,filter_order=4,axis=0):
    """filter signal data using butter filter.

    Parameters
    ==================
    data: N-D array
    cutoff: scalar
        the critical frequency
    fs: scalar
        the sampling frequency
    btype: string
        'low' for lowpass, 'high' for highpass, 'bandpass' for bandpass
    filter_order: scalar
        The order for the filter
    axis: scalar
        The axis of data to which the filter is applied

    Output
    ===============
    N-D array of the filtered data

    """

    import numpy as np
    from scipy import signal

    if btype=='bandpass':
            normal_cutoff=[cutoff[0]/0.5/fs,cutoff[1]/0.5/fs]
    else:
        normal_cutoff=cutoff/(0.5*fs)  #normalize cutoff frequency
    b,a=signal.butter(filter_order, normal_cutoff,btype)
    y = signal.filtfilt(b, a, data, axis=axis)

    return y