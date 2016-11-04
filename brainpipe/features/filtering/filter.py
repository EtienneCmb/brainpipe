#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from .firls import fir1, fir_order
from scipy.signal import filtfilt, butter, bessel


def filter_fcn(sf, npts, f, filtname='fir1', cycle=3, order=3, ftype='bandpass', padlen=None, axis=0):
    """Get a filter according to filtname.

    :sf: sampling frequency
    :f: frequency band for filtering
    :npts: number of time points
    :filtname: name of the filter, either 'fir1', 'bessel' 'butter', None
    :cycle: number of cycles (fir1)
    :order: order of the filter (for butter / bessel)
    :ftype: filter type, either 'bandpass', 'lowpass' or 'highpass'. Only avaible for butter and bessel filters
    :padlen: pad length for fir1. If none, is set to forder
    :axis: filt along the axis
    :returns:the filtering function

    """
    # Input management :
    if filtname not in ['bessel', 'butter', 'fir1']:
        raise ValueError(
            "filtname must be either 'bessel', 'butter' or 'fir1'")
    if ftype not in ['bandpass', 'lowpass', 'highpass']:
        raise ValueError(
            "ftype must be either 'bandpass', 'lowpass' or 'highpass'")
    if (ftype is not 'bandpass') and filtname == 'fir1':
        raise ValueError(
            "For lowpass and highpass filters, use either 'butter' or 'bessel'")
    if (ftype is not 'bandpass') and not isinstance(f, (int, float)):
        raise ValueError('For lowpass and highpass filters, f must be a float or int')
    if (ftype == 'bandpass') and not isinstance(f, np.ndarray):
        f = np.array(f)
    if filtname is not 'fir1':
        f = np.multiply(f, 2 / sf)

    # fir1 filter :
    if filtname == 'fir1':
        fOrder = fir_order(sf, npts, f[0], cycle=cycle)
        b, a = fir1(fOrder, f / (sf / 2))
    # butterworth filter :
    elif filtname == 'butter':
        b, a = scisig.butter(order, f, btype=ftype)
        fOrder = None

    # bessel filter :
    elif filtname == 'bessel':
        b, a = scisig.bessel(order, f, btype=ftype)
        fOrder = None

    # Padlen in case of fir1 filter :
    if (padlen is None) and (filtname == 'fir1'):
        padlen = fOrder

    # Construct output function :
    if filtname is None:
        def filtSignal(x):
            return x
    else:
        def filtSignal(x):
            return scisig.filtfilt(b, a, x, padlen=fOrder, axis=axis)

    return filtSignal

