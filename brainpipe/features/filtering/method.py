#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import scipy.signal as scisig
from .firls import fir1, fir_order
from .wavelet import morlet

__all__ = ['filter_fcn', 'transfo_fcn', 'feat_fcn', 'preproc_fcn']


def filter_fcn(sf, npts, f=None, filtname='fir1', cycle=3, order=3, ftype='bandpass', padlen=None):
    """Get a filter according to filtname.

    :sf: sampling frequency
    :npts: number of time points
    :f: frequency band for filtering
    :filtname: name of the filter, either 'fir1', 'bessel' 'butter', None
    :cycle: number of cycles (fir1)
    :order: order of the filter (for butter / bessel)
    :ftype: filter type, either 'bandpass', 'lowpass' or 'highpass'. Only avaible for butter and bessel filters
    :padlen: pad length for fir1. If none, is set to forder
    :returns:the filtering function

    """
    # -------------------------------------------------------------
    # Input management :
    if f is None:
        filtname = None
    if filtname not in ['bessel', 'butter', 'fir1', None]:
        raise ValueError(
            "filtname must be either 'bessel', 'butter' or 'fir1'")
    if ftype not in [None, 'bandpass', 'lowpass', 'highpass']:
        raise ValueError(
            "ftype must be either 'bandpass', 'lowpass' or 'highpass'")
    if (ftype is not 'bandpass') and filtname == 'fir1':
        raise ValueError(
            "For lowpass and highpass filters, use either 'butter' or 'bessel'")
    if (ftype is not 'bandpass') and not isinstance(f, (int, float)):
        if isinstance(f, (int, float)):
            raise ValueError(
                'For lowpass and highpass filters, f must be a float or int')
    if (ftype == 'bandpass') and not isinstance(f, np.ndarray):
        f = np.array(f)
    if (filtname in ['butter', 'bessel']) and (f is not None):
        f = np.multiply(f, 2 / sf)

    # -------------------------------------------------------------
    # Function management :
    # fir1 filter :
    if filtname == 'fir1':
        fOrder = fir_order(sf, npts, f[0], cycle=cycle)
        b, a = fir1(fOrder, f / (sf / 2))
        sup = ', order=' + str(fOrder) + ', cycle=' + str(cycle)

    # butterworth filter :
    elif filtname == 'butter':
        b, a = scisig.butter(order, f, btype=ftype)
        fOrder = None
        sup = ', order=' + str(order) + ', type=' + ftype

    # bessel filter :
    elif filtname == 'bessel':
        b, a = scisig.bessel(order, f, btype=ftype)
        fOrder = None
        sup = ', order=' + str(order) + ', type=' + ftype

    # None :
    else:
        sup = ''

    # Padlen in case of fir1 filter :
    if (padlen is None) and (filtname == 'fir1'):
        padlen = fOrder

    # Construct output function :
    if filtname is None:
        def filtFcn(x):
            return x
    else:
        def filtFcn(x):
            return scisig.filtfilt(b, a, x, padlen=fOrder)

    # -------------------------------------------------------------
    # String management :
    filtStr = 'Filter(name=' + str(filtname) + '{})'.format(sup)

    return filtFcn, filtStr


def transfo_fcn(sf, f=None, transname='hilbert', width=7.0):
    """Get either hilbert or wavelet transform.

    :sf: sampling frequency, float
    :f: frequency for filtering (wavelet)
    :transname: either 'hilbert' or 'wavelet'
    :returns: a transformation function

    """
    # Check inputs :
    if transname not in ['hilbert', 'wavelet', None]:
        raise ValueError(
            "transname must be either 'hilbert' or 'wavelet' or None")
    if (transname == 'wavelet') and (f is None):
        raise ValueError(
            'When using wavelet, f can not be None. Change f before.')
    # Transformation function :
    if transname == 'hilbert':
        def transFcn(x):
            return scisig.hilbert(x)
        sup = ''
    elif transname == 'wavelet':
        def transFcn(x):
            return morlet(x, sf, f, width)
        sup = ', width=' + str(width)
    elif transname is None:
        def transFcn(x):
            return x
        sup = ''
    transStr = 'Transformation(name=' + str(transname) + '{})'.format(sup)

    return transFcn, transStr


def feat_fcn(featinfo=None):
    """Return a function to compute either the amplitude, power, phase or
    identity features.

    :featinfo: either 'amplitude', 'power', 'phase', None
    :returns: the features featinfo function

    """
    # Check input :
    if featinfo not in ['amplitude', 'power', 'phase', None]:
        raise ValueError(
            "featinfo must be either 'amplitude', 'power', 'phase' or None")
    # Features featinfo function :
    if featinfo == 'amplitude':
        def featinfoFcn(x):
            return np.abs(x)
    elif featinfo == 'power':
        def featinfoFcn(x):
            return np.square(np.abs(x))
    elif featinfo == 'phase':
        def featinfoFcn(x):
            return np.angle(x)
    elif featinfo is None:
        def featinfoFcn(x):
            return x
    # Features featinfo string :
    featinfoStr = 'Feature(featinfo=' + str(featinfo) + ')'

    return featinfoFcn, featinfoStr


def preproc_fcn(detrend=False, demean=False, axis=0):
    """Preprocess x.

    :detrend: remove linear trend, bool
    :demean: remove the mean, bool
    :axis: to apply transformation
    :returns: preprocessed x

    """
    # Detrend function :
    if detrend:
        def dtrd_fcn(x, axis=0):
            return scisig.detrend(x, axis=axis)
    else:
        def dtrd_fcn(x, axis=0):
            return x
    # Demean function
    if demean:
        def demn_fcn(x, axis=0):
            return np.subtract(x, x.mean(axis))
    else:
        def demn_fcn(x, axis=0):
            return x
    # Link both :

    def preprocFcn(x, axis=0):
        x = demn_fcn(x, axis=axis)
        x = dtrd_fcn(x, axis=axis)
        return x
    # String :
    preprocStr = 'Preprocessing({})'
    if detrend or demean:
        sup = 'detrend=' + str(detrend) + ', demean=' + str(demean)
    else:
        sup = 'None'

    return preprocFcn, preprocStr.format(sup)
