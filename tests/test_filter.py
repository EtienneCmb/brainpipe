#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

from brainpipe.features.filtering import firls
from brainpipe.features.filtering import wavelet
from brainpipe.features.filtering import method

def _printRsl(st):
    print('->'+st)

def test_fir1():
    """Test the fir1 filter"""
    # Define variables :
    fs = 1024
    f = np.array([10, 15])
    L = 512
    cycle = 6
    # Compute forder :
    forder = firls.fir_order(fs, L, f[0], cycle)
    b, a = firls.fir1(forder, f/(fs/2))

    assert forder == 170


def test_wavelet():
    """Test wavelet"""
    x = np.random.rand(4000,)
    sf = 1024
    f =100
    wavelet.morlet(x, sf, f, width=21.0)


def test_filter():
    """Test filter function"""
    sf = 1024
    npts = 2789
    f_band = np.array([10, 12])
    f_lowhigh = 50
    print('\n')
    _, firStr = method.filter_fcn(sf, npts, f=f_band, cycle=6)
    _printRsl(firStr)
    _, NoneStr = method.filter_fcn(sf, npts, f=f_band, filtname=None)
    _printRsl(NoneStr)
    _, butterBand = method.filter_fcn(sf, npts, f=f_band, filtname='butter', order=2)
    _printRsl(butterBand)
    _, besselBand = method.filter_fcn(sf, npts, f=f_band, filtname='bessel', order=3)
    _printRsl(besselBand)
    _, butterLow = method.filter_fcn(sf, npts, f=f_lowhigh, filtname='butter', ftype='lowpass', order=4)
    _printRsl(butterLow)
    _, butterHigh = method.filter_fcn(sf, npts, f=f_lowhigh, filtname='butter', ftype='highpass', order=5)
    _printRsl(butterHigh)
    _, besselLow = method.filter_fcn(sf, npts, f=f_lowhigh, filtname='bessel', ftype='lowpass', order=6)
    _printRsl(besselLow)
    _, besselHigh = method.filter_fcn(sf, npts, f=f_lowhigh, filtname='bessel', ftype='highpass', order=7)
    _printRsl(besselHigh)


def test_transfo():
    """Test transformation function"""
    sf = 1024
    print('\n')
    _, transStr = method.transfo_fcn(sf)
    _printRsl(transStr)
    _, transStr = method.transfo_fcn(sf, transname='wavelet', f=[2, 4])
    _printRsl(transStr)
    _, transStr = method.transfo_fcn(sf, transname='wavelet', width=21.0, f=[13, 30])
    _printRsl(transStr)
    _, transStr = method.transfo_fcn(sf, transname=None)
    _printRsl(transStr)


def test_featkind():
    """Test kind function"""
    print('\n')
    _, featStr = method.feat_fcn(kind='amplitude')
    _printRsl(featStr)
    _, featStr = method.feat_fcn(kind='power')
    _printRsl(featStr)
    _, featStr = method.feat_fcn(kind='phase')
    _printRsl(featStr)
    _, featStr = method.feat_fcn(kind=None)
    _printRsl(featStr)


def test_preproc():
    """Test preprocessing function"""
    print('\n')
    a = np.arange(100)
    # Get functions :
    idFcn, idStr = method.preproc_fcn()
    dtrdFcn, dtrdStr = method.preproc_fcn(detrend=True)
    _printRsl(dtrdStr)
    demnFcn, demnStr = method.preproc_fcn(demean=True)
    _printRsl(demnStr)
    # a unchanged :
    a_bool = np.array_equal(idFcn(a), a)
    # De-trending :
    a_detrend = dtrdFcn(a)
    a_detrend_bool = all(np.isclose(a_detrend, np.zeros_like(a)))
    # De-meaning :
    a_demean = demnFcn(a)
    a_demean_bool = np.isclose(a_demean.mean(), 0)
    assert a_bool and a_detrend_bool and a_demean_bool
