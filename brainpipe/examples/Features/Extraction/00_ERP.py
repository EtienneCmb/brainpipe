#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This script explain how to extract Event Related Potential
of a Neural signal.
"""
import numpy as np
import matplotlib.pyplot as plt
from brainpipe.features import ERP, Window, Normalization, Chain


def _plot(t, x, title='', **kwargs):
    """Basic plot function"""
    plt.plot(t, x, **kwargs), plt.xlabel('Time'), plt.ylabel('uV')
    plt.title(title)
    plt.axis('tight')


if __name__ == "__main__":
    # First, create a random signal :
    sf = 1024.
    npts = 8467
    f = 500
    t = np.arange(npts)/sf
    x = np.sin(2*np.pi*f*t)+np.random.rand(npts)
    # Define a basic ERP object and apply to the signal :
    erp_bsc = ERP(sf, npts)
    xf0 = erp_bsc.apply(x).ravel()
    print('Basic configuration: ', str(erp_bsc))
    # Plot this basic configuration :
    plt.figure()
    plt.subplot(311)
    _plot(t, x, 'Original signal')
    plt.subplot(312)
    _plot(t, xf0, 'ERP (basic configuration)')
    plt.subplot(313)
    plt.psd(xf0, 256, Fs=sf)
    plt.title('PSD of the basic ERP'), plt.xlim([0, 50]), plt.ylim([-100, 0])

    # Advanced example : you can normalize the ERP and take the mean using
    # a custom Normalization() and Window() objects :
    win = Window(sf, auto=(0, npts, 10, 5), unit='sample')
    norm = Normalization(sf, kind='zscore', baseline=(0, 1), unit='s')
    ch = Chain(sf, npts, f=15, filtname='bessel', ftype='lowpass', order=1)
    erp_adv = ERP(sf, npts, chain=ch, win=win, norm=norm)
    print('Advanced configuration: ', str(erp_adv))
    x = np.tile(x, (3, 10, 1))
    x += np.random.rand(*x.shape)
    xf1 = np.squeeze(erp_adv.apply(x, axis=2))
    # Plot one example :
    plt.figure()
    plt.subplot(211)
    _plot(t, x[:, 0 ,:].T, '3 random signals')
    plt.subplot(212)
    _plot(erp_adv.time, xf1[:, 0 ,:].T, 'ERP of the 3 random signals')

    plt.show()
