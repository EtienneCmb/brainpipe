#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This script illustrate how to normalize an array."""
import numpy as np
import matplotlib.pyplot as plt
from brainpipe.features import Normalization


def _plot(xvec, yvec, mat, title=''):
    """Plotting function."""
    plt.pcolormesh(xvec, yvec, mat, cmap='inferno')
    plt.colorbar()
    plt.xlabel('Time (s)'), plt.ylabel('Axis 2')
    plt.title(title, y=1.02)
    plt.axis('tight')


if __name__ == '__main__':
    # ---------- Create a random signal ------------
    sf = 256.0      # Sampling frequency
    npts = 5000     # Number of points
    nrep = 50       # An other dimension
    t = np.arange(npts)/sf
    yvec = np.arange(nrep)
    x = np.meshgrid(np.arange(nrep), np.arange(npts))[1].T
    x = x.astype(float) / x.max()

    # ---------- Normalize -----------
    # No normalization :
    no = Normalization(sf)
    x0 = no.apply(x)
    # Normalization A-B :
    no1 = Normalization(sf, norm=1, baseline=(5, 10), unit='s')
    x1 = no1.apply(x, axis=1)
    # Normalization A/B :
    no2 = Normalization(sf, norm='A/B', baseline=(2250, 15257), unit='ms')
    no2Str = str(no2)
    x2 = no2.apply(x, axis=1)
    # Normalization A-B/B :
    no2.norm = 3 # Update precedent object
    x3 = no2.apply(x, axis=1)
    # Normalization z-score :
    no3 = Normalization(sf, norm=4, baseline=(1000, 1500))
    x4 = no3.apply(x, axis=1)

    # Plot :
    plt.subplot(611)
    _plot(t, yvec, x, title='Original signal')
    plt.subplot(612)
    _plot(t, yvec, x0, str(no))
    plt.subplot(613)
    _plot(t, yvec, x1, str(no1))
    plt.subplot(614)
    _plot(t, yvec, x2, no2Str)
    plt.subplot(615)
    _plot(t, yvec, x3, str(no2))
    plt.subplot(616)
    _plot(t, yvec, x4, str(no3))
    plt.show()
