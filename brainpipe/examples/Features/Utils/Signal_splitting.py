#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This script explain how to split a signal. For instance, if you have a
long time series, you may want to split it in 30 seconds window.
"""
import numpy as np
import matplotlib.pyplot as plt

from brainpipe.features import TimeSplit


if __name__ == "__main__":
    # Define a random signal :
    sf = 256.      # Sampling frequency
    npts = 30000   # Number of time points
    t = np.arange(npts) / sf
    sig = np.sin(2*np.pi*0.10*t)

    # Split signal :
    sp = TimeSplit(sf, 22, unit='s')
    xsp = sp.apply(sig, axis=0)
    tsp = sp.apply(t)[0, :]
    print(tsp.min(), tsp.max())

    # Plot :
    plt.figure()
    plt.subplot(211)
    plt.plot(t, sig, color='lightgray', label='Original signal')
    plt.subplot(212)
    plt.plot(tsp, xsp.T, color='firebrick', label='Splitted signal')
    plt.show()
