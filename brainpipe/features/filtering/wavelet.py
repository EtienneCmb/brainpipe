#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Functions for complex decomposition using Morlet wavelets
"""

import numpy as np
from scipy import signal

__all__ = ['morlet', 'ndmorlet']

def _morlet_wlt(sf, f, width=7.0):
    """Get a wavelet of Morlet.
    
    Parameters
    ----------
    sf: float
        Sampling frequency

    f: array, shape (2,)
        Frequency vector

    width: float, optional, (def: 7.0)
        Width of the wavelet

    Returns
    -------
    wlt: array
        Morlet wavelet
    """
    sf, f, width = float(sf), float(f), float(width)
    dt = 1 / sf
    sf = f / width
    st = 1 / (2 * np.pi * sf)

    # Build morlet wavelet :
    t = np.arange(-width * st / 2, width * st / 2, dt)
    A = 1 / np.sqrt((st * np.sqrt(np.pi)))
    wlt = A * np.exp(-np.square(t) / (2 * np.square(st))) * np.exp(1j * 2 * np.pi * f * t)

    return wlt


def morlet(x, sf, f, width=7.0):
    """Complex decomposition of a signal x using the morlet
    wavelet.
    
    Parameters
    ----------
    x: array, shape (N,)
        The signal to use for the complex decomposition. Must be
        a vector of length N.

    sf: float
        Sampling frequency

    f: array, shape (2,)
        Frequency vector

    width: float, optional, (def: 7.0)
        Width of the wavelet

    Returns
    -------
    xout: array, shape (N,)
        The complex decomposition of the signal x.
    """
    # Get the wavelet :
    m = _morlet_wlt(sf, f, width)

    # Compute morlet :
    y = np.convolve(x, m)
    xout = y[int(np.ceil(len(m) / 2)) - 1:int(len(y) - np.floor(len(m) / 2))]

    return xout


def ndmorlet(x, sf, f, axis=0, get=None, width=7.0):
    """Apply the complex decomposition using the Morlet's wavelet
    for a multi-dimentional array.

    Parameters
    ----------
    x: array
        The signal to use for the complex decomposition.

    sf: float
        Sampling frequency

    f: array, shape (2,)
        Frequency vector

    axis: integer, optional, (def: 0)
        Specify the axis where is located the time dimension

    get: string, optional, (def: None)
        Specify the type of information to extract from the
        complex decomposition. Use either None, 'amplitude',
        'power' or 'phase'.

    width: float, optional, (def: 7.0)
        Width of the wavelet

    Returns
    -------
    xout: array, same shape as x 
        Array containing the type of information specified in get.
    """
    # Get the wavelet :
    m = _morlet_wlt(sf, f, width)

    # Define a morlet function :
    def morletFcn(xt):
        # Compute morlet :
        y = np.convolve(xt, m)
        return y[int(np.ceil(len(m) / 2)) - 1:int(len(y) - np.floor(len(m) / 2))]

    # Apply to x :
    xmorlet = np.apply_along_axis(morletFcn, axis, x)

    # Get complex/amplitude/phase :
    if get is None:
        return xmorlet
    elif get == 'amplitude':
        return np.abs(xmorlet)
    elif get == 'power':
        return np.square(2 * np.abs(xmorlet) / sf)
    elif get == 'phase':
        return np.angle(xmorlet)
