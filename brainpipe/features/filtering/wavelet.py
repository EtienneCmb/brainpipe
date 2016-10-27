#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from scipy import signal


def morlet_wlt(sf, f, width=7.0):
    """Get a wavelet of Morlet."""
    sf, f, width = float(sf), float(f), float(width)
    dt = 1 / sf
    sf = f / width
    st = 1 / (2 * np.pi * sf)

    # Build morlet wavelet :
    t = np.arange(-3.5 * st, 3.5 * st, dt)
    A = 1 / np.sqrt((st * np.sqrt(np.pi)))
    return A * np.exp(-np.square(t) / (2 * np.square(st))) * np.exp(1j * 2 * np.pi * f * t)


def morlet(x, sf, f, width=7.0):
    """Compute wavelet transform on vector.

    :x: data, vector
    :sf:sampling frequency, float
    :f: frequency for extraction, float
    :width: width of the wavelet, float
    :returns: complex vector

    """
    # Get the wavelet :
    m = morlet_wlt(sf, f, width)

    # Compute morlet :
    y = np.convolve(x, m)
    return y[int(np.ceil(len(m) / 2)) - 1:int(len(y) - np.floor(len(m) / 2))]


def ndmorlet(x, sf, f, axis=0, get='amplitude', width=7.0):
    """Compute morlet on ndarray."""
    # Get the wavelet :
    m = morlet_wlt(sf, f, width)

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
