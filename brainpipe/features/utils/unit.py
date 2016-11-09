#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

__all__ = ['time_to_sample', 'sample_to_time']

def time_to_sample(x, sf, from_unit='s'):
    """Convert x from a time unit to a sample unit using the sampling
    frequency.

    :x: int/float/np.ndarray
    :sf: float, sampling frequency
    :from_unit: string, either 's' or 'ms' for seconde or milliseconde
    :returns: same type as x

    """
    # ---------------------------------------------
    #              INPUT CHECKING
    # ---------------------------------------------
    # Check sampling frequency :
    if not isinstance(sf, (int, float)):
        raise ValueError(
            'Sampling frequency must be either a integer or a float')
    else:
        sf = float(sf)
    # Check x :
    if isinstance(x, (int, float)):
        x = float(x)
    elif isinstance(x, np.ndarray):
        x = x.astype(float)
    else:
        raise ValueError('x must be either an integer, a float or an array')
    # Check unit :
    if from_unit is 's':
        mult = sf
    elif from_unit is 'ms':
        mult = sf / 1000
    else:
        raise ValueError(
            "from_unit must be either 's' (seconde) or 'ms' (milliseconde)")

    # Conversion :
    if isinstance(x, (int, float)):
        x *= mult
        x = int(np.rint(x))
    else:
        np.multiply(x, mult, out=x)
        np.rint(x, out=x)
        x = x.astype(int)
    return x


def sample_to_time(x, sf, to_unit='s'):
    """Convert x from a sample unit to a time unit using the sampling
    frequency.

    :x: int/float/np.ndarray
    :sf: float, sampling frequency
    :to_unit: string, either 's' or 'ms' for seconde or milliseconde
    :returns: same type as x

    """
    # ---------------------------------------------
    #              INPUT CHECKING
    # ---------------------------------------------
    # Check sampling frequency :
    if not isinstance(sf, (int, float)):
        raise ValueError(
            'Sampling frequency must be either a integer or a float')
    else:
        sf = float(sf)
    # Check x :
    if isinstance(x, (int, float)):
        x = float(x)
    elif isinstance(x, np.ndarray):
        x = x.astype(float)
    else:
        raise ValueError('x must be either an integer, a float or an array')
    # Check unit :
    if to_unit is 's':
        mult = sf
    elif to_unit is 'ms':
        mult = sf / 1000
    else:
        raise ValueError(
            "to_unit must be either 's' (seconde) or 'ms' (milliseconde)")

    # Conversion :
    if isinstance(x, (int, float)):
        x /= mult
    else:
        np.divide(x, mult, out=x)
    return x
