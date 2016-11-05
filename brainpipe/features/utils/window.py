#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from warnings import warn

from .makebin import binarize, binarray
from .unit import time_to_sample, sample_to_time


class Window(object):

    """Docstring for Window."""

    def __init__(self, sf, window=None, auto=None, unit='sample'):
        """Take the mean of a signal inside defined windows.

        Arg:
            sf: float
                Sampling frequency

        Kargs:
            window: list/array, optional, (def: None)
                Manually defined a window.You can precise if each values
                inside have to be considered in sample, second or millisecond
                using the unit parameter.

            auto: tuple/list/array, optional, (def: None)
                Automatically defined a sliding window. The auto parameter
                must be a tuple/list/array of length 4. This tuple represent
                (start, end, width step) with start/end control the begining
                and the end of the baseline, each window has a length of width
                and move with step. You can precise if each values inside
                have to be considered in sample, second or millisecond
                using the unit parameter.

            unit: string, optional, (def: 'sample')
                Define if window's should be either in 'sample' or, 's' or
                'ms' for respectively second and miliseconds unit.

        Return:
            A window object with a .apply() method.

        """
        self._sf = sf
        self._window = window
        self._auto = auto
        self._unit = unit
        self._WinCheckInputs()

    def __str__(self):
        pass

    ########################################################
    #                   USER FUNCTIONS
    ########################################################

    def apply(self, x, axis=0):
        """Apply the window on the array x.

        Specify the dimension to binarize using the axis parameter.
        Return a binarize version of x.

        """
        if self._window is not None:
            return binarray(x, np.ndarray.tolist(self._window), axis)
        else:
            warn('No window detected.')
            return x

    ########################################################
    #                   SUB FUNCTIONS
    ########################################################

    def _WinCheckInputs(self):
        """Check inputs."""
        # ------------- TYPE CHECKING -----------
        # Sampling frequency checking :
        if not isinstance(self._sf, (int, float)):
            raise ValueError(
                'Sampling frequency must be either a float or an integer.')
        else:
            self._sf = float(self._sf)
        # Unit checking :
        if self._unit not in ['sample', 's', 'ms']:
            raise ValueError(
                "'unit' parameter must be either 'sample' or 's' or 'ms'")
        # auto checking :
        if self._auto is not None:
            if not isinstance(self._auto, (list, tuple, np.ndarray)):
                raise ValueError('auto must be either a tuple/list/array')
            else:
                self._auto = np.ravel(self._auto)
                if len(self._auto) is not 4:
                    raise ValueError(
                        'auto parameter must have a length of 4: (start, end, width, step)')
        # Window and start/end/step/width defined :
        if (self._window is not None) and (self._auto is not None):
            warn('window and auto or both defined. Only window is consider.')
            self._auto = None

        # ------------- VALUE CHECKING -----------
        # start > end :
        if self._auto is not None:
            if self._auto[0] > self._auto[1]:
                raise ValueError(
                    'Using the auto parameter, start must be < to end.')
            else:
                if self._unit in ['s', 'ms']:
                    self._auto = time_to_sample(self._auto, self._sf,
                                                from_unit=self._unit)
                    self._unit = 'sample'
                self._window = binarize(*tuple(self._auto))
        # window shape checking :
        if self._window is not None:
            # Tansform in a 2D array :
            self._window = np.atleast_2d(np.asarray(self._window))
            winshape = self._window.shape
            # Check shape :
            if 2 not in winshape:
                raise ValueError("'window' must be a (2, N) array")
            elif winshape == (2, 1):
                self._window = self._window.T
            # Convert if not already done :
            if self._unit in ['s', 'ms']:
                self._window = time_to_sample(self._window, self._sf,
                                              from_unit=self._unit)
                self._unit = 'sample'

    ########################################################
    #                   PROPERTIES
    ########################################################

    @property
    def sf(self):
        return self._sf

    @property
    def auto(self):
        return self._auto

    @property
    def window(self):
        return self._window

    @property
    def unit(self):
        return self._unit


class TimeSplit(object):

    """Docstring for TimeSplit."""

    def __init__(self, sf, duration, unit='s', keepout=False):
        """Split an array according to defined duration.

        Args:
            sf: float
                Sampling frequency

            duration: int/float
                Splitting duration

        Kargs:
            unit: string, optional, (def: 's')
                Control the unit of the duration parameter. Use either
                'sample', 's' (seconde) or 'ms' (milliseconde)

            keepout: bool, optional, (def: False)
                If keeput is True, this allow to split an array into
                multiple sub-arrays of non-equal size.

        Return:
            A TimeSplit object with a .apply() method.

        """
        self._sf = sf
        self._duration = duration
        self._unit = unit
        self._keepout = keepout

        # -----------------------------------------
        #             INPUTS CHECKING
        # -----------------------------------------
        # Sampling frequency :
        if not isinstance(self._sf, (int, float)):
            raise ValueError(
                'Sampling frequency must be either a float or an interger.')
        else:
            self._sf = float(self._sf)
        # Duration checking :
        if not isinstance(self._duration, (int, float)):
            raise ValueError('duration must be either a float or an interger.')
        else:
            self._duration = float(self._duration)
        # Unit :
        if self._unit is 'sample':
            pass
        elif self._unit in ['s', 'ms']:
            self._duration = time_to_sample(self._duration,
                                            self._sf, from_unit=self._unit)
        else:
            raise ValueError("unit must be either 'sample', 's' or 'ms'")

    #####################################################
    #                   USER FUNCTION
    #####################################################

    def apply(self, x, axis=0):
        """Apply the time split object to an array x.

        Specify the axis along which splitting.

        """
        # Get axis :
        npts = x.shape[axis]
        # Get the number of section :
        nsec = int(npts // self._duration)
        # Split x :
        if self._keepout:
            return np.array_split(x, nsec, axis=axis)
        else:
            # Truncate the matrix before splitting :
            idx = [slice(None)] * x.ndim
            # Find the correct number of points :
            idx[axis] = slice(int(nsec * self._duration))
            return np.array(np.split(x[idx], nsec, axis=axis))

    #####################################################
    #                   PROPERTIES
    #####################################################
    @property
    def sf(self):
        return self._sf

    @sf.setter
    def sf(self, value):
        self._sf = value

    @property
    def duration(self):
        return self._duration

    @duration.setter
    def duration(self, value):
        self._duration = value

    @property
    def unit(self):
        return self._unit

    @unit.setter
    def unit(self, value):
        self._unit = value

    @property
    def keepout(self):
        return self._keepout

    @keepout.setter
    def keepout(self, value):
        self._keepout = value
