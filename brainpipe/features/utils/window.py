#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from warnings import warn
from .makebin import binarize, binarray


class Window(object):

    """Docstring for Window."""

    def __init__(self, sf, start=None, end=None, width=None, step=None,
                 window=None, wintype='shift', unit='sample'):
        """Take the mean of a signal inside defined windows.

        Arg:
            sf: float
                Sampling frequency

        Kargs:
            start/end/width/step: int/float, optional, (def: None)
                Parameters to define automatic window behavior.

            window: list/array, optional, (def: None)
                Manually defined a window.

            wintype: int/float, optional, (def: None)
                Define the type of window like. Use either:
                    - 'shift': [[start, start+width], [start+step, start+width+step], ..., [end-width, end]]
                    - 'center': [[start-width/2, start+width/2], ..., [end-width/2, end+width/2]]

            unit: string, optional, (def: 'sample')
                Define if window's should be either in 'sample' or, 's' or
                'ms' for respectively second and miliseconds unit.

        Return:
            A window object with a .get() and .apply() methods.

        """
        self._sf = float(sf)
        self._start = start
        self._end = end
        self._width = width
        self._step = step
        self._window = window
        self._wintype = wintype
        self._unit = unit
        self._windefined = False
        self._allpara = False
        self._alreadydefined = False
        self._force = False
        self._converted = False
        self._paraconverted = [False, False, False, False]
        self._WinUpdate()

    def __str__(self):
        pass

    ########################################################
    #                   USER FUNCTIONS
    ########################################################

    def get(self):
        pass

    def apply(self, x, axis=0):
        """Apply the window on the array x.

        Specify the dimension to binarize using the axis parameter.
        Return a binarize version of x.

        """
        if self._window is not None:
            return binarray(x, self._window, axis)
        else:
            warn('No window detected.')

    def clear(self):
        """Clear all predifined parameters and window."""
        self._start, self._end = None, None
        self._width, self._step = None, None
        self._window = None
        self._windefined = False
        self._allpara = False
        self._alreadydefined = False
        self._force = False

    ########################################################
    #                   SUB FUNCTIONS
    ########################################################

    def _WinUpdate(self):
        """Update window configuration."""
        # Check inputs :
        self._WinCheckInputs()
        # Build window :
        self._BuildWin()

    def _WinCheckInputs(self):
        """Check inputs."""
        # ------------- TYPE CHECKING -----------
        # Unit checking :
        if not isinstance(self._unit, str) or self._unit not in ['sample', 's', 'ms']:
            raise ValueError(
                "'unit' parameter must be either 'sample' or 's' or 'ms'")
        # wintype checking :
        if not isinstance(self._wintype, str) or self._wintype not in ['center', 'shift']:
            raise ValueError(
                "'wintype' parameter must be either 'center' or 'shift'")
        # start/end/width/step checking :
        paraNone = [k is None or isinstance(k, (int, float)) for k in [
            self._start, self._end, self._width, self._step]]
        if not all(paraNone):
            raise ValueError(
                "'start', 'end', 'width' and 'step' parameters must be either int or float.")
        # Window and start/end/step/width defined :
        paraFloat = [isinstance(k, (int, float)) for k in [
            self._start, self._end, self._width, self._step]]
        if (self._window is not None) and any(paraFloat) and not self._alreadydefined:
            warn(
                "'window' is defined, 'start', 'end', 'width' and 'step' are going to be ignored.")
            self._start, self._end, self._step, self._width = None, None, None, None

        # ------------- VALUE CHECKING -----------
        # start > end :
        if (self._start is not None) and (self._end is not None) and (self._start >= self._end):
            raise ValueError(
                "'start' must be strictly inferior to 'end' parameter")
        # Bool (start, end, step, width) :
        if all(paraFloat):
            self._allpara = True
        # window shape checking :
        if self._window is not None:
            self._window = np.atleast_2d(np.array(self._window))
            winshape = self._window.shape
            if 2 not in winshape:
                raise ValueError("'window' must be a (2, N) array")
            elif ((2 in winshape) and (winshape[0] != 2)) or (winshape[0] is 1):
                self._window = self._window.T
            if (winshape[0] == 1) and (winshape[1] == 2):
                self._window = self._window.T
            else:
                self._nwin = self._window.shape[1]
                self._windefined = True
        self._BuildWin()

    def _BuildWin(self):
        """Build the window."""
        # Check unit conversion :
        self._UnitConvert()
        paraFloat = [isinstance(k, (int, float)) for k in [
            self._start, self._end, self._width, self._step]]
        # In case of start/end/width/step :
        if (not self._windefined and self._allpara) or (self._force and self._allpara):
            self._window = np.atleast_2d(binarize(self._start, self._end,
                                                  self._width, self._step))
            self._force = False
            self._alreadydefined = True
        # Array to list :
        if (self._window is not None) and not isinstance(self._window, list):
            self._window = np.ndarray.tolist(self._window)

    def _UnitConvert(self):
        """Convert time unit to sample unit."""
        paraFloat = [isinstance(k, (int, float)) for k in [
            self._start, self._end, self._width, self._step]]
        if self._unit in ['s', 'ms'] and not self._converted:
            # Define multiplicative coefficient :
            if self._unit is 's':
                self._mult = self._sf
            elif self._unit is 'ms':
                self._mult = self._sf / 1000.0
            # Apply on window :
            if self._window is not None:
                self._window = np.array(self._window).astype(float)
                self._window = np.multiply(self._window, self._mult).astype(int)
                self._converted = True
            elif any(paraFloat):
                if (self._start is not None) and not self._paraconverted[0]:
                    self._start = int(self._start * self._mult)
                    self._paraconverted[0] = True
                if (self._end is not None) and not self._paraconverted[1]:
                    self._end = int(self._end * self._mult)
                    self._paraconverted[1] = True
                if (self._width is not None) and not self._paraconverted[2]:
                    self._width = int(self._width * self._mult)
                    self._paraconverted[2] = True
                if (self._step is not None) and not self._paraconverted[3]:
                    self._step = int(self._step * self._mult)
                    self._paraconverted[3] = True
                self._converted = True

    ########################################################
    #                   PROPERTIES
    ########################################################

    @property
    def sf(self):
        return self._sf

    @sf.setter
    def sf(self, value):
        self._sf = value
        self._WinUpdate()

    @property
    def start(self):
        return self._start

    @start.setter
    def start(self, value):
        self._start = value
        self._force = True
        self._window = None
        self._converted = False
        self._paraconverted[0] = False
        self._WinUpdate()

    @property
    def end(self):
        return self._end

    @end.setter
    def end(self, value):
        self._end = value
        self._force = True
        self._window = None
        self._converted = False
        self._paraconverted[1] = False
        self._WinUpdate()

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        self._width = value
        self._force = True
        self._window = None
        self._converted = False
        self._paraconverted[2] = False
        self._WinUpdate()

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, value):
        self._step = value
        self._force = True
        self._window = None
        self._converted = False
        self._paraconverted[3] = False
        self._WinUpdate()

    @property
    def window(self):
        return self._window

    @window.setter
    def window(self, value):
        self._window = value
        self._WinUpdate()

    @property
    def wintype(self):
        return self._wintype

    @wintype.setter
    def wintype(self, value):
        self._wintype = value
        self._WinUpdate()

    @property
    def unit(self):
        return self._unit

    @unit.setter
    def unit(self, value):
        self._unit = value
        self._WinUpdate()


# class TimeSplit(object):

#     """Docstring for TimeSplit."""

#     def __init__(self):
#         """TODO: to be defined1."""
#         pass
