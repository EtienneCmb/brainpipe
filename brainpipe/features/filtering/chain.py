#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from .method import *
from ..utils import _listcheck


class ChainTransform(object):
    """Design a chain of transformations to apply to multi-dimentional array.
    This class can be used to pre-processed your data, to extract spectral
    informations (using the hilbert or wavelet transform) or to compute phase /
    amplitude /power features.

    Args:
        sf: float
            Sampling frequency

        npt: float
            The number of time points. This parameter is then used to defined
            an apropriate filter.

    Kargs:
        f: list/float, optional, (def: None)
            Frequency bands for filtering the data. For a bandpass filter,
            use f=[[f1, f2], ..., [fn, fm]]. For a lowpass/highpass filter,
            use f=[f1, f2, ..., fn]. The f parameter is only active if
            filtname is not None.

        filtname: string, optional, (def: None)
            Define a filter in order to extract spectral informations in specific
            frequency bands. Use either 'fir1', 'butter' or 'bessel'. The 'fir1'
            filter is equivalent to eegfilt in the eeglab matlab toolbox.

        cycle: int, optional, (def: 3)
            Number of cycles for the 'fir1' filter. A good practice is to use 3
            for low frequency and 6 cycles for high frequency (>= 60hz).

        order: int, optional, (def: 3)
            Order of the 'butter' and 'bessel' filter. Using an order > 3 can
            provoc signal distortion.

        ftype: string, optional, (def: 'bandpass')
            Type of filter. Use either 'bandpass', 'lowpass' or 'highpass'.

        padlen: int, optional, (def: None)
            Parameter for the filtfilt function (see scipy.signal.filtfilt
            documentation).

        transname: string, optional, (def: 'hilbert')
            Name of a transformation to transform a signal from time domain
            to complex domain. Use either 'hilbert' or 'wavelet' (morlet).

        width: float, optional, (def: 7.0)
            Width of the wavelet transform. It's preferable to use width >= 5
            (Bertrand, 1996)

        kind: string, optional, (def: None)
            Define the type of information to extract from the signal. Use
            either None, 'amplitude', 'power' or 'phase'.

        detrend: bool, optional, (def: False)
            Remove the linear trend of a signal.

        demean: bool, optional, (def: False)
            Remove the mean of the signal.

        verbose: int, optional, (def: 0)
            Control to display or not informations when defining your chain
            of transformations. Use 0 for no output, 1 inform when the chain
            is updated and > 1 for more details.

    Return:
        A chain object. Inside this object, you can change parameters
        dynamically. Use the .get() method to get a list of transformations
        or .apply() to directly apply your transformations to your data.
        Finally, use str(chain_obect) to display current configuration.

    """

    def __init__(self, sf, npts, f=None, filtname='fir1', cycle=3, order=3,
                 ftype='bandpass', padlen=None, transname='hilbert', width=7.0,
                 kind=None, detrend=False, demean=False, verbose=0):
        self._sf = float(sf)
        self._npts = float(npts)
        self._f = f
        self._filtname = filtname
        self._cycle = cycle
        self._order = order
        self._ftype = ftype
        self._padlen = padlen
        self._transname = transname
        self._width = float(width)
        self._kind = kind
        self._detrend = detrend
        self._demean = demean
        self._verbose = verbose
        # Get associated functions and build chain :
        self._update()

    def __str__(self):
        chain = 'Chain(Settings(sf=' + str(self._sf) + ', npts=' + \
            str(self._npts) + '), {pre}, {filter}, {trans}, {feat})'
        return chain.format(filter=self._filtStr, trans=self._transStr, feat=self._featStr, pre=self._preprocStr)

    # -------------------------------------------------------
    #                    USER FUNCTIONS
    # -------------------------------------------------------

    def get(self):
        """Get the list of chain transformations.

        Each element of the list is a function for one specific band
        inside the f parameter. You can then manually applied those
        transformations to your data. Otherwise, use .apply() to
        automatically apply on your data.

        """
        # Get latest update :
        self._update()
        # Return list of function :
        return self._chain

    def apply(self, x, axis=0):
        """Apply the list of transformations to the data x.
        Specify where is the time dimension using axis. The 'axis'
        dimension must correspond to the input parameter npts.
        This function return an array of shape (n_f, x.shape) with
        n_f the number of frequencies in f.
        """
        # Get latest update :
        self._update()

    # -------------------------------------------------------
    #                    DEEP FUNCTIONS
    # -------------------------------------------------------
    def _inputCompatibility(self):
        """Check if inputs are compatible."""
        if self._transname == 'wavelet':
            self._filtname = None

    def _fSpecificChain(self, f):
        """Build a chain of transformations for one f."""
        # Get preprocessing function :
        preprocFcn, self._preprocStr = preproc_fcn(detrend=self._detrend,
                                                   demean=self._demean)
        # Get the filter function:
        filtFcn, self._filtStr = filter_fcn(self._sf, self._npts, f=f, filtname=self._filtname,
                                            cycle=self._cycle, order=self._order, ftype=self._ftype,
                                            padlen=self._padlen)
        # Get the transformation function:
        transFcn, self._transStr = transfo_fcn(self._sf, f=f, transname=self._transname,
                                               width=self._width)
        # Get feature type function :
        featFcn, self._featStr = feat_fcn(kind=self._kind)

        # Now, build a unique function for this frequency :
        def fSpecific(x, axis=0):
            """Apply transformation on x.

            This function mix vector and matrix operations

            """
            # Matricial preprocessing :
            x = preprocFcn(x, axis=axis)
            # Filt and transformation (on vector) :

            def _fSpecific(v):
                """Transformation chain."""
                v = filtFcn(v)
                v = transFcn(v)
                return v
            # Apply on axis :
            x = np.apply_along_axis(_fSpecific, axis, x)
            # Extract features using ndarray :
            x = featFcn(x)
            return x
        return fSpecific

    def _buildchain(self):
        """Build the final chain of transformation and return a list for each
        frequency in self._f."""
        self._chain = [self._fSpecificChain(k) for k in self._f]
        if (self._verbose > 0) and (self._verbose <= 1):
            print('Chain updated')
        elif self._verbose > 1:
            print('Chain updated to: ' + self.__str__())

    def _update(self):
        """Update configuration."""
        # Check inputs compatibility :
        self._inputCompatibility()
        # Check inputs :
        if self._f is not None:
            self._f = _listcheck(self._f)
        else:
            self._f = [None]
        # Get list of functions :
        self._buildchain()

    # -------------------------------------------------------
    #                    PROPERTIES
    # -------------------------------------------------------
    # Sampling frequency :
    @property
    def sf(self):
        return self._sf

    @sf.setter
    def sf(self, value):
        self._sf = value
        self._update()

    # Time points :
    @property
    def npts(self):
        return self._npts

    @npts.setter
    def npts(self, value):
        self._npts = value
        self._update()

    # Frequency :
    @property
    def f(self):
        return self._f

    @f.setter
    def f(self, value):
        self._f = value
        self._update()

    # Filter name :
    @property
    def filtname(self):
        return self._filtname

    @filtname.setter
    def filtname(self, value):
        self._filtname = value
        self._update()

    # Number of cycle (firls)
    @property
    def cycle(self):
        return self._cycle

    @cycle.setter
    def cycle(self, value):
        self._cycle = value
        self._update()

    # Filter order (bessel / butter)
    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, value):
        self._order = value
        self._update()

    # Filter type :
    @property
    def ftype(self):
        return self._ftype

    @ftype.setter
    def ftype(self, value):
        self._ftype = value
        self._update()

    # Padlen (for fir1)
    @property
    def padlen(self):
        return self._padlen

    @padlen.setter
    def padlen(self, value):
        self._padlen = value
        self._update()

    # Transformation name :
    @property
    def transname(self):
        return self._transname

    @transname.setter
    def transname(self, value):
        self._transname = value
        self._update()

    # Wavelet width :
    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        self._width = value
        self._update()

    # Feature kind :
    @property
    def kind(self):
        return self._kind

    @kind.setter
    def kind(self, value):
        self._kind = value
        self._update()

    # De-trending :
    @property
    def detrend(self):
        return self._detrend

    @detrend.setter
    def detrend(self, value):
        self._detrend = value
        self._update()

    # De-meaning :
    @property
    def demean(self):
        return self._demean

    @demean.setter
    def demean(self, value):
        self._demean = value
        self._update()

    # Verbose :
    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        self._verbose = value
        self._update()
