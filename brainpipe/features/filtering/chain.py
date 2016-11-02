#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from .method import *
from joblib import Parallel, delayed


class Chain(object):
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

        featinfo: string, optional, (def: None)
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

    def __init__(self, sf, npts, f=None, filtname=None, cycle=3, order=3,
                 ftype='bandpass', padlen=None, transname=None, width=7.0,
                 featinfo=None, detrend=False, demean=False, verbose=0):
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
        self._featinfo = featinfo
        self._detrend = detrend
        self._demean = demean
        self._verbose = verbose
        # Get associated functions and build chain :
        self._ConfUpdate()

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
        self._ConfUpdate()
        # Return list of function :
        return self._chain

    def apply(self, x, n_jobs=1, axis=0):
        """Apply the list of transformations to the data x.

        Specify where is the time dimension using axis. The 'axis'
        dimension must correspond to the input parameter npts. This
        function return an array of shape (n_f, x.shape) with n_f the
        number of frequencies in f.

        """
        # Check shape of x :
        xaxis = float(x.shape[axis])
        if not np.equal(xaxis, self._npts):
            raise ValueError('The dimension of x on axis ' + str(axis) + ' is ' + str(
                x.shape[axis]) + " and doesn't correspond to the variable npts=" + str(self._npts))
        # Check n_jobs n_frequencies = 1:
        if len(self._f) == 1:
            n_jobs = 1
        # Apply function :
        xf = Parallel(n_jobs=n_jobs)(delayed(_apply)(x, self, k, axis=axis)
                                     for k in self._f)
        return np.array(xf)

    # -------------------------------------------------------
    #                    DEEP FUNCTIONS
    # -------------------------------------------------------
    def _checkInputs(self):
        """Check if inputs are compatible."""
        # -----------------------------------------
        # Transname checking :
        # -----------------------------------------
        # Force to add a feature kind in case of complex transformation :
        if (self._transname is not None) and (self._featinfo is None):
            raise ValueError("""Using a complex decomposition like 'hilbert'
                    or 'wavelet', you must define the featinfo parameter
                    (either 'amplitude', 'power' or 'phase')""")
        # Wavelet checking :
        if self._transname is 'wavelet':
            self._filtname = None

        # -----------------------------------------
        # Frequency checking :
        # -----------------------------------------
        # No filtering if no frequency :
        if self._f is None:
            self._filtname = None

        if (self._f is not None) or (self._filtname is not None):
            # Check frequency vector :
            if not isinstance(self._f, np.ndarray) or np.array(self._f).ndim == 1:
                self._f = np.atleast_2d(self._f)
            fshape = self._f.shape
            # f must be a (1, N) or (2, N) array :
            if (1 not in fshape) and (2 not in fshape):
                raise ValueError('Shape of frequency vector is not compatible.')
            # Shape of f checking according to ftype :
            if (self._ftype is 'bandpass') and (self._transname is not 'wavelet'):
                if 2 not in fshape:
                    raise ValueError(
                        'For a bandpass filter, f must be a (2, n_frequency) array')
                elif fshape[0] is not 2:
                    self._f = self._f.T
            elif (self._ftype is not 'bandpass'):
                if 1 not in fshape:
                    raise ValueError(
                        "For a 'lowpass'/'highpass' filter, f must be a (1, n_frequency)")
                elif fshape[0] is not 1:
                    self._f = self._f.T
            # Shape of f checking according to transname :
            if (self._transname is 'wavelet') and (fshape[0] != 1):
                self._f = np.atleast_2d(self._f.mean(0))
            # No filtering / No wavelet :
            if (self._transname is not 'wavelet') and (self._filtname is None):
                self._f = [None]
        else:
            self._f = [None]
        self._f = list(np.array(self._f).astype(float).T)

    def _ConfUpdate(self):
        """Update configuration."""
        # Check inputs compatibility :
        self._checkInputs()
        # Get list of functions :
        _, propStr = _getFilterProperties(self, self._f[0])
        # Update string :
        self._preprocStr, self._filtStr, self._transStr, self._featStr = propStr
        # Print output :
        if (self._verbose > 0) and (self._verbose <= 1):
            print('Chain updated')
        elif self._verbose > 1:
            print('Chain updated to: ' + self.__str__())

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
        self._ConfUpdate()

    # Time points :
    @property
    def npts(self):
        return self._npts

    @npts.setter
    def npts(self, value):
        self._npts = value
        self._ConfUpdate()

    # Frequency :
    @property
    def f(self):
        return self._f

    @f.setter
    def f(self, value):
        self._f = value
        self._ConfUpdate()

    # Filter name :
    @property
    def filtname(self):
        return self._filtname

    @filtname.setter
    def filtname(self, value):
        self._filtname = value
        self._ConfUpdate()

    # Number of cycle (firls)
    @property
    def cycle(self):
        return self._cycle

    @cycle.setter
    def cycle(self, value):
        self._cycle = value
        self._ConfUpdate()

    # Filter order (bessel / butter)
    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, value):
        self._order = value
        self._ConfUpdate()

    # Filter type :
    @property
    def ftype(self):
        return self._ftype

    @ftype.setter
    def ftype(self, value):
        self._ftype = value
        self._ConfUpdate()

    # Padlen (for fir1)
    @property
    def padlen(self):
        return self._padlen

    @padlen.setter
    def padlen(self, value):
        self._padlen = value
        self._ConfUpdate()

    # Transformation name :
    @property
    def transname(self):
        return self._transname

    @transname.setter
    def transname(self, value):
        self._transname = value
        self._ConfUpdate()

    # Wavelet width :
    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        self._width = value
        self._ConfUpdate()

    # Feature featinfo :
    @property
    def featinfo(self):
        return self._featinfo

    @featinfo.setter
    def featinfo(self, value):
        self._featinfo = value
        self._ConfUpdate()

    # De-trending :
    @property
    def detrend(self):
        return self._detrend

    @detrend.setter
    def detrend(self, value):
        self._detrend = value
        self._ConfUpdate()

    # De-meaning :
    @property
    def demean(self):
        return self._demean

    @demean.setter
    def demean(self, value):
        self._demean = value
        self._ConfUpdate()

    # Verbose :
    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        self._verbose = value
        self._ConfUpdate()


def _getFilterProperties(self, f):
    """This function return the diffrents functions for pre-processing,
    filtering, complex transformation and apply a feature type.

    This function is outside the class because of parallel processing.

    """
    # Get preprocessing function :
    preprocFcn, preprocStr = preproc_fcn(detrend=self._detrend,
                                         demean=self._demean)
    # Get the filter function:
    filtFcn, filtStr = filter_fcn(self._sf, self._npts, f=f, filtname=self._filtname,
                                  cycle=self._cycle, order=self._order, ftype=self._ftype,
                                  padlen=self._padlen)
    # Get the transformation function:
    transFcn, transStr = transfo_fcn(self._sf, f=f, transname=self._transname,
                                     width=self._width)
    # Get feature type function :
    featFcn, featStr = feat_fcn(featinfo=self._featinfo)

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
            return featFcn(transFcn(v))
        # Apply on axis :
        return np.apply_along_axis(_fSpecific, axis, x)
    return fSpecific, (preprocStr, filtStr, transStr, featStr)


def _apply(x, self, f, axis=0):
    # Get latest updated function (for frequency f) :
    fcn, _ = _getFilterProperties(self, f)
    # Apply this function for f :
    x = fcn(x, axis=axis)
    return x
