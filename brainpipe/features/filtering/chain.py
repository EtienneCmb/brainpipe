#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from warnings import warn
from joblib import Parallel, delayed
from .method import *
from ..utils import BandSplit


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

        bandsplit: list, optional, (def: None)
            Use this parameter to split a frequency band in multiple sub
            frequency bands. bandsplit must be a list of integers with the
            same length as f.

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

    Return:
        A chain object. Inside this object, you can change parameters
        dynamically. Use the .get() method to get a list of transformations
        or .apply() to directly apply your transformations to your data.
        Finally, use str(chain_obect) to display current configuration.

    """

    def __init__(self, sf, npts, f=None, bandsplit=None, filtname=None,
                 cycle=3, order=3, ftype='bandpass', padlen=None,
                 transname=None, width=7.0, featinfo=None, detrend=False,
                 demean=False):
        self._sf = float(sf)
        self._npts = float(npts)
        self._f = f
        self._bandsplit = bandsplit
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
        # Split or not the frequency band :
        if (self._filtname is not None) and (self._ftype is 'bandpass') or (self._transname is 'wavelet'):
            self.split = BandSplit(f, bandsplit)
        else:
            self.split = BandSplit([0, 1], None)
        self._splitStr = str(self.split)
        # Get associated functions and build chain :
        self._ConfUpdate()

    def __str__(self):
        chain = 'Chain(Settings(sf=' + str(self._sf) + ', npts=' + \
            str(self._npts) + '), {split}, {pre}, {filter}, {trans}, {feat})'
        return chain.format(filter=self._filtStr, trans=self._transStr,
                            feat=self._featStr, pre=self._preprocStr,
                            split=self._splitStr)

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
        if self._f is None:
            fapply = [[None]]
        else:
            if self.split.splitted is None:
                fapply = self._f
            else:
                fapply = self.split.fsplit
        if len(fapply) == 1:
            n_jobs = 1
        # Apply function :
        xf = Parallel(n_jobs=n_jobs)(delayed(_apply)(x, self, k, axis=axis)
                                     for k in fapply)
        return np.array(xf)

    def update(self):
        """Update the chain configuration."""
        self._ConfUpdate()

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
            self._ftype = None

        # -----------------------------------------
        # Frequency checking :
        # -----------------------------------------
        # No filtering if no frequency :
        if self._f is None:
            if self._filtname is not None:
                warn('f is None but not filtname.')
            if self._transname is not None:
                warn('f is None but not transname.')
            self._filtname = None
            self._transname = None
        else:
            if (self._filtname is None) and (self._transname is None):
                raise ValueError(
                    'f is not None but there is no filtname or no transformation.')
            else:
                # Force f to be a 2D array :
                self._f = np.atleast_2d(self._f)
                fshape = self._f.shape
                ######### f.shape = (N,) ##########
                if (self._ftype in ['lowpass', 'highpass']) or (self._transname is 'wavelet'):
                    self._f = np.ravel(self._f)
                ######### f.shape = (N, 2) ##########
                elif (self._ftype is 'bandpass') and (self._transname is not 'wavelet'):
                    if 2 not in fshape:
                        raise ValueError(
                            'Using a bandpass filter, the f parameter must be a (N, 2) array')
                    elif fshape[1] is not 2:
                        self._f = self._f.T

    def _ConfUpdate(self):
        """Update configuration."""
        # Check inputs compatibility :
        self._checkInputs()
        # Get string of configuration (send f None)
        if self._f is None:
            fapply = None
        else:
            fapply = self._f[0]
        _, propStr = _getFilterProperties(self, fapply)
        # Update string :
        self._preprocStr, self._filtStr, self._transStr, self._featStr = propStr

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

    # Time points :
    @property
    def npts(self):
        return self._npts

    @npts.setter
    def npts(self, value):
        self._npts = value

    # Frequency :
    @property
    def f(self):
        return self._f

    @f.setter
    def f(self, value):
        self._f = value

    # Filter name :
    @property
    def filtname(self):
        return self._filtname

    @filtname.setter
    def filtname(self, value):
        self._filtname = value

    # Number of cycle (firls)
    @property
    def cycle(self):
        return self._cycle

    @cycle.setter
    def cycle(self, value):
        self._cycle = value

    # Filter order (bessel / butter)
    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, value):
        self._order = value

    # Filter type :
    @property
    def ftype(self):
        return self._ftype

    @ftype.setter
    def ftype(self, value):
        self._ftype = value

    # Padlen (for fir1)
    @property
    def padlen(self):
        return self._padlen

    @padlen.setter
    def padlen(self, value):
        self._padlen = value

    # Transformation name :
    @property
    def transname(self):
        return self._transname

    @transname.setter
    def transname(self, value):
        self._transname = value

    # Wavelet width :
    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        self._width = value

    # Feature featinfo :
    @property
    def featinfo(self):
        return self._featinfo

    @featinfo.setter
    def featinfo(self, value):
        self._featinfo = value

    # De-trending :
    @property
    def detrend(self):
        return self._detrend

    @detrend.setter
    def detrend(self, value):
        self._detrend = value

    # De-meaning :
    @property
    def demean(self):
        return self._demean

    @demean.setter
    def demean(self, value):
        self._demean = value

    # Verbose :
    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        self._verbose = value


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
