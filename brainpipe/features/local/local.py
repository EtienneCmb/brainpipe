#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from ._localbase import _local
from ..filtering import Chain


__all__ = ['ERP', 'SigFilt', 'Amplitude', 'Power',
           'TimeFrequencyMap', 'Phase']


class ERP(_local):

    """Extract Event Related Potential from an N-dimentional signal.

    Args:
        sf: int/float
            Sampling frequency

    Kargs:
        chain: Chain, optional, (def: None)
            Define a chain object for controling the lowpass frequency,
            the filter type, the order... By default, the ERP use a lowpass
            Butterworth filter under 10Hz, with an order 3. Alternatively,
            you can use a dictionnary to directly pass arguments.


        win: Window, optional, (def: None)
            Pass a Window object to get the mean of an N-dimentional signal
            inside sliding windows. Alternatively, you can use a dictionnary
            to directly pass arguments.

        norm: Normalization, optional, (def: None)
            Pass a Normalization object to normalize an N-dimentional signal
            by a baseline period. Alternatively, you can use a dictionnary to
            directly pass arguments.

    Return:
        A ERP object with an apply() method. After defining the object, use
        str() to check parameters. See examples/Features/00_ERP.py to see an
        exemple of use.

    """

    def __init__(self, sf, npts, chain=None, win=None, norm=None):
        _local.__init__(self, sf, npts, chain, win, norm)
        # Force ERP :
        self._feat = 'ERP'
        # Check frequency and filter :
        self.chain.featinfo = None
        self.chain.ftype = 'lowpass'
        if self.chain.filtname not in ['butter', 'bessel']:
            self.chain.filtname = 'butter'
        if not isinstance(self.chain.f, (int, float)) or (self.chain.f is None):
            self.chain.f = 10.0
        # Update chain and base :
        self.chain.update()
        self._update()


class SigFilt(_local):

    """Docstring for SigFilt."""

    def __init__(self, sf, npts, chain=None, win=None, norm=None):
        _local.__init__(self, sf, npts, chain, win, norm)
        # Force power :
        self._feat = 'SigFilt'
        self.chain.featinfo = None


class Amplitude(_local):

    """Docstring for Amplitude."""

    def __init__(self, sf, npts, chain=None, win=None, norm=None):
        _local.__init__(self, sf, npts, chain, win, norm)
        # Force power :
        self._feat = 'Amplitude'
        self.chain.featinfo = 'amplitude'


class Power(_local):

    """Docstring for Power."""

    def __init__(self, sf, npts, chain=None, win=None, norm=None):
        _local.__init__(self, sf, npts, chain, win, norm)
        # Force power :
        self._feat = 'Power'
        self.chain.featinfo = 'power'


class TimeFrequencyMap(_local):

    """Docstring for TimeFrequencyMap."""

    def __init__(self, sf, npts, chain=None, win=None, norm=None):
        _local.__init__(self, sf, npts, chain, win, norm)
        # Force power :
        self._feat = 'Power'
        self.chain.featinfo = 'power'


class Phase(_local):

    """Docstring for Phase."""

    def __init__(self, sf, npts, chain=None, win=None, norm=None):
        _local.__init__(self, sf, npts, chain, win, norm)
        # Force power :
        self._feat = 'Phase'
        self.chain.featinfo = 'phase'
