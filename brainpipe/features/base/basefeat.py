#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from .base import _base
from ..filtering import Chain


__all__ = ['ERP', 'SigFilt', 'Amplitude', 'Power',
           'TimeFrequencyMap', 'Phase']


class ERP(_base):

    """Docstring for ERP."""

    def __init__(self, sf, npts, chain=None, win=None, norm=None):
        """TODO: to be defined1."""
        _base.__init__(self, sf, npts, chain, win, norm)
        # Force ERP :
        self._feat = 'ERP'
        # Check frequency and filter :
        self.chain.featinfo = None
        self.chain.ftype = 'lowpass'
        if self.chain.filtname not in ['butter', 'bessel']:
            self.chain.filtname = 'butter'
        if not isinstance(self.chain.f, (int, float)) or (self.chain.f is None):
            self.chain.f = 10.0
        self.chain.update()
        self._update()


class SigFilt(_base):

    """Docstring for SigFilt."""

    def __init__(self, sf, npts, chain=None, win=None, norm=None):
        """TODO: to be defined1."""
        _base.__init__(self, sf, npts, chain, win, norm)
        # Force power :
        self._feat = 'SigFilt'
        self.chain.featinfo = None


class Amplitude(_base):

    """Docstring for Amplitude."""

    def __init__(self, sf, npts, chain=None, win=None, norm=None):
        """TODO: to be defined1."""
        _base.__init__(self, sf, npts, chain, win, norm)
        # Force power :
        self._feat = 'Amplitude'
        self.chain.featinfo = 'amplitude'


class Power(_base):

    """Docstring for Power."""

    def __init__(self, sf, npts, chain=None, win=None, norm=None):
        """TODO: to be defined1."""
        _base.__init__(self, sf, npts, chain, win, norm)
        # Force power :
        self._feat = 'Power'
        self.chain.featinfo = 'power'


class TimeFrequencyMap(_base):

    """Docstring for TimeFrequencyMap."""

    def __init__(self, sf, npts, chain=None, win=None, norm=None):
        """TODO: to be defined1."""
        _base.__init__(self, sf, npts, chain, win, norm)
        # Force power :
        self._feat = 'Power'
        self.chain.featinfo = 'power'


class Phase(_base):

    """Docstring for Phase."""

    def __init__(self, sf, npts, chain=None, win=None, norm=None):
        """TODO: to be defined1."""
        _base.__init__(self, sf, npts, chain, win, norm)
        # Force power :
        self._feat = 'Phase'
        self.chain.featinfo = 'phase'
