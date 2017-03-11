#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import abc
from ..utils import Window, Normalization
from ..filtering import Chain


class _local(object):

    """Base class for basics features (Power, Phase, Amplitude, SigFilt, TF)"""

    def __init__(self, sf, npts, chain=None, win=None, norm=None, time=None):
        self._sf = sf
        self._npts = npts
        self._str = ''
        self._feat = 'local'
        self.chain = chain
        self.win = win
        self.norm = norm
        self.time = time
        self._update()

    def __str__(self):
        # first, update :
        self._update()
        # String build :
        self._str = '{feat}(' + ', \n'.join([str(self.chain),
                                             str(self.norm), str(self.win)]) + ')'
        return self._str.format(feat=self._feat)

    def _update(self):
        """Update the configuration."""
        # Default chain :
        if self.chain is None:
            self.chain = Chain(self._sf, self._npts)
        elif type(self.chain) is Chain:
            pass
        elif isinstance(self.chain, dict):
            self.chain = Chain(self._sf, self._npts, **self.chain)
        else:
            raise ValueError(
                'chain must be None, a Chain object or a dictionnary of supplementar arguments')
        # Default window :
        if self.win is None:
            self.win = Window(self._sf)
        elif type(self.win) is Window:
            pass
        elif isinstance(win, dict):
            self.win = Window(self._sf, **self.win)
        else:
            raise ValueError(
                'win must be None, a Window object or a dictionnary of supplementar arguments')
        # Default normalization :
        if self.norm is None:
            self.norm = Normalization(self._sf)
        elif type(self.norm) is Normalization:
            pass
        elif isinstance(self.norm, dict):
            self.norm = Normalization(self._sf, **self.norm)
        else:
            raise ValueError(
                'norm must be None, a Normalization object or a dictionnary of supplementar arguments')
        # Default time :
        if self.time is None:
            self.time = np.arange(self._npts)/self._sf
        else:
            if not isinstance(self.time, np.ndarray):
                self.time = np.ravel(self.time)
            if not np.equal(len(self.time), self._npts):
                raise ValueError("The time vector must have the length as npts.")

    ##########################################################
    #                   USER FUNCTION
    ##########################################################
    def apply(self, x, axis=0, n_jobs=1):
        """"""
        # Apply chain / normalization / window objects :
        if self.chain is not None:
            x = self.chain.apply(x, axis=axis)
            if self.chain.f is not None:
                axis += 1
        if self.norm is not None:
            x = self.norm.apply(x, axis=axis)
        if self.win is not None:
            self.time = self.win.apply(self.time).ravel()
            x = self.win.apply(x, axis=axis)
        return x

    @abc.abstractmethod
    def stat(self, arg):
        """"""
