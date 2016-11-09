#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from .makebin import binarize
from ...utils import uorderlst


__all__ = ['defband', 'bandname', 'bandfcy', 'BandSplit']


stdF = pd.DataFrame({'vlfc': [[0, 1.5]], 'delta': [[2, 4]], 'theta': [[5, 7]],
                     'alpha': [[8, 13]], 'beta': [[13, 30]], 'low-gamma': [[30, 60]],
                     'high-gamma': [[60, 200]]},
                    columns=['vlfc', 'delta', 'theta', 'alpha', 'beta', 'low-gamma',
                             'high-gamma'])


def defband(return_as='table'):
    """Get the default physiological frequency bands of interest.

    Args:     return_as: string, optional, [def: 'table]         Say how
    to return bands. Use either 'table' to get a pandas Dataframe or
    'list' to have two list containing bands name and definition.

    """
    if return_as is 'table':
        return stdF
    elif return_as is 'list':
        return list(stdF.keys()), list(stdF.values[0])


def bandname(band):
    """Find the physiological name of a frequency band.

    Args:
        band: list/np.ndarray/tuple/int/float
            List of frequency bands

    Returns:
        List of band names

    Example:
        >>> band = [40, 0.5, 6]
        >>> findBandName(band)
        >>> ['Low-gamma', 'VLFC', 'Theta']

    """
    # Check input type :
    cref = np.atleast_2d(list(stdF.values[0])).mean(1)
    bname = list(stdF.keys())
    return [bname[np.abs(cref - k).argmin()] for k in np.ravel(band)]


def bandfcy(band):
    """Find the physiological frequency band for a given name.

    Args:
        band: list
            List of frequency band name

    Returns:
        List of frequency bands

    """
    # Check input type :
    if isinstance(band, str):
        band = [band]
    cref = list(stdF.values[0])
    fband = []
    for k in band:
        try:
            fband.append(list(stdF[k.lower()])[0])
        except:
            raise ValueError(
                k + ' not found. Please, search for ' + str(list(stdF.keys())))
    return fband


class BandSplit(object):

    """Split frequency band in multiple sub-bands.

    Args:
        f: list/np.ndarray
            List of starting and ending frequency bands.

    Kargs:
        bandsplit: list/tuple, optional, (def: None)
            List of integers or None to define how to split each band inside f.
            The lenght of split must be the same as f.

    Return:
        A BandSplit object with a get() and a join() method.

    """

    ########################################################
    #                   SUB-FUNCTIONS
    ########################################################

    def __init__(self, f, bandsplit=None):
        self._f = f
        self._bandsplit = bandsplit
        self._splitted = None
        self._fsplit = []
        self._sup = 'None'
        # Frequency checking (N, 2) :
        self._f = np.atleast_2d(self._f)
        fshape = self._f.shape
        if 2 not in fshape:
            raise ValueError('f must be a (N, 2)')
        elif fshape[1] is not 2:
            self._f = self._f.T
        self._nf = self._f.shape[0]
        # Split checking :
        if self._bandsplit is None:
            pass
        elif not isinstance(self._bandsplit, (list, tuple)):
            raise ValueError(
                'bandsplit must be list/tuple of integers/float/None')
        elif isinstance(self._bandsplit, (list, tuple)) and (len(self._bandsplit) is not self._nf):
            raise ValueError(
                'The length of bandsplit must be the same as the number of frequency bands')
        elif not all([k is None or isinstance(k, int) for k in self._bandsplit]):
            self._bandsplit = [None if k is None else int(
                k) for k in self._bandsplit]
            # raise ValueError('bandsplit must be list/tuple of integers/float/None')
        # Build fsplit and splitted :
        if self._bandsplit is None:
            self._fsplit = self._f
            self._splitted = None
        else:
            # Get usefull variables :
            self._splitted = []
            self._sup = 'bandsplit='+str(self._bandsplit)
            q = 0
            for num, k in enumerate(self._bandsplit):
                # Don't split this band :
                if k is None:
                    self._fsplit.append(np.ndarray.tolist(self._f[num, :]))
                    self._splitted.append(np.nan)
                # Split this band :
                elif isinstance(k, int):
                    reflist = binarize(self._f[num, 0], self._f[num, 1], k, k)
                    self._fsplit.extend(reflist)
                    self._splitted.extend([q] * len(reflist))
                    q += 1
            # Convert splitted into masked array :
            self._splitted = np.asarray(self._splitted).ravel()
            self._splitted = np.ma.masked_array(
            self._splitted, mask=np.isnan(self._splitted))
            # Get unique ordered values inside self._splitted :
            self._usplit = uorderlst(self._splitted.data)

    def __len__(self):
        return len(self._splitted)


    def __str__(self):
        return 'Split({})'.format(self._sup)

    ########################################################
    #                   USER FUNCTIONS
    ########################################################

    def getsplit(self):
        """Get the splitted frequency bands.

        This method return an array of splitted frequencies.

        """
        return self._fsplit

    def joinsplit(self, x, axis=0):
        """Join a splitted array according to the splitted frequency vector."""
        # Don't join :
        if self._splitted is None:
            return x
        # Join on axis :
        else:
            if x.shape[axis] is not len(self._splitted):
                raise ValueError(
                    'On axis ' + str(axis) + ', x must have a dim of ' + str(len(self._splitted)))
            else:
                # Allocate x :
                xshape = list(x.shape)
                xshape[axis] = self._nf
                xsplit = np.zeros(xshape, dtype=float)
                # First, get values that don't need to be meaned. idxnan_sp
                # represent dim of splitted x and idxnan of future meaned
                # array :
                idxnan_sp = [slice(None)] * x.ndim
                idxnan_sp[axis] = np.where(self._splitted.mask)[0]
                idxnan_sp = idxnan_sp.copy()
                idxnan = [slice(None)] * x.ndim
                idxnan[axis] = np.where(np.isnan(self._usplit))[0]
                xsplit[idxnan] = x[idxnan_sp]
                # Now, take the mean inside splitted bands :
                for val in self._splitted.compressed():
                    # Get where splitted is val :
                    idxnan_sp[axis] = np.where(self._splitted.data == val)[0]
                    # Index of future array :
                    idxnan[axis] = np.where(self._usplit == val)[0]
                    # Finally take the mean :
                    xsplit[idxnan] = np.mean(
                        x[idxnan_sp], axis=axis, keepdims=True)
            return xsplit

    ########################################################
    #                   PROPERTIES
    ########################################################

    @property
    def f(self):
        return self._f

    @property
    def bandsplit(self):
        return self._bandsplit

    @property
    def fsplit(self):
        return self._fsplit

    @property
    def splitted(self):
        return self._splitted
