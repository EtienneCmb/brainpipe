#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from warnings import warn
from .unit import time_to_sample


def normalize(A, B, kind=0, axis=0):
    """normalize A by B using the 'kind' parameter.

    Parameters
    ----------
    A : np.ndarray
        Array to normalize

    B : np.ndarray
        Array used for normalization.

    kind : int, optional [def : 0]
        0 // None : No normalisation
        1 // 'A-B' : Substraction
        2 // 'A/B' : Division
        3 // 'A-B/B' : Substract then divide
        4 // 'zscore': Z-score

    axis: int, optional, (def: 0)
        Specify the axis to take the mean or deviation of B.

    Return
    ------
    A normalize version of A.

    """
    # ----------------- INPUT CHECKING -------------------
    # Array checking :
    if not isinstance(A, np.ndarray):
        raise ValueError('A should be an array')
    else:
        A = A.astype(float)
    if not isinstance(B, np.ndarray):
        raise ValueError('B should be an array')
    else:
        B = B.astype(float)
    # Check existing norm :
    if kind not in [0, 1, 2, 3, 4, None, 'A-B', 'A/B', 'A-B/B', 'zscore']:
        raise ValueError(kind + ' is not an existing normalization. Use ' + str(0) + ', ' + str(
            1) + ', ' + str(2) + ', ' + str(3) + ', ' + str(4) + ", None, 'A-B', 'A/B', 'A-B/B' or 'zscore'")

    # ----------------- NORMALIZATION ------------------
    # Get mean of B and deviation of B :
    Bm = np.mean(B, axis=axis, keepdims=True)
    Bstd = np.std(B, axis=axis, keepdims=True)
    # No normalisation
    if kind in [0, None]:
        return A
    # Substraction
    elif kind in [1, 'A-B']:
        np.subtract(A, Bm, out=A)
        return A
    # Division
    elif kind in [2, 'A/B']:
        np.divide(A, Bm, out=A)
        return A
    # Substract then divide
    elif kind in [3, 'A-B/B']:
        np.subtract(A, Bm, out=A)
        np.divide(A, Bm, out=A)
        return A
    # Z-score
    elif kind in [4, 'zscore']:
        np.subtract(A, Bm, out=A)
        np.divide(A, Bstd, out=A)
        return A


class Normalization(object):

    """Normalize an array by a defined baseline."""

    def __init__(self, sf, kind=None, baseline=None, unit='sample'):
        """Normalize an array.

        Args:
            sf: int/float
                The sampling frequency

        Kargs:
            kind : int, optional [def : 0]
                0 // None : No normalisation
                1 // 'A-B' : Substraction
                2 // 'A/B' : Division
                3 // 'A-B/B' : Substract then divide
                4 // 'zscore': Z-score

            baseline: tuple/list/np.ndarray, optional, (def: None)
                Baseline period. Should be a list, tuple or array
                of lenght 2 (ex: baseline=(0, 255)). The baseline can be
                defined either in sample, second or millisecond.

            unit: string, optional, (def: 'sample')
                Define the unit of the baseline. Can be either 'sample'
                's' (second) or 'ms' (millisecond)

        Return:
            A normalization obejct with a .apply() method.

        """
        self._sf = sf
        self._kind = kind
        self._baseline = baseline
        self._unit = unit
        self._str = ''
        self._NormCheckInputs()

    def __str__(self):
        return 'Normalization({})'.format(self._str)

    ########################################################
    #                   SUB FUNCTIONS
    ########################################################
    def _NormCheckInputs(self):
        """Check inputs for normalization."""
        # Check sampling frequency :
        if not isinstance(self._sf, (int, float)):
            raise ValueError(
                'Sampling frequency must be either a float or an integer')
        else:
            self._sf = float(self._sf)
        # Check kind :
        if self._kind not in [0, 1, 2, 3, 4, None, 'A-B', 'A/B', 'A-B/B', 'zscore']:
            raise ValueError(self._kind + ' is not an existing normalization. Use ' + str(0) + ', ' + str(
                1) + ', ' + str(2) + ', ' + str(3) + ', ' + str(4) + ", None, 'A-B', 'A/B', 'A-B/B' or 'zscore'")
        else:
            if (self._kind not in [0, None]) and (self._baseline is None):
                warn(
                    'If kind is not None (or 0), you should not let the baseline parameter to None')
                self._kind = None
            if self._kind in [0, None]:
                self._baseline = None
                self._kind = None
        # Check the baseline :
        if self._baseline is not None:
            if not isinstance(self._baseline, (tuple, list, np.ndarray)):
                raise ValueError(
                    'The baseline parameter should be either  list, tuple or array')
            else:
                self._baseline = np.array(self._baseline).ravel()
                if len(self._baseline) is not 2:
                    raise ValueError('The lentgh of the baseline should be 2.')
                if self._baseline[0] > self._baseline[1]:
                    raise ValueError(
                        'baseline ending shoud be >= at baseline starting.')
        # Unit checking :
        if (self._unit not in ['sample', 's', 'ms']) or not isinstance(self._unit, str):
            raise ValueError("unit must be either 'sample', 's' or 'ms'")
        else:
            if (self._unit in ['s', 'ms']) and (self._baseline is not None):
                self._baseline = time_to_sample(self._baseline,
                                                self._sf, from_unit=self._unit)
                self._unit = 'sample'
        # String management :
        if (self._baseline is None) or (self._kind is None):
            self._str = 'None'
        else:
            # Detect normalization type and set string :
            if self._kind in [1, 'A-B']:
                sup = 'A - mean(Baseline)'
            elif self._kind in [2, 'A/B']:
                sup = 'A / mean(Baseline)'
            elif self._kind in [3, 'A-B/B']:
                sup = '(A - mean(Baseline) / mean(Baseline)'
            elif self._kind in [4, 'zscore']:
                sup = 'A - mean(Baseline) / std(Baseline)'
            # Build string :
            self._str = 'kind=' + sup + \
                ', baseline=(' + str(self._baseline[0]) + ', ' + \
                str(self._baseline[1]) + '), unit=' + self._unit

    ########################################################
    #                   USER FUNCTIONS
    ########################################################
    def apply(self, x, axis=0):
        """Apply the defined normalization to a matrix x along axis."""
        # Check shape :
        npts = x.shape[axis]
        if self._baseline is not None:
            if any(self._baseline > npts):
                raise ValueError(
                    'Defined baseline use points larger than the shape of x')
            else:
                idx = [slice(None)] * x.ndim
                idx[axis] = slice(self._baseline[0], self._baseline[1])
        else:
            idx = [slice(None)] * x.ndim
        return normalize(x, x[idx], self._kind, axis=axis)

    ########################################################
    #                   PROPERTIES
    ########################################################
    @property
    def sf(self):
        return self._sf

    @sf.setter
    def sf(self, value):
        self._sf = value
        self._NormCheckInputs()

    @property
    def kind(self):
        return self._kind

    @kind.setter
    def kind(self, value):
        self._kind = value
        self._NormCheckInputs()

    @property
    def baseline(self):
        return self._baseline

    @baseline.setter
    def baseline(self, value):
        self._baseline = value
        self._NormCheckInputs()

    @property
    def unit(self):
        return self._unit

    @unit.setter
    def unit(self, value):
        self._unit = value
        self._NormCheckInputs()
