#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


def normalize(A, B, norm=0, axis=0):
    """normalize A by B using the 'norm' parameter.

    Parameters
    ----------
    A : np.ndarray
        Array to normalize

    B : np.ndarray
        Array used for normalization.

    norm : int, optional [def : 0]
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
        raise ValueError("A should be an array")
    else:
        A = A.astype(float)
    if not isinstance(B, np.ndarray):
        raise ValueError("B should be an array")
    else:
        B = B.astype(float)
    # Check existing norm :
    if norm not in [0, 1, 2, 3, 4, None, 'A-B', 'A/B', 'A-B/B', 'zscore']:
        raise ValueError(norm + ' is not an existing normalization. Use ' + str(0) + ', ' + str(
            1) + ', ' + str(2) + ', ' + str(3) + ', ' + str(4) + ", None, 'A-B', 'A/B', 'A-B/B' or 'zscore'")

    # ----------------- NORMALIZATION ------------------
    # Get mean of B and deviation of B :
    Bm = np.mean(B, axis=axis, keepdims=True)
    Bstd = np.std(B, axis=axis, keepdims=True)
    # No normalisation
    if (norm == 0) or (norm is None):
        return A
    # Substraction
    elif norm == 1:
        np.subtract(A, Bm, out=A)
        return A
    # Division
    elif norm == 2:
        np.divide(A, Bm, out=A)
        return A
    # Substract then divide
    elif norm == 3:
        np.subtract(A, Bm, out=A)
        np.divide(A, Bm, out=A)
        return A
    # Z-score
    elif norm == 4:
        np.subtract(A, Bm, out=A)
        np.divide(A, Bstd, out=A)
        return A
