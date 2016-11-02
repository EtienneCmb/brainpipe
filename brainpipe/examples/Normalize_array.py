#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This script illustrate how to normalize an array."""
import numpy as np
import matplotlib.pyplot as plt
from brainpipe.features import normalize


def _plot(mat, title=''):
    """Plotting function."""
    plt.pcolormesh(mat, cmap='inferno')
    plt.colorbar()
    plt.xlabel('Axis 1'), plt.ylabel('Axis 2')
    plt.title(title, y=1.02)
    plt.axis('tight')


if __name__ == '__main__':
    # Define an array A to normalize by B :
    A = np.random.rand(200, 4000)
    B = A[:, 20:100]

    # Use the diffrent types of normalization :
    A0 = normalize(A, B, norm=0, axis=1)
    t0 = 'No normalization'
    A1 = normalize(A, B, norm=1, axis=1)
    t1 = 'A - mean(B)'
    A2 = normalize(A, B, norm=2, axis=1)
    t2 = 'A / mean(B)'
    A3 = normalize(A, B, norm=3, axis=1)
    t3 = '(A - mean(B))/mean(B)'
    A4 = normalize(A, B, norm=4, axis=1)
    t4 = 'Z-score (A - mean(B))/std(B)'

    # Plot :
    plt.figure()
    plt.subplot(151), _plot(A0, t0)
    plt.subplot(152), _plot(A1, t1)
    plt.subplot(153), _plot(A2, t2)
    plt.subplot(154), _plot(A3, t3)
    plt.subplot(155), _plot(A4, t4)

    plt.show()
