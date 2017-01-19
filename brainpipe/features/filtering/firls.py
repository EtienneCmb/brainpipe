#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from numpy import matlib

def NoddFcn(F, M, W, L):  # N is odd
    # Variables :
    b0 = 0
    m = np.arange(int(L + 1))
    k = m[1:len(m)]
    b = np.zeros(k.shape)

    # Run Loop :
    for s in range(0, len(F), 2):
        m = (M[s + 1] - M[s]) / (F[s + 1] - F[s])
        b1 = M[s] - m * F[s]
        b0 = b0 + (b1 * (F[s + 1] - F[s]) + m / 2 * (
            F[s + 1] * F[s + 1] - F[s] * F[s])) * abs(
            np.square(W[round((s + 1) / 2)]))
        b = b + (m / (4 * np.pi * np.pi) * (
            np.cos(2 * np.pi * k * F[s + 1]) - np.cos(2 * np.pi * k * F[s])
        ) / (k * k)) * abs(np.square(W[round((s + 1) / 2)]))
        b = b + (F[s + 1] * (m * F[s + 1] + b1) * np.sinc(2 * k * F[
            s + 1]) - F[s] * (m * F[s] + b1) * np.sinc(2 * k * F[s])) * abs(
            np.square(W[round((s + 1) / 2)]))

    b = np.insert(b, 0, b0)
    a = (np.square(W[0])) * 4 * b
    a[0] = a[0] / 2
    aud = np.flipud(a[1:len(a)]) / 2
    a2 = np.insert(aud, len(aud), a[0])
    h = np.concatenate((a2, a[1:] / 2))

    return h


def NevenFcn(F, M, W, L):  # N is even
    # Variables :
    k = np.arange(0, int(L) + 1, 1) + 0.5
    b = np.zeros(k.shape)

    # Run Loop :
    for s in range(0, len(F), 2):
        m = (M[s + 1] - M[s]) / (F[s + 1] - F[s])
        b1 = M[s] - m * F[s]
        b = b + (m / (4 * np.pi * np.pi) * (np.cos(2 * np.pi * k * F[
            s + 1]) - np.cos(2 * np.pi * k * F[s])) / (
            k * k)) * abs(np.square(W[round((s + 1) / 2)]))
        b = b + (F[s + 1] * (m * F[s + 1] + b1) * np.sinc(2 * k * F[
            s + 1]) - F[s] * (m * F[s] + b1) * np.sinc(2 * k * F[s])) * abs(
            np.square(W[round((s + 1) / 2)]))

    a = (np.square(W[0])) * 4 * b
    h = 0.5 * np.concatenate((np.flipud(a), a))

    return h


def firls(N, F, M):
    """Switch between odd/even."""
    # Variables definition :
    W = np.ones(round(len(F) / 2), dtype=float)
    N += 1
    F /= 2
    L = (N - 1) / 2

    Nodd = bool(N % 2)

    if Nodd:    # Odd case
        h = NoddFcn(F, M, W, L)
    else:       # Even case
        h = NevenFcn(F, M, W, L)

    return h


def fir1(N, Wn):
    """Get fir1 coefficients."""
    # Variables definition :
    nbands = len(Wn) + 1
    ff = np.array((0, Wn[0], Wn[0], Wn[1], Wn[1], 1))

    f0 = np.mean(ff[2:4])
    L = N + 1

    mags = np.arange(nbands) % 2
    aa = np.ravel(matlib.repmat(mags, 2, 1), order='F')

    # Get filter coefficients :
    h = firls(L - 1, ff, aa)

    # Apply a window to coefficients :
    Wind = np.hamming(L)
    b = np.array(h.T * Wind)
    c = np.exp(-1j * 2 * np.pi * (f0 / 2) * np.arange(L))
    b = b / abs(c * b.T)

    return np.squeeze(b), 1


def fir_order(Fs, sizevec, flow, cycle=3):
    """Get firls order filter."""
    filtorder = cycle * (Fs // flow)

    if (sizevec < 3 * filtorder):
        filtorder = (sizevec - 1) // 3

    return int(filtorder)
