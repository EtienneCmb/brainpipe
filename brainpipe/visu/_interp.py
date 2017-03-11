import numpy as n

__all__ = ['mapinterpolation']


def mapinterpolation(data, x=None, y=None, interpx=1, interpy=1):
    """Interpolate a 2D map"""
    # Get data size :
    dim2, dim1 = data.shape
    ndim2, ndim1 = dim2/interpy, dim1/interpx
    # Define xticklabel and yticklabel :
    if x is None:
        x = n.arange(0, dim1, interpx)
    if y is None:
        y = n.arange(0, dim2, interpy)
    # Define the meshgrid :
    Xi, Yi = n.meshgrid(
        n.arange(0, dim1-1, interpx), n.arange(0, dim2-1, interpy))
    # 2D interpolation :
    datainterp = interp2(data, Xi, Yi)
    # Linearly interpolate vectors :
    xvecI = n.linspace(x[0], x[-1], datainterp.shape[0])
    yvecI = n.linspace(y[0], y[-1], datainterp.shape[1])

    return datainterp, xvecI, yvecI


def interplinearvec(x, interpx):
    return list(n.linspace(x[0], x[-1], len(x)/interpx))


def interp2(z, xi, yi, extrapval=0):  # n.nan):
    """
    Linear interpolation equivalent to interp2(z, xi, yi,'linear') in MATLAB
    @param z: function defined on square lattice [0..width(z))X[0..height(z))
    @param xi: matrix of x coordinates where interpolation is required
    @param yi: matrix of y coordinates where interpolation is required
    @param extrapval: value for out of range positions. default is numpy.nan
    @return: interpolated values in [xi,yi] points
    @raise Exception:
    """

    x = xi.copy()
    y = yi.copy()
    nrows, ncols = z.shape

    if nrows < 2 or ncols < 2:
        raise Exception("z shape is too small")

    if not x.shape == y.shape:
        raise Exception("sizes of X indexes and Y-indexes must match")

    # find x values out of range
    x_bad = ((x < 0) | (x > ncols-1))
    if x_bad.any():
        x[x_bad] = 0

    # find y values out of range
    y_bad = ((y < 0) | (y > nrows-1))
    if y_bad.any():
        y[y_bad] = 0

    # linear indexing. z must be in 'C' order
    ndx = n.floor(y) * ncols + n.floor(x)
    ndx = ndx.astype('int32')

    # fix parameters on x border
    d = (x == ncols - 1)
    x = (x - n.floor(x))
    if d.any():
        x[d] += 1
        ndx[d] -= 1

    # fix parameters on y border
    d = (y == nrows - 1)
    y = (y - n.floor(y))
    if d.any():
        y[d] += 1
        ndx[d] -= ncols

    # interpolate
    one_minus_t = 1 - y
    z = z.ravel()
    f = (z[ndx] * one_minus_t + z[ndx + ncols] * y) * (1 - x) + (
        z[ndx + 1] * one_minus_t + z[ndx + ncols + 1] * y) * x

    # Set out of range positions to extrapval
    if x_bad.any():
        f[x_bad] = extrapval
    if y_bad.any():
        f[y_bad] = extrapval

    return f
