"""Mutual information measures."""
import numpy as np


def shannon_entropy(x):
    """Compute the Shannon entropy.

    H(x) = sum{ x * log2(x) }

    Parameters
    ----------
    x : array_like
        Probability array.

    Notes
    -----
    Schreiber, Measuring information transfer, 2000
    """
    assert (x.max() <= 1) and (0 <= x.min())
    # Take only non-zero values as log(0) = 0 :
    nnz_x = x[np.nonzero(x)]
    return -np.sum(nnz_x * np.log2(nnz_x))


def _mi_xy(xy, bins):
    # Rebuild x and y :
    n_pts = int(len(xy) / 2)
    x, y = xy[0:n_pts], xy[n_pts::]
    return _mi(x, y, bins)


def _mi(x, y, bins=64):
    """MI on two time-series."""
    # Compute probabilities :
    p_x = np.histogram(x, bins)[0]
    p_y = np.histogram(y, bins)[0]
    p_xy = np.histogram2d(x, y, bins)[0]
    p_x = p_x / p_x.sum()
    p_y = p_y / p_y.sum()
    p_xy = p_xy / p_xy.sum()
    # Get entropy :
    h_x = shannon_entropy(p_x)
    h_y = shannon_entropy(p_y)
    h_xy = shannon_entropy(p_xy)
    # Compute mutual information :
    return h_x + h_y - h_xy


# def _mi_stat(xy, bins):
#     """Function equivalent to the _mi_xy."""
#     from scipy.stats import chi2_contingency
#     # Rebuild x and y :
#     n_pts = int(len(xy) / 2)
#     x, y = xy[0:n_pts], xy[n_pts::]
#     c_xy = np.histogram2d(x, y, bins)[0]
#     g, p, dof, expected = chi2_contingency(c_xy, lambda_="log-likelihood")
#     mi = 0.5 * g / c_xy.sum()
#     return mi / np.log(2)


def cmi(x, y, axis=0, bins=64):
    """Compute the cross mutual information of two arrays.

    MI = H(x) + H(y) - H((x, y))

    Parameters
    ----------
    x, y : array_like
        Arrays used to compute MI.
    axis : int | 0
        Location of the time axis.
    bins : int | 64
        Number of bins used for the histogram to get the probability of each
        ellement.

    Returns
    -------
    mi : array_like
        Array of mutual informations.

    Notes
    -----
    Schreiber, Measuring information transfer, 2000
    """
    assert all([isinstance(k, np.ndarray) for k in [x, y]])
    assert x.shape == y.shape
    xy = np.concatenate((x, y), axis=axis)
    return np.apply_along_axis(_mi_xy, axis, xy, bins)
