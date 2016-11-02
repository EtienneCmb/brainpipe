import numpy as np
from scipy.stats import binom

__all__ = ['bino_da2p', 'bino_p2da', 'bino_signifeat']


def bino_da2p(y, da):
    """For a given vector label, get p-values of a decoding accuracy
    using the binomial law.

    Args:
        y : array
            The vector label

        da: int / float / list /array [0 <= da <= 100]
            The decoding accuracy array. Ex : da = [75, 33, 25, 17].

    Return:
        p: ndarray
            The p-value associate to each decoding accuracy
    """
    y = np.ravel(y)
    nbepoch = len(y)
    nbclass = len(np.unique(y))
    if not isinstance(da, np.ndarray):
        da = np.array(da)
    if (da.max() > 100) or (da.min() < 0):
        raise ValueError('Consider 0<=da<=100')
    return 1 - binom.cdf(nbepoch * da / 100, nbepoch, 1 / nbclass)


def bino_p2da(y, p):
    """For a given vector label, get the decoding accuracy of p-values
    using the binomial law.

    Args:
        y: array
            The vector label

        p: int / float / list / array [0 <= p < 1]
            p-value. Ex : p = [0.05, 0.01, 0.001, 0.00001]

    Return:
        da: ndarray
            The decoding accuracy associate to each p-value
    """
    y = np.ravel(y)
    nbepoch = len(y)
    nbclass = len(np.unique(y))
    if not isinstance(p, np.ndarray):
        p = np.array(p)
    if (p.max() >= 1):
        raise ValueError('Consider 0<=p<1')
    return binom.ppf(1 - p, nbepoch, 1 / nbclass) * 100 / nbepoch


def bino_signifeat(feat, th):
    """Get significants features.

    Args:
        feat: array
            Array containing either decoding accuracy (da) either p-values

        th: int / float
            The threshold in order to find significants features.

    Return:
        index: array
            The index corresponding to significants features

        signi_feat: array
            The significants features
    """
    # If feat is p-values :
    if (feat.min() >= 0) and (feat.max() < 1) and (th >= 0) and (th < 1):
        index = np.where(feat < th)
    # If feat is da :
    elif (feat.min() >= 0) and (feat.max() <= 100) and (th >= 0) and (th <= 100):
        index = np.where(feat >= th)
    # Any other case :
    else:
        raise ValueError('Cannot recognize the type of feat.')

    return index, feat[index]
