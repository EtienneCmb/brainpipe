from scipy.stats import binom
import numpy as n

####################################################################
# - Statistical binomial threshold :
####################################################################


def binostat(y, p):
    y = n.ravel(y)
    nbepoch = len(y)
    nbclass = len(n.unique(y))
    return binom.ppf(1 - p, nbepoch, 1 / nbclass) * 100 / nbepoch


def binostatinv(y, da):
    y = n.ravel(y)
    nbepoch = len(y)
    nbclass = len(n.unique(y))
    return 1 - binom.cdf(nbepoch * da / 100, nbepoch, 1 / nbclass)


def binofeat(y, da, p):
    th = binostat(y, p)
    signifeat = da >= th
    try:
        return signifeat[0], th
    except:
        return [signifeat], th
