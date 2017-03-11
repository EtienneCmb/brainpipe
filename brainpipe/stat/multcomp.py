import numpy as np
from scipy.stats import binom
from itertools import product


__all__ = ["bonferroni", "fdr", "maxstat"]


def bonferroni(p, axis=-1):
    """Bonferroni correction

    Args:
        p: array
            Array of p-values

    Kargs:
        axis: int, optional, [def: -1]
            Axis to apply the Bonferroni correction. If axis is -1,
            the correction is applied through all dimensions.

    Return:
        Corrected pvalues
    """
    if axis == -1:
        fact = len(np.ravel(p))
    else:
        fact = p.shape[axis]
    return fact*p


def fdr(p, q):
    """False Discovery Rate correction

    Args:
        p: array
            Array of p-values

        q: float
            Thresholding p-value

    Return:
        pcorr: array
            Corrected p-values
    """
    # Thresholding function :
    def c(V):
        # See Genovese, Lazar and Holmes (2002) page 872, second column,
        # first paragraph:
        if V < 1000:
            # Compute it exactly
            s = np.sum(1./np.arange(1, V+1, 1))
        else:
            # Approximate it
            s = np.log(V) + 0.57721566490153286060651209008240243104215933593992359880576723488486772677766467093694706329174674951463144724980708248096050401448654283622417399764492353625350033374293733773767394279259525824709491600873520394816567
        return s
    # Convert the input into a row vector and sort
    pback = p.copy()
    dim = p.shape
    p = np.ravel(p)
    idx = np.argsort(p, axis=0)
    p = p[idx]
    V = len(p)
    # Compute the threshold probability for each voxel
    pi = (np.arange(1, V+1)/V) * q / c(V)
    h = p <= pi
    # Undo the sorting and convert the output back into the original format
    idxrev = np.argsort(idx)
    h = h[idxrev].reshape(dim)
    pback[~h] = 1
    return pback


def maxstat(perm, axis=-1):
    """Correction by the maximum statistic

    Args:
        perm: array
            The permutations.

        axis: integer, optional, [def: -1]
            Use -1 to correct through all dimensions. Otherwise,
            use d1, d2, ... or dn to correct through a specific dimension.

    Kargs:
        permR: array
            The re-aranged permutations according to the selectionned
            axis. Then use perm_2pvalues to get the p-value according to this
            distribution.
    """
    # Max through all dimensions :
    if axis == -1:
        permR = perm.max()*np.ones(perm.shape)
    # Max through specific dimension :
    elif axis >= 0:
        permR = np.max(perm, axis=axis, keepdims=True)
        psh, pRsh = perm.shape, permR.shape
        permR = np.tile(permR, [int(psh[k]/pRsh[k]) for k in range(perm.ndim)])
    # Any other values :
    else:
        raise ValueError('axis must be an integer between'
                         ' -1 and '+str(perm.ndim-1))
    return permR


# class _multcomp(object):

#     """Class for multiple comparison inheritance
#     """
#     def __init__(self):
#         pass

#     @staticmethod
#     def maxstat(perm, axis=-1):
#         """
#         """
#         return maxstat(perm, axis=axis)

#     @staticmethod
#     def fdr(p, q):
#         """
#         """
#         return fdr(p, q)

#     @staticmethod
#     def bonferroni(p, axis=-1):
#         """
#         """
#         return bonferroni(p, axis=axis)

# _multcomp.maxstat.__doc__ += maxstat.__doc__
# _multcomp.fdr.__doc__ += fdr.__doc__
# _multcomp.bonferroni.__doc__ += bonferroni.__doc__
