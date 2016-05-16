import numpy as np
from itertools import product
from types import FunctionType

__all__ = ["perm_rndDatasets",
           "perm_swap",
           "perm_array",
           "perm_rep",
           "perm_metric",
           "perm_2pvalue"
           ]


def perm_2pvalue(data, perm, n_perm, threshold=None, tail=2):
    """Get the associated p-value of a permutation distribution

    Arg:
        data: array
            Array of real data. The shape must be (d1, d2, ..., dn)

        perm: array
            Array of permutations. The shape must be (n_perm, d1, d2, ..., dn)

        n_perm: int
            Number of permutations

    Kargs:
        threshold: int / float, optional, [def: None]
            Every values upper to threshold are going to be set to 1.

        tail: int, optional, [def: 2]
            Define if the calculation of p-value must take into account one
            or two tails of the permutation distribution

    Return:
        pvalue : array
            Array of associated p-values
    """
    # Check data type :
    if isinstance(data, (int, float)):
        data = np.matrix(data)
    if perm.ndim <= data.ndim:
        perm = perm[..., np.newaxis]

    # Check permutations shape :
    psh, dsh = np.array(perm.shape), np.array(data.shape)
    cond = int(psh[0] - n_perm)
    if (cond is not 0) or not np.array_equal(psh[1::], dsh):
        raise ValueError('perm must have a shape of'
                         ' '+str(tuple([n_perm]+list(dsh)))+' instead of '+str(perm.shape))

    # Loop on data values :
    dataLoop = product(*tuple([range(k) for k in data.shape]))

    # Get the permutation function :
    fcn = _tailfcn(perm, n_perm, tail)

    # Apply the function :
    pval = np.ones(data.shape)
    for k in dataLoop:
        pval[k] = fcn(perm[(np.arange(n_perm), *k)], data[k], n_perm)

    # Replace 0 by /n_perm :
    pval[np.where(pval == 0)] = 1/n_perm

    # Threshold results :
    if threshold is not None:
        pval[np.where(pval >= threshold)] = 1

    return pval


def permtail(perm, data, n_perm, tail=2):
    """Compute p-values from either superior or inferior part of the distribution.
    """
    # One tail (lower) :
    if tail == -1:
        return (np.sum(perm <= data)) / n_perm
    # One tail (upper) :
    elif tail == 1:
        return (np.sum(perm >= data)) / n_perm
    # Two tails :
    elif tail == 2:
        return (np.sum(np.abs(perm) >= np.abs(data))) / n_perm


def _tailfcn(perm, n_perm, tail):
    """Return either function for calculation of one or two tails
    """
    # One tail (lower) :
    if tail == -1:
        def evalp(perm, data, n_perm):
            return permtail(perm, data, n_perm, tail=tail)
    # One tail (upper) :
    elif tail == 1:
        def evalp(perm, data, n_perm):
            return permtail(perm, data, n_perm, tail=tail)
    # Two tails :
    elif tail == 2:
        def evalp(perm, data, n_perm):
            return permtail(perm, data, n_perm, tail=tail)

    return evalp


def perm_rndDatasets(mu=0, sigma=1, dmu=0.1, dsigma=0.1, size=(5, 5),
                     rndstate=0):
    """Generate data and permutations uniformly distributed.

    Kargs:
        mu: int/float
            Center of the distribution

        sigma: int/float
            Deviation of the distribution

        dmu: int/float
            Introduce a shift to the mean of data

        dsigma: int/float
            Introduce a shift to the deviation of data

        size: tuple
            Size of the generated data and permutations (d1, d2, ..., d3)

        rndstate: int
            Fix the random state of the machine

    Returns:
        data: array
            Simulated data of shape (n_perm, d1, d2, ..., d3)

        perm: array
            Simulated permutations of shape (n_perm, d1, d2, ..., d3)
    """
    rnd = np.random.RandomState(rndstate)
    perm = rnd.normal(loc=mu, scale=sigma, size=size)
    data = rnd.normal(loc=mu+dmu, scale=sigma+dsigma, size=size)
    return data, perm


def perm_swap(a, b, n_perm=200, axis=-1, rndstate=0):
    """Permute values between two arrays and generate a number of permutations.

    Args:
        a: ndarray
            Array of shape (d1, d2, ...)

        b: ndarray
            Array with the same shape of a

        n_pem: int
            Number of permutations

    Kargs:
        axis: int, optional, [def: -1]
            Axis for swapping values. If axis is -1, this mean that all
            values across all dimensions are going to be swap.

        rndstate: int
            Fix the random state of the machine

    Return:
        ash, bsh: array
            The two swapping arrays with a new shape of (n_perm ,d1, d2, ...)
    """
    # Check if size a == size b:
    if a.shape != b.shape:
        raise ValueError("Shape of 'a' is "+str(a.shape)+" shape of"
                         " b is "+str(b.shape)+". Both must be equal")
    ab_sh = a.shape

    # Shuffle both entire matrix
    if axis == -1:
        ab_backup = np.concatenate((np.ravel(a), np.ravel(b)))
        absh_mat = np.zeros((n_perm, len(ab_backup)))
        # For each permutation :
        for k in range(n_perm):
            # Backup of ab :
            ab = ab_backup.copy()
            # New random state :
            rnd = np.random.RandomState(rndstate+k)
            # Shuffle copy :
            rnd.shuffle(np.ravel(ab))
            absh_mat[k, :] = ab
        # Finally reshape data :
        absplit = np.split(absh_mat, 2, axis=1)
        return absplit[0].reshape(n_perm, *ab_sh), absplit[1].reshape(n_perm, *ab_sh)
    else:
        # Swap axes if axis != 0:
        if axis != 0:
            a = np.swapaxes(a, 0, axis)
            b = np.swapaxes(b, 0, axis)
        # Swap a & b along axis :
        ab_backup = np.concatenate((a, b), axis=0)
        absh_mat = np.zeros((n_perm, 2*a.shape[0], *tuple(list(a.shape)[1::])))
        # For each permutation :
        for k in range(n_perm):
            # Backup of ab :
            ab = ab_backup.copy()
            # New random state :
            rnd = np.random.RandomState(rndstate+k)
            # Shuffle copy :
            rnd.shuffle(ab)
            absh_mat[k, ...] = ab
        # Re-order a & b :
        absplit = np.split(absh_mat, 2, axis=1)
        a, b = absplit[0], absplit[1]
        if axis != 0:
            a = np.swapaxes(a, 1, axis+1)
            b = np.swapaxes(b, 1, axis+1)
        return a, b


def perm_array(x, n_perm=200, rndstate=0):
    """Generate n_perm permutations of a ndarray

    Args:
        x: array
            Data to repeat of shape (d1, d2, ..., d3)

        n_perm: int
            Number of permutations

        rndstate: int
            Fix the random state of the machine

    Returns:
        perm: array
            Repeated data of shape (n_perm, d1, d2, ..., d3)

        idx: array
            Index of permutations of shape (n_perm, d1, d2, ..., d3)
    """
    dim = tuple([n_perm] + list(x.shape))
    xrep = perm_rep(np.ravel(x), n_perm)
    xrep, idx = _scramble2D(xrep, rndstate=rndstate)

    return xrep.reshape(dim), idx.reshape(dim)


def perm_rep(x, n_perm):
    """Repeat a ndarray x n_perm times

    Args:
        x: array
            Data to repeat of shape (d1, d2, ..., d3)

        n_perm: int
            Number of permutations

    Returns:
        xrep: array
            Repeated data of shape (n_perm, d1, d2, ..., d3)
    """
    return np.tile(x[np.newaxis, ...], tuple([n_perm] + [1]*x.ndim))


def perm_metric(metric):
    """Get the metric
    """
    # Pre-defined metrics :
    if isinstance(metric, str):
        # None:
        if metric is None:
            def fcn(A, B, axis=0):
                return A
        # A - B :
        elif metric == 'm_minus':
            def fcn(A, B, axis=0):
                return A - B
        # (A - B) / std(B)
        elif metric == 'm_zscore':
            def fcn(A, B, axis=0):
                return (A - B) / np.std(B)
        # (A - B) / B
        elif metric == 'm_center':
            def fcn(A, B, axis=0):
                return (A - B) / np.mean(B)
        # Others :
        else:
            raise ValueError('No metric called "'+metric+'" is defined.')
    # User-defined metric :
    elif isinstance(metric, FunctionType):
        fcn = metric
    # Other
    else:
        raise ValueError("Unknown type of metric. Choose either pre-defined"
                         " metrics ('minus', 'zscore', 'center', 'divide') "
                         "or use your own function.")
    return fcn


def _scramble2D(a, rndstate=0):
    """Return an array with the values of `a` independently shuffled.
    """
    rnd = np.random.RandomState(rndstate)
    b = rnd.rand(*a.shape)
    idx = np.argsort(b)
    shuffled = a[np.arange(a.shape[0])[:, None], idx]
    return shuffled, idx
