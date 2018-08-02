"""Statistics for connectivity."""
import logging

import numpy as np
from itertools import permutations
from scipy import spatial

from .correction import get_pairs
from ..sys import set_log_level
from ..statistics import perm_2pvalue


logger = logging.getLogger('brainpipe')


def fc_summarize(ts, axis=0, method='std', verbose=None):
    """Summarize dFC measure across the time dimension.

    Parameters
    ----------
    ts : array_like
        Connectivity array to summarize.
    axis : int | 0
        Location of dimension to summarize.
    method : {'std', 'coefvar', 'mean', 'l2'}
        Method used to summarize connectivity arrays across the time dimension.
        Use 'std' for standard deviation, 'mean', 'coefvar' (see notes) or 'l2'
        for L2 norm.
    verbose : bool, str, int, or None
        The verbosity of messages to print. If a str, it can be either
        PROFILER, DEBUG, INFO, WARNING, ERROR, or CRITICAL.

    Notes
    -----
    Javier Gonzalez-Castillo, The spatial structure of resting state
    connectivity stability on the scale of minutes, 2014
    """
    set_log_level(verbose)
    assert isinstance(ts, np.ndarray)
    assert isinstance(axis, int) and (axis <= ts.ndim)
    assert method in ['std', 'coefvar', 'mean', 'l2']
    # Summarize :
    if method == 'std':
        logger.info("Summarize using the standard deviation")
        return ts.std(axis=axis)
    elif method == 'mean':
        logger.info("Summarize using the mean")
        return ts.mean(axis=axis)
    elif method == 'coefvar':
        logger.info("Summarize using the coefficient of variation")
        # Correlation -> Fisher's z-score :
        ts_zs = np.arctanh(ts)
        # Summarized :
        ts_m = np.mean(ts_zs, axis=axis)
        ts_m[ts_m == 0.] = 1.
        ts_sum = np.std(ts_zs, axis=axis) / ts_m
        # Fisher's z-score -> Correlation :
        return np.tanh(ts_sum)
    elif method == 'l2':
        logger.info("Summarize using L2 norm")
        return np.sqrt(np.sum(ts ** 2, axis=axis))


def permute_connectivity(connect, n_perm=200, rndstate=0, part='upper'):
    """Permute a connectivity array.

    Parameters
    ----------
    connect : array_like
        Connectivit array of shape (n_sites, n_sites)
    n_perm : int | 200
        Number of permutations to perform.
    rndstate : int | 0
        Random state to use for reproducibility.
    part : {'upper', 'lower', 'both'}
        Randomize the array along either the 'upper', 'lower' or 'both' part
        of the connectivity array.

    Returns
    -------
    r2d_perm : array_like
        Permuted connectivity array of shape (n_perm, n_sites, n_sites).
    """
    assert isinstance(connect, np.ndarray) and (connect.ndim == 2)
    assert connect.shape[0] == connect.shape[1]
    assert isinstance(n_perm, int) and (n_perm > 0)
    assert isinstance(rndstate, int)
    n_sites = connect.shape[0]
    # Ravel connectivity array :
    pairs = get_pairs(n_sites, part=part, as_array=False)
    r_connect = connect[pairs]
    assert r_connect.ndim == 1
    # Shuffle the ravel version of the connectivity array :
    rnd_state = np.random.RandomState(rndstate)
    r_perm = np.zeros((n_perm, len(r_connect)), dtype=r_connect.dtype)
    for k in range(n_perm):
        r_perm[k, :] = rnd_state.permutation(r_connect)
    # Reconstuct the 2D connectivity array :
    r2d_perm = np.zeros((n_perm, n_sites, n_sites), dtype=connect.dtype)
    r2d_perm[:, pairs[0], pairs[1]] = r_perm
    return r2d_perm


def statistical_summary(connect, n_perm=200, part='upper', method='l2',
                        as_pval=True, tail=1, verbose=None):
    """Get a statistical summary across a time dimension.

    Parameters
    ----------
    connect : array_like
        Time-resolved connectivity array of shape (n_sites, n_sites, n_times)
    n_perm : int | 200
        Number of permutations to perform.
    part : {'upper', 'lower', 'both'}
        Randomize the array along either the 'upper', 'lower' or 'both' part
        of the connectivity array.
    method : {'std', 'coefvar', 'mean', 'l2'}
        Method used to summarize connectivity arrays across the time dimension.
        Use 'std' for standard deviation, 'mean', 'coefvar' (see notes) or 'l2'
        for L2 norm.
    as_pval : bool | True
        If True, p-values array of shape (n_sites, n_sites) are returned. If
        False, summarized permutations of shape (n_perm, n_sites, n_sites) are
        returned instead.
    tail : int | 1
        If `as_pval` is True, specify if one tail or two should be considered
        when p-values are computed.
    verbose : bool, str, int, or None
        The verbosity of messages to print. If a str, it can be either
        PROFILER, DEBUG, INFO, WARNING, ERROR, or CRITICAL.

    Returns
    -------
    arr : array_like
        P-values if `as_pval` is True otherwise summarized permutation.
    """
    assert isinstance(connect, np.ndarray) and (connect.ndim == 3)
    assert connect.shape[0] == connect.shape[1]
    n_times = connect.shape[-1]
    set_log_level(verbose)
    logger.info("Statistical summary of %s part using %s method on %i "
                "permutations" % (part, method, n_perm))

    connect_p = np.zeros((n_perm, *connect.shape), dtype=connect.dtype)
    for k in range(n_times):
        # Randomize the connectivity array :
        connect_p[..., k] = permute_connectivity(connect[..., k],
                                                 n_perm=n_perm, part='upper',
                                                 rndstate=k)
    # Summarize the array along the time dimension :
    connect_psum = fc_summarize(connect_p, axis=3, method=method,
                                verbose=False)
    assert connect_psum.ndim == 3
    if not as_pval:
        return connect_psum
    # If needed, return p-values instead of permutations
    connect_sum = fc_summarize(connect, axis=2, method=method, verbose=False)
    assert connect_sum.ndim == 2
    return perm_2pvalue(connect_sum, connect_psum, n_perm, tail=tail)


def random_phase(ts, axis=0):
    """Random phase for statistical assessment of connectivity.

    Parameters
    ----------
    ts : array_like
        Time series.
    axis : int | 0
        Location of the time axis.

    Returns
    -------
    ts_p : array_like
        Time-series with a random phase added to it.

    Notes
    -----
    Liegeois, R., Laumann, T. O., Snyder, A. Z., Zhou, J., and Yeo, B. T.
    (2017). Interpreting temporal fluctuations in resting-state functional
    connectivity MRI. Neuroimage.
    """
    n_pts = ts.shape[axis]
    # Demean the time-series :
    # ts_m = ts.mean(axis=axis, keepdims=True)
    ts_d = ts  # - ts_m
    # Compute the DFT :
    ts_dft = np.fft.rfft(ts_d, axis=axis)
    # Prepare the random phase :
    phi_shape = [1] * ts_dft.ndim
    phi_shape[axis] = ts_dft.shape[axis]
    phi = np.random.uniform(0, 2 * np.pi, phi_shape)
    # Repeat the phase across all dimensions :
    phi_rep = list(ts.shape)
    phi_rep[axis] = 1
    phi = np.tile(phi, phi_rep)
    # Add this phase to the DFT :
    ts_dft = np.abs(ts_dft) * np.exp(1j * phi)
    return np.fft.irfft(ts_dft, n=n_pts, axis=axis)  # + ts_m


def mantel(x, y, perms=100, method='pearson', tail='two-tail'):
    """Perform the mantel test.

    Takes two distance matrices (either redundant matrices or condensed
    vectors) and performs a Mantel test. The Mantel test is a significance test
    of the correlation between two distance matrices.

    Parameters
    ----------
    x : array_like
      First distance matrix (condensed or redundant).
    y : array_like
      Second distance matrix (condensed or redundant), where the order of
      elements corresponds to the order of elements in the first matrix.
    perms : int | 100
      The number of permutations to perform. A larger number gives more
      reliable results but takes longer to run. If the actual number
      of possible permutations is smaller, the program will enumerate all
      permutations. Enumeration can be forced by setting this argument to 0.
    method : {'pearson', 'spearman'}
      Type of correlation coefficient to use.
    tail : {'upper', 'lower', 'two-tail'}
      Which tail to test in the calculation of the empirical p-value.

    Returns
    -------
    r : float
      Veridical correlation
    p : float
      Empirical p-value
    z : float
      Standard score (z-score)
    """
    assert method in ['spearman', 'pearson']
    assert tail in ['upper', 'lower', 'two-tail']
    assert x.shape == y.shape
    # Ensure that x and y are formatted as Numpy arrays.
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)

    # Check that x and y are valid distance matrices.
    if not spatial.distance.is_valid_dm(x) and spatial.distance.is_valid_y(x) == False:
        raise ValueError("x is not a valid condensed or redundant distance matrix")
    if spatial.distance.is_valid_dm(y) == False and spatial.distance.is_valid_y(y) == False:
        raise ValueError('y is not a valid condensed or redundant distance matrix')

    # If x or y is a redundant distance matrix, reduce it to a condensed distance matrix.
    if x.ndim == 2:
        x = spatial.distance.squareform(x, force='tovector', checks=False)
    if y.ndim == 2:
        y = spatial.distance.squareform(y, force='tovector', checks=False)

    # Check for minimum size.
    if x.shape[0] < 3:
        raise ValueError('x and y should represent at least 3 objects')

    # Now we're ready to start the Mantel test using a number of optimizations:
    #
    # 1. We don't need to recalculate the pairwise distances between the objects
    #    on every permutation. They've already been calculated, so we can use a
    #    simple matrix shuffling technique to avoid recomputing them. This works
    #    like memoization.
    #
    # 2. Rather than compute correlation coefficients, we'll just compute the
    #    covariances. This works because the denominator in the equation for the
    #    correlation coefficient will yield the same result however the objects
    #    are permuted, making it redundant. Removing the denominator leaves us
    #    with the covariance.
    #
    # 3. Rather than permute the y distances and derive the residuals to calculate
    #    the covariance with the x distances, we'll represent the y residuals in
    #    the matrix and shuffle those directly.
    #
    # 4. If the number of possible permutations is less than the number of
    #    permutations that were requested, we'll run a deterministic test where
    #    we try all possible permutations rather than sample the permutation
    #    space. This gives a faster, deterministic result.

    # Calculate the x and y residuals, which will be used to compute the
    # covariance under each permutation.
    x_residuals, y_residuals = x - x.mean(), y - y.mean()

    # Expand the y residuals to a redundant matrix.
    y_residuals_as_matrix = spatial.distance.squareform(y_residuals, force='tomatrix', checks=False)

    # Get the number of objects.
    m = y_residuals_as_matrix.shape[0]

    # Calculate the number of possible matrix permutations.
    n = np.math.factorial(m)

    # Initialize an empty array to store temporary permutations of y_residuals.
    y_residuals_permuted = np.zeros(y_residuals.shape[0], dtype=float)

    # If the number of requested permutations is greater than the number of
    # possible permutations (m!) or the perms parameter is set to 0, then run a
    # deterministic Mantel test ...
    if perms >= n or perms == 0:

        # Initialize an empty array to store the covariances.
        covariances = np.zeros(n, dtype=float)

        # Enumerate all permutations of row/column orders and iterate over them.
        for i, order in enumerate(permutations(range(m))):

            # Take a permutation of the matrix.
            y_residuals_as_matrix_permuted = y_residuals_as_matrix[order, :][:, order]

            # Condense the permuted version of the matrix. Rather than use
            # distance.squareform(), we call directly into the C wrapper for speed.
            spatial.distance._distance_wrap.to_vector_from_squareform_wrap(y_residuals_as_matrix_permuted, y_residuals_permuted)

            # Compute and store the covariance.
            covariances[i] = (x_residuals * y_residuals_permuted).sum()

    # ... otherwise run a stochastic Mantel test.
    else:

        # Initialize an empty array to store the covariances.
        covariances = np.zeros(perms, dtype=float)

        # Initialize an array to store the permutation order.
        order = np.arange(m)

        # Store the veridical covariance in 0th position...
        covariances[0] = (x_residuals * y_residuals).sum()

        # ...and then run the random permutations.
        for i in range(1, perms):

            # Choose a random order in which to permute the rows and columns.
            np.random.shuffle(order)

            # Take a permutation of the matrix.
            y_residuals_as_matrix_permuted = y_residuals_as_matrix[order, :][:, order]

            # Condense the permuted version of the matrix. Rather than use
            # distance.squareform(), we call directly into the C wrapper for speed.
            spatial.distance._distance_wrap.to_vector_from_squareform_wrap(y_residuals_as_matrix_permuted, y_residuals_permuted)

            # Compute and store the covariance.
            covariances[i] = (x_residuals * y_residuals_permuted).sum()

    # Calculate the veridical correlation coefficient from the veridical covariance.
    r = covariances[0] / np.sqrt((x_residuals ** 2).sum() * (y_residuals ** 2).sum())

    # Calculate the empirical p-value for the upper or lower tail.
    if tail == 'upper':
        p = (covariances >= covariances[0]).sum() / float(covariances.shape[0])
    elif tail == 'lower':
        p = (covariances <= covariances[0]).sum() / float(covariances.shape[0])
    elif tail == 'two-tail':
        p = (abs(covariances) >= abs(covariances[0])).sum() / float(covariances.shape[0])

    # Calculate the standard score.
    z = (covariances[0] - covariances.mean()) / covariances.std()

    return r, p, z
