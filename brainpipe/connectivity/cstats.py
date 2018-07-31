"""Statistics for connectivity."""
import numpy as np
from itertools import permutations
from scipy import spatial, stats


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
    pos, neg = n_pts // 2, n_pts // 2 + 1
    # Demean the time-series :
    ts_m = ts.mean(axis=axis, keepdims=True)
    ts_d = ts - ts_m
    # Compute the DFT :
    ts_dft = np.fft.fft(ts_d, axis=axis)
    # ------------------------------------------------------------
    # RANDOM PHASE
    # ------------------------------------------------------------
    # Prepare phase before broadcasting :
    sz = [1] * ts.ndim
    sz[axis] = ts.shape[axis]
    phi = np.zeros(sz)
    # Define posive phase :
    sz_pos = [1] * ts.ndim
    sz_pos[axis] = pos - 1
    phi_pos = np.random.uniform(0, 2 * np.pi, sz_pos)
    pos_slice = [slice(None)] * ts.ndim
    pos_slice[axis] = slice(1, pos)
    phi[pos_slice] = phi_pos
    pos_slice[axis] = slice(neg, n_pts)
    phi[pos_slice] = -np.flip(phi_pos, axis)
    ts_dft += 1j * phi
    # Perform inverse DFT and keep only real :
    ts_idft = np.fft.ifft(ts_dft, axis=axis).real
    return ts_idft + ts_m


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
