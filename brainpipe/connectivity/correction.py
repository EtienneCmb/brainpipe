"""Connectivity correction function."""
import numpy as np


def _axes_correction(axis, ndim, num):
    """Get a slice at a specific axis."""
    axes = [slice(None)] * ndim
    axes[axis] = num
    return tuple(axes)


def get_pairs(n, part='upper', as_array=True):
    """Get connectivity pairs of the upper triangle.

    Parameters
    ----------
    n : int
        Number of electrodes.
    part : {'upper', 'lower', 'both'}
        Part of the connectivity array to get.
    as_array : bool | True
        Specifify if returned pairs should be a (n_pairs, 2) array or a tuple
        of indices.

    Returns
    -------
    pairs : array_like
        A (n_pairs, 2) array of integers.
    """
    assert part in ['upper', 'lower', 'both']
    if part == 'upper':
        idx = np.triu_indices(n, k=1)
    elif part == 'lower':
        idx = np.tril_indices(n, k=-1)
    elif part == 'both':
        high = np.c_[np.triu_indices(n, k=1)]
        low = np.c_[np.tril_indices(n, k=-1)]
        _idx = np.r_[high, low]
        idx = (_idx[:, 0], _idx[:, 1])
    if as_array:
        return np.r_[idx]
    else:
        return idx


def remove_site_contact(mat, channels, mode='soft', remove_lower=False):
    """Remove proximate contacts for SEEG electrodes in a connectivity array.

    Parameters
    ----------
    mat : array_like
        A (n_elec, n_elec) array of connectivity.
    channels : list
        List of channel names of length n_elec.
    mode : {'soft', 'hard'}
        Use 'soft' to only remove successive contacts and 'hard' to remove all
        connectivty that come from the same electrode.
    remove_lower : bool | False
        Remove lower triangle.

    Returns
    -------
    select : array_like
        Array of boolean values with True values in the array that need to be
        removed.
    """
    from re import findall
    n_elec = len(channels)
    assert (mat.shape == (n_elec, n_elec)) and mode in ['soft', 'hard']
    # Define the boolean array to return :
    select = np.zeros_like(mat, dtype=bool)
    # Find site letter and num :
    r = [[findall(r'\D+', k)[0]] + findall(r'\d+', k) for k in channels]
    r = np.asarray(r)
    for i, k in enumerate(r):
        letter, digit_1, digit_2 = [k[0], int(k[1]), int(k[2])]
        if mode is 'soft':
            next_contact = [letter, str(digit_1 + 1), str(digit_2 + 1)]
            to_remove = np.all(r == next_contact, axis=1)
        else:
            to_remove = r[:, 0] == letter
            to_remove[i] = False
        select[i, to_remove] = True
    # Remove lower triangle :
    select[np.tril_indices(n_elec)] = remove_lower
    return select


def anat_based_reorder(c, df, col):
    """Reorder and connectivity array according to anatomy.

    Parameters
    ----------
    c : array_like
        Array of (N, N) connectivity.
    df : pd.DataFrame
        DataFrame containing anamical informations.
    col : str
        Name of the column to use in the DataFrame.

    Returns
    -------
    c_r : array_like
        Anat based reorganized connectivity array.
    labels : array_like
        Array of reorganized labels.
    index : array_like
        Array of indices used for the reorganization.
    """
    assert isinstance(c, np.ndarray) and c.ndim == 2
    assert col in df.keys()
    n_elec = c.shape[0]
    # Group DataFrame column :
    grp = df.groupby(col).groups
    labels = list(df.keys())
    index = np.concatenate([list(k) for k in grp.values()])
    # Get pairs :
    pairs = np.c_[np.triu_indices(n_elec, k=1)]
    # Reconstruct the array :
    c_r = np.zeros_like(c)
    for k, i in pairs:
        row, col = min(index[k], index[i]), max(index[k], index[i])
        c_r[row, col] = c[k, i]
    return c_r, labels, index


def anat_based_mean(x, df, col, xyz=None):
    """Take the mean of a connectivity array according to anatomcal structures.

    Parameters
    ----------
    x : array_like
        Array of (N, N) connectivity.
    df : pd.DataFrame
        DataFrame containing anamical informations.
    col : str
        Name of the column to use in the DataFrame.
    xyz : array_like | None
        Array of coordinate of each electrode.

    Returns
    -------
    x_r : array_like
        Mean array of connectivity inside structures.
    labels : array_like
        Array of labels used to take the mean.
    xyz_r : array_like
        Array of mean coordinates. Return only if `xyz` is not None.
    """
    assert isinstance(x, np.ndarray) and x.ndim == 2
    assert col in df.keys()
    # Force masked array conversion :
    is_masked = np.ma.is_masked(x)
    if not is_masked:
        x = np.ma.masked_array(x, mask=False)
    mask = x.mask
    # Group DataFrame column :
    grp = df.groupby(col).groups
    labels = list(grp.keys())
    index = list(grp.values())
    n_labels = len(labels)
    x_r = np.ma.zeros((n_labels, n_labels), dtype=float)
    mask_r = np.zeros((n_labels, n_labels), dtype=bool)
    for r in range(n_labels):
        for c in range(n_labels):
            r_idx, c_idx = np.array(index[r]), np.array(index[c])
            mask_r[r, c] = (~mask[r_idx, :][:, c_idx]).any()
            x_r[r, c] = x[r_idx, :][:, c_idx].mean()
    x_r = np.ma.masked_array(x_r.data, mask=~mask_r)
    if not is_masked:
        x_r = np.array(x_r)
    if isinstance(xyz, np.ndarray) and (xyz.shape[0] == x.shape[0]):
        xyz_r = np.zeros((n_labels, 3))
        for k, i in enumerate(index):
            xyz_r[k, :] = xyz[i, :].mean(0)
        return x_r, labels, xyz_r
    else:
        return x_r, labels
