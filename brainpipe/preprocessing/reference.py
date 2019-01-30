"""iEEG referencing."""
import logging

import numpy as np
import pandas as pd
from re import findall

from ..system.logging import set_log_level

logger = logging.getLogger('brainpipe')


def ieeg_referencing(data, channels, xyz=None, method='bipolar', sep='.',
                     ignore=None, verbose=None):
    """Rereferencing intracranial data.

    Parameters
    ----------
    data : array_like
        Array of data of shape (n_channels, n_pts, n_trials)
    channels : list
        List of channels with a length of (n_channels).
    xyz : array_like | None
        Array of MNI/Talairach coordinates of shape (n_channels, 3)
    method : {'bipolar', 'laplacian'}
        Referencing method.
    sep : string | '.'
        Channel name separator (e.g "v1.124" will be considered as "v1" with
        `sep='.'`)
    ignore : list | None
        List of channel index to ignore.
    verbose : bool, str, int, or None
        The verbosity of messages to print. If a str, it can be either
        PROFILER, DEBUG, INFO, WARNING, ERROR, or CRITICAL.

    Returns
    -------
    data_b : array_like
        Bipolarized data.
    chan_b : array_like
        Name of the bipolarized channels.
    xyz_b : array_like
        Bipolarized coordinates.
    """
    methods = dict(bipolar=_ref_bipolar, laplacian=_ref_laplacian)
    n_chan_data, n_pts, n_trials = data.shape
    n_chan = len(channels)
    set_log_level(verbose)
    channels = np.asarray(channels)
    # Checking :
    assert isinstance(data, np.ndarray), "data should be an array"
    assert data.ndim == 3, "data should be (n_channels, n_pts, n_trials)"
    assert n_chan_data == n_chan, ("The number of channels along dimension 0 "
                                   "should be %i" % (n_chan))
    if ignore is not None:
        msg = "ignore should either be a list, a tuple or an array of integers"
        assert isinstance(ignore, (list, tuple, np.ndarray)), msg
        assert len(ignore), msg
        assert all([isinstance(k, int) for k in ignore]), msg
        ignore = np.asarray(ignore)
    consider = np.ones((n_chan,), dtype=bool)
    assert method in methods, "method should be %s" % ', '.join(methods)
    logger.info("Referencing %i channels using %s method" % (n_chan, method))
    if not isinstance(xyz, np.ndarray) or (xyz.shape[0] != n_chan):
        xyz = np.zeros((n_chan, 3))
        logger.info("    No coordinates detected")

    # Preprocess channel names by separating channel names / number:
    chnames, chnums = [], []
    for num, k in enumerate(channels):
        # Remove spaces and separation :
        channels[num] = k.strip().replace(' ', '').split(sep)[0]
        # Get only the name / number :
        if findall(r'\d+', k):
            number = findall(r'\d+', k)[0]
            chnums.append(int(number))
            chnames.append(k.split(number)[0])
        else:
            chnums.append(-1)
            chnames.append(k)
    chnums, chnames = np.asarray(chnums), np.asarray(chnames)

    # Find if some channels have to be ignored :
    if isinstance(ignore, (tuple, list, np.ndarray)):
        ignore = np.asarray(ignore)
        consider[ignore] = False
    consider[chnums == -1] = False
    logger.info('    %i channels are going to be ignored (%s)' % (
                (~consider).sum(), ', '.join(channels[~consider].tolist())))

    # Get index to bipolarize :
    _fcn = methods[method]
    return _fcn(data, xyz, channels, chnames, chnums, consider)


def _ref_bipolar(data, xyz, channels, chnames, chnums, consider):
    """Referencing using bipolarization."""
    idx = []
    for num in range(len(channels)):
        if not consider[num]:
            continue
        # Get the name of the current electrode and the needed one :
        need_elec = str(chnames[num]) + str(int(chnums[num]) - 1)
        is_present = channels == need_elec
        if not any(is_present):
            continue
        # Find where is located the electrode :
        idx_need = np.where(is_present)[0]
        assert len(idx_need) == 1, ("Multiple channels have the same name "
                                    "%s" % ', '.join(['%s (%i)' % (channels[k],
                                    k) for k in idx_need]))  # noqa
        idx += [[num, idx_need[0]]]

    logger.info("    Reference iEEG data using bipolarization")
    chan_b = []
    n_pts, n_trials = data.shape[1], data.shape[2]
    data_b = np.zeros((len(idx), n_pts, n_trials), dtype=data.dtype)
    xyz_b = np.zeros((len(idx), 3))
    for k, i in enumerate(idx):
        chan_b += ['%s - %s' % (channels[i[0]], channels[i[1]])]
        data_b[k, ...] = data[i[0], ...] - data[i[1], ...]
        xyz_b[k, ...] = np.c_[xyz[i[0], :], xyz[i[1], :]].mean(1)
    return data_b, chan_b, xyz_b


def _ref_laplacian(data, xyz, channels, chnames, chnums, consider):
    """Referencing using laplacian."""
    idx = []
    for num in range(len(channels)):
        if not consider[num]:
            continue
        # Get the name of the current electrode and the needed one :
        need_elec_left = str(chnames[num]) + str(int(chnums[num]) - 1)
        need_elec_right = str(chnames[num]) + str(int(chnums[num]) + 1)
        is_present_left = channels == need_elec_left
        is_present_right = channels == need_elec_right
        if not any(is_present_left) and not any(is_present_right):
            continue
        # Find where are located left / right electrodes :
        idx_need_left = np.where(is_present_left)[0]
        idx_need_right = np.where(is_present_right)[0]
        assert (len(idx_need_left) <= 1) and (len(idx_need_right) <= 1)
        idx += [[num, np.r_[idx_need_left, idx_need_right].tolist()]]

    logger.info("    Reference iEEG data using laplacian")
    chan_b = []
    n_pts, n_trials = data.shape[1], data.shape[2]
    data_b = np.zeros((len(idx), n_pts, n_trials), dtype=data.dtype)
    xyz_b = np.zeros((len(idx), 3))
    for k, i in enumerate(idx):
        chan_b += ['%s - m(%s)' % (channels[i[0]], ', '.join(channels[i[1]]))]
        data_b[k, ...] = data[i[0], ...] - data[i[1], ...].mean(axis=0)
        xyz_b[k, ...] = np.c_[xyz[i[0], :], xyz[i[1], :].mean(axis=0)].mean(1)
    return data_b, chan_b, xyz_b


def contact_bipo_to_mono(contact):
    """Convert a list of bipolar contacts into unique monopolar sites.

    Parameters
    ----------
    contact : list
        List of bipolar contact.

    Returns
    -------
    contact_r : list
        List of unsorted monopolar contacts.
    """
    from textwrap import wrap
    contact = [k.strip().replace(' ', '').replace('-', '') for k in contact]
    _split = []
    for k in contact:
        _k = wrap(k, int(np.ceil(len(k) / 2)))
        assert len(_k) == 2, "Wrong channel conversion %s" % str(_k)
        _split += list(_k)
    _split = np.ravel(_split)
    c_unique = []
    _ = [c_unique.append(k) for k in _split if k not in c_unique]  # noqa
    return c_unique


def contact_mono_to_bipo(contact, sep='-'):
    """Convert a list of monopolar contacts into bipolar contacts.

    Parameters
    ----------
    contact : list
        List of monopolar contact.
    sep : string | '-'
        String separator between bipolar contact.

    Returns
    -------
    contact_r : list
        List of bipolar contacts.
    """
    bip = []
    for k in contact:
        try:
            letter = ''.join([i for i in k if not i.isdigit()])
            number = int(findall(r'\d+', k)[0])
            previous_contact = '%s%i' % (letter, number - 1)
            if previous_contact in contact:
                bip += ['%s%s%s' % (k, sep, previous_contact)]
        except:
            logger.info('%s is not an SEEG channel' % k)
    return bip


def flat_bipolar_contact(contact):
    """Get a flatten version of bipolar contacts.

    For example, "A'12 - A'11" -> "A'12A'11"

    Parameters
    ----------
    contact : list
        List of contacts.

    Returns
    -------
    contact_c : list
        List of flattened contacts.
    """
    repl = {' ': '', '-': ''}
    for i, k in enumerate(contact):
        for o, n in repl.items():
            contact[i] = k.replace(o, n)
    return contact


def clean_contact(contact):
    """Clean contact's name.

    For example "A02 - A01" -> "A2-A1"

    Parameters
    ----------
    contact : list
        List of contacts.

    Returns
    -------
    contact_c : list
        List of cleaned contacts.
    """
    chan_repl = {'01': '1', '02': '2', '03': '3', '04': '4', '05': '5',
                 '06': '6', '07': '7', '08': '8', '09': '9', ' ': '', 'p': "'"}
    if not isinstance(contact, pd.Series):
        contact = pd.Series(data=contact, name='contact')
    contact.replace(chan_repl, regex=True, inplace=True)
    contact = contact.str.upper()
    contact = contact.str.strip()
    return list(contact)
