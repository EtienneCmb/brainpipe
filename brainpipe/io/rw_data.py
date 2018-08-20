"""Read and writes files."""
import os

import numpy as np
import pandas as pd
import pickle
from scipy.io import loadmat, savemat

from .read_json import load_json, save_json


def save_file(name, *arg, compress=False, **kwargs):
    """Save a file without carrying of extension.

    Parameters
    ----------
    name : string
        Full path to the file (could be pickle, mat, npy, npz, txt, json,
        xslx, csv).
    """
    name = safety_save(name)
    file_name, file_ext = os.path.splitext(name)
    if file_ext == '.pickle':  # Pickle
        with open(name, 'wb') as f:
            pickle.dump(kwargs, f)
    elif file_ext == '.mat':  # Matlab
        savemat(name, kwargs)
    elif file_ext == '.npy':  # Numpy (single array)
        np.save(name, *arg)
    elif file_ext == '.npz':  # Numpy (multi array)
        if compress:
            np.savez_compressed(name, kwargs)
        else:
            np.savez(name, kwargs)
    elif file_ext == '.json':  # JSON
        save_json(name, kwargs)
    else:
        raise IOError("Extension %s not supported." % file_ext)


def load_file(name):
    """Load a file without carrying of extension.

    Parameters
    ----------
    name : string
        Full path to the file (could be pickle, mat, npy, npz, txt, json, xlsx,
        xls, csv).
    """
    assert os.path.isfile(name)
    file_name, file_ext = os.path.splitext(name)
    if file_ext == '.pickle':  # Pickle :
        with open(name, "rb") as f:
            arch = pickle.load(f)
        return arch
    elif file_ext == '.mat':  # Matlab :
        return loadmat(name)
    elif file_ext in ['.npy', '.npz']:  # Numpy (single / multi array)
        return np.load(name)
    elif file_ext == '.txt':  # text file
        return np.genfromtxt(name)
    elif file_ext == '.json':  # JSON
        return load_json(name)
    elif file_ext in ['.xls', '.xlsx']:  # Excel
        return pd.read_excel(name)
    elif file_ext in ['.xls', '.xlsx']:  # CSV
        return pd.read_csv(name)
    else:
        raise IOError("Extension %s not supported." % file_ext)


def safety_save(name):
    """Check if a file name exist.

    If it exist, increment it with '(x)'.
    """
    k = 1
    while os.path.isfile(name):
        fname, fext = os.path.splitext(name)
        if fname.find('(') + 1:
            name = fname[0:fname.find('(') + 1] + str(k) + ')' + fext
        else:
            name = fname + '(' + str(k) + ')' + fext
        k += 1
    return name
