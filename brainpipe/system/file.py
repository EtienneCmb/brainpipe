#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from scipy.io import loadmat, savemat
import pickle
import os

__all__ = ['savefile', 'loadfile', 'searchfile']


def savefile(name, *arg, check=True, **kwargs):
    """Save variables into *.mat, *.npy, *.npz,

    *.pickle files

    :name: string, file name (must contain an extension)
    :check: bool, verify and increment in case of already defined files
    :*arg: supplementar arguments for *.npy extension
    :**kwargs: supplementar arguments for others extensions

    """
    # Check existing extension :
    ext = os.path.splitext(name)[1]
    if ext not in ['.mat', '.npy', '.npz', '.pickle']:
        raise ValueError(
            'The file must contain an extension as *.mat, *.npy, *.npz, *.pickle')
    # Check name :
    if check:
        name = _safetySave(name)
    # Pickle :
    if ext == '.pickle':
        with open(name, 'wb') as f:
            pickle.dump(kwargs, f)
    # Matlab :
    elif ext == '.mat':
        savemat(name, kwargs)
    # Numpy (.npy) :
    elif ext == '.npy':
        np.save(name, arg)
    # Numpy (.npz) :
    elif ext == '.npz':
        np.savez(name, **kwargs)


def _safetySave(name):
    """Check if a file name exist.

    If it exist, increment it with '(x)'

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


def loadfile(name, var=None):
    """Load a *.mat, *.npy, *.npz and *.pickle files.

    :name: string, file name with one of the extension above
    :var: list, list of variables to keep (not for *.npy files)
    :returns: a dictionnary with selected variables (for *.mat, *.pickle and
    *.npz files) or directly the array (for *.npy files)

    """
    # Check existing extension :
    ext = os.path.splitext(name)[1]
    # Check extension :
    if ext not in ['.mat', '.npy', '.npz', '.pickle']:
        raise ValueError(
            'The file must contain an extension as *.mat, *.npy, *.npz, *.pickle')
    # Pickle :
    if ext == '.pickle':
        with open(name, 'rb') as f:
            data = pickle.load(f)
    # Matlab :
    elif ext == '.mat':
        data = loadmat(name)
    # Numpy (.npy) :
    elif ext == '.npy':
        data = np.load(name)
    # Numpy (.npz) :
    elif ext == '.npz':
        data = dict(np.load(name))
    # Return only var :
    if (var is not None) and (ext != '.npy'):
        if not isinstance(var, list):
            raise ValueError(
                'var must be a list of strings describing variable names.')
        else:
            if not all([isinstance(k, str) for k in var]):
                raise ValueError(
                    'Variables inside var must be a list of strings')
            else:
                keys = list(data.keys())
                for k in keys:
                    if k not in var:
                        del data[k]
    return data


def searchfile(path, *pattern, recursive=False, fullpath=False):
    """Search files in folders / subfolders and use pattern to filter the
    results.

    :path: string, path to search files
    :*pattern: pattern to filter files in folders / subfolders
    :recursive: bool, search in subfolders
    :fullpath: bool, specify if output files have to contain full path
    :returns: a list of files

    """
    # Get list of files :
    if recursive:
        if fullpath:
            listfiles = [os.path.join(dp, f) for dp, dn, fn in os.walk(
                os.path.expanduser(path)) for f in fn]
        else:
            listfiles = [f for dp, dn, fn in os.walk(
                os.path.expanduser(path)) for f in fn]
    else:
        if fullpath:
            listfiles = [os.path.join(path, k) for k in os.listdir(path)]
        else:
            listfiles = os.listdir(path)
    # Filter results :
    if pattern:
        filesout = []
        for num, files in enumerate(listfiles):
            # Search each pattern :
            patbool = []
            for pat in pattern:
                # Normal search :
                if not bool(pat.find('*') + 1):
                    patbool.append(bool(files.find(pat) + 1))
                # Search extension :
                else:
                    patbool.append(os.path.splitext(files)[1] == pat[1:])
            # Keep (or not) files :
            if all(patbool):
                filesout.append(files)
    else:
        filesout = listfiles.copy()
    return filesout
