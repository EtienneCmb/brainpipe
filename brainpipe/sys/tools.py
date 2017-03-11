import pickle
from scipy.io import loadmat, savemat
from os.path import splitext, isfile
import numpy as np

__all__ = ['savefile',
           'loadfile',
           'adaptsize'
           ]


def savefile(name, *arg, **kwargs):
    """Save a file without carrying of extension.

    arg: for .npy extension
    kwargs: for .pickle or .mat extensions
    """
    name = _safetySave(name)
    fileName, fileExt = splitext(name)
    # Pickle :
    if fileExt == '.pickle':
        with open(name, 'wb') as f:
            pickle.dump(kwargs, f)
    # Matlab :
    elif fileExt == '.mat':
        data = savemat(name, kwargs)
    # Numpy (single array) :
    elif fileExt == '.npy':
        data = np.save(name, arg)


def loadfile(name):
    """Load a file without carrying of extension. The function return
    a dictionnary data.
    """
    fileName, fileExt = splitext(name)
    # Pickle :
    if fileExt == '.pickle':
        with open(name, "rb") as f:
            data = pickle.load(f)
    # Matlab :
    elif fileExt == '.mat':
        data = loadmat(name)
    # Numpy (single array)
    elif fileExt == '.npy':
        data = np.load(name)
    return data


def _safetySave(name):
    """Check if a file name exist. If it exist, increment it with '(x)'
    """
    k = 1
    while isfile(name):
        fname, fext = splitext(name)
        if fname.find('(')+1:
            name = fname[0:fname.find('(')+1]+str(k)+')'+fext
        else:
            name = fname+'('+str(k)+')'+fext
        k += 1
    return name


def adaptsize(x, where):
    """Adapt the dimension of an array depending of the tuple dim

    Args:
        x : the signal for swaping axis
        where : where each dimension should be put

    Example:
        >>> x = np.random.rand(2,4001,160)
        >>> adaptsize(x, (1,2,0)).shape -> (160, 2, 4001)
    """
    if not isinstance(where, np.ndarray):
        where = np.array(where)

    where_t = list(where)
    for k in range(len(x.shape)-1):
        # Find where "where" is equal to "k" :
        idx = np.where(where == k)[0]
        # Roll axis :
        x = np.rollaxis(x, idx, k)
        # Update the where variable :
        where_t.remove(k)
        where = np.array(list(np.arange(k+1)) + where_t)

    return x