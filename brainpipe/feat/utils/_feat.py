from brainpipe.tools import binarize
# from brainpipe._utils._plot import _plot, _2Dplot

import numpy as n
import warnings

__all__ = [
            '_manageWindow',
            '_manageFrequencies',
            'normalize',
            '_checkref'
          ]


def _manageFrequencies(f, split=None):
    """Manage frequency bands definition

    Parameters
    ----------
    f : list
        List containing the couple of frequency bands.
        Each couple can be either a list or a tuple.
        Ex: f = [(2,4),(5,7)]

    split : int or list of int, optional [def: None]
        Split the frequency band f in "split" band width.
        If f is a list which contain couple of frequencies,
        split is a list too.

    Returns
    ----------
    f : the modified list
    fSplit : the splitted f
    fSplitIndex : index of the splitted bands
    """
    if (len(f) == 2) and (type(f[0]) in [int, float]):
        f = [f]
    nf = len(f)
    if (split is None) or (type(split) == int):
        split = [split]
    if len(split) != nf:
        split = split*nf
    fSplit, fSplitIndex, lenOld = [], [], 0
    for k, i in enumerate(f):
        if split[k] is None:
            fSplit.append(f[k])
        elif type(split[k]) == int:
            fSplit.extend(binarize(i[0], i[1], split[k], split[k],
                          kind='tuple'))
        fSplitIndex.append([lenOld, len(fSplit)])
        lenOld = len(fSplit)
    return f, fSplit, fSplitIndex


def _manageWindow(npts, window=None, width=None, step=None, kind='tuple',
                  time=None):
    """Manage window definition

    Parameters
    ----------
    npts : int
        Number of points in the time signal

    window : tuple, list, None, optional [def: None]
        List/tuple: [100,1500]
        List of list/tuple: [(100,500),(200,4000)]
        None and the width and step paameters will be considered

    width : int, optional [def : None]
        width of a single window

    step : int, optional [def : None]
        Each window will be spaced by the "step" value

    kind : string, optional, [def: 'list']
        Return either a list or a tuple
    """
    if window and (len(window) == 2) and (type(window[0]) == int):
        window = [window]
    else:
        if width is not None:
            if step is None:
                step = round(width/2)
            window = binarize(0, npts, width, step, kind=kind)
    if window:
        xvec = [round((k[0]+k[1])/2) for k in window]
    else:
        xvec = list(n.arange(0, npts))
    if (time is not None) and not isinstance(time, list):
        tme = list(time)
    if (time is not None) and (len(time) == npts):
        xvec = time
    elif time and (len(time) != npts):
        warnings.warn("The length of 'time' ["+str(len(time))+"] must be equal"
                      "to the length of the defined window ["+str(len(xvec)) +
                      ". A default vector is going to used.")

    return window, xvec


def normalize(A, B, norm=0):
    """normalize A by B using the 'norm' parameter

    Parameters
    ----------
    A : array
        array to normalize

    B : array
        List/tuple: [100,1500]
        List of list/tuple: [(100,500),(200,4000)]
        None and the width and step paameters will be considered

    norm : int, optional [def : 0]
        0 : No normalisation
        1 : Substraction
        2 : Division
        3 : Substract then divide
        4 : Z-score
    """
    # No normalisation
    if norm == 0:
        return A
    # Substraction
    elif norm == 1:
        return A - B
    # Division
    elif norm == 2:
        return A / B
    # Substract then divide
    elif norm == 3:
        return (A - B) / B
    # Z-score
    elif norm == 4:
        return (A - B) / n.std(B, axis=0)


def _checkref(name, tofind, ref):
    """Check if the method 'tofind' of type 'name' exist in ref"""
    if tofind not in ref:
        raise ValueError('No "'+name+'" called "'+str(tofind)+'" is defined.'
                         ' Choose between "'+'", "'.join(ref)+'"')
