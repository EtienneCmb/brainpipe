import numpy as np


def binarize(start, end, width, step, kind='shift'):
    """Generate a window to binarize a signal.

    start: int
        Define starting point

    endtime : int
        Define ending point

    width : int
        Width of a single window

    step : int
        Each window will be spaced by the "step" value

    kind : string, optional, (def: 'shift')
        Define either a shifted window ('shift') or a square 2D window
        ('square')

    """
    # Inputs checking :
    if any([not isinstance(k, int) for k in (start, end, width, end)]):
        raise ValueError(
            "'start', 'end', 'width' and 'step' parameters must be integers.")
    if kind not in ['shift', 'square']:
        raise ValueError("kind must be either 'shift' or 'square'")
    # Sub-bin function :

    def _binarize(start, end, width, step):
        X = np.vstack((np.arange(start, end - width + step, step),
                       np.arange(start + width, end + step, step))).astype(int)
        if X[1, -1] > end:
            X = X[:, 0:-1]
        return np.ndarray.tolist(X.T)
    # Build either a 'shift' or a 'square' window :
    if kind is 'shift':
        win = _binarize(start, end, width, step)
    elif kind is 'square':
        win = []
        center = np.arange(start, end, width)
        [win.extend(_binarize(k, end, width, step)) for k in center]
    return win


def binarray(x, bins, axis=0):
    """Binarize an array according to defined bins (see binarize function)

    Arg:
        x: np.ndarray
            Array of data

        bins: list
            List of integers in order to binarize x

    Kargs:
        axis: int, optional, (def: 0)
            Specify axis to binarize x

    Return:
        A binarize version of x

    """
    # Sub binarray function :
    def _binarray(x, bins):
        xbin = np.zeros((len(bins),), dtype=float)
        for num, k in enumerate(bins):
            xbin[num] = x[k[0]:k[1]].mean()
        return xbin
    # Ndimentional binarize :
    return np.apply_along_axis(_binarray, axis, x, bins)
