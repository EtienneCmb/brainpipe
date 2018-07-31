import numpy as np


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
    for k in range(len(x.shape) - 1):
        # Find where "where" is equal to "k" :
        idx = np.where(where == k)[0]
        # Roll axis :
        x = np.rollaxis(x, idx, k)
        # Update the where variable :
        where_t.remove(k)
        where = np.array(list(np.arange(k + 1)) + where_t)

    return x
