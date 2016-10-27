import numpy as np

__all__ = ['_listcheck']

def _listcheck(lst):
    """Check if a list is like [[1, 2], [2, 3]...]

    :lst: TODO
    :returns: TODO

    """
    if isinstance(lst, (int, float)):
        lst = [lst]
    if isinstance(lst, list) and isinstance(lst[0], (int, float)):
        lst = [lst]
    lst = [[k] if isinstance(k, (int, float)) else k for k in lst]
    return lst


