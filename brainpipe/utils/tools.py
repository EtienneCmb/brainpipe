#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

__all__ = ['uorderlst']

def uorderlst(lst):
    """Return a unique set of a list, and preserve order of appearance
    """
    seen = set()
    seen_add = seen.add
    return [x for x in lst if not (x in seen or seen_add(x))]
