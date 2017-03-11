#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base class for all features
"""

class _base(object):

    """Base class for all features
    """

    def __init__(self):
        raise NotImplementedError("Implement this method")

    def _check_params():
        """Check parameters consistency
        """
        raise NotImplementedError("Implement this method")

    def save():
        """Save the feature object
        """
        pass

    def apply():
        """Appply the feature class
        """
        raise NotImplementedError("Implement this method")
