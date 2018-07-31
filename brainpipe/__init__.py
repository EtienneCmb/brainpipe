import logging

from .preprocessing import *
from .classification import *
from .feature import *
from .sys import set_log_level

"""Set 'info' as the default logging level
"""
logger = logging.getLogger('brainpipe')
set_log_level('info')

__version__ = '0.1.8'
