"""
Design a filter, filt a signal, extract the phase, amplitude or power
"""

import numpy as np

from .utils._filtering import _get_method, _apply_method
from .utils._feat import _checkref

__all__ = [
    'fdesign',
    'fextract'
]

docfilter = """
        filtname: string, optional [def: 'fir1']
            Name of the filter. Possible values are:
                - 'fir1': Window-based FIR filter design
                - 'butter': butterworth filter
                - 'bessel': bessel filter

        cycle: int, optional [def: 3]
            Number of cycle to use for the filter. This parameter
            is only avaible for the 'fir1' method

        order: int, optional [def: 3]
            Order of the 'butter' or 'bessel' filter

        axis: int, optional [def: 0]
            Filter accross the dimension 'axis'
"""

docType = """"""


class fdesign(object):

    """Design a filter

    Args:"""
    __doc__ += docfilter

    def __init__(self, filtname='fir1', cycle=3, order=3, axis=0):
        _checkref('filtname', filtname, ['fir1', 'butter', 'bessel',
                  'wavelet'])
        self._filtname = filtname
        self._cycle = cycle
        self._order = order
        self._axis = axis

    def __str__(self):
        if self._filtname == 'fir1':
            filtStr = 'Filter(name='+self._filtname+', cycle='+str(
                self._cycle)+', axis='+str(self._axis)+')'
        else:
            filtStr = 'Filter(name='+self._filtname+', order='+str(
                self._order)+', axis='+str(self._axis)+')'
        return filtStr


class fextract(fdesign):

    """Extract informations from a signal

    Args:
        method: string
            Method to transform the signal. Possible values are:
                - 'hilbert': apply a hilbert transform to each column
                - 'hilbert1': hilbert transform to a whole matrix
                - 'hilbert2': 2D hilbert transform
                - 'wavelet': wavelet transform
                - 'filter': filtered signal

        kind: string
            Type of information to extract to the transformed signal.
            Possible values are:
                - 'signal': return the transform signal
                - 'phase': phase of the the transform signal
                - 'amplitude': amplitude of the transform signal
                - 'power': power of the transform signal

    Kargs:
        dtrd: bool, optional [def: False]
            Detrend the signal

        wltWidth: int, optional [def: 7]
            Width of the wavelet

        wltCorr: int, optional [def: 3]
            Correction of the edgde effect for the wavelet
    """
    __doc__ += docfilter

    def __init__(self, method, kind, filtname='fir1', cycle=3, order=3,
                 axis=0, dtrd=False, wltWidth=7, wltCorr=3):
        # Check the defined method :
        _checkref('method', method, ['hilbert', 'hilbert1', 'hilbert2',
                  'wavelet', 'filter'])
        _checkref('kind', kind, ['signal', 'phase', 'amplitude', 'power'])
        self._method = method
        self._kind = kind
        self._wltWidth = wltWidth
        self._wltCorr = wltCorr
        self._dtrd = dtrd
        super().__init__(filtname=filtname, cycle=cycle, order=order,
                         axis=axis)

    def __str__(self):
        filtStr = super().__str__()
        if self._method == 'wavelet':
            supStr = ', wavelet(width='+str(
                self._wltWidth)+', correction='+str(self._wltCorr)+')'
        else:
            supStr = ''

        return 'Extract(kind='+self._kind+', method='+self._method + \
            ', detrend='+str(self._dtrd)+supStr+',\n'+filtStr+')'

    def get(self, sf, f, npts):
        """Get the methods

        Args:
            sf: integer
                Sampling frequency

            f: tuple/list
                List containing the couple of frequency bands. Each couple can be
                either a list or a tuple. Example: f=[ [2,4], [5,7], [60,250] ]

        Return:
            fMeth: list
                List of methods for filtering
        """
        if type(f[0]) == int:
            f = [f]
        fMeth = _get_method(sf, f, npts, self._filtname, self._cycle,
                            self._order, self._axis, self._method,
                            self._wltWidth, self._kind)
        return fMeth

    def apply(self, x, fMeth):
        """Apply the defined methods

        Args:
            x: array
                Array to filt. Shape of x must be (npts x ntrials)

            fMeth: list
                List of methods for filtering
        Return:
            xf: array
                The filtered signal of shape (n frequency x npts x ntrials)
        """
        return _apply_method(x, fMeth, self._dtrd, self._method,
                             self._wltCorr, self._wltWidth)
