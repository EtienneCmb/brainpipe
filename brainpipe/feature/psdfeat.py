import numpy as np
from scipy.signal import welch
from itertools import product

from brainpipe.tools import binarize
from brainpipe.feature.utils._feat import _manageWindow
from brainpipe.visual.cmon_plt import tilerplot


__all__ = ['PSD', 'powerPSD', 'SpectralEntropy']

commondoc = """

    Args:
        sf: int
            Sampling frequency

        npts: int
            Number of points of the time serie

        width: int, optional [def: None]
            width of a single window.

        step: int, optional [def: None]
            Each window will be spaced by the "step" value.

        time: list/array, optional [def: None]
            Define a specific time vector
"""

supdoc = """
        f: tuple/list
            List containing the couple of frequency bands to extract spectral
            informations. Alternatively, f can be define with the form
            f=(fstart, fend, fwidth, fstep) where fstart and fend are the
            starting and endind frequencies, fwidth and fstep are the width
            and step of each band.

                >>> # Specify each band:
                >>> f = [ [15, 35], [25, 45], [35, 55], [45, 60], [55, 75] ]
                >>> # Second option:
                >>> f = (15, 75, 20, 10)

"""

getdoc = """Get the {feat}

        Args:
            x: array
                Data to get {feat}. x can be a vector of (npts), a matrix
                (npts, ntrials) or 3D array (nelectrodes, npts, ntrials).

        Kargs:
            kwargs: any supplementar argument are directly passed to the welch
            function of scipy.
"""


class PSD(tilerplot):

    """Compute the power spectral density of multiple electrodes.
    """
    __doc__ += commondoc

    def __init__(self, sf, npts, step=None, width=None, time=None):
        self._sf, self._npts = sf, npts
        # Manage time and frequencies:
        self._window, self.xvec = _manageWindow(npts, window=None, width=width,
                                                step=step, time=time)

    def get(self, x, **kwargs):
        """
        """
        # Check size of x:
        if len(x.shape) == 1:
            x = x.reshape(1, len(x), 1)
        elif len(x.shape) == 2:
            x = x[np.newaxis, ...]
        self._nelec, self._npts, self._ntrials = x.shape

        # Split x in window:
        if self._window is not None:
            x = np.transpose(
                np.array([x[:, k[0]:k[1], :] for k in self._window]), (
                                                           1, 2, 0, 3))

        # Compute PSD:
        return welch(x, fs=self._sf, axis=1, **kwargs)

PSD.get.__doc__ += getdoc.format(feat='PSD')+"""
        Returns:
            f: frequency vector of shape (nfce,)

            amp: PSD array of shape (nelectrodes, nfce, nwin, ntrials)
"""


class powerPSD(tilerplot):

    """Compute the power based on psd of multiple electrodes.
    """
    __doc__ += commondoc + supdoc

    def __init__(self, sf, npts, f=[60, 200], step=None, width=None,
                 time=None):
        # Check the type of f:
        if (len(f) == 4) and isinstance(f[0], (int, float)):
            self.yvec = binarize(f[0], f[1], f[2], f[3], kind='list')
        else:
            self.yvec = f
        if not isinstance(f[0], list):
            self.yvec = [f]
        self._psd = PSD(sf, npts, step=step, width=width, time=time)

    def get(self, x, **kwargs):
        """
        """
        fpsd, amp = self._psd.get(x, **kwargs)
        return np.array([amp[:, (fpsd >= k[0])*(fpsd <= k[1]), ...].mean(
                                                     1) for k in self.yvec])

powerPSD.get.__doc__ += getdoc.format(feat='PSD power')+"""
        Return:
            xpow: power array of shape (nfce, nelectrodes, nwin, ntrials)
"""


class SpectralEntropy(tilerplot):

    """Compute the spectral entropy based on psd of multiple electrodes.
    """
    __doc__ += commondoc

    def __init__(self, sf, npts, step=None, width=None, time=None):
        self._psd = PSD(sf, npts, step=step, width=width, time=time)

    def get(self, x, **kwargs):
        """
        """
        fpsd, amp = self._psd.get(x, **kwargs)
        N = len(fpsd)
        ne, namp, nw, nt = amp.shape
        iteract = product(range(ne), range(nw), range(nt))

        xentropy = np.zeros((ne, nw, nt))
        for e, w, t in iteract:
            curamp = amp[e, :, w, t]
            curamp /= np.sum(curamp)
            xentropy[e, w, t] = -(1/np.log(N))*np.sum(curamp*np.log(curamp))

        return xentropy

SpectralEntropy.get.__doc__ += getdoc.format(feat='spectral entropy')+"""
        Return:
            xent: spectral entropy of x of shape (nelectrodes, nwin, ntrials)
"""
