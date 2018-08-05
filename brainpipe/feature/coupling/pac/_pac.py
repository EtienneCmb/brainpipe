import numpy as np
from scipy.signal import hilbert
from itertools import product
from joblib import Parallel, delayed

from .pacmeth import *
from brainpipe.tools import groupInList, list2index, adaptsize

__all__ = [
            '_cfcCheck',
            '_cfcFiltSuro'
          ]


def _cfcFiltSuro(xPha, xAmp, surJob, self):
    """SUP: Get the cfc and surrogates

    The function return:
        - The unormalized cfc
        - All the surrogates (for pvalue)
        - The mean of surrogates (for normalization)
        - The deviation of surrogates (for normalization)
    """
    # Check input variables :
    npts, ntrial = xPha.shape
    W = self._window
    nwin = len(W)

    # Get the filter for phase/amplitude properties :
    phaMeth = self._pha.get(self._sf, self._pha.f, self._npts)
    ampMeth = self._amp.get(self._sf, self._amp.f, self._npts)

    # Filt the phase and amplitude :
    xPha = self._pha.apply(xPha, phaMeth)
    xAmp = self._amp.apply(xAmp, ampMeth)

    # Extract phase of amplitude for PLV method:
    if self.Id[0] in ['4']:
        for a in range(xAmp.shape[0]):
            for t in range(xAmp.shape[2]):
                xAmp[a, :, t] = np.angle(hilbert(np.ravel(xAmp[a, :, t])))

    # 2D loop trick :
    claIdx, listWin, listTrial = list2index(nwin, ntrial)

    # Get the unormalized cfc :
    uCfc = [_cfcGet(np.squeeze(xPha[:, W[k[0]][0]:W[k[0]][1], k[1]]),
                    np.squeeze(xAmp[:, W[k[0]][0]:W[k[0]][1], k[1]]),
                    self.Id, self._nbins) for k in claIdx]
    uCfc = np.array(groupInList(uCfc, listWin))

    # Run surogates on each window :
    if (self.n_perm != 0) and (self.Id[0] is not '5') and (self.Id[1] is not '0'):
        Suro = Parallel(n_jobs=surJob)(delayed(_cfcGetSuro)(
            xPha[:, k[0]:k[1], :], xAmp[:, k[0]:k[1], :],
            self.Id, self.n_perm, self._nbins, self._matricial) for k in self._window)
        mSuro = [np.mean(k, 3) for k in Suro]
        stdSuro = [np.std(k, 3) for k in Suro]
    else:
        Suro, mSuro, stdSuro = None, None, None

    return uCfc, Suro, mSuro, stdSuro


def _cfcGet(pha, amp, Id, nbins):
    """Compute the basic cfc model
    """
    # Get the cfc model :
    Model, _, _, _, _, _ = CfcSettings(Id, nbins=nbins)

    return Model(np.matrix(pha), np.matrix(amp), nbins)


def _cfcGetSuro(pha, amp, Id, n_perm, nbins, matricial):
    """Compute the basic cfc model
    """
    # Get the cfc model :
    Model, Sur, _, _, _, _ = CfcSettings(Id, nbins=nbins, matricial=matricial)

    return Sur(pha, amp, Model, n_perm, matricial)


def _cfcCheck(xPha, xAmp, npts):
    """Manage xPha and xAmp size
    """
    if xPha.shape == xAmp.shape:
        if xPha.ndim == 2:
            xPha = xPha[np.newaxis, ...]
            xAmp = xAmp[np.newaxis, ...]
        if xPha.shape[1] != npts:
            raise ValueError('Second dimension must be '+str(npts))
    else:
        raise ValueError('xPha and xAmp must have the same size')

    return xPha, xAmp
