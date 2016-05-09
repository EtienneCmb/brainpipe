from joblib import Parallel, delayed
from psutil import cpu_count
import numpy as np
from itertools import product

from brainpipe.feat.utils._feat import (_manageWindow, _manageFrequencies,
                                        _checkref)
from brainpipe.feat.filtering import fextract, docfilter
from brainpipe.feat.coupling.pac._pac import *
from brainpipe.feat.coupling.pac.pacmeth import *

__all__ = ['pac']


windoc = """
        window: tuple/list/None, optional [def: None]
            List/tuple: [100,1500]
            List of list/tuple: [(100,500),(200,4000)]
            None and the width and step parameters will be considered

        width: int, optional [def: None]
            width of a single window.

        step: int, optional [def: None]
            Each window will be spaced by the "step" value.

        time: list/array, optional [def: None]
            Define a specific time vector

    """

Footnotes = """

    .. rubric:: Footnotes
    .. [#f1] Canolty et al, 2006 `see <http://www.ncbi.nlm.nih.gov/pmc/articles/PMC2628289/>`_
    .. [#f2] Tort et al, 2010 `see <http://www.ncbi.nlm.nih.gov/pmc/articles/PMC2941206/>`_
    .. [#f3] See Ozkurt et al, 2012 `see <http://www.ncbi.nlm.nih.gov/pubmed/22531738/>`_

"""


class _coupling(object):

    """
    """

    def __init__(self, pha_f, pha_kind, pha_meth, pha_cycle,
                 amp_f, amp_kind, amp_meth, amp_cycle,
                 sf, npts, window, width, step, time, **kwargs):
        # Define windows and frequency :
        self._pha = fextract(kind=pha_kind, method=pha_meth,
                             cycle=pha_cycle, **kwargs)
        self._amp = fextract(kind=amp_kind, method=amp_meth,
                             cycle=amp_cycle, **kwargs)
        self._window, self._xvec = _manageWindow(npts, window=window,
                                                 width=width, step=step,
                                                 time=time)
        self._pha.f, _, _ = _manageFrequencies(pha_f, split=None)
        self._amp.f, _, _ = _manageFrequencies(amp_f, split=None)
        if self._window is None:
            self._window = [(0, npts)]
            self.xvec = [0, npts]

        # Get variables :
        self._width = width
        self._step = step
        self._nPha = len(self._pha.f)
        self._nAmp = len(self._amp.f)
        self._sf = sf
        self._npts = npts
        self._nwin = len(self._window)


class pac(_coupling):

    """Compute the phase-amplitude coupling (pac) either in local or
    distant coupling. PAC require three things:

        - Main method for compute it
        - Surrogates to correct the true pac value
        - A normalization method to correct true pas value bu surrogates

    Contributor: Juan LP Soto  `Python <http://www.python.org/>`_.

    Args:
        sf: int
            Sampling frequency

        npts: int
            Number of points of the time serie

    Kargs:
        Id: string, optional, [def: '114']
            The Id correspond to the way of computing pac. Id is composed of
            three digits [ex : Id='210']

                * First digit: refer to the pac method:

                    - '1': Modulation Index [#f1]_
                    - '2': Kullback-Leibler Distance [#f2]_
                    - '3': Phase synchrony
                    - '4': Amplitude PSD
                    - '5': Heights Ratio
                    - '6': ndPAC [#f3]_

                * Second digit: refer to the method for computing surrogates:

                    - '0': No surrogates
                    - '1': Shuffle phase values
                    - '2': Time lag
                    - '3': Swap phase/amplitude through trials
                    - '4': Swap amplitude
                    - '5': circular shifting

                * Third digit: refer to the normalization method for correction:

                    - '0': No normalization
                    - '1': Substract the mean of surrogates
                    - '2': Divide by the mean of surrogates
                    - '3': Substract then divide by the mean of surrogates
                    - '4': Z-score

            So, if Id='123', this mean that pac will be evaluate using the
            Modulation Index ('1'), then surrogates will be find by introducing a
            time lag ('2') and finally, the true pac value will be normalized by
            substracting then dividing by the mean of surrogates.

        pha_f: tuple/list, optional, [def: [2,4]]
                List containing the couple of frequency bands for the phase.
                Example: f=[ [2,4], [5,7], [60,250] ]

        pha_meth: string, optional, [def : 'hilbert']
            Method for the phase extraction.

        pha_cycle: integer, optional, [def : 3]
            Number of cycles for filtering the phase.

        amp_f: tuple/list, optional, [def : [60,200]]
                List containing the couple of frequency bands for the amplitude.
                Each couple can be either a list or a tuple.

        amp_meth: string, optional, [def : 'hilbert']
            Method for the amplitude extraction.

        amp_cycle: integer, optional, [def : 6]
            Number of cycles for filtering the amplitude.

        nbins: integer, optional, [def: 18]
            Some pac method (like Kullback-Leibler Distance or Heights Ratio) need
            a binarization of the phase. nbins control the number of bins.

    """
    __doc__ += windoc + docfilter + Footnotes

    def __init__(self, sf, npts, Id='114', pha_f=[2, 4], pha_meth='hilbert',
                 pha_cycle=3, amp_f=[60, 200], amp_meth='hilbert', amp_cycle=6,
                 nbins=18, window=None, width=None, step=None, time=None, **kwargs):
        # Check pha and amp methods:
        _checkref('pha_meth', pha_meth, ['hilbert', 'hilbert1', 'hilbert2'])
        _checkref('amp_meth', amp_meth, ['hilbert', 'hilbert1', 'hilbert2'])

        # Initalize pac object :
        self.Id = Id
        pha_kind = 'phase'
        amp_kind = 'amplitude'
        _coupling.__init__(self, pha_f, pha_kind, pha_meth, pha_cycle,
                           amp_f, amp_kind, amp_meth, amp_cycle,
                           sf, npts, window, width, step, time, **kwargs)
        # Get pac model :
        _, _, _, ModelStr, SurStr, NormStr = CfcSettings(Id, nbins)
        self._model = ['Method : '+ModelStr, 'Surrogates : '+SurStr,
                       'Normalization : '+NormStr]
        self._nbins = nbins

    def __str__(self):
        phafilt = 'Phase : '+str(self._pha)
        ampfilt = 'Amplitude : '+str(self._amp)
        met = self._model[0]+',\n'+self._model[1]+',\n'+self._model[2]+',\n'
        cfcStr = 'Crossfrequency Coupling(step='+str(self._step)+', width='+str(
            self._width)+', Id='+self.Id+', nbins='+str(self._nbins)+',\n'+met

        return cfcStr+phafilt+',\n'+ampfilt+')'

    def get(self, xpha, xamp, n_perm=200, n_jobs=-1):
        """Get the normalized cfc mesure between an xpha and xamp signals.

        Args:
            xpha: array
                Signal for phase. The shape of xpha should be :
                (n_electrodes x n_pts x n_trials)

            xamp: array
                Signal for amplitude. The shape of xamp should be :
                (n_electrodes x n_pts x n_trials)

        Kargs:
            n_perm: integer, optional, [def: 200]
                Number of permutations for normalizing the cfc mesure.

            n_jobs: integer, optional, [def: -1]
                Control the number of jobs for parallel computing. Use 1, 2, ..
                depending of your number or cores. -1 for all the cores.

            If the same signal is used (example : xpha=x and xamp=x), this mean
            the program compute a local cfc.

        Returns:
            ncfc: array
                The unormalized cfc mesure of size :
                (n_phase x n_amplitude x n_electrodes x n_windows x n_trials)

            pvalue: array
                The associated p-values of size :
                (n_phase x n_amplitude x n_electrodes x n_windows)
        """
        # Check the inputs variables :
        xpha, xamp = _cfcCheck(xpha, xamp, self._npts)
        self.n_perm = n_perm
        if n_perm != 0:
            self.p = 1/n_perm
        else:
            self.p = None
        N = xpha.shape[0]

        # Manage jobs repartition :
        if (N < cpu_count()) and (n_jobs != 1):
            surJob = n_jobs
            elecJob = 1
        elif (N >= cpu_count()) and (n_jobs != 1):
            surJob = 1
            elecJob = n_jobs
        else:
            surJob, elecJob = 1, 1

        # Get the unormalized cfc and surogates:
        cfcsu = Parallel(n_jobs=elecJob)(delayed(_cfcFiltSuro)(
            xpha[k, ...], xamp[k, ...], surJob, self) for k in range(N))
        uCfc, Suro, mSuro, stdSuro = zip(*cfcsu)

        # Compute permutations :
        if self.n_perm != 0:
            uCfc, Suro, mSuro = np.array(uCfc), np.array(Suro), np.array(mSuro)
            stdSuro = np.array(stdSuro)

            # Normalize each cfc:
            _, _, Norm, _, _, _ = CfcSettings(self.Id)
            nCfc = Norm(uCfc, mSuro, stdSuro)

            # Confidence interval :
            pvalue = np.array([_cfcPvalue(nCfc[k, ...], Suro[
                k, ...]) for k in range(N)])

            return nCfc.transpose(4, 3, 0, 1, 2), pvalue.transpose(3, 2, 0, 1)
        else:
            return uCfc.transpose(4, 3, 0, 1, 2), None