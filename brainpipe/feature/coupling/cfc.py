from joblib import Parallel, delayed
from psutil import cpu_count

import numpy as np
from itertools import product
from scipy.special import erfinv
from scipy.stats import norm 

from brainpipe.feature.utils._feat import (_manageWindow, _manageFrequencies,
                                           _checkref)
from brainpipe.feature.filtering import fextract, docfilter
from brainpipe.feature.coupling.pac._pac import *
from brainpipe.feature.coupling.pac.pacmeth import *
from brainpipe.feature.utils._feat import normalize
from brainpipe.feature import power, phase, sigfilt
from brainpipe.tools import binarize, binArray
from brainpipe.statistics import perm_2pvalue, circ_corrcc, circ_rtest
from brainpipe.visual.cmon_plt import tilerplot
from brainpipe.visual import addLines

__all__ = ['pac',
           'PhaseLockedPower',
           'erpac', 
           'pfdphase',
           'PLV'
           ]


windoc = """
        window: tuple/list/None, optional [def: None]
            List/tuple: [100,1500]
            List of list/tuple: [(100,500),(200,4000)]
            Width and step parameters will be ignored.

        width: int, optional [def: None]
            width of a single window.

        step: int, optional [def: None]
            Each window will be spaced by the "step" value.

        time: list/array, optional [def: None]
            Define a specific time vector

    """

Footnotes = """

    .. rubric:: Footnotes
    .. [#f1] `Canolty et al, 2006 <http://www.ncbi.nlm.nih.gov/pmc/articles/PMC2628289/>`_
    .. [#f2] `Tort et al, 2010 <http://www.ncbi.nlm.nih.gov/pmc/articles/PMC2941206/>`_
    .. [#f3] `Ozkurt et al, 2012 <http://www.ncbi.nlm.nih.gov/pubmed/22531738/>`_
    .. [#f4] `Bahramisharif et al, 2013 <http://www.jneurosci.org/content/33/48/18849.short/>`_
    .. [#f5] `Penny et al, 2008 <http://www.sciencedirect.com/science/article/pii/S0165027008003816>`_

"""

def cfcparafilt(xpha, xamp, n_jobs, self):
    """Parallel filtering through electrode dimension
    """
    nelec = xpha.shape[0]
    # Run para filtering :
    data = Parallel(n_jobs=n_jobs)(delayed(_cfcparafilt)(
            xpha[e, ...], xamp[e, ...], self,
            ) for e in range(nelec))
    pha, amp = zip(*data)
    return np.array(pha), np.array(amp)


def _cfcparafilt(xpha, xamp, self):
    """Sub parallel filtering function
    """
    # Get the filter for phase/amplitude properties :
    phaMeth = self._pha.get(self._sf, self._pha.f, self._npts)
    ampMeth = self._amp.get(self._sf, self._amp.f, self._npts)
    # Filt phase and amplitude :
    pha = self._pha.apply(xpha, phaMeth)
    amp = self._amp.apply(xamp, ampMeth)
    return pha, amp


class _coupling(tilerplot):

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
        self._window, xvec = _manageWindow(npts, window=window,
                                           width=width, step=step,
                                           time=time)
        self._pha.f, _, _ = _manageFrequencies(pha_f, split=None)
        self._amp.f, _, _ = _manageFrequencies(amp_f, split=None)
        if time is None:
            time = np.arange(npts)
        if self._window is None:
            self._window = [(0, npts)]
            self.time = np.array(self._window).mean()
            # self.xvec = [0, npts]
        else:
            self.time = binArray(time, self._window)[0]

        # Get variables :
        self._width = width
        self._step = step
        self._nPha = len(self._pha.f)
        self._nAmp = len(self._amp.f)
        self._sf = sf
        self._npts = npts
        self._nwin = len(self._window)
        self.pha = [np.mean(k) for k in self._pha.f]
        self.amp = [np.mean(k) for k in self._amp.f]


class pac(_coupling):

    """Compute the phase-amplitude coupling (pac) either in local or
    distant coupling. PAC require three things:

        - Main method to compute it
        - Surrogates to correct the true pac estimation
        - A normalization method to correct pas by surrogates

    Contributor: Juan LP Soto.

    Args:
        sf: int
            Sampling frequency

        npts: int
            Number of points of the time serie

    Kargs:
        Id: string, optional, [def: '113']
            The Id correspond to the way of computing pac. Id is composed of
            three digits [ex : Id='210']

                * First digit: refer to the pac method:

                    - '1': Mean Vector Length [#f1]_
                    - '2': Kullback-Leibler Divergence [#f2]_
                    - '3': Heights Ratio
                    - '4': Phase synchrony (or adapted PLV) [#f5]_
                    - '5': ndPAC [#f3]_

                * Second digit: refer to the method for computing surrogates:

                    - '0': No surrogates
                    - '1': Swap trials phase/amplitude [#f2]_
                    - '2': Swap trials amplitude [#f4]_
                    - '3': Shuffle phase time-series
                    - '4': Shuffle amplitude time-series
                    - '5': Time lag [#f1]_ [NOT IMPLEMENTED YET]
                    - '6': Circular shifting [NOT IMPLEMENTED YET]

                * Third digit: refer to the normalization method for correction:

                    - '0': No normalization
                    - '1': Substract the mean of surrogates
                    - '2': Divide by the mean of surrogates
                    - '3': Substract then divide by the mean of surrogates
                    - '4': Z-score

            So, if Id='143', this mean that pac will be evaluate using the
            Modulation Index ('1'), then surrogates are computing by randomly
            shuffle amplitude values ('4') and finally, the true pac value
            will be normalized by substracting then dividing by the mean of surrogates.

        pha_f: tuple/list, optional, [def: [2,4]]
            List containing the couple of frequency bands for the phase.
            Example: f=[ [2,4], [5,7], [60,250] ]

        pha_meth: string, optional, [def: 'hilbert']
            Method for the phase extraction.

        pha_cycle: integer, optional, [def: 3]
            Number of cycles for filtering the phase.

        amp_f: tuple/list, optional, [def: [60,200]]
            List containing the couple of frequency bands for the amplitude.
            Each couple can be either a list or a tuple.

        amp_meth: string, optional, [def: 'hilbert']
            Method for the amplitude extraction.

        amp_cycle: integer, optional, [def: 6]
            Number of cycles for filtering the amplitude.

        nbins: integer, optional, [def: 18]
            Some pac method (like Kullback-Leibler Distance or Heights Ratio) need
            a binarization of the phase. nbins control the number of bins.

    """
    __doc__ += windoc + docfilter + Footnotes

    def __init__(self, sf, npts, Id='113', pha_f=[2, 4], pha_meth='hilbert',
                 pha_cycle=3, amp_f=[60, 200], amp_meth='hilbert', amp_cycle=6,
                 nbins=18, window=None, width=None, step=None, time=None,
                 **kwargs):
        # Check pha and amp methods:
        _checkref('pha_meth', pha_meth, ['hilbert', 'hilbert1', 'hilbert2'])
        _checkref('amp_meth', amp_meth, ['hilbert', 'hilbert1', 'hilbert2'])

        # Check the type of f:
        if (len(pha_f) == 4) and isinstance(pha_f[0], (int, float)):
            pha_f = binarize(
                pha_f[0], pha_f[1], pha_f[2], pha_f[3], kind='list')
        if (len(amp_f) == 4) and isinstance(amp_f[0], (int, float)):
            amp_f = binarize(
                amp_f[0], amp_f[1], amp_f[2], amp_f[3], kind='list')
        self.xvec = []

        # Initalize pac object :
        self.Id = Id
        me = Id[0]
        # Manage settings :
        #   1 - Choose if we extract phase or amplitude :
        #       - Methods using phase // amplitude :
        if me in ['1', '2', '3', '5', '6']:
            pha_kind, amp_kind = 'phase', 'amplitude'
        #       - Methods using phase // phase :
        elif me in ['4']:
            pha_kind, amp_kind = 'phase', 'amplitude'
        #   2 - Specific case of Ozkurt :
        if me == '5':
            Id = '500'
        # Initialize cfc :
        _coupling.__init__(self, pha_f, pha_kind, pha_meth, pha_cycle,
                           amp_f, amp_kind, amp_meth, amp_cycle,
                           sf, npts, window, width, step, time, **kwargs)
        # Get pac model :
        _, _, _, ModelStr, SurStr, NormStr = CfcSettings(Id, nbins)
        self.model = ['Method : '+ModelStr, 'Surrogates : '+SurStr,
                      'Normalization : '+NormStr]
        self._nbins = nbins

    def __str__(self):
        phafilt = 'Phase : '+str(self._pha)
        ampfilt = 'Amplitude : '+str(self._amp)
        met = self.model[0]+',\n'+self.model[1]+',\n'+self.model[2]+',\n'
        cfcStr = 'Crossfrequency Coupling(step='+str(self._step)+', width='+str(
            self._width)+', Id='+self.Id+', nbins='+str(self._nbins)+',\n'+met

        return cfcStr+phafilt+',\n'+ampfilt+')'

    def get(self, xpha, xamp, n_perm=200, p=0.05, matricial=False, n_jobs=-1):
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

            p: float, optional, [def: 0.05]
                p-value for the statistical method of Ozkurt 2012.

            matricial: bool, optional, [def: False]
                Some methods can work in matricial computation. This can lead
                to a 10x or 30x time faster. But, please, monitor your RAM usage
                beacause this parameter can use a lot of RAM. So, turn this parameter
                in case of small computation.

            n_jobs: integer, optional, [def: -1]
                Control the number of jobs for parallel computing. Use 1, 2, ..
                depending of your number or cores. -1 for all the cores.

            If the same signal is used (example : xpha=x and xamp=x), this mean
            the program compute a local cfc.

        Returns:
            ncfc: array
                The cfc mesure of size :
                (n_amplitude x n_phase x n_electrodes x n_windows x n_trials)

            pvalue: array
                The associated p-values of size :
                (n_amplitude x n_phase x n_electrodes x n_windows)
        """
        # Check the inputs variables :
        xpha, xamp = _cfcCheck(xpha, xamp, self._npts)
        self.n_perm = n_perm
        self._matricial = matricial
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
        uCfc = np.array(uCfc)

        # Permutations ans stat:
        if (self.Id[0] is not '5'):
            # Compute permutations :
            if (self.n_perm is not 0) and (self.Id[1] is not '0'):
                Suro, mSuro, stdSuro = np.array(
                    Suro), np.array(mSuro), np.array(stdSuro)

                # Normalize each cfc:
                _, _, Norm, _, _, _ = CfcSettings(self.Id)
                nCfc = Norm(uCfc, mSuro, stdSuro)

                # Confidence interval :
                pvalue = perm_2pvalue(uCfc.mean(2), np.rollaxis(Suro.mean(2), 4),
                                      self.n_perm, tail=1)

                return nCfc.transpose(3, 4, 0, 1, 2), pvalue.transpose(2, 3, 0, 1)
            else:
                return uCfc.transpose(3, 4, 0, 1, 2), None
        elif self.Id[0] is '5':
            # Ozkurt threshold :
            xlim = (erfinv(1-p)**2)
            # Set to zero non-significant values:
            idxUn = np.where(uCfc <= 2*xlim)
            uCfc[idxUn] = 0
            return uCfc.transpose(3, 4, 0, 1, 2), None


class PhaseLockedPower(object):

    """Extract phase-locked power and visualize shifted time-frequency map
    according to phase peak.

    Args:
        sf: int
            Sampling frequency

        npts: int
            Number of points of the time serie

    Kargs:
        f: tuple/list, optional, [def: (2, 200, 10, 5)]
            The frequency vector (fstart, fend, fwidth, fstep)

        pha: tuple/list, optional, [def: [8, 13]]
            Frequency for phase.

        time: array/list, optional, [def: None]
            The time vector to use

        baseline: tuple/list, optional, [def: None]
            Location of baseline (in sample)

        norm: integer, optional, [def: None]
            Normalize method
                - 0: No normalisation
                - 1: Substraction
                - 2: Division
                - 3: Substract then divide
                - 4: Z-score

        powArgs: any supplementar arguments are directly passed to the power
        function.
    """

    def __init__(self, sf, npts, f=(2, 200, 10, 5), pha=[8, 13], time=None,
                 baseline=None, norm=None, **powArgs):
        # Define objects:
        self._normBck = norm
        self._baseline = baseline
        self._powObj = power(
            sf, npts, f=f, baseline=baseline, norm=0, time=time, **powArgs)
        self._phaObj = phase(sf, npts, f=pha)
        self._sigObj = sigfilt(sf, npts, f=pha)

    def get(self, x, cue):
        """Get power phase locked

        Args:
            x: array
                Data of shape (npt, ntrials)

            cue: integer
                Cue to align time-frequency maps.

        Returns:
            xpow, xpha, xsig: repectively realigned power, phase and filtered
            signal
        """
        # Find cue according to define time vector
        self._cue = cue
        xvec = self._powObj.xvec
        cue = np.abs(np.array(xvec)-cue).argmin()
        self._cueIdx = cue
        # Extact power, phase and filtered signal:
        xpow = np.squeeze(self._powObj.get(x)[0])
        xpha = np.squeeze(self._phaObj.get(x)[0])
        xsig = np.squeeze(self._sigObj.get(x)[0])
        # Re-align:
        xpha_s, xpow_s, xsig_s = np.empty_like(
            xpha), np.empty_like(xpow), np.empty_like(xsig)
        nTrials = xsig.shape[1]
        for k in range(nTrials):
            # Get shifting:
            move = self._PeakDetection(xsig[:, k], cue)
            # Apply shifting:
            xpha_s[:, k] = self._ShiftSignal(np.matrix(xpha[:, k]), move)
            xsig_s[:, k] = self._ShiftSignal(np.matrix(xsig[:, k]), move)
            xpow_s[:, :, k] = self._ShiftSignal(xpow[:, :, k], move)
        xpow_s = np.mean(xpow_s, 2)
        # Normalize mean power:
        if self._normBck is not 0:
            bsl = self._baseline
            xFm = np.mean(xpow_s[:, bsl[0]:bsl[1]], 1)
            baseline = np.tile(xFm[:, np.newaxis], [1, xpow_s.shape[1]])
            xpow_s = normalize(xpow_s, baseline, norm=self._normBck)

        return xpow_s, xpha_s, xsig_s

    def tflockedplot(self, xpow, sig, cmap='viridis', vmin=None, vmax=None,
                     ylim=None, alpha=0.3, kind='std', vColor='r',
                     sigcolor='slateblue', fignum=0):
        """Plot realigned time-frequency maps.

        Args:
            xpow, sig: output of the get() method. sig can either be the phase
            or the filtered signal.

        Kargs:
            cmap: string, optional, [def: 'viridis']
                The colormap to use

            vmin, vmax: int/float, otpional, [def: None, None]
                Limits of the colorbar

            ylim: tuple/list, optional, [def: None]
                Limit for the plot of the signal

            alpha: float, optional, [def: 0.3]
                Transparency of deviation/sem

            kind: string, optional, [def: 'std']
                Choose between 'std' or 'sem' to either display standard
                deviation or standard error on the mean for the signal plot

            vColor: string, optional, [def: 'r']
                Color of the vertical line which materialized the choosen cue

            sigcolor: string, optional, [def: 'slateblue']
                Color of the signal

            fignum: integer, optional, [def: 0]
                Number of the figure

        Returns:
            figure, axes1 (TF plot), axes2 (signal plot), axes3 (colorbar)
        """
        import matplotlib.gridspec as gridspec
        import matplotlib.pyplot as plt

        xvec = self._powObj.xvec
        yvec = self._powObj.yvec

        fig = plt.figure(fignum, figsize=(8, 9))
        gs = gridspec.GridSpec(11, 11)
        ax1 = plt.subplot(gs[0:-2, 0:-1])
        ax2 = plt.subplot(gs[-2::, 0:-1])
        ax3 = plt.subplot(gs[1:-3, -1])

        # TF:
        im = ax1.imshow(xpow, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax,
                        extent=[xvec[0], xvec[-1], yvec[-1], yvec[0]])
        addLines(ax1, vLines=[self._cue], vShape=['-'], vWidth=[3],
                 vColor=[vColor])
        ax1.set_xticks([])
        ax1.set_xticklabels('')
        ax1.set_ylabel('Frequency (hz)')
        ax1.invert_yaxis()
        ax1.tick_params(axis='both', which='both', top='off', right='off')
        ax1.axis('tight')

        cb = plt.colorbar(im, cax=ax3)
        cb.set_ticks(cb.get_clim())
        cb.set_label('Power modulations', labelpad=-10)

        # Signal:
        xm = sig.mean(1)
        if kind == 'std':
            x2add = sig.std(1)
        elif kind == 'sem':
            x2add = sig.std(1)/np.sqrt(len(xm)-1)
        xlow, xhigh = xm-x2add, xm+x2add
        ax = ax2.plot(xvec, xm, lw=2, color=sigcolor)
        ax2.fill_between(xvec, xlow, xhigh, alpha=alpha,
                         color=ax[0].get_color())
        ax2.set_yticks(ax2.get_ylim())
        ax2.tick_params(axis='both', which='both', top='off', right='off')
        ax2.set_xlabel('Time')
        if ylim is not None:
            ax2.set_ylim(ylim)
        else:
            ax2.axis('tight')
        addLines(ax2, vLines=[self._cue], vShape=['-'], vWidth=[3],
                 vColor=[vColor])

        return plt.gcf(), ax1, ax2, ax3

    @staticmethod
    def _PeakDetection(sig, cue):
        """Detect peaks in a signal and return the shifting length
        corresponding to the defined cue
        sig: vector
        cue: integer (in sample)
        """
        peaks = []
        for k in range(len(sig)-1):
            if (sig[k-1] < sig[k]) and (sig[k] > sig[k+1]):
                peaks.append(k)
        minPeak = peaks[np.abs(np.array(peaks)-cue).argmin()]
        return minPeak-cue

    @staticmethod
    def _ShiftSignal(sig, move):
        """
        """
        npts = sig.shape[1]
        sigShift = np.zeros(sig.shape)
        if move >= 0:
            sigShift[:, 0:npts-move] = sig[:, move::]
        elif move < 0:
            sigShift[:, np.abs(move)::] = sig[:, 0:npts-np.abs(move)]
        return sigShift


class erpac(_coupling):

    """Compute Event Related Phase-Amplitude coupling. See [#f6]_

    .. rubric:: Footnotes
    .. [#f6] `Voytek et al, 2013 <http://www.ncbi.nlm.nih.gov/pubmed/22986076>`_

    Args:
        sf: int
            Sampling frequency

        npts: int
            Number of points of the time serie

    Kargs:
        pha_f: tuple/list, optional, [def: [2,4]]
            List containing the couple of frequency bands for the phase.
            Example: f=[ [2,4], [5,7], [60,250] ]

        pha_meth: string, optional, [def: 'hilbert']
            Method for the phase extraction.

        pha_cycle: integer, optional, [def: 3]
            Number of cycles for filtering the phase.

        amp_f: tuple/list, optional, [def: [60,200]]
            List containing the couple of frequency bands for the amplitude.
            Each couple can be either a list or a tuple.

        amp_meth: string, optional, [def: 'hilbert']
            Method for the amplitude extraction.

        amp_cycle: integer, optional, [def: 6]
            Number of cycles for filtering the amplitude.

    """
    __doc__ += windoc

    def __init__(self, sf, npts, pha_f=[2, 4], pha_meth='hilbert',
                 pha_cycle=3, amp_f=[60, 200], amp_meth='hilbert', amp_cycle=6,
                 window=None, step=None, width=None, time=None, **kwargs):
        # Check pha and amp methods:
        _checkref('pha_meth', pha_meth, ['hilbert', 'hilbert1', 'hilbert2'])
        _checkref('amp_meth', amp_meth, ['hilbert', 'hilbert1', 'hilbert2'])

        # Check the type of f:
        if (len(pha_f) == 4) and isinstance(pha_f[0], (int, float)):
            pha_f = binarize(
                pha_f[0], pha_f[1], pha_f[2], pha_f[3], kind='list')
        if (len(amp_f) == 4) and isinstance(amp_f[0], (int, float)):
            amp_f = binarize(
                amp_f[0], amp_f[1], amp_f[2], amp_f[3], kind='list')

        # Initialize cfc :
        _coupling.__init__(self, pha_f, 'phase', pha_meth, pha_cycle,
                           amp_f, 'amplitude', amp_meth, amp_cycle,
                           sf, npts, window, width, step, time, **kwargs)

    def get(self, xpha, xamp, n_perm=200, n_jobs=-1):
        """Get the erpac mesure between an xpha and xamp signals.

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
            the program compute a local erpac.

        Returns:
            xerpac: array
                The erpac mesure of size :
                (n_amplitude x n_phase x n_electrodes x n_windows)

            pvalue: array
                The associated p-values of size :
                (n_amplitude x n_phase x n_electrodes x n_windows)
        """
        # Check and get methods:
        xpha, xamp = _cfcCheck(xpha, xamp, self._npts)
        npha, namp = self._nPha, self._nAmp
        phaMeth = self._pha.get(self._sf, self._pha.f, self._npts)
        ampMeth = self._amp.get(self._sf, self._amp.f, self._npts)
        
        # Extract phase and amplitude:
        nelec, npts, ntrials = xpha.shape
        xp, xa = cfcparafilt(xpha, xamp, n_jobs, self)

        # Window:
        if not (self._window == [(0, npts)]):
            xp = binArray(xp, self._window, axis=2)[0]
            xa = binArray(xa, self._window, axis=2)[0]
            npts = xp.shape[2]

        # Extract ERPAC and surrogates:
        iteract = product(range(nelec), range(npha), range(namp))
        xerpac = np.zeros((nelec, npha, namp, npts))
        pval = np.empty_like(xerpac)
        for e, p, a in iteract:
            xerpac[e, p, a, :], pval[e, p, a, :] = _erpac(xp[e, p, ...],
                                  xa[e, a, ...], n_perm, n_jobs)
        
        return xerpac, pval

def _erpac(xp, xa, n_perm, n_jobs):
    """Sub erpac function
    [xp] = [xa] = (npts, ntrials)
    """
    npts, ntrials = xp.shape
    # Compute ERPAC
    xerpac = np.zeros((npts,))
    for t in range(npts):
        xerpac[t] = circ_corrcc(xp[t, :], xa[t, :])[0]

    # Compute surrogates:
    data = Parallel(n_jobs=n_jobs)(delayed(_erpacSuro)(
            xp, xa, npts, ntrials) for pe in range(n_perm))
    suro = np.array(data)

    # Normalize erpac:
    xerpac = (xerpac - suro.mean(0))/suro.std(0)

    # Get p-value:
    pvalue = norm.cdf(-np.abs(xerpac))*2

    return xerpac, pvalue

def _erpacSuro(xp, xa, npts, ntrials):
    """Parallel surrogates
    """
    # Permute ntrials (only for amplitude):
    perm = np.random.permutation(ntrials)
    for t in range(npts):
        suro = circ_corrcc(xp[t, :], xa[t, perm])[0]
    return suro


class pfdphase(_coupling):

    """Get the preferred phase of a phase-amplitude coupling

    Args:
        sf: int
            Sampling frequency

        npts: int
            Number of points of the time serie

    Kargs:
        nbins: integer, optional, [def: 18]
            Number of bins to binarize the amplitude.

        pha_f: tuple/list, optional, [def: [2,4]]
            List containing the couple of frequency bands for the phase.
            Example: f=[ [2,4], [5,7], [60,250] ]

        pha_meth: string, optional, [def: 'hilbert']
            Method for the phase extraction.

        pha_cycle: integer, optional, [def: 3]
            Number of cycles for filtering the phase.

        amp_f: tuple/list, optional, [def: [60,200]]
            List containing the couple of frequency bands for the amplitude.
            Each couple can be either a list or a tuple.

        amp_meth: string, optional, [def: 'hilbert']
            Method for the amplitude extraction.

        amp_cycle: integer, optional, [def: 6]
            Number of cycles for filtering the amplitude.

    """
    __doc__ += windoc

    def __init__(self, sf, npts, nbins=18, pha_f=[2, 4], pha_meth='hilbert',
                 pha_cycle=3, amp_f=[60, 200], amp_meth='hilbert', amp_cycle=6,
                 window=None, width=None, step=None, time=None,
                 **kwargs):
        # Check pha and amp methods:
        _checkref('pha_meth', pha_meth, ['hilbert', 'hilbert1', 'hilbert2'])
        _checkref('amp_meth', amp_meth, ['hilbert', 'hilbert1', 'hilbert2'])

        # Check the type of f:
        if (len(pha_f) == 4) and isinstance(pha_f[0], (int, float)):
            pha_f = binarize(
                pha_f[0], pha_f[1], pha_f[2], pha_f[3], kind='list')
        if (len(amp_f) == 4) and isinstance(amp_f[0], (int, float)):
            amp_f = binarize(
                amp_f[0], amp_f[1], amp_f[2], amp_f[3], kind='list')
        self.xvec = []
        
        # Binarize phase vector :
        self._binsize = 360 / nbins
        self._phabin = np.arange(0, 360, self._binsize)
        self.phabin = np.concatenate((self._phabin[:, np.newaxis],
                                      self._phabin[:, np.newaxis]+self._binsize), axis=1)

        # Initialize coupling:
        _coupling.__init__(self, pha_f, 'phase', pha_meth, pha_cycle,
                           amp_f, 'amplitude', amp_meth, amp_cycle,
                           sf, npts, window, width, step, time, **kwargs)
        self._nbins = nbins

    def get(self, xpha, xamp, n_jobs=-1):
        """Get the preferred phase

        Args:
            xpha: array
                Signal for phase. The shape of xpha should be :
                (n_electrodes x n_pts x n_trials)

            xamp: array
                Signal for amplitude. The shape of xamp should be :
                (n_electrodes x n_pts x n_trials)

        Kargs:
            n_jobs: integer, optional, [def: -1]
                Control the number of jobs for parallel computing. Use 1, 2, ..
                depending of your number or cores. -1 for all the cores.

            If the same signal is used (example : xpha=x and xamp=x), this mean
            the program compute a local cfc.

        Returns:
            pfp: array
                The preferred phase extracted from the mean of trials of size :
                (n_amplitude x n_phase x n_electrodes x n_windows)

            prf: array
                The preferred phase extracted from each trial of size :
                (n_amplitude x n_phase x n_electrodes x n_windows x n_trials)

            ambin: array
                The binarized amplitude of size :
                (n_amplitude x n_phase x n_electrodes x n_windows x n_bins x n_trials)

            pvalue: array
                The associated p-values of size :
                (n_amplitude x n_phase x n_electrodes x n_windows)
        """
        # Check the inputs variables :
        xpha, xamp = _cfcCheck(xpha, xamp, self._npts)
        nelec, npts, ntrials = xamp.shape
        namp, npha, nwin, nbins = self._nAmp, self._nPha, self._nwin, self._nbins
        phabin, binsize = self._phabin, self._binsize

        # Get filtered phase and amplitude ;
        pha, amp = cfcparafilt(xpha, xamp, n_jobs, self)

        # Bring phase from [-pi,pi] to [0, 360]
        pha = np.rad2deg((pha+2*np.pi)%(2*np.pi))

        # Windowing phase an amplitude :
        pha = [pha[:, :, k[0]:k[1], :] for k in self._window]
        amp = [amp[:, :, k[0]:k[1], :] for k in self._window]

        # Define iter product :
        iteract = product(range(namp), range(npha), range(nelec), range(nwin))
        data = Parallel(n_jobs=n_jobs)(delayed(_pfp)(
                pha[w][e, p, ...], amp[w][e, a, ...],
                phabin, binsize) for a, p, e, w in iteract)

        # Manage dim and output :
        pfp, prf, pval, ampbin = zip(*data)
        del pha, amp, data
        ls = [namp, npha, nelec, nwin, nbins, ntrials]
        ampbin = np.array(ampbin).reshape(*tuple(ls))
        prf = np.array(prf).reshape(*tuple(ls[0:-2]))
        pval = np.array(pval).reshape(*tuple(ls[0:-2]))
        ls.pop(4)
        pfp = np.array(pfp).reshape(*tuple(ls))

        return pfp, prf, ampbin, pval
        
def _pfp(pha, amp, phabin, binsize):
    """Sub prefered phase function
    """
    nbin, nt = len(phabin), pha.shape[1]
    ampbin = np.zeros((len(phabin), nt), dtype=float)
    # Binarize amplitude accros all trials :
    for t in range(nt):
        curpha, curamp = pha[:, t], amp[:, t]
        for i, p in enumerate(phabin):
            idx = np.logical_and(curpha >= p, curpha < p+binsize)
            if idx.astype(int).sum() != 0:
                ampbin[i, t] = curamp[idx].mean()
            else:
                ampbin[i, t] = 0
        ampbin[:, t] /= ampbin[:, t].sum()
    # Find prefered phase and p-values :
    pfp = np.array([phabin[k]+binsize/2 for k in ampbin.argmax(axis=0)])
    pvalue = circ_rtest(pfp)[0]
    prf = phabin[ampbin.mean(axis=1).argmax()]+binsize/2
    
    return pfp, prf, pvalue, ampbin


class PLV(_coupling):

    """Compute the Phase-Locking Value [#f7]_

    Args:
        sf: int
            Sampling frequency

        npts: int
            Number of points of the time serie

    Kargs:
        f: tuple/list, optional, [def: [2,4]]
            List containing the couple of frequency bands for the phase.
            Example: f=[ [2,4], [5,7], [60,250] ]

        method: string, optional, [def: 'hilbert']
            Method for the phase extraction.

        cycle: integer, optional, [def: 3]
            Number of cycles for filtering the phase.

        sample: list, optional, [def: None]
            Select samples in the time series to compute the plv

        time: list/array, optional [def: None]
            Define a specific time vector

        amp_cycle: integer, optional, [def: 6]
            Number of cycles for filtering the amplitude.

    .. rubric:: Footnotes
    .. [#f7] `Lachaux et al, 1999 <http://www.ma.utexas.edu/users/davis/reu/ch3/cwt/lachaux.pdf>`_
    """

    def __init__(self, sf, npts, f=[2, 4], method='hilbert', cycle=3,
                 sample=None, time=None, **kwargs):
        # Check pha and amp methods:
        _checkref('pha_meth', method, ['hilbert', 'hilbert1', 'hilbert2'])

        # Check the type of f:
        if (len(f) == 4) and isinstance(f[0], (int, float)):
            f = binarize(f[0], f[1], f[2], f[3], kind='list')

        # Initialize PLV :
        _coupling.__init__(self, f, 'phase', method, cycle,
                           f, 'phase', method, cycle,
                           sf, npts, None, None, None, time, **kwargs)
        if time is None:
            time = np.arange(npts)
        else:
            time = time

        if sample is None:
            sample = slice(npts)
        self._sample = sample
        self.time = time[sample]
        del self.amp

    def get(self, xelec1, xelec2, n_perm=200, n_jobs=-1):
        """Get Phase-Locking Values for a set of distant sites

        Args:
            xelec1, xelec2: array
                PLV will be compute between xelec1 and xelec2. Both matrix
                contains times-series of each trial pear electrodes. It's
                not forced that both have the same size but they must have at least
                the same number of time points (npts) and trials (ntrials).
                [xelec1] = (n_elec1, npts, ntrials), [xelec2] = (n_elec2, npts, ntrials)

        Kargs:
            n_perm: int, optional, [def: 200]
                Number of permutations to estimate the statistical significiancy
                of the plv mesure

            n_jobs: integer, optional, [def: -1]
                Control the number of jobs for parallel computing. Use 1, 2, ..
                depending of your number or cores. -1 for all the cores.

        Returns:
            plv: array
                The plv mesure for each phase and across electrodes of size:
                [plv] = (n_pha, n_elec1, n_elec2, n_sample)

            pvalues: array
                The p-values with the same shape of plv
        """
        # Check the inputs variables :
        if xelec1.ndim == 2:
            xelec1 = xelec1[np.newaxis, ...]
        if xelec2.ndim == 2:
            xelec2 = xelec2[np.newaxis, ...]
        if (xelec1.shape[1] != self._npts) or (xelec2.shape[1] != self._npts):
            raise ValueError("The second dimension of xelec1 and xelec2 must be "+str(self._npts))
        if not np.array_equal(np.array(xelec1.shape[1::]), np.array(xelec2.shape[1::])):
            raise ValueError("xelec1 and xelec2 could have a diffrent number of electrodes"
                             " but the number of time points and trials must be the same.")
        nelec1, npts, ntrials, nelec2, npha = *xelec1.shape, xelec2.shape[0], self._nPha

        # Get filtered phase for xelec1 and xelec2 :
        xcat = np.concatenate((xelec1, xelec2), axis=0)
        del xelec1, xelec2
        data = np.array(Parallel(n_jobs=n_jobs)(delayed(_plvfilt)(
                    xcat[e, ...], self) for e in range(xcat.shape[0])))
        xp1, xp2 = data[0:nelec1, ...], data[nelec1::, ...]
        del data

        # Select samples :
        xp1, xp2 = xp1[:, :, self._sample, :], xp2[:, :, self._sample, :]
        npts = xp1.shape[2]

        # Compute true PLV:
        iteract = product(range(nelec1), range(nelec2))
        plv = np.array(Parallel(n_jobs=n_jobs)(delayed(_plv)(
                    xp1[e1, ...], xp2[e2, ...]) for e1, e2 in iteract))
        plv = np.transpose(plv.reshape(nelec1, nelec2, npha, npts), (2, 0, 1, 3))

        # Compute surrogates:
        pvalues = np.zeros_like(plv)
        perm = [np.random.permutation(ntrials) for k in range(n_perm)]
        iteract = product(range(nelec1), range(nelec2))
        for e1, e2 in iteract:
            pvalues[:, e1, e2, ...] = _plvstat(xp1[e1, ...], xp2[e2, ...],
                                               plv[:, e1, e2, ...], n_perm, n_jobs, perm)

        return plv, pvalues
    
def _plvfilt(x, self):
    """Sub PLV filt
    """
    # Get the filter for phase/amplitude properties :
    fMeth = self._pha.get(self._sf, self._pha.f, self._npts)
    return self._pha.apply(x, fMeth)

def _plvstat(xp1, xp2, plv, n_perm, n_jobs, perm):
    """Sub plv-stat function
    """
    # Compute plv for each permutation of xp2 trials :
    plvs = np.array(Parallel(n_jobs=n_jobs)(delayed(_plv)(
                 xp1, xp2[..., p]) for p in perm))

    # Get p-values from permutations :
    return perm_2pvalue(plv, plvs, n_perm, tail=1)

def _plv(phi1, phi2):
    """PLV, (lachaux et al, 1999)
    """
    return np.abs(np.exp(1j*(phi1-phi2)).mean(axis=-1))
