from brainpipe.feat.filtering import fextract, docfilter
from brainpipe.tools import binarize, binArray
from brainpipe.feat.utils._feat import (_manageWindow, _manageFrequencies,
                                        normalize, _checkref)
from brainpipe.visu.cmon_plt import tilerplot
from brainpipe.statistics import (perm_swap, perm_metric, perm_2pvalue,
                                  maxstat, circ_rtest)

from scipy.stats import wilcoxon, kruskal

from joblib import Parallel, delayed
import numpy as np
from itertools import product
from warnings import warn


__all__ = ['sigfilt',
           'amplitude',
           'power',
           'TF',
           'phase',
           'PLF'
           ]


# ------------------------------------------------------------
# DOCUMENTATION
# ------------------------------------------------------------
docsignal = """norm: int, optional [def: None]
            Number to choose the normalization method
                - 0: No normalisation
                - 1: Substraction
                - 2: Division
                - 3: Substract then divide
                - 4: Z-score

        baseline: tuple/list of int [def: None]
            Define a window to normalize the power

        split: int or list of int, optional [def: None]
            Split the frequency band f in "split" band width.
            If f is a list which contain couple of frequencies,
            split is a list too.
"""

supfilter = """
        kwargs: supplementar arguments for filtering
""" + docfilter + """

.. automethod:: get"""


# ------------------------------------------------------------
# MAIN SPECTRAL CLASS
# ------------------------------------------------------------
class _spectral(tilerplot):
    """This class is optimized for 3D arrays.

    Args:
        sf: int
            Sampling frequency

        npts: int
            Number of points of the time serie

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

    def __init__(self, sf, npts, kind, f, baseline, norm, method, window,
                 width, step, split, time, meanT, **kwargs):

        # Check the type of f:
        if (len(f) == 4) and isinstance(f[0], (int, float)):
            f = binarize(f[0], f[1], f[2], f[3], kind='list')
        # Manage time and frequencies:
        self._window, self.xvec = _manageWindow(
            npts, window=window, width=width, step=step, time=time)
        self.f, self._fSplit, self._fSplitIndex = _manageFrequencies(
            f, split=split)
        # Get variables :
        self._baseline = baseline
        self._norm = norm
        self._width = width
        self._step = step
        self._split = split
        self._nf = len(self.f)
        self._sf = sf
        self._npts = npts
        self.yvec = [round((k[0]+k[1])/2) for k in self.f]
        self._kind = kind
        self._fobj = fextract(method, kind, **kwargs)
        self._meanT = meanT
        if (self._window is not None) and (time is not None):
            self.xvec = binArray(time, self._window)[0]
        self.xvec = list(self.xvec)

    def __str__(self):
        extractStr = str(self._fobj)
        powStr = '{ki}(norm={no}, step={st}, width={wi}, split={sp},\n'
        powStr = powStr.format(ki=self._kind, no=self._norm, st=self._step,
                               wi=self._width, sp=self._split)
        return powStr+extractStr+')'

    def get(self, x, statmeth=None, tail=2, n_perm=200, metric='m_center',
            maxstat=False, n_jobs=-1):
        """Get the spectral feature of the signal x.

        Args:
            x: array
                Data with a shape of (n_electrodes x n_pts x n_trials)

        Kargs:
            statmeth: string, optional, [def: None]
                Method to evaluate the statistical significiancy. To get p-values,
                the program will compare real values with a defined baseline. As a
                consequence, the 'norm' and 'baseline' parameter should not be None.

                - 'permutation': randomly shuffle real data with baseline.Control the number of permutations with the n_perm parameter. For example, if n_perm = 1000, this mean that minimum p-valueswill be 0.001.
                - 'wilcoxon': Wilcoxon signed-rank test
                - 'kruskal': Kruskal-Wallis H-test

            n_perm: integer, optional, [def: 200]
                Number of permutations for assessing statistical significiancy.

            tail: int, optional, [def: 2]
                For the permutation method, get p-values from one or two tails of
                the distribution. Use -1 for testing A<B, 1 for A>B and 2 for A~=B.

            metric: string/function type, optional, [def: 'm_center']
                Use diffrent metrics to normalize data and permutations by the
                defined baseline. Use:

                - None: compare directly values without transformation
                - 'm_center': (A-B)/mean(B) transformation
                - 'm_zscore': (A-B)/std(B) transformation
                - 'm_minus': (A-B) transformation
                - function: user defined function [def myfcn(A, B): return array_like]

            maxstat: bool, optional, [def: False]
                Correct p-values with maximum statistique. If maxstat is True,
                the correction will be applied only trhough frequencies.

            n_jobs: integer, optional, [def: -1]
                Control the number of jobs to extract features. If
                n_jobs = -1, all the jobs are used.

        Return:
            xF: array
                The un/normalized feature of x, with a shape of
                (n_frequency x n_electrodes x n_window x n_trials)

            pvalues: array
                p-values with a shape of (n_frequency x n_electrodes x n_window)
        """
        # Get variables :
        self._statmeth = statmeth
        self._n_perm = n_perm
        self._2t = tail
        self._mxst = maxstat
        self._metric = metric

        # Check input size :
        if len(x.shape) == 2:
            x = x[np.newaxis, ...]
        if x.shape[1] != self._npts:
            raise ValueError('The second dimension must be '+str(self._npts))
        nfeat = x.shape[0]
        warnmsg = 'You define a normalization but no baseline has been' + \
                  ' specified. Normalization will be ignore'
        if (self._norm is not None) and (self._baseline is None):
            warn(warnmsg)
            self._norm = None

        # Check statistical method :
        if statmeth is not None:
            _checkref('statmeth', statmeth, ['permutation', 'wilcoxon', 'kruskal'])

        # run feature computation:
        data = Parallel(n_jobs=n_jobs)(
            delayed(_get)(x[k, ...], self) for k in range(nfeat))
        xF, pvalues = zip(*data)

        # Re-organize data :
        xF = np.swapaxes(np.array(xF), 0, 1)
        if pvalues[0] is not None:
            pvalues = np.swapaxes(np.array(pvalues), 0, 1)
        else:
            pvalues = None

        # Remove last dimension (for TF):
        if self._meanT:
            xF = xF[..., 0]

        return xF, pvalues


# ------------------------------------------------------------
# SUB GET FUNCTION
# ------------------------------------------------------------
def _get(x, self):
    """Sub get function.

    Get the spectral info of x.
    """
    # Unpack args :
    bsl, norm = self._baseline, self._norm
    n_perm, statmeth = self._n_perm, self._statmeth

    # Get the filter properties and apply:
    fMeth = self._fobj.get(self._sf, self._fSplit, self._npts)
    xF = self._fobj.apply(x, fMeth)
    nf, npts, nt = xF.shape

    # Statistical evaluation :
    if (n_perm is not 0) and (bsl is not None) and (statmeth is not None):
        pvalues = _evalstat(self, xF, bsl)
    else:
        pvalues = None

    # Mean through trials:
    if self._meanT:
        xF = np.mean(xF[..., np.newaxis], 2)

    # Normalize power :
    if (norm is not None) and (bsl is not None):
        xFm = np.mean(xF[:, bsl[0]:bsl[1], :], 1)
        baseline = np.tile(xFm[:, np.newaxis, :], [1, xF.shape[1], 1])
        xF = normalize(xF, baseline, norm=norm)

    # Mean Frequencies :
    xF, _ = binArray(xF, self._fSplitIndex, axis=0)

    # Mean time :
    if self._window is not None:
        xF, _ = binArray(xF, self._window, axis=1)

    return xF, pvalues


def _evalstat(self, x, bsl):
    """Statistical evaluation of features

    [x] = [xn] = (nFce, npts, nTrials)
    """
    # Unpack variables:
    statmeth = self._statmeth
    n_perm = self._n_perm
    tail = self._2t
    maxst = self._mxst

    # Mean Frequencies :
    x, _ = binArray(x, self._fSplitIndex, axis=0)

    # Get the baseline and set to same shape of x:
    xFm = np.mean(x[:, bsl[0]:bsl[1], :], 1)

    # Mean time :
    if self._window is not None:
        x, _ = binArray(x, self._window, axis=1)

    # Repeat baseline:
    baseline = np.tile(xFm[:, np.newaxis, :], [1, x.shape[1], 1])

    # Get shape of x:
    nf, npts, nt = x.shape
    pvalues = np.ones((nf, npts))

    # Switch between methods:
    #   -> Permutations
    # Loops on time and matrix for frequency (avoid RAM usage but increase speed)
    if statmeth == 'permutation':
        # Get metric:
        fcn = perm_metric(self._metric)
        # Apply metric to x and baseline:
        xN = fcn(x, baseline).mean(axis=2)
        # For each time points:
        for pts in range(npts):
            # Randomly swap x // baseline :
            perm = perm_swap(x[:, pts, :], baseline[:, pts, :],
                             n_perm=n_perm, axis=1, rndstate=0+pts)[0]
            # Normalize permutations by baline:
            perm = fcn(perm, baseline[:, pts, :]).mean(2)
            # Maximum stat (correct through frequencies):
            if maxst:
                perm = maxstat(perm, axis=1)
            # Get pvalues :
            pvalues[:, pts] = perm_2pvalue(xN[:, pts], perm, n_perm, tail=tail)

    #   -> Wilcoxon // Kruskal-Wallis:
    else:
        # Get the method:
        if statmeth == 'wilcoxon':
            def fcn(a, b): return wilcoxon(a, b)[1]
        elif statmeth == 'kruskal':
            def fcn(a, b): return kruskal(a, b)[1]

        # Apply:
        ite = product(range(nf), range(npts))
        for k, i in ite:
            pvalues[k, i] = fcn(x[k, i, :], xFm[k, :])

    return pvalues


# ------------------------------------------------------------
# FILTERED SIGNAL
# ------------------------------------------------------------
class sigfilt(_spectral):

    """Extract the filtered signal. """
    __doc__ += _spectral.__doc__ + docsignal + supfilter

    def __init__(self, sf, npts, f=[60, 200], baseline=None, norm=None,
                 window=None, width=None, step=None, split=None, time=None,
                 **kwargs):
        _spectral.__init__(self, sf, npts, 'signal', f, baseline, norm,
                           'filter', window, width, step, split, time,
                           False, **kwargs)


# ------------------------------------------------------------
# AMPLITUDE
# ------------------------------------------------------------
class amplitude(_spectral):

    """Extract the amplitude of the signal. """
    __doc__ += _spectral.__doc__ + docsignal
    __doc__ += """
    method: string
        Method to transform the signal. Possible values are:
            - 'hilbert': apply a hilbert transform to each column
            - 'hilbert1': hilbert transform to a whole matrix
            - 'hilbert2': 2D hilbert transform
            - 'wavelet': wavelet transform
    """ + supfilter

    def __init__(self, sf, npts, f=[60, 200], baseline=None, norm=None,
                 method='hilbert1', window=None, width=None, step=None,
                 split=None, time=None, **kwargs):
        _checkref('method', method, ['hilbert', 'hilbert1', 'hilbert2',
                  'wavelet'])
        _spectral.__init__(self, sf, npts, 'amplitude', f, baseline, norm,
                           method, window, width, step, split, time,
                           False, **kwargs)


# ------------------------------------------------------------
# POWER
# ------------------------------------------------------------
class power(_spectral):

    """Extract the power of the signal. """
    __doc__ += _spectral.__doc__ + docsignal
    __doc__ += """
    method: string
        Method to transform the signal. Possible values are:
            - 'hilbert': apply a hilbert transform to each column
            - 'hilbert1': hilbert transform to a whole matrix
            - 'hilbert2': 2D hilbert transform
            - 'wavelet': wavelet transform
    """ + supfilter

    def __init__(self, sf, npts, f=[60, 200], baseline=None, norm=None,
                 method='hilbert1', window=None, width=None, step=None,
                 split=None, time=None, **kwargs):
        _checkref('method', method, ['hilbert', 'hilbert1', 'hilbert2',
                  'wavelet'])
        _spectral.__init__(self, sf, npts, 'power', f, baseline, norm,
                           method, window, width, step, split, time,
                           False, **kwargs)


# ------------------------------------------------------------
# TIME-FREQUENCY MAP
# ------------------------------------------------------------
class TF(_spectral):

    """Extract the time-frequency map of the signal. """
    __doc__ += _spectral.__doc__ + docsignal
    __doc__ += """
    method: string
        Method to transform the signal. Possible values are:
            - 'hilbert': apply a hilbert transform to each column
            - 'hilbert1': hilbert transform to a whole matrix
            - 'hilbert2': 2D hilbert transform
            - 'wavelet': wavelet transform
    """ + supfilter

    def __init__(self, sf, npts, f=(2, 200, 10, 5), baseline=None, norm=None,
                 method='hilbert1', window=None, width=None, step=None,
                 time=None, **kwargs):
        _checkref('method', method, ['hilbert', 'hilbert1', 'hilbert2',
                  'wavelet'])
        _spectral.__init__(self, sf, npts, 'power', f, baseline, norm,
                           method, window, width, step, None, time,
                           True, **kwargs)


# ------------------------------------------------------------
# PHASE
# ------------------------------------------------------------
class phase(_spectral):

    """Extract the phase of a signal. """
    __doc__ += _spectral.__doc__
    __doc__ += """method: string
        Method to transform the signal. Possible values are:
            - 'hilbert': apply a hilbert transform to each column
            - 'hilbert1': hilbert transform to a whole matrix
            - 'hilbert2': 2D hilbert transform
    """ + supfilter

    def __init__(self, sf, npts, f=[2, 4], method='hilbert', window=None,
                 width=None, step=None, time=None, **kwargs):
        _checkref('method', method, ['hilbert', 'hilbert1', 'hilbert2'])
        _spectral.__init__(self, sf, npts, 'phase', f, None, None, method,
                           window, width, step, None, time, False, **kwargs)

    def get(self, x, getstat=True, n_jobs=-1):
        """Get the spectral phase of the signal x.

        Args:
            x: array
                Data with a shape of (n_electrodes x n_pts x n_trials)

        Kargs:
            getstat: bool, optional, [def: True]
                Set it to True if p-values should be computed. Satistical
                p-values are comptuted using Rayleigh test.

            n_jobs: integer, optional, [def: -1]
                Control the number of jobs to extract features. If
                n_jobs = -1, all the jobs are used.

        Return:
            xF: array
                The phase of x, with a shape of
                (n_frequency x n_electrodes x n_window x n_trials)

            pvalues: array
                p-values with a shape of (n_frequency x n_electrodes x n_window)
        """
        # Check input size :
        if len(x.shape) == 2:
            x = x[np.newaxis, ...]
        if x.shape[1] != self._npts:
            raise ValueError('The second dimension must be '+str(self._npts))
        nfeat = x.shape[0]
        self._getstat = getstat
        # run feature computation:
        data = Parallel(n_jobs=n_jobs)(
            delayed(_phase)(x[k, ...], self) for k in range(nfeat))
        xF, pvalues = zip(*data)
        # Manage output type:
        if pvalues[0] is None:
            pvalues = None
        else:
            pvalues = np.array(pvalues)

        return np.array(xF), pvalues

def _phase(x, self):
    """Sub-phase function
    """
    # Get the filter properties and apply:
    fMeth = self._fobj.get(self._sf, self._fSplit, self._npts)
    xF = self._fobj.apply(x, fMeth)

    # Mean time :
    if self._window is not None:
        xF, _ = binArray(xF, self._window, axis=1)
    nf, npts, nt = xF.shape

    # Get p-value:
    if self._getstat:
        pvalues = np.zeros((nf, npts), dtype=float)
        for f in range(nf):
            for k in range(npts):
                pvalues[f, k] = circ_rtest(xF[f, k, :])[0]
    else:
        pvalues = None

    return xF, pvalues


class PLF(phase):

    """Extract the phase-locking factor of a signal. """
    __doc__ += _spectral.__doc__
    __doc__ += """method: string
        Method to transform the signal. Possible values are:
            - 'hilbert': apply a hilbert transform to each column
            - 'hilbert1': hilbert transform to a whole matrix
            - 'hilbert2': 2D hilbert transform
    """ + supfilter

    def __init__(self, sf, npts, f=[2, 4], method='hilbert', window=None,
                 width=None, step=None, time=None, **kwargs):
        phase.__init__(self, sf, npts, f=f, method=method, window=window,
                       width=width, step=step, time=time, **kwargs)
        self.__phaO = phase(sf, npts, f=f, method=method, window=window,
                            width=width, step=step, time=time, **kwargs)

    def get(self, x, getstat=True, n_jobs=-1):
        """Get the phase-locking factor of the signal x.

        Args:
            x: array
                Data with a shape of (n_electrodes x n_pts x n_trials)

        Kargs:
            getstat: bool, optional, [def: True]
                Set it to True if p-values should be computed. Satistical
                p-values are comptuted using Rayleigh test.

            n_jobs: integer, optional, [def: -1]
                Control the number of jobs to extract features. If
                n_jobs = -1, all the jobs are used.

        Return:
            plf: array
                The PLF of x, with a shape of
                (n_frequency x n_electrodes x n_window)

            pvalues: array
                p-values with a shape of (n_frequency x n_electrodes x n_window)
        """
        # Get phase and p-values
        xf, pval = self.__phaO.get(x, getstat=getstat, n_jobs=n_jobs)
        # Get plf
        plf = np.abs(np.exp(1j*(xf)).mean(3))
        return plf, pval