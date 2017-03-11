import numpy as np
from scipy.special import erfinv
from itertools import product

from brainpipe.tools import binarize
from brainpipe.statistics import perm_swap, perm_array

__all__ = [
    'CfcSettings',
]


# ----------------------------------------------------------------------------
#                            ID to CFC MODEL
# ----------------------------------------------------------------------------
def CfcSettings(Id, nbins=18, n_perm=200, tlag=[0, 0], matricial=True):
    """From an Id, get the model of cfc composed with:
    - Method : how to compute cfc
    - Surrogates : method for computing surrogates
    - Normalization : how to normalize the cfc with surrogates

    For each of the three components, this function return a string and a
    function to apply.
    """
    # Define the method of PAC :
    [CfcModel, CfcModelStr] = CfcMethodList(int(Id[0]), nbins=nbins)

    # Define the way to compute surrogates :
    [CfcSur, CfcSurStr] = CfcSurrogatesList(int(Id[1]), CfcModel,
                                            n_perm=n_perm, tlag=tlag,
                                            matricial=matricial)

    # Define the way to normalize the Cfc with surrogates :
    [CfcNorm, CfcNormStr] = CfcNormalizationList(int(Id[2]))

    return CfcModel, CfcSur, CfcNorm, CfcModelStr, CfcSurStr, CfcNormStr


# ----------------------------------------------------------------------------
#                                 METHODS
# ----------------------------------------------------------------------------

def CfcMethodList(Id, nbins=18):
    """List of methods to compute the cfc. This list include methods for
    Phase-Amplitude, phase-phase or amplitude-amplitude coupling. Here's the
    list of the implemented methods :
    - Mean Vector Length
    - Kullback-Leibler Divergence
    - Heights Ratio
    - Phase synchrony
    - ndPAC


    Each method take at least a pha and amp array with the respective
    dimensions:
    pha.shape = (Nb phase     x Time points)
    amp.shape = (Nb amplitude x Time points)
    And each method should return a (Nb amplitude x Nb phase)
    """
    # Mean Vector Length (Canolty, 2006)
    if Id == 1:
        def CfcModel(pha, amp, *arg):
            return MVL(pha, amp)
        CfcModelStr = 'Mean Vector Length (Canolty, 2006)'

    # Kullback-Leiber divergence (Tort, 2010)
    elif Id == 2:
        def CfcModel(pha, amp, nbins=nbins):
            return KullbackLeiblerDivergence(pha, amp, nbins)
        CfcModelStr = 'Kullback-Leibler Divergence ['+str(
            nbins)+' bins] (Tort, 2010)'

    # Heights ratio
    elif Id == 3:
        def CfcModel(pha, amp, nbins=nbins):
            return HeightsRatio(pha, amp, nbins)
        CfcModelStr = 'Heights ratio ['+str(nbins)+' bins]'

    # Phase synchrony
    elif Id == 4:
        def CfcModel(pha, amp, *arg):
            return PhaseSynchrony(pha, amp)
        CfcModelStr = 'Phase synchrony (PLV, (Penny, 2008))'

    # ndPac (Ozkurt, 2012)
    elif Id == 5:
        def CfcModel(pha, amp, *arg):
            return ndCfc(pha, amp)
        CfcModelStr = 'Normalized direct Pac (Ozkurt, 2012)'

    return CfcModel, CfcModelStr


def MVL(pha, amp):
    """Mean Vector Length (Canolty, 2006)

    Method :
    abs(amplitude x exp(phase)) <-> sum modulations of the
    complex radius accross time. MI = resultant radius
    """
    return np.array(abs(amp*np.exp(1j*pha).T)/pha.shape[1])


def KullbackLeiblerDivergence(pha, amp, nbins):
    """Kullback Leibler Divergence (Tort, 2010)
    """
    # Get the phase locked binarized amplitude :
    abin, abinsum = _kl_hr(pha, amp, nbins)
    abin = np.divide(abin, np.rollaxis(abinsum, 0, start=3))
    abin[abin == 0] = 1
    abin = abin * np.log2(abin)

    return (1 + abin.sum(axis=2)/np.log2(nbins))


def HeightsRatio(pha, amp, nbins):
    """Heights Ratio
    """
    # Get the phase locked binarized amplitude :
    abin, abinsum = _kl_hr(pha, amp, nbins)
    M, m = abin.max(axis=2), abin.min(axis=2)
    MDown = M.copy()
    MDown[MDown == 0] = 1

    return (M-m)/MDown


def _kl_hr(pha, amp, nbins):
    nPha, npts, nAmp = *pha.shape, amp.shape[0]
    step = 2*np.pi/nbins
    vecbin = binarize(-np.pi, np.pi+step, step, step)
    if len(vecbin) > nbins:
        vecbin = vecbin[0:-1]

    abin = np.zeros((nAmp, nPha, nbins))
    for k, i in enumerate(vecbin):
        # Find where phase take vecbin values :
        pL, pC = np.where((pha >= i[0]) & (pha < i[1]))

        # Matrix to do amp x binMat :
        binMat = np.zeros((npts, nPha))
        binMat[pC, pL] = 1
        meanMat = np.matlib.repmat(binMat.sum(axis=0), nAmp, 1)
        meanMat[meanMat == 0] = 1

        # Multiply matrix :
        abin[:, :, k] = np.divide(np.dot(amp, binMat), meanMat)
    abinsum = np.array([abin.sum(axis=2) for k in range(nbins)])

    return abin, abinsum


def PhaseSynchrony(pha, amp):
    """Phase Synchrony
    """
    return np.array(abs((np.exp(-1j*amp)*(np.exp(1j*pha).T))/pha.shape[1]))


def ndCfc(pha, amp):
    """Normalized direct Pac (Ozkurt, 2012)
    """
    npts = amp.shape[1]
    # Get mean and deviation of amplitude :
    amp_m = np.tile(np.mean(amp, 1)[..., np.newaxis], (1, npts))
    amp_std = np.tile(np.std(amp, 1)[..., np.newaxis], (1, npts))
    # Normalize amplitude :
    amp = np.divide(amp - amp_m, amp_std)
    # Compute pac :
    return np.square(np.abs(amp*np.exp(1j*pha.T)))/npts

# ----------------------------------------------------------------------------
#                                 SURROGATES
# ----------------------------------------------------------------------------


def CfcSurrogatesList(Id, CfcModel, n_perm=200, tlag=[0, 0], matricial=True):
    """List of methods to compute surrogates.

    The surrogates are used to normalized the cfc value. It help to determine
    if the cfc is reliable or not. Usually, the surrogates used the same cfc
    method on surrogates data.
    Here's the list of methods to compute surrogates:
    - No surrogates
    - Swap phase/amplitude through trials
    - Swap amplitude
    - Shuffle phase time-series
    - Shuffle amplitude time-series
    - Time lag
    - circular shifting

    Each method should return the surrogates, the mean of the surrogates and
    the deviation of the surrogates.
    """
    # No surrogates
    if Id == 0:
        def CfcSuroModel(pha, amp, CfcModel, n_perm, *args):
            return (None, None, None)
        CfcSuroModelStr = 'No surrogates'

    # Swap phase/amplitude through trials
    elif Id == 1:
        def CfcSuroModel(pha, amp, CfcModel, n_perm, matricial, *args):
            return CfcTrialSwap(pha, amp, CfcModel, n_perm=n_perm,
                                matricial=matricial)
        CfcSuroModelStr = 'Swap phase/amplitude through trials, (Tort, 2010)'

    # Swap amplitude
    elif Id == 2:
        def CfcSuroModel(pha, amp, CfcModel, n_perm, matricial, *args):
            return CfcAmpSwap(pha, amp, CfcModel, n_perm=n_perm,
                              matricial=matricial)
        CfcSuroModelStr = 'Swap amplitude, (Bahramisharif, 2013)'

    # Shuffle phase values
    elif Id == 3:
        def CfcSuroModel(pha, amp, CfcModel, n_perm, *args):
            return CfcShufflePhase(pha, amp, CfcModel, n_perm=n_perm)
        CfcSuroModelStr = 'Shuffle phase time-series'

    # Shuffle amplitude values
    elif Id == 4:
        def CfcSuroModel(pha, amp, CfcModel, n_perm, *args):
            return CfcShuffleAmp(pha, amp, CfcModel, n_perm=n_perm)
        CfcSuroModelStr = 'Shuffle amplitude time-series'

    # Introduce a time lag
    elif Id == 5:
        def CfcSuroModel(pha, amp, CfcModel, n_perm, tlag, *args):
            return CfcTimeLag(pha, amp, CfcModel, n_perm=n_perm, tlag=tlag)
        CfcSuroModelStr = 'Time lag on amplitude between ['+int(
            tlag[0])+';'+int(tlag[1])+'] , (Canolty, 2006)'

    # Circular shifting
    elif Id == 6:
        def CfcSuroModel(pha, amp, CfcModel, n_perm):
            return CfcCircShift(pha, amp, CfcModel, n_perm=n_perm)
        CfcSuroModelStr = 'Circular shifting'

    return CfcSuroModel, CfcSuroModelStr


def CfcTrialSwap(xfP, xfA, CfcModel, n_perm=200, matricial=True):
    """Swap phase/amplitude trials (Tort, 2010)

    [xfP] = (nPha, npts, ntrials)
    [xfA] = (nAmp, npts, ntrials)
    """
    # Get sizes :
    nPha, timeL, nbTrials = xfP.shape
    nAmp = xfA.shape[0]
    Suro = np.zeros((nbTrials, nAmp, nPha, n_perm))
    if matricial:
        # Swap trials phase/amplitude :
        phamp = np.concatenate((xfP, xfA))
        phampSh1, phampSh2 = perm_swap(phamp, phamp, axis=2, n_perm=n_perm)
        phaSh, ampSh = phampSh1[:, 0:nPha, ...], phampSh2[:, nPha::, ...]
        del phamp, phampSh1, phampSh2
        # Get pac :
        iteract = product(range(nbTrials), range(n_perm))
        for tr, pe in iteract:
            Suro[tr, :, :, pe] = CfcModel(np.matrix(phaSh[pe, :, :, tr]),
                                          ampSh[pe, :, :, tr])
    else:
        # Swap trials phase/amplitude :
        phampiter = product(range(nPha), range(nAmp))
        for ipha, iamp in phampiter:
            # Concatenate selected pha/amp :
            phamp = np.concatenate((xfP[[ipha], ...], xfA[[iamp], ...]))
            # Swap:
            phampSh1, phampSh2 = perm_swap(phamp, phamp, axis=2, n_perm=n_perm)
            phaSh, ampSh = phampSh1[:, 0, ...], phampSh2[:, 1, ...]
            # Get pac :
            iteract = product(range(nbTrials), range(n_perm))
            for tr, pe in iteract:
                Suro[tr, iamp, ipha, pe] = CfcModel(np.matrix(phaSh[pe, :, tr]),
                                                    ampSh[pe, :, tr])

    return Suro


def CfcAmpSwap(xfP, xfA, CfcModel, n_perm=200, matricial=True):
    """Swap phase/amplitude trials, (Bahramisharif, 2013)

    [xfP] = (nPha, npts, ntrials)
    [xfA] = (nAmp, npts, ntrials)
    """
    # Get sizes :
    nPha, timeL, nbTrials = xfP.shape
    nAmp = xfA.shape[0]
    Suro = np.zeros((nbTrials, nAmp, nPha, n_perm))
    if matricial:
        # Swap trials phase/amplitude :
        ampSh, _ = perm_swap(xfA, xfA, axis=2, n_perm=n_perm)
        # Get pac :
        iteract = product(range(nbTrials), range(n_perm))
        for tr, pe in iteract:
            Suro[tr, :, :, pe] = CfcModel(xfP[:, :, tr],
                                          np.matrix(ampSh[pe, :, :, tr]))
    else:
        for iamp in range(nAmp):
            ampSh, _ = perm_swap(xfA[iamp, ...], xfA[iamp, ...], axis=1, n_perm=n_perm)
            # Get pac :
            iteract = product(range(nbTrials), range(n_perm))
            for tr, pe in iteract:
                Suro[tr, iamp, :, pe] = CfcModel(xfP[:, :, tr],
                                                 np.matrix(ampSh[pe, :, tr]))

    return Suro


def CfcShufflePhase(xfP, xfA, CfcModel, n_perm=200):
    """Randomly shuffle phase

    [xfP] = (nPha, npts, ntrials)
    [xfA] = (nAmp, npts, ntrials)
    """
    # Get sizes :
    nPha, timeL, nbTrials = xfP.shape
    nAmp = xfA.shape[0]
    perm = [np.random.permutation(timeL) for k in range(n_perm)]
    # Compute surrogates :
    Suro = np.zeros((nbTrials, nAmp, nPha, n_perm))
    for k in range(nbTrials):
        curPha, curAmp = xfP[:, :, k], np.matrix(xfA[:, :, k])
        for i in range(n_perm):
            # Randomly permute phase time-series :
            CurPhaShuffle = curPha[:, perm[i]]
            # compute new Cfc :
            Suro[k, :, :, i] = CfcModel(np.matrix(CurPhaShuffle), curAmp)

    return Suro


def CfcShuffleAmp(xfP, xfA, CfcModel, n_perm=200):
    """Randomly shuffle amplitudes

    [xfP] = (nPha, npts, ntrials)
    [xfA] = (nAmp, npts, ntrials)
    """
    # Get sizes :
    nPha, timeL, nbTrials = xfP.shape
    nAmp = xfA.shape[0]
    perm = [np.random.permutation(timeL) for k in range(n_perm)]
    # Compute surrogates :
    Suro = np.zeros((nbTrials, nAmp, nPha, n_perm))
    for k in range(nbTrials):
        curPha, curAmp = xfP[:, :, k], np.matrix(xfA[:, :, k])
        for i in range(n_perm):
            # Randomly permute amplitude time-series :
            CurAmpShuffle = curAmp[:, perm[i]]
            # compute new Cfc :
            Suro[k, :, :, i] = CfcModel(np.matrix(curPha), CurAmpShuffle)

    return Suro


def CfcShufflePhaAmp(xfP, xfA, CfcModel, n_perm=200):
    """Randomly shuffle amplitudes

    [xfP] = (nPha, npts, ntrials)
    [xfA] = (nAmp, npts, ntrials)
    """
    # Get sizes :
    nPha, timeL, nbTrials = xfP.shape
    nAmp = xfA.shape[0]
    perm = [np.random.permutation(timeL) for k in range(n_perm)]
    # Compute surrogates :
    Suro = np.zeros((nbTrials, nAmp, nPha, n_perm))
    for k in range(nbTrials):
        curPha, curAmp = xfP[:, :, k], np.matrix(xfA[:, :, k])
        for i in range(n_perm):
            # Randomly permute phase time-series :
            CurAmpShuffle = curAmp[:, perm[i]]
            CurPhaShuffle = curPha[:, perm[i]]
            # compute new Cfc :
            Suro[k, :, :, i] = CfcModel(np.matrix(CurPhaShuffle), CurAmpShuffle)

    return Suro

# ----------------------------------------------------------------------------
#                               NORMALIZATION
# ----------------------------------------------------------------------------
def CfcNormalizationList(Id):
    """List of the normalization methods.

    Use a normalization to normalize the true cfc value by the surrogates.
    Here's the list of the normalization methods :
    - No normalization
    - Substraction : substract the mean of surrogates
    - Divide : divide by the mean of surrogates
    - Substract then divide : substract then divide by the mean of surrogates
    - Z-score : substract the mean and divide by the deviation of the
                surrogates

    The normalized method only return the normalized cfc.
    """
    # No normalisation
    if Id == 0:
        def CfcNormModel(uCfc, SuroMean, SuroStd):
            return uCfc
        CfcNormModelStr = 'No normalisation'

    # Substraction
    if Id == 1:
        def CfcNormModel(uCfc, SuroMean, SuroStd):
            return ucfc-SuroMean
        CfcNormModelStr = 'Substract the mean of surrogates'

    # Divide
    if Id == 2:
        def CfcNormModel(uCfc, SuroMean, SuroStd):
            return uCfc/SuroMean
        CfcNormModelStr = 'Divide by the mean of surrogates'

    # Substract then divide
    if Id == 3:
        def CfcNormModel(uCfc, SuroMean, SuroStd):
            SuroMean[SuroMean == 0] = 1
            return (uCfc-SuroMean)/SuroMean
        CfcNormModelStr = 'Substract then divide by the mean of surrogates'

    # Z-score
    if Id == 4:
        def CfcNormModel(uCfc, SuroMean, SuroStd):
            SuroStd[SuroStd == 0] = 1
            return (uCfc-SuroMean)/SuroStd
        CfcNormModelStr = 'Z-score'

    return CfcNormModel, CfcNormModelStr
