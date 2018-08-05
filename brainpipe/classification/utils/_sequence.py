import numpy as np
from joblib import Parallel, delayed
from itertools import combinations

__all__ = ['_sequence']


def _sequence(x, y, clfObj, direction, grp, n_jobs, display, cwi):
    """Run the inner sequence forward/backward/exhaustiv.
    """
    # Check inputs size :
    y = np.ravel(y)
    grpwrd = 'group'
    if x.shape[0] != len(y):
        x = x.T
    if not isinstance(grp, list):
        grp = list(grp)
    if not grp:
        grp = np.arange(x.shape[1])
        grpwrd = 'feature'
    grp = {'g'+str(l): [i for i, j in enumerate(grp) if j == k]
           for l, k in enumerate(set(grp))}
    grpu = [k for k in range(len(set(grp)))]
    nfeat = len(grpu)
    ntrial, _ = x.shape

    # Initialize list :
    if direction == 'forward':
        cst, dyn = [], grpu.copy()
        dirwrd = 'added'
        new_score = 0.05
    elif direction == 'backward':
        cst, dyn = grpu.copy(), grpu.copy()
        dirwrd = 'removed'
        new_score = clfObj.fit(x, n_jobs=1, mf=True)[0][0]
    elif direction == 'exhaustive':
        cst, dyn = grpu.copy(), []
        new_score = 0.05
    combi = _seqcombination(cst, dyn, direction, grp)

    if display:
        print('-> Run the '+direction+' feature selection')

    # 2 - Classify all features to find the best one :
    old_score = 0
    k = 0
    flist = []
    while k <= nfeat - 1 and old_score < new_score:
        # Classify all combinations :
        old_score = new_score
        new_score, indu, all_scores = clfcombi(clfObj, x, combi, n_jobs=n_jobs)
        ind = combi[indu]

        # Update variables:
        if direction == 'forward':
            igrp = grpu[indu]
            cst.append(igrp)
            grpu = np.delete(grpu, indu)
            dyn = grpu.copy()
            flist = cst[0:-1]
        elif direction == 'backward':
            flist = cst
            igrp = grpu[indu]
            grpu = np.delete(grpu, indu)
            cst = grpu.copy()
            dyn = grpu.copy()
        elif direction == 'exhaustive':
            flist = ind
            k = nfeat + 1

        if direction != 'exhaustive':
            k += 1
            if display:
                print('Step', k, '- ', grpwrd, dirwrd,
                      ':', igrp, '|| DA =', new_score)
            combi = _seqcombination(cst, dyn, direction, grp)

            if cwi:
                old_score = new_score
            flist = [j for k in flist for j in grp['g'+str(k)]]

    return flist


def _seqcombination(cst, dyn, direction, grp):
    """Generate combi for forward/backward/exhaustive sequence.

    cst : list containing the index of all the features
    dyn : features to add or remove
    direction : direction of the sequence
    grp : group features
    """
    if direction == 'forward':
        combi = [cst + [y]
                 for y in dyn if not list(set(cst).intersection([y]))]
    elif direction == 'backward':
        combi = [list(set(cst).difference([x])) for x in dyn]
    elif direction == 'exhaustive':
        combi = [list(k) for i in range(1, len(cst)+1)
                 for k in combinations(cst, i)]

    return [[j for i in k for j in grp['g'+str(i)]] for k in combi]


def clfcombi(clfObj, x, combi, n_jobs=1):
    """Classify each combi and return scores and the best one location.
    """
    # - Classify each combination :
    score = np.array(Parallel(n_jobs=n_jobs)(
        delayed(_clfcombi)(clfObj, x[..., k]) for k in combi))

    return score.max(), score.argmax(), score


def _clfcombi(clfObj, x):
    """Classify one combination
    """
    if x.shape[1] != 0:
        return np.mean(clfObj.fit(x, n_jobs=1, mf=True)[0])
    else:
        return 0
