import numpy as n
import pandas as pd
from itertools import combinations

from brainpipe.clf.utils.methods import *
from brainpipe.clf._classification import classify, defCv, defClf


__all__ = ['_mf', 'combineGroups']


def _mf(x, Id, grp, self, display, probOccur):
    """cross-validation multifeatures
    """

    # Unpack arguments :
    y = self._y
    cvOut = defCv(y, **self._cvOut).cvr
    clfOut = defClf(y, **self._clfOut)

    # Repetitions loop :
    idxCvOut, da = [], []
    for k, i in enumerate(cvOut):

        predCv, yTestCv = [], []
        for train_index, test_index in i.split(x,y):

            # Get training and testing sets:
            xTrain, xTest, yTrain, yTest = x[train_index, :], x[
                test_index, :], y[train_index], y[test_index]

            # Define the classification object :
            clfObj = classify(yTrain, clf=defClf(yTrain, **self._clfIn),
                              cvtype=defCv(yTrain, **self._cvIn))

            # Get the MF model:
            MFmeth, MFstr = Id2methods(Id, yTrain, clfObj, p=self._p,
                                       display=display,
                                       direction=self._direction,
                                       nbest=self._nbest,
                                       stat=self._stat, n_perm=self._n_perm)

            # Apply the MF model:
            xCv, idxCvIn, grpIn, MFstrCascade = applyMethods(
                MFmeth, MFstr, xTrain, grp)

            # Select the xTest features:
            xTest = xTest[:, idxCvIn]

            # Classify:
            if xCv.size:
                predCv.extend(clfOut.fit(xCv, yTrain).predict(xTest))
            else:
                predCv.extend(n.array([None] * len(test_index)))

            # Keep info:
            idxCvOut.extend(idxCvIn), yTestCv.extend(yTest)

        # Get the decoding accuracy :
        da.append(100 * sum([1 if predCv[k] == yTestCv[k]
                             else 0 for k in range(0, len(predCv))
                             ]) / len(predCv))

    prob = occurProba(idxCvOut, list(range(x.shape[1])), kind=probOccur)

    return da, prob, MFstrCascade

####################################################################
# - Compute the occurence probability :
####################################################################


def occurProba(x, ref, kind='i%'):
    """Get the probability of occurence of a feature.
    -> '%' : in percentage (float)
    -> 'i%' : in integer percentage (int)
    -> 'c' : count (= number of times the feature has been selected)
    """
    if kind == '%':
        return [100 * x.count(k) / len(x) if x else 0 for k in ref]
    if kind == 'i%':
        return [round(100 * x.count(k) / len(x)) if x else 0 for k in ref]
    if kind == 'c':
        return [x.count(k) if x else 0 for k in ref]


####################################################################
# - Generate combinaition of features and get info :
####################################################################


def combineGroups(grp, nfeat, combine=True, grpas='single', grplen=[]):
    """Combine or not, groups of features. The grp parameter
    should be a list of string. This function return a dataframe
    which contain the diffrents groups and their associating index

    grp = list of names for each features
    combine = combine or not groups
    grpas = consider groups as single//multi-features
    """
    # If the grp is not empty :
    if grp:
        # Group features of grp in a Dataframe :
        gpFeat = pd.DataFrame({'group': grp})
        gp = gpFeat.groupby(['group'])

        # Get the unique index of each group :
        seen = set()
        seen_add = seen.add
        unqOrder = [k for k in grp if not (k in seen or seen_add(k))]
        idxGp = [gp.groups[k] for k in unqOrder]

        # Combine or not the group of index :
        if combine:
            if not grplen:
                grplen = list(n.arange(1, len(idxGp)+1))
        else:
            if not grplen:
                grplen = list(n.arange(1, 2))
        idxComb = _getCombi(idxGp, grplen=grplen, kind=int)
        featNameComb = _getCombi(unqOrder, grplen=grplen, kind=str, sep=' + ')

        # If 'single', each group will have its own number :
        if grpas == 'single':
            group = [list(n.arange(len(i))) for k, i in enumerate(idxComb)]
        elif grpas == 'group':
            group = _getCombi([[k]*len(i) for k, i in enumerate(idxGp)],
                              grplen=grplen, kind=int)

        return pd.DataFrame({'name': featNameComb, 'idx': idxComb,
                             'group': group},
                            columns=['name', 'idx', 'group'])
    else:
        return pd.DataFrame({'name': 'none', 'idx': [list(n.arange(nfeat))],
                             'group': [list(n.arange(nfeat))]}, columns=[
                             'name', 'idx', 'group'])


####################################################################
# - Combine string and int elements :
####################################################################


def _getCombi(stuff, grplen=[], kind=int, sep=''):
    """Function which return combinations of string/integer objects.
    -> (start, stop) : number of combine elements from initial list
    """
    allComb = []
    if not grplen:
        grplen = list(n.arange(1, len(idxGp)+1))
    for L in grplen:
        for subset in combinations(stuff, L):
            if kind == str:
                allComb.extend([sep.join(subset)])
            elif kind == int:
                t = []
                [t.extend(k) for k in subset]
                allComb.extend([t])
    return allComb
