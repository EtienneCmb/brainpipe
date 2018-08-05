import numpy as np

from ._sequence import *

__all__ = ['Id2methods', 'applyMethods']


####################################################################
# - Select and return a mf model :
####################################################################

def Id2methods(Id, y, clf, p=0.05, display=False, direction='forward',
               nbest=10, stat='bino', n_perm=200):
    """Transform a Id to a list of methods

    '0': No selectionp. All the features are used
    '1': Select <p significant features using either a binomial law
         or permutations
    '2': select 'nbest' features
    '3': use 'forward'/'backward'/'exhaustive' to  select features
    """

    # Define all the methods :
    def submeth(Idx):

        # Select all features
        if Idx == '0':
            def MFmeth(x, GRP):
                return select_all(x)
            StrMeth = 'SelectAll'

        # use a binomial law to select features
        if Idx == '1':
            def MFmeth(x, GRP):
                return select_stat(x, y, clf, stat, GRP, p, n_perm)

            if stat == 'bino':
                StrMeth = 'Binomial selection at p<'+str(p)
            elif stat.lower().find('_rnd')+1:
                StrMeth = 'Permutations ('+stat+') selection at p<'+str(p)

        # nbest features
        if Idx == '2':
            def MFmeth(x, GRP):
                return select_nbest(x, y, clf, nbest, GRP)
            StrMeth = str(nbest)+' best features'

        # use 'forward'/'backward'/'exhaustive'to  select features
        if Idx == '3':

            def MFmeth(x, GRP):
                return _sequence(x, y, clf, direction, GRP, 1, display, False)
            StrMeth = direction+' feature selection'

        return MFmeth, StrMeth

    # Define a list containing the methods:
    return [submeth(k)[0] for k in Id], [submeth(k)[1] for k in Id]


####################################################################
# - Apply the selected model to the dataset :
####################################################################


def applyMethods(MFmeth, MFstr, x, grp):
    """Apply a list of method to the features x. MFstr contain the name of each
    method"""

    idx = list(np.arange(x.shape[1]))

    # Each method find features and return a reduce set:
    def findAndSelect(meth, x, grp, idxO):
        idxN = meth(x, grp)
        return x[:, idxN], [grp[k] for k in idxN], [idxO[k] for k in idxN]

    # Apply each method and get the final set of features:
    for k in MFmeth:
        if not x.size or not idx:
            break
        x, grp, idx = findAndSelect(k, x, grp, idx)

    # String of method application:
    MFstrCascade = ' => '.join(MFstr)

    return x, idx, grp, MFstrCascade


####################################################################
# - MF methods :
####################################################################


def select_all(x):
    """Select and return idx of every features"""
    return list(np.arange(x.shape[1]))


def select_stat(x, y, clf, meth, grp, p, n_perm):
    """Select and return idx of significant features"""
    _, pvalue, _ = clf.fit(x, mf=True, grp=grp, method=meth,
                           n_perm=n_perm, n_jobs=1)
    return grp2idx(grp, [k for k, i in enumerate(pvalue) if i <= p])


def select_nbest(x, y, clf, nbest, grp):
    """Select nbest features"""
    da = np.mean(clf.fit(x, mf=True, grp=grp, n_jobs=1)[0], 1)
    return grp2idx(grp, list(np.ravel(da.T).argsort()[-nbest:][::-1]))


def grp2idx(grp, idx):
    """This function return the index of a group list"""
    return [k for k, i in enumerate(grp) for j, l in enumerate(idx) if i == l]
