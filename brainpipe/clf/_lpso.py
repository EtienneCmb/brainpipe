import numpy as np
import pandas as pd
from itertools import product

from sklearn.model_selection import LeavePGroupsOut
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed

from brainpipe.clf.utils._classif import _info
from brainpipe.clf.utils._classif import *
from brainpipe.tools import uorderlst
from brainpipe.clf.utils._clfplt import clfplt
from brainpipe.statistics import (bino_da2p, bino_p2da,
                                  perm_2pvalue, permIntraClass)

__all__ = ['LeavePSubjectOut']


class LeavePSubjectOut(clfplt):

    """Leave p-subbject out cross-validation

    Args:
        y: list
            List of label vectors for each subject.

        nsuj: int
            Number of subjects

    Kargs:
        pout: int
            Number of subjects to leave out for testing. If pout=1,
            this is a leave one-subject out

        clf: int / string / classifier object, optional, [def: 0]
            Define a classifier. If clf is an integer or a string, the
            classifier will be defined inside classify. Otherwise, it is
            possible to define a classifier before with defClf and past it in clf.

        clfArg: supplementar arguments
            This dictionnary can be used to define supplementar arguments for the
            classifier. See the documentation of defClf.
    """

    def __init__(self, y, nsuj, pout=1, clf='lda', **clfArg):
        self._y = y
        self._ry = np.ravel(np.concatenate(y))
        self._nsuj = nsuj
        self._pout = pout
        # Manage cross-validation:
        self._cv = LeavePGroupsOut(pout)
        self._cv.shStr = 'Leave '+str(pout)+' subjects out'
        self._cv.lgStr = self._cv.shStr
        self._cv.rep = 1
        self._cv.y = y[0]
        # Manage classifier :
        if isinstance(clf, (int, str)):
            clf = defClf(self._ry, clf=clf, **clfArg)
        self._clf = clf
        # Manage info:
        self._updatestring()
        # Stat tools:
        self.stat = clfstat()
        

    def fit(self, x, mf=False, center=False, grp=None,
            method='bino', n_perm=200, rndstate=0, n_jobs=-1):
        """Apply the classification and cross-validation objects to the array x.

        Args:
            x: list
                List of dataset for each subject. All the dataset in the list
                should have the same number of columns but the number of lines
                could be diffrent for each subject and must correspond to the 
                same number of lines each each label vector of y.

        Kargs:
            mf: bool, optional, [def: False]
                If mf=False, the returned decoding accuracy (da) will have a
                shape of (1, rep) where rep, is the number of repetitions.
                This mean that all the features are used together. If mf=True,
                da.shape = (M, rep), where M is the number of columns of x.

            center: optional, bool, [def: False]
                Normalize fatures with a zero mean by substracting then dividing
                by the mean. The center parameter should be set to True if the
                classifier is a svm.

            grp: array, optional, [def: None]
                If mf=True, the grp parameter allow to define group of features.
                If x.shape = (N, 5) and grp=np.array([0,0,1,2,1]), this mean that
                3 groups of features will be considered : (0,1,2)

            method: string, optional, [def: 'bino']
                Four methods are implemented to test the statistical significiance
                of the decoding accuracy :

                    - 'bino': binomial test
                    - 'label_rnd': randomly shuffle the labels

                Methods 2 and 3 are based on permutations. They should provide
                similar results. But 4 should be more conservative.

            n_perm: integer, optional, [def: 200]
                Number of permutations for the methods 2, 3 and 4

            rndstate: integer, optional, [def: 0]
                Fix the random state of the machine. Usefull to reproduce results.

            n_jobs: integer, optional, [def: -1]
                Control the number of jobs to cumpute the decoding accuracy. If
                n_jobs = -1, all the jobs are used.

        Return:
            da: array
                The decoding accuracy of shape n_repetitions x n_features

            pvalue: array
                Array of associated pvalue of shape n_features

            daPerm: array
                Array of all the decodings obtained for each permutations of shape
                n_perm x n_features

        .. rubric:: Footnotes
        .. [#f8] `Ojala and Garriga, 2010 <http://www.jmlr.org/papers/volume11/ojala10a/ojala10a.pdf>`_
        .. [#f9] `Combrisson and Jerbi, 2015 <http://www.ncbi.nlm.nih.gov/pubmed/25596422/>`_
        """
        # Check x, y:
        xbk = x.copy()
        x, y, train, test = _checkXY(x, self._y, mf, grp, center, self)
        nsuj, nfeat = x.shape

        # Run classification:
        da, ytrue, ypred = _fit(x, y, train, test, self, n_jobs)

        # Get statistics:
        # -------------------------------------------------------------
        # Binomial :
        # -------------------------------------------------------------
        if method == 'bino':
            pvalue = bino_da2p(self._ry, da)
            daperm = None
            pperm = None
        # -------------------------------------------------------------
        # Permutations :
        # -------------------------------------------------------------
        # -> Shuffle the labels :
        elif method == 'label_rnd':
            # Get length of each label vector :
            l = [0] + [len(k) for k in self._y]
            # Get consecutive sum elements :
            cm = np.cumsum(l)
            # Build a slice list containing shape of each label vector :
            sl = [slice(cm[k], cm[k+1]) for k in range(len(cm)-1)]
            # Define intra-shuffle :
            def _intraSh():
                # Ravel full y :
                yc = np.concatenate(self._y).ravel()
                # Permute yc :
                yc_perm = np.random.permutation(yc)
                # Reshape yc_perm to fit to each label shape :
                return [yc_perm[k] for k in sl]
            # Build shuffle y :
            y_sh = [_checkXY(xbk, _intraSh(),
                             mf, grp, center, self)[1] for k in range(n_perm)]
            cvs = Parallel(n_jobs=n_jobs)(delayed(_fit)(
                    x, y_sh[k], train, test, self, 1)
                    for k in range(n_perm))

            # Reconstruct daperm and get the associated p-value:
            daperm, _, _ = zip(*cvs)
            daperm = np.array(daperm).reshape(n_perm, nfeat)
            pvalue = perm_2pvalue(da, daperm, n_perm, tail=1)
            pperm = pvalue

        else:
            raise ValueError('No statistical method '+method+' found')

        # Try to get featinfo:
        try:
            if grp is not None:
                grp = uorderlst(grp)
            else:
                grp = np.arange(nfeat)
            self.info.featinfo = self.info._featinfo(self._clf, self._cv,
                                                     da[:, np.newaxis], grp=grp,
                                                     pperm=pperm)
        except:
            pass

        return da, pvalue, daperm

    def change_clf(self, clf='lda', **clfArg):
        """Change the classifier
        """
        if isinstance(clf, (int, str)):
            clf = defClf(self._ry, clf=clf, **clfArg)
        self._clf = clf
        tempinfo = _info(np.ravel(self._ry), self._cv,
                                  self._clf)._clfinfo(self._cv, self._clf)

    def _updatestring(self):
        """Update info
        """
        try:
            self.info = _info(np.ravel(self._ry), self._cv, self._clf)
        except:
            pass

def _fit(x, y, train, test, self, n_jobs):
    """Sub fit function
    """
    nsuj, nfeat = x.shape
    iteract = product(range(nfeat), zip(train, test))
    ya = Parallel(n_jobs=n_jobs)(delayed(_subfit)(
            np.concatenate(tuple(x[i].iloc[k[0]])),
            np.concatenate(tuple(x[i].iloc[k[1]])),
            np.concatenate(tuple(y[0].iloc[k[0]])),
            np.concatenate(tuple(y[0].iloc[k[1]])),
            self) for i, k in iteract)
    # Re-arrange ypred and ytrue:
    ypred, ytrue = zip(*ya)
    ypred = [np.concatenate(tuple(k)) for k in np.split(np.array(ypred), nfeat)]
    ytrue = [np.concatenate(tuple(k)) for k in np.split(np.array(ytrue), nfeat)]
    da = np.ravel([100*accuracy_score(ytrue[k], ypred[k]) for k in range(nfeat)])
    return da, ytrue, ypred

def _subfit(xtrain, xtest, ytrain, ytest, self):
    """Sub sub-fitting function
    """
    # Check size :
    if xtrain.ndim == 1:
        xtrain = xtrain[:, np.newaxis]
    if xtest.ndim == 1:
        xtest = xtest[:, np.newaxis]
    nfeat = xtrain.shape[1]

    # Train & classify:
    self._clf.fit(xtrain, ytrain)
    return self._clf.predict(xtest), ytest

def _checkXY(x, y, mf, grp, center, self):
    # Size checking:
    if not all([k.shape[1]==x[0].shape[1] for k in x]):
        raise ValueError('All features across subjects should have '
                         'the same number of features')

    # Center data:
    if center:
        x = [(k-np.tile(k.mean(0), (k.shape[0], 1)))/k.mean(0) for k in x]

    # Manage MF:
    if grp is not None:
        mf = True
        grp = np.ravel(grp)
        if not all([k.shape[1]==len(grp) for k in x]):
            raise ValueError('The length of the grp parameter must be equal '
                             'to the number of features for each subject.')
    
    if mf:
        if grp is None:
            x = pd.DataFrame([[k] for k in x])
        else:
            ugrp = uorderlst(grp)
            x = pd.DataFrame([[k[:, np.where(grp == i)[0]] for i in ugrp] for k in x])
    else:
        x = pd.DataFrame([np.ndarray.tolist(k.T) for k in x])

    # Create training and testing set:
    train, test = [], []
    iteract = self._cv.split(np.random.rand(self._nsuj), np.arange(self._nsuj), np.arange(self._nsuj))
    for training, testing in iteract:
        train.append(list(training))
        test.append(list(testing))
    y = pd.DataFrame([[k] for k in y])
    return x, y, train, test

def dfshuffle(df, axis=0, rnd=0):
    """Shuffle dataframe
    """
    rnd = np.random.RandomState(rnd)
    df = df.copy()
    df.apply(rnd.shuffle, axis=axis)
    return df