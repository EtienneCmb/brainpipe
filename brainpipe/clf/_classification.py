import numpy as np

from sklearn.base import clone
from sklearn.metrics import accuracy_score, confusion_matrix
from joblib import Parallel, delayed

from brainpipe.statistics import (bino_da2p, bino_p2da,
                                  perm_2pvalue, permIntraClass)
from brainpipe.tools import groupInList, list2index, uorderlst
from brainpipe.sys.tools import adaptsize
from brainpipe.clf.utils._classif import *
from brainpipe.clf.utils._clfplt import clfplt
import matplotlib.pyplot as plt

from itertools import product
import pandas as pd

__all__ = ['classify',
           'defClf',
           'defCv',
           'defVoting',
           'generalization'
           ]

class classify(_classification, clfplt):
    """Define a classification object and apply to classify data.
    This class can be consider as a centralization of scikit-learn
    tools, with a few more options.

    To classify data, two objects are necessary :
    - A classifier object (lda, svm, knn...)
    - A cross-validation object which is used to validate a classification
    performance.
    This two objects can either be defined before the classify object with
    defCv and defClf, or they can be directly defined inside the classify
    class.

    Args:
        y: array
            The vector label

    Kwargs:
        clf: int / string / classifier object, optional, [def: 0]
            Define a classifier. If clf is an integer or a string, the
            classifier will be defined inside classify. Otherwise, it is
            possible to define a classifier before with defClf and past it in clf.

        cvtype: string / cross-validation object, optional, [def: 'skfold']
            Define a cross-validation. If cvtype is a string, the
            cross-validation will be defined inside classify. Otherwise, it is
            possible to define a cross-validation before with defCv and past it
            in cvtype.

        clfArg: dictionnary, optional, [def: {}]
            This dictionnary can be used to define supplementar arguments for the
            classifier. See the documentation of defClf.

        cvArg: dictionnary, optional, [def: {}]
            This dictionnary can be used to define supplementar arguments for the
            cross-validation. See the documentation of defCv.

    Example:

            >>> # 1) Define a classifier and a cross-validation before classify():
            >>> # Define a 50 times 5-folds cross-validation :
            >>> cv = defCv(y, cvtype='kfold', rep=50, n_folds=5)
            >>> # Define a Random Forest with 200 trees :
            >>> clf = defClf(y, clf='rf', n_tree=200, random_state=100)
            >>> # Past the two objects inside classify :
            >>> clfObj = classify(y, clf=clf, cvtype=cv)

            >>> # 2) Define a classifier and a cross-validation inside classify():
            >>> clfObj = classify(y, clf = 'rf', cvtype = 'kfold',
            >>>        clfArg = {'n_tree':200, 'random_state':100},
            >>>                  cvArg = {'rep':50, 'n_folds':5})
            >>> # 1) and 2) are equivalent. Then use clfObj.fit() to classify data.
    """

    def __str__(self):
        return self.lgStr

    def fit(self, x, mf=False, center=False, grp=None,
            method='bino', n_perm=200, rndstate=0, n_jobs=-1):
        """Apply the classification and cross-validation objects to the array x.

        Args:
            x: array
                Data to classify. Consider that x.shape = (N, M), N is the number
                of trials (which should be the length of y). M, the number of
                colums, is a supplementar dimension for classifying data. If M = 1,
                the data is consider as a single feature. If M > 1, use the
                parameter mf to say if x should be consider as a single feature
                (mf=False) or multi-features (mf=True)

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
                    - 'full_rnd': randomly shuffle the whole array x
                    - 'intra_rnd': randomly shuffle x inside each class and each feature

                Methods 2, 3 and 4 are based on permutations. The method 2 and 3
                should provide similar results. But 4 should be more conservative.

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
        # Get the true decoding accuracy:
        da, x, y, self._ytrue, self._ypred = _fit(x, self._y, self._clf, self._cv.cvr,
                                                  mf, grp, center, n_jobs)
        nfeat = len(x)
        rndstate = np.random.RandomState(rndstate)
        score = np.array([np.mean(k) for k in da])

        # Get statistics:
        # -------------------------------------------------------------
        # Binomial :
        # -------------------------------------------------------------
        if method == 'bino':
            pvalue = bino_da2p(y, score)
            daPerm = None
            pperm = None
        # -------------------------------------------------------------
        # Permutations :
        # -------------------------------------------------------------
        elif method.lower().find('_rnd')+1:

            # Generate idx tricks :
            iteract = product(range(n_perm), range(nfeat))

            # -> Shuffle the labels :
            if method == 'label_rnd':
                y_sh = [rndstate.permutation(y) for k in range(n_perm)]
                cvs = Parallel(n_jobs=n_jobs)(delayed(_cvscore)(
                        x[k], y_sh[i], clone(self._clf), self._cv.cvr[0])
                        for i, k in iteract)

            # -> Full randomization :
            elif method == 'full_rnd':
                cvs = Parallel(n_jobs=n_jobs)(delayed(_cvscore)(
                        rndstate.permutation(x[k]), y, clone(self._clf),
                        self._cv.cvr[0]) for i, k in iteract)

            # -> Shuffle intra-class :
            elif method == 'intra_rnd':
                cvs = Parallel(n_jobs=n_jobs)(delayed(_cvscore)(
                        x[k][permIntraClass(y, rnd=i), :], y, clone(self._clf),
                        self._cv.cvr[0]) for i, k in iteract)

            # Reconstruct daPerm and get the associated p-value:
            daPerm, _, _ = zip(*cvs)
            daPerm = np.array(daPerm).reshape(n_perm, nfeat)
            pvalue = perm_2pvalue(score, daPerm, n_perm, tail=1)
            pperm = pvalue

        else:
            raise ValueError('No statistical method '+method+' found')

        # Get features informations:
        try:
            if grp is not None:
                grp = uorderlst(grp)
            self.info.featinfo = self.info._featinfo(self._clf, self._cv, da,
                                                     grp=grp, pperm=pperm)
        except:
            pass

        return da.T, pvalue, daPerm

    def cm(self, normalize=True):
        """Get the confusion matrix of each feature.

        Kargs:
            normalize: bool, optional, [def: True]
                Normalize or not the confusion matrix

            update: bool, optional, [def: True]
                If update is True, the data will be re-classified. But, if update
                is set to False, and if the methods .fit() or .fit_stat() have been
                run before, the data won't we re-classified. Instead, the labels
                previously found will be used to get confusion matrix.

        Return:
            CM: array
                Array of confusion matrix of shape (n_features x n_class x n_class)
        """
        # Re-classify data or use the already existing labels :
        if not ((hasattr(self, '_ytrue')) and (hasattr(self, '_ypred'))):
            raise ValueError("No labels found. Please run .fit()")
        else:
            # Get variables and compute confusion matrix:
            y_pred, y_true = self._ypred, self._ytrue
            nfeat, nrep = len(y_true), len(y_true[0])
            CM = [np.mean(np.array([confusion_matrix(y_true[k][i], y_pred[
                k][i]) for i in range(nrep)]), 0) for k in range(nfeat)]

            # Normalize the confusion matrix :
            if normalize:
                CM = [100*k/k.sum(axis=1)[:, np.newaxis] for k in CM]

            return np.array(CM)


def _fit(x, y, clf, cv, mf, grp, center, n_jobs):
    """Sub function for fitting
    """
    # Check the inputs size :
    x, y = checkXY(x, y, mf, grp, center)
    rep, nfeat = len(cv), len(x)

    # Tricks : construct a list of tuple containing the index of
    # (repetitions,features) & loop on it. Optimal for parallel computing :
    claIdx, listRep, listFeat = list2index(rep, nfeat)

    # Run the classification :
    cvs = Parallel(n_jobs=n_jobs)(delayed(_cvscore)(
        x[k[1]], y, clone(clf), cv[k[0]]) for k in claIdx)
    da, y_true, y_pred = zip(*cvs)

    # Reconstruct elements :
    da = np.array(groupInList(da, listFeat))
    y_true = groupInList(y_true, listFeat)
    y_pred = groupInList(y_pred, listFeat)

    return da, x, y, y_true, y_pred


class generalization(object):
    """Generalize the decoding performance of features.
    The generalization consist of training and testing at diffrents
    moments. The use is to see if a feature is consistent and performant
    in diffrents period of time.

    Args:
        time: array/list
            The time vector of dimension npts

        y: array
            The vector label of dimension ntrials

        x: array
            The data to generalize. If x is a 2D array, the dimension of x
            should be (ntrials, npts). If x is 3D array, the third dimension
            is consider as multi-features. This can be usefull to do time
            generalization in multi-features.

    Kargs:
        clf: int / string / classifier object, optional, [def: 0]
            Define a classifier. If clf is an integer or a string, the
            classifier will be defined inside classify. Otherwise, it is
            possible to define a classifier before with defClf and past it in clf.

        cvtype: string / cross-validation object, optional, [def: None]
            Define a cross-validation. If cvtype is None, the diagonal of the
            matrix of decoding accuracy will be set at zero. If cvtype is defined,
            a cross-validation will be performed on the diagonal. If cvtype is a
            string, the cross-validation will be defined inside classify.
            Otherwise, it is possible to define a cross-validation before with
            defCv and past it in cvtype.

        clfArg: dictionnary, optional, [def: {}]
            This dictionnary can be used to define supplementar arguments for the
            classifier. See the documentation of defClf.

        cvArg: dictionnary, optional, [def: {}]
            This dictionnary can be used to define supplementar arguments for the
            cross-validation. See the documentation of defCv.

    Return:
        An array of dimension (npts, npts) containing the decoding accuracy. The y
        axis is the training time and the x axis is the testing time (also known
        as "generalization time")
    """
    def __init__(time, y, x, clf='lda', cvtype=None, clfArg={},
                 cvArg={}):
        pass

    def __new__(self, time, y, x, clf='lda', cvtype=None, clfArg={},
                cvArg={}):

        self.y = np.ravel(y)
        self.time = time

        # Define clf if it's not defined :
        if isinstance(clf, (int, str)):
            clf = defClf(y, clf=clf, **clfArg)
        self.clf = clf

        # Define cv if it's not defined :
        if isinstance(cvtype, str) and (cvtype is not None):
            cvtype = defCv(y, cvtype=cvtype, rep=1, **cvArg)
        self.cv = cvtype
        if isinstance(cvtype, list):
            cvtype = cvtype[0]
        # Check the size of x:
        x = np.atleast_3d(x)
        npts, ntrials = len(time), len(y)
        if x.shape[0] is not npts:
            raise ValueError('First dimension of x must be '+str(npts))
        if x.shape[1] is not ntrials:
            raise ValueError('Second dimension of x must be '+str(ntrials))

        da = np.zeros([npts, npts])
        # Training dimension
        for k in range(npts):
            xx = x[k, ...]
            # Testing dimension
            for i in range(npts):
                xy = x[i, ...]
                # If cv is defined, do a cv on the diagonal
                if (k == i) and (cvtype is not None):
                    da[i, k] = _cvscore(xx, y, clf, self.cv.cvr[0])[0]/100
                # If cv is not defined, let the diagonal at zero
                elif (k == i) and (cvtype is None):
                    pass
                else:
                    da[i, k] = accuracy_score(y, clf.fit(xx, y).predict(xy))
        return 100*da
