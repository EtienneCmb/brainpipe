import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from brainpipe.clf.utils._sequence import *
from brainpipe.clf.utils._mf import *

from brainpipe.clf._classification import defClf, defCv
from brainpipe.statistics import bino_da2p


__all__ = ['mf', 'sequence']


class mf(object):

    """Compute multi-features (mf) with the possibility of using methods in
    cascade and run the mf on particular groups.

    Args:
        y: array-like
            The target variable to try to predict in the case of
            supervised learning

    Kargs:
        Id: string, optional, [def: '0']
            Use this parameter to define a cascade of methods. Here is the list
            of the current implemented methods:

            * '0': No selection. All the features are used
            * '1': Select <p significant features using either a binomial law or permutations
            * '2': select 'nbest' features
            * '3': use 'forward'/'backward'/'exhaustive' to  select features

            If is for example Id='12', the program will first select significants
            features, then, on this subset, it will find the nbest features. All
            the methods can be serialized.

        p: float, optional, [def: 0.05]
            The pvalue to select features for the Id='1' method

        n_perm: integer, optional, [def: 200]
            Number of permutations for the Id='1' method

        stat : string, optional, [def: 'bino']
            Statisical test for selecting features for the Id='1' method. Choose
            between:

            * 'bino': binomial test
            * 'label_rnd': randomly shuffle the labels
            * 'full_rnd': randomly shuffle the whole array x
            * 'intra_rnd': randomly shuffle x inside each class and each feature

            Methods 2, 3 and 4 are based on permutations. The method 2 and 3
            should provide similar results. But 4 should be more conservative.

        threshold: integer/float, optional, [def: None]
            Define a decoding accuracy for thresholding features. equivalent to the
            p parameter.

        nbest: integer, optional, [def: 10]
            For the Id='2', use this parameter to control the number of features
            to select. If nbest=10, the program will classify each feature and then
            select the 10 best of them.

        direction: string, optional, [def: 'forward']
            For the method Id='3', use:

            * 'forward'
            * 'backward'
            * 'exhaustive'

            to control the direction of the feature selection.

        occurence: string, optional, [def: 'i%']
            Use this parameter to modify the way of visualizing the occurence of
            each feature apparition. Choose between :

            * '%' : in percentage (float)
            * 'i%' : in integer percentage (int)
            * 'c' : count (= number of times the feature has been selected)

        clfIn // clfOut : dictionnary, optional
            Use those dictionnaries to control the classifier to use.

                * clfIn : the classifier use for the training [def: LDA]
                * clfOut : the classifier use for the testing [def: LDA]

            To have more controlable classifiers, see the defClf() class inside
            the classification module.

        cvIn // cvOut : dictionnary, optional
            Use those dictionnaries to control the cross-validations (cv) to use.

            * cvIn : the cv to use for the training [def: 1 time stratified
                     10-folds]
            * cvOut : the more extern cv, to separate training and testing and
                      to avoid over-fitting [def: 10 time stratified 10-folds]

            To have more controlable cross-validation, see the defCv() class inside
            the classification module.

    Return
        A multi-features object with a fit() method to apply to model to the data.

    """

    def __init__(self, y, Id='0', p=0.05, n_perm=200, stat='bino',
                 threshold=None, nbest=10, direction='forward', occurence='i%',
                 clfIn={'clf': 'lda'}, clfOut={'clf': 'lda'},
                 cvIn={'cvtype': 'skfold', 'n_folds': 10, 'rep': 1},
                 cvOut={'cvtype': 'skfold', 'n_folds': 10, 'rep': 10}):
        self._Id = Id
        self._stat = stat
        self._y = np.ravel(y)
        if threshold is not None:
            p = bino_da2p(y, threshold)
        self._p = p
        self._nbest = nbest
        self._n_perm = n_perm
        self._direction = direction
        self._threshold = threshold
        self._occurence = occurence
        self._clfIn = clfIn
        self._clfOut = clfOut
        self._cvIn = cvIn
        self._cvOut = cvOut
        self.setup = {'Id': Id, 'p': p, 'n_perm': n_perm, 'stat': stat,
                      'nbest': nbest, 'threshold': threshold,
                      'direction': direction, 'occurence': occurence,
                      'clfIn': defClf(y, **clfIn).lgStr,
                      'clfOut': defClf(y, **clfOut).lgStr,
                      'cvIn': defCv(y, **cvIn).lgStr,
                      'cvOut': defCv(y, **cvOut).lgStr}

    def str(self):
        pass

    def fit(self, x, grp=[], center=False, combine=False, grpas='single',
            grplen=[], display=True, n_jobs=-1):
        """Run the model on the matrix of features x

        Args:
            x: array-like
                The features. Dimension [n trials x n features]

        Kargs:
            grp: list of strings, optional, [def: []]
                Group features by using a list of strings. The length of grp must
                be the same as the number of features. If grp is not empty, the
                program will run the feature selection inside each group.

            center: optional, bool, [def: False]
                Normalize fatures with a zero mean by substracting then dividing
                by the mean. The center parameter should be set to True if the
                classifier is a svm.

            combine: boolean, optional, [def: False]
                If a group of features is specified using the grp parameter,
                combine give the access of combining or not groups. For example,
                if there is three unique groups, combining them will compute the mf
                model on each combination : [[1],[2],[3],[1,2],[1,3],[2,3],[1,2,3]]

            grpas: string, optional, [def: 'single']
                Specify how to consider features inside each group. If the
                parameter grpas ("group as") is:

                    * 'single': inside each combination of group, the features are considered as independant.
                    * 'group': inside each combination of group, the features are going to be associated. So the mf model will search to add a one by one feature, but it will add groups of features.

            grplen: list, optional, [def: []]
                Control the number of combinations by specifying the number of
                elements to associate. If there is three unique groups, all
                possible combinations are : [[1],[2],[3],[1,2],[1,3],[2,3],[1,2,3]]
                but if grplen is specify, for example grplen=[1,3], this will
                consider combinations of groups only with a length of 1 and 3 and
                remove combinations of 2 elements: [[1],[2],[3],[1,2,3]]

            display: boolean, optional, [def: True]
                Display informations for each step of the mf selection. If n_jobs
                is -1, it is advise to set the display to False.

            n_jobs: integer, optional, [def: -1]
                Control the number of jobs to cumpute the decoding accuracy. If
                n_jobs=-1, all the jobs are used.

        Returns:
            da: list
                The decoding accuracy (da) for each group with the selected number
                of repetitions, which by default is set to 10 (see : cvOut // rep)

            prob: list
                The appearance probability of each feature. The size of prob is the
                same as da.

            groupinfo: pandas Dataframe
                Dataframe to resume the mf feature selection.

        """
        # - Check and get elements sizes:
        y = self._y
        if x.shape[0] != len(y):
            x = x.T
        y = np.ravel(y)
        ntrial, nfeat = x.shape

        # Normalize features :
        if center:
            x_m = np.tile(np.mean(x, 0), (x.shape[0], 1))
            x = (x-x_m)/x_m

        # Combine groups :
        grp_c = combineGroups(grp, nfeat, combine, grpas=grpas, grplen=grplen)
        grp_name, grp_idx = list(grp_c['name']), list(grp_c['idx'])
        ngrp = len(grp_name)

        # - Run the MF model for each combinaition:
        mfdata = Parallel(n_jobs=n_jobs)(delayed(_fit)(
            x, y, grp_c, k, combine, display, self) for k in range(len(grp_c)))

        # Get data & complete the Dataframe :
        da, prob, MFstr = zip(*mfdata)
        self.MFstr = MFstr[-1]
        grp_c['da'], grp_c['occurrence'] = [sum(k) / len(k) for k in da], prob

        return da, prob, grp_c


def _fit(x, y, grp_c, K, combine, display, self):
    """Run the mf for each combination. Subfunction for parallel
    computing
    """
    # Display informations :
    if display:
        print('=> Group : ' + grp_c['name'].iloc[K] +
              ' ('+str(K+1)+'/'+str(len(grp_c))+')', end='\r')

    # Check mf inputs :
    idxgrp, group = grp_c['idx'].iloc[K], grp_c['group'].iloc[K]
    group = [j for k, i in enumerate(group)
             for j, l in enumerate(set(group)) if i == l]
    if combine and len(set(group)) == 1:
        Id_t = '0'
    else:
        Id_t = self._Id

    return _mf(x[:, idxgrp], Id_t, group, self, display, self._occurence)


class sequence(object):

    """Define a sequence object and run a forward/backward/exhaustive feature
    selection.

    Parameters
    ----------
    clfObj : classify object
        Define an object with the classify function. This object will include
        a classifier an potentially a cross-validation

    direction : string, optional, [def: 'forward']
        Use 'forward', 'backward' or 'exhaustive'

    display : boolean, optional, [def: True]
        At each step, print the evolution of the features selection.

    cwi : boolean, optional, [def: False]
        cwi stand for "continue while increasing". If this parameter is set to
        True, the feature selection will stop if a feature decrease the
        performance.
    """

    def __init__(self, clfObj, direction, display=True, cwi=False):
        self._clf = clfObj
        self._direction = direction
        self._display = display
        self._cwi = cwi

    def fit(self, x, y, grp=[], n_jobs=-1):
        """Run the sequence and get the selected features.

        x : array-like
            The data to fit. x should have a shape of (n trials x n features)

        y : array-like
            The target variable to try to predict in the case of
            supervised learning.

        grp : list/array, optionnal, [def: []]
            The grp parameter can be used to define groups of features. If grp
            is not an empty list, the feature will not be applied on single
            features but on group of features.

        n_jobs : integer, optional, default : 1
            The number of CPUs to use to do the computation. -1 means all CPUs
        """
        return _sequence(x, y, self._clf, self._direction, grp, n_jobs,
                         self._display, self._cwi)

    def fit_cv(self):
        """TO ADD
        """
        pass
