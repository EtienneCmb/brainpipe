import numpy as np
import pandas as pd
from pandas import ExcelWriter
from warnings import warn

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.base import clone

from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import (StratifiedKFold, KFold, LeaveOneOut,
                                     StratifiedShuffleSplit, ShuffleSplit,
                                     LeaveOneGroupOut)

from brainpipe.statistics import *
from brainpipe.tools import uorderlst

__all__ = ['_classification',
           '_info',
           '_cvscore',
           'checkXY',
           'defClf',
           'defCv',
           'defVoting',
           'clfstat'
           ]


class _classification(object):

    """Sub-classification class
    """

    def __init__(self, y, clf='lda', cvtype='skfold', clfArg={}, cvArg={}):

        self._y = y
        if not hasattr(clf, 'shStr') or not hasattr(clf, 'lgStr'):
            clf.shStr, clf.lgStr = 'custom', 'Custom classifier'
        self._lgStr = ''
        # Defineclassifier and cross-valdation :
        self._defineClf(clf, **clfArg)
        self._defineCv(cvtype, **cvArg)
        # Get info:
        self.info = _info(y, self._cv, self._clf)
        self._updatestring()
        # Stat tools:
        self.stat = clfstat()


    def __str__(self):
        return self._lgStr

    def _defineClf(self, clf, **clfArg):
        """Sub-function for defining classifier
        """
        if isinstance(clf, (int, str)):
            clf = defClf(self._y, clf=clf, **clfArg)
        self._clf = clf

    def _defineCv(self, cvtype='skfold', **cvArg):
        """Sub-function for defining cross-validation
        """
        if isinstance(cvtype, str):
            cvtype = defCv(self._y, cvtype=cvtype, **cvArg)
        self._cv = cvtype

    def change_clf(self, clf='lda', **clfArg):
        """Change the classifier
        """
        oldclf = self._clf.lgStr
        self._defineClf(clf, **clfArg)
        self._updatestring()
        self.info = _info(self._y, self._cv, self._clf)
        print('Classifier updated from a '+oldclf+' to a '+self._clf.lgStr)

    def change_cv(self, cvtype='skfold', **cvArg):
        oldcv = self._cv.lgStr
        self._defineCv(cvtype, **cvArg)
        self._updatestring()
        self.info = _info(self._y, self._cv, self._clf)
        print('Cross-validation updated from a '+oldcv+' to a '+self._cv.lgStr)

    def _updatestring(self):
        self._shStr = self._clf.shStr +' / '+self._cv.shStr
        self._lgStr = self._clf.lgStr +' / '+self._cv.lgStr


class _info(object):

    """Get info of current classification settings
    """

    def __init__(self, y, cv, clf):
        self.y = y
        self.clfinfo = self._clfinfo(cv, clf)
        self.statinfo = self._statinfo()

    def _clfinfo(self, cv, clf):
        info = pd.DataFrame({'Classifier':clf.lgStr,
                             'Cross-validation':cv.lgStr,
                             'Repetition':[cv.rep],
                             'Class':[list(set(cv.y))],
                             'Chance (theorical, %)':[100/len(set(cv.y))]
                            })
        return info

    def _statinfo(self):
        info = pd.DataFrame({'Class':[list(set(self.y))],
                             'N-class':[len(set(self.y))],
                             'Chance (theorical, %)':[100/len(set(self.y))],
                             'Chance (binomial, %)':[{'p_0.05':bino_p2da(self.y, 0.05),
                                                      'p_0.01':bino_p2da(self.y, 0.01),
                                                      'p_0.001':bino_p2da(self.y, 0.001)
                                                     }]
                            })
        return info

    def _featinfo(self, clf, cv, da, grp=None, pbino=None, pperm=None):
        # Manage input arguments :
        dastd = np.round(100*da.std(axis=1))/100
        dam = da.mean(axis=1)
        if grp is None:
            grp = np.array([str(k) for k in range(len(dam))])
        if pbino is None:
            pbino = bino_da2p(self.y, dam)
        if pperm is None:
            pperm = np.ones((len(dam),))
        array = np.array([np.ravel(dam), np.ravel(dastd), np.ravel(pbino), np.ravel(pperm), np.ravel(grp)]).T

        # Create the dataframe:
        subcol = ['DA (%)', 'STD (+/-)', 'p-values (Binomial)', 'p-values (Permutations)', 'Group']
        str2repeat = clf.shStr+' / '+cv.shStr
        idxtuple = list(zip(*[[str2repeat]*len(subcol), subcol]))
        index = pd.MultiIndex.from_tuples(idxtuple, names=['Settings', 'Results'])
        return pd.DataFrame(array, columns=index)

    def to_excel(self, filename='myfile.xlsx'):
        """Export informations to a excel file

        Kargs:
            filename: string
                Name of the excel file ex: filename='myfile.xlsx'
        """
        writer = ExcelWriter(filename)
        self.clfinfo.to_excel(writer,'Classifier')
        self.statinfo.to_excel(writer,'Statistics')
        try:
            self.featinfo.to_excel(writer,'Features')
        except:
            warn('Informations about features has been ignored. Run fit()')
        writer.save()



def _cvscore(x, y, clf, cvS):
    """Get the decoding accuracy of one cross-validation
    """
    y_pred, y_true = [], []
    iterator = cvS.split(x, y=y)
    for trainidx, testidx in iterator:
        xtrain, xtest = x[trainidx, ...], x[testidx, ...]
        ytrain, ytest = y[trainidx, ...], y[testidx, ...]
        clf.fit(xtrain, ytrain)
        y_pred.extend(clf.predict(xtest))
        y_true.extend(ytest)
    return 100*accuracy_score(y_true, y_pred), y_true, y_pred


def checkXY(x, y, mf, grp, center):
    """Prepare the inputs x and y
    x.shape = (ntrials, nfeat)
    """
    x, y = np.matrix(x), np.ravel(y)
    if x.shape[0] != len(y):
        x = x.T

    # Normalize features :
    if center:
        x_m = np.tile(np.mean(x, 0), (x.shape[0], 1))
        x = (x-x_m)/x_m

    # Group parameter :
    if grp is not None:
        mf = True
        grp = np.ravel(grp)
    if mf:
        if grp is None:
            x = [x]
        elif (grp is not None) and (grp.size == x.shape[1]):
            ugrp = uorderlst(grp)
            x = [np.array(x[:, np.where(grp == k)[0]]) for k in ugrp]
        elif (grp is not None) and (grp.size != x.shape[1]):
            raise ValueError('The grp parameter must have the same size as the'
                             ' number of features ('+str(x.shape[1])+')')
    else:
        x = [np.array(x[:, k]) for k in range(x.shape[1])]

    return x, y




class defClf(object):

    """Choose a classifier and switch easyly between classifiers
    implemented in scikit-learn.

    Args:
        y: array
            The vector label

    clf: int or string, optional, [def: 0]
        Define a classifier. Use either an integer or a string
        Choose between:

            - 0 / 'lda': Linear Discriminant Analysis (LDA)
            - 1 / 'svm': Support Vector Machine (SVM)
            - 2 / 'linearsvm' : Linear SVM
            - 3 / 'nusvm': Nu SVM
            - 4 / 'nb': Naive Bayesian
            - 5 / 'knn': k-Nearest Neighbor
            - 6 / 'rf': Random Forest
            - 7 / 'lr': Logistic Regression
            - 8 / 'qda': Quadratic Discriminant Analysis

    kern: string, optional, [def: 'rbf']
        Kernel of the 'svm' classifier

    n_knn: int, optional, [def: 10]
        Number of neighbors for the 'knn' classifier

    n_tree: int, optional, [def: 100]
        Number of trees for the 'rf' classifier

    Kargs:
        optional arguments. To define other parameters, see the description of
        scikit-learn.

    Return:
        A scikit-learn classification objects with two supplementar arguments :

            - lgStr : long description of the classifier
            - shStr : short description of the classifier
    """

    def __init__(self, y, clf='lda', kern='rbf', n_knn=10, n_tree=100,
                 priors=False, **kwargs):
        pass

    def __str__(self):
        return self.lgStr

    def __new__(self, y, clf='lda', kern='rbf', n_knn=10, n_tree=100,
                priors=False, **kwargs):

        # Use a pre-defined classifier :
        if isinstance(clf, (str, int)):
            # Default value for priors :
            priors = np.array([1/len(np.unique(y))]*len(np.unique(y)))

            if isinstance(clf, str):
                clf = clf.lower()

            # LDA :
            if clf == 'lda' or clf == 0:
                clfObj = LinearDiscriminantAnalysis(
                    priors=priors, **kwargs)
                clfObj.lgStr = 'Linear Discriminant Analysis'
                clfObj.shStr = 'LDA'

            # SVM : ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable
            elif clf == 'svm' or clf == 1:
                clfObj = SVC(kernel=kern, probability=True, **kwargs)
                clfObj.lgStr = 'Support Vector Machine (kernel=' + kern + ')'
                clfObj.shStr = 'SVM-' + kern

            # Linear SVM:
            elif clf == 'linearsvm' or clf == 2:
                clfObj = LinearSVC(**kwargs)
                clfObj.lgStr = 'Linear Support Vector Machine'
                clfObj.shStr = 'LSVM'

            # Nu SVM :
            elif clf == 'nusvm' or clf == 3:
                clfObj = NuSVC(**kwargs)
                clfObj.lgStr = 'Nu Support Vector Machine'
                clfObj.shStr = 'NuSVM'

            # Naive Bayesian :
            elif clf == 'nb' or clf == 4:
                clfObj = GaussianNB(**kwargs)
                clfObj.lgStr = 'Naive Baysian'
                clfObj.shStr = 'NB'

            # KNN :
            elif clf == 'knn' or clf == 5:
                clfObj = KNeighborsClassifier(n_neighbors=n_knn, **kwargs)
                clfObj.lgStr = 'k-Nearest Neighbor (neighbor=' + str(n_knn) + ')'
                clfObj.shStr = 'KNN-' + str(n_knn)

            # Random forest :
            elif clf == 'rf' or clf == 6:
                clfObj = RandomForestClassifier(n_estimators=n_tree, **kwargs)
                clfObj.lgStr = 'Random Forest (tree=' + str(n_tree) + ')'
                clfObj.shStr = 'RF-' + str(n_tree)

            # Logistic regression :
            elif clf == 'lr' or clf == 7:
                clfObj = LogisticRegression(**kwargs)
                clfObj.lgStr = 'Logistic Regression'
                clfObj.shStr = 'LogReg'

            # QDA :
            elif clf == 'qda' or clf == 8:
                clfObj = QuadraticDiscriminantAnalysis(**kwargs)
                clfObj.lgStr = 'Quadratic Discriminant Analysis'
                clfObj.shStr = 'QDA'

            else:
                raise ValueError('No classifier "'+str(clf)+'"" found')

        # Use a custom classifier :
        else:
            clfObj = clf
            clfObj.shStr = 'custom'
            clfObj.lgStr = 'Custom classifier'


        return clfObj



class defVoting(object):

    """Define a voting classifier

    Args:
        y: list/array
            The label vector

    Kargs:
        estimators: list, optional, (def: [])
            A list a pre-build estimators using defClf()

        clfArgs: list, optional, (def: [])
            Instead of sending a list of pre-defined classifiers,
            build estimators directly using a list of dictionnaries
            to pass to the defClf class.

    Return
        clf: a classifier object

    Example:
        >>> # Build two classifiers :
        >>> clf0 = defClf(y, clf='lda')
        >>> clf1 = defClf(y, clf='svm')
        >>> # Build the voting classifier :
        >>> clf = defVoting(y, estimators=[clf0, clf1])
        >>> # Alternatively, thos steps can be done in one line :
        >>> clf = defVoting(y, clfArgs=[{'clf':'lda'}, {'clf':'svm'}])
    """


    def __new__(self, y, estimators=[], clfArgs=[], **kwargs):

        # estimators is not list :
        if not isinstance(estimators, list):
            raise ValueError('estimators must be a list of classifiers define with defClf')
        # estimators is list and not empty :
        elif isinstance(estimators, list) and (len(estimators)):
            clfObj = self._voting(estimators, **kwargs)
        # estimators is list :
        elif not len(estimators) and len(clfArgs):
            estimators = [defClf(y, **k) for k in clfArgs]
            clfObj = self._voting(estimators, **kwargs)

        return clfObj

    def _voting(estimators, **kwargs):
        """Build the classifier
        """
        clfObj = VotingClassifier([(k.shStr, k) for k in estimators], n_jobs=1, **kwargs)
        clfObj.lgStr = ' + '.join([k.lgStr for k in estimators])
        clfObj.shStr = ' + '.join([k.shStr for k in estimators])
        return clfObj


class defCv(object):

    """Choose a cross_validation (CV) and switch easyly between
    CV implemented in scikit-learn.

    Args:
        y: array
            The vector label

    kargs:
        cvtype: string, optional, [def: skfold]
            Define a cross_validation. Choose between :

                - 'skfold': Stratified k-Fold
                - 'kfold': k-fold
                - 'sss': Stratified Shuffle Split
                - 'ss': Shuffle Split
                - 'loo': Leave One Out
                - 'lolo': Leave One Label Out

        n_folds: integer, optional, [def: 10]
            Number of folds

        rndstate: integer, optional, [def: 0]
            Define a random state. Usefull to replicate a result

        rep: integer, optional, [def: 10]
            Number of repetitions

        kwargs: optional arguments. To define other parameters,
        see the description of scikit-learn.

    Return:
        A list of scikit-learn cross-validation objects with two supplementar
        arguments:

            - lgStr: long description of the cross_validation
            - shStr: short description of the cross_validation
    """

    def __init__(self, y, cvtype='skfold', n_folds=10, rndstate=0, rep=10,
                 **kwargs):
        self.y = np.ravel(y)
        self.cvr = [0]*rep
        self.rep = rep
        for k in range(rep):
            self.cvr[k], self.lgStr, self.shStr = _define(y, cvtype=cvtype, n_splits=n_folds,
                                                          rndstate=k, rep=rep, **kwargs)


    def __str__(self):
        return self.lgStr


def _define(y, cvtype='skfold', n_splits=10, rndstate=0, rep=10,
            **kwargs):
    # Stratified k-fold :
    if cvtype == 'skfold':
        cvT = StratifiedKFold(n_splits=n_splits, shuffle=True,
                              random_state=rndstate, **kwargs)
        lgStr = str(rep)+'-times, '+str(n_splits)+' Stratified k-folds'
        shStr = str(rep)+'-rep_'+str(n_splits)+'-'+cvtype

    # k-fold :
    elif cvtype == 'kfold':
        cvT = KFold(n_splits=n_splits, shuffle=True,
                    random_state=rndstate, **kwargs)
        lgStr = str(rep)+'-times, '+str(n_splits)+' k-folds'
        shStr = str(rep)+'-rep_'+str(n_splits)+'-'+cvtype

    # Shuffle stratified k-fold :
    elif cvtype == 'sss':
        cvT = StratifiedShuffleSplit(n_splits=n_splits,
                                     test_size=1/n_splits,
                                     random_state=rndstate, **kwargs)
        lgStr = str(rep)+'-times, test size 1/' + \
            str(n_splits)+' Shuffle Stratified Split'
        shStr = str(rep)+'-rep_'+str(n_splits)+'-'+cvtype

    # Shuffle stratified :
    elif cvtype == 'ss':
        cvT = ShuffleSplit(n_splits=rep, test_size=1/n_splits,
                           random_state=rndstate, **kwargs)
        lgStr = str(rep)+'-times, test size 1/' + \
            str(n_splits)+' Shuffle Stratified'
        shStr = str(rep)+'-rep_'+str(n_splits)+'-'+cvtype

    # Leave One Out :
    elif cvtype == 'loo':
        cvT = LeaveOneOut(len(y))
        lgStr = str(rep)+'-times, Leave One Out'
        shStr = str(rep)+'-rep_'+cvtype

    # Leave One Label Out :
    elif cvtype == 'lolo':
        cvT = LeaveOneGroupOut(y)
        lgStr = str(rep)+'-times, leave One Label Out'
        shStr = str(rep)+'-rep_'+cvtype

    else:
        raise ValueError('No cross-validation "'+cvtype+'"" found')

    return cvT, lgStr, shStr



class clfstat(object):

    """Specific statistic class for classification
    """

    def bino_da2pvalue(self, y, da):
        """"""
        return bino_da2p(y, da)

    def bino_pvalue2da(self, y, p):
        """"""
        return bino_p2da(y, p)

    def perm_da2pvalue(self, da, daperm, multcomp=None, multp=0.05):
        """Get p-values (corrected or not) from decoding accuracies
        computed using permutations

        Args:
            da: array
                Array of true decoding of shape n_features

            daperm:
                Array of decoding for permutations of shape
                n_perm x n_features

        Kargs:
            multcomp: string, optional, [def: None]
                Apply a multiple correction to p-values. Use either:

                - 'maxstat': maximum statistics
                - 'bonferroni': bonferroni correction
                - 'fdr': False Discovery Rate correction

            multp: float, optional, [def: 0.05]
                p-value to use for multiplt comparison

        Return:
            pvalue: array
                Corrected (or not) p-values of shape n_features
        """
        if da.ndim == 2:
            da = da.mean(0)
        da = np.ravel(da)
        # Maximum statistic correction :
        if multcomp == 'maxstat':
            daperm = maxstat(daperm, axis=0)
        pvalue = perm_2pvalue(da, daperm, daperm.shape[0], tail=1)
        # bonferroni, fdr correction:
        if multcomp == 'bonferroni':
            pvalue = bonferroni(pvalue, axis=0)
        if multcomp == 'fdr':
            pvalue = fdr(pvalue, multp)
        return pvalue

    def perm_pvalue2da(self, daperm, p=0.05, maxst=False):
        """Get the decoding accuracy from permutations from
        which you can consider that it's p-significant;

        Args:
            daperm:
                Array of decoding for permutations of shape
                n_perm x n_features

        Kargs:
            p: float, optional, [def: 0.05]
                p-value to search in permutation distribution

            maxst: bool, optional, [def: False]
                Correct permutations with maximum statistics

        Return:
            dapval: float
                The decoding accuracy from which you can consider
                your results as p-significants using permutations
        """
        return perm_pvalue2level(daperm, p=p, maxst=maxst)

clfstat.bino_da2pvalue.__doc__ += bino_da2p.__doc__
clfstat.bino_pvalue2da.__doc__ += bino_p2da.__doc__
