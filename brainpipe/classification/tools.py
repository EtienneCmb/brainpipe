import numpy as np

__all__ = ["data2classif"]


class data2classif(object):

    """Organize and prepare data for classification. This function
    concatenate datasets in order to be usuable by machine learning algorithms.

    Kargs:
        trial_dim: int, optional, [def: 0]
            Specfify where to find the trial dimension inside each array

        feat_dim: int, optional, [def: 1]
            Specfify where to find the feature dimension inside each array

        **condition: list
            Define conditions to separate and features to use. Each condition
            should have the form cond=[feature1, feature2, ..., featureN]

    Return:
        x: array
            The final dataset

        y: list
            List of labels of each condition

    Example:
        >>> # Construct random datasets:
        >>> supdim = 104
        >>> n_feat1, n_feat2 = 60, 64     # Number of features
        >>> n_trial1, n_trial2 = 160, 254 # Number of trials
        >>> cond1_feat1 = np.random.rand( supdim, n_feat1, n_trial1 )
        >>> cond1_feat2 = np.random.rand( supdim, n_feat2, n_trial1 )
        >>> cond2_feat1 = np.random.rand( supdim, n_feat1, n_trial2 )
        >>> cond2_feat2 = np.random.rand( supdim, n_feat2, n_trial2 )
        >>> # Build dataset (features and trials are respectively in dimension 1 and 2)
        >>> feat_dim, trial_dim = 1, 2
        >>> x, y = data2classif(cond1=[cond1_feat1, cond1_feat2],
        >>>                     cond2=[cond2_feat1, cond2_feat2],
        >>>                     trial_dim=2, feat_dim=1)
        >>> # Check final sizes:
        >>> print('x shape = ', x.shape, '\ny length =', len(y))
    """
    def __init__(trial_dim=0, feat_dim=1, **condition):
        pass

    def __new__(self, trial_dim=0, feat_dim=1, featname=None,
                order=None, **condition):
        # Check order parameter :
        if order is not None:
            condLst = order
        else:
            condLst = condition.keys()
        # Shape scanning :
        #   -> First, check number inside each cond :
        allDim = [len(condition[k]) for k in condLst]
        errorMsg = 'Inside each condition, there is not the same number of array {cond}'
        self._2bool(allDim, msg=errorMsg.format(cond=str({
                        k: allDim[i] for i, k in enumerate(condLst)})))

        #   -> Check the trial dimension :
        trialAll = [[i.shape[trial_dim] for i in condition[k]] for k in condLst]
        errorMsg = 'Not the same number of trials (trial_dim='+str(trial_dim)+') in condition {cond} = {nb}'
        [self._2bool(trialAll[i], msg=errorMsg.format(cond=k,
                  nb=str(trialAll[i]))) for i, k in enumerate(condLst)]

        #   -> Check the feature dimension :
        featAll = [[i.shape[feat_dim] for i in condition[k]] for k in condLst]
        errorMsg = 'Not the same number of features (feat_dim='+str(feat_dim)+') in condition {cond}'
        featCond = {k:[j.shape[feat_dim] for j in condition[k]] for i, k in enumerate(condLst)}
        self._2bool(featAll, msg=errorMsg.format(cond=str(featCond)))

        # Concatenate data :
        #   -> Concatenate features inside each condition :
        for k in condLst:
            xFeat = np.array([])
            for i in condition[k]:
                xFeat = np.concatenate((xFeat, i), axis=feat_dim) if xFeat.size else i
            condition[k] = xFeat

        #   -> Concatenate conditions :
        x, y = np.array([]), []
        if len(condLst) > 1:
            for num, k in enumerate(condLst):
                y += [num]*condition[k].shape[trial_dim]
                x = np.concatenate((x, condition[k]), axis=trial_dim) if x.size else condition[k]
        else:
            x = xFeat
            y = [0]*condition[list(condLst)[0]].shape[trial_dim]

        return x, y

    @staticmethod
    def _2bool(x, msg=False):
        """Check size
        """
        # Check dim in list of int :
        if ((np.array(x)-x[0]).sum() == 0) and (isinstance(x[0], int)):
            return True
        # Check dim in list of list :
        elif isinstance(x[0], list) and len(x) > 1:
            xs = [set(x[0]).intersection(set(k)) for k in x[1::]][0]
            if (np.array(list(xs))-np.array(x[0])).sum() == 0:
                pass
            else:
                raise ValueError(msg)
        elif isinstance(x[0], list) and len(x) == 1:
            pass
        else:
            if msg:
                raise ValueError(msg)
            else:
                return False
