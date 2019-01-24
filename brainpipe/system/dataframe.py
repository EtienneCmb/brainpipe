import pandas as pd
from warnings import warn
import numpy as n


class pdTools(object):

    """Tools for pandas DataFrame

    Syntax:
        - To search if arg1 is in column1:

        >>> keep = ('column1', arg1)

        - AND condition for ar1 and arg2 in column1:

        >>> keep = ('column1', [arg1, 2])

        - AND condition for arg1 in column1 and arg2 in column2:

        >>> keep = [('column1', arg1), ('column2', arg2)]

        - OR condition:

        >>> keep = ('column1', arg1), ('column2', arg2), ...


    """

    def search(self, df, *keep):
        """Search in a pandas dataframe.

        Args:
            df: pandas dataframe
                The dataframe to filter

            keep: tuple/list
                Control the informations to search. See the syntax definition

        Return:
            List of row index for informations found.
        """
        return _pdSearch(df, *keep)

    def keep(self, df, *keep, keep_idx=True):
        """Filter a pandas dataframe and keep only interresting rows.

        Args:
            df: pandas dataframe
                The dataframe to filter

            keep: tuple/list
                Control the informations to keep. See the syntax definition

            keep_idx: bool, optional [def : True]
                Add a column to df to check what are the rows that has been
                kept.

        Return:
            A pandas Dataframe with only the informations to keep.
        """
        return _pdKeep(df, *keep, keep_idx=keep_idx)[0]

    def remove(self, df, *rm, rm_idx=True):
        """Filter a pandas dataframe and remove only interresting rows.

        Args:
            df: pandas dataframe
                The dataframe to be filter

            rm: tuple/list
                Control the informations to remove. See the syntax definition

            rm_idx: bool, optional [def : True]
                Add a column to df to check what are the rows that has been
                removed.

        Return:
            A pandas Dataframe without the removed informations.
        """
        return _pdRm(df, *rm, rm_idx=rm_idx)[0]


def _pdSearch(df, *keep):
    keyList = list(df.keys())
    # For each keep:
    index, indexCol = [], []
    for k in keep:
        if type(k) is not list:
            k = [k]
        # For each & condition:
        for iN, iS in enumerate(k):
            colName, toKeep = iS[0], iS[1]
            # Check if the column exist :
            if keyList.count(colName):
                if type(toKeep) is not list:
                    toKeep = [toKeep]
                # Then search in df:
                indS = []
                for jN, jS in enumerate(toKeep):
                    indS.extend(list(n.where(df[colName] == jS)[0]))
                if iN == 0:
                    indSet = set(indS)
                else:
                    indSet = indSet.intersection(set(indS))
            else:
                warn('No column "'+colName +
                     '" found. This argument has been ignored from your search'
                     )
                indSet = []

        index.append(list(indSet))

    return index


def _pdKeep(df, *keep, keep_idx=True):
    """Sub keep function
    """
    dfKeep = pd.DataFrame()
    keepIdx = _pdSearch(df, *keep)
    indCol = []
    for k in keepIdx:
        dfKeep = dfKeep.append(df.iloc[k])
        indCol.extend(k)

    if keep_idx:
        dfKeep['keep_idx'] = indCol

    return dfKeep.set_index([list(n.arange(dfKeep.shape[0]))]), keepIdx


def _pdRm(df, *rm, rm_idx=True):
    """Filter a pandas dataframe and keep only interresting rows.
    This function can combine "and" and "or" conditionnal test for
    a more flexible control of a dataframe.

    df : pandas dataframe
        The dataframe to be filtered

    keep : tuple/list
        Control the informations to keep

    indCol : bool, optional [def : True]
        Add a column to df to check what are the rows that has been
        kept.

    & : inside the list
    + : for the number of arg
    Each keep: [(Column1,[s1,s2,...]),(Column2,[s21,s22])]
    """
    dfT = df.copy()
    rmIdx = _pdSearch(dfT, *rm)
    indCol = []
    for k in rmIdx:
        indCol.extend(k)
    setList = list(set(indCol))
    dfT.drop(setList, inplace=True)
    fullList = list(n.arange(dfT.shape[0]))

    if rm_idx:
        dfT['rm_idx'] = list(set(fullList).difference(setList))

    return dfT.set_index([list(n.arange(dfT.shape[0]))]), rmIdx
