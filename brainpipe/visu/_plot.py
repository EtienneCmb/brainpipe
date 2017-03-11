import numpy as n
import matplotlib.pyplot as plt
from brainpipe._utils._interp import mapinterpolation

__all__ = ['_plot', '_2Dplot', 'mapinterpolation']


def _sub(X, fcn, maxplot=10):
    """Manage the subplot"""
    if type(X) is n.ndarray:
        return fcn(X)
    elif type(X) is list:
        L = len(X)
        if L <= maxplot:
            fig = plt.figure()
            if L < 4:
                ncol, nrow = L, 1
            else:
                ncol = round(n.sqrt(L)).astype(int)
                nrow = round(L/ncol).astype(int)
                while nrow*ncol < L:
                    nrow += 1
            for k in range(0, L):
                fig.add_subplot(nrow, ncol, k+1)
                fcn(X[k])
            return plt.gca()
        else:
            raise ValueError('Warning : the "maxplot" parameter prevent to a'
                             'large number of plot. To increase the number'
                             ' of plot, change "maxplot"')


def _plot(xvec, yvec, title='', xlabel='', ylabel='', maxplot=10, **kwargs):
    """Simple plot"""
    def _subplot(yvec):
        dimLen = len(yvec.shape)
        if dimLen == 1:
            X = [yvec]
        elif dimLen == 2:
            X = [n.mean(yvec, 1)]
        elif dimLen == 3:
            X = [n.mean(yvec[k, :, :], 1) for k in range(0, yvec.shape[0])]
        [plt.plot(xvec, k, label=str(i), **kwargs) for i, k in enumerate(X)]
        ax = plt.gca()
        ax.set_xlabel(xlabel), ax.set_ylabel(ylabel)
        ax.set_title(title), ax.legend()
        plt.autoscale(tight=True)
        return ax

    return _sub(yvec, _subplot, maxplot=maxplot)


def _2Dplot(X, xvec, yvec, title='', xlabel='', ylabel='', cmap='viridis',
            cblabel='', maxplot=10, interp=(1, 1), **kwargs):
    """Simple 2D plot"""
    def _sub2Dplot(X):
        if (interp[0], interp[1]) != (1, 1):
            X, xV, yV = mapinterpolation(X, x=xvec, y=yvec, interpx=interp[1],
                                         interpy=interp[0])
        else:
            xV, yV = xvec, yvec
        im = plt.imshow(X, cmap=cmap, aspect='auto', extent=[xV[
            0], xV[-1], yV[-1], yV[0]], **kwargs)
        ax = plt.gca()
        ax.set_xlabel(xlabel), ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.invert_yaxis()
        cb = plt.colorbar(im)
        cb.set_label(cblabel)
        return ax

    return _sub(X, _sub2Dplot, maxplot=maxplot)
