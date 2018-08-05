from warnings import warn

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection

from ._interp import mapinterpolation


__all__ = ['addLines', 'BorderPlot', 'tilerplot', 'addPval', 'rmaxis',
           'despine', 'continuouscol']


class _pltutils(object):

    """
    **kwargs:
        title:
            title of plot [def: '']

        xlabel:
            label of x-axis [def: '']

        ylabel:
            label of y-axis [def: '']

        xlim:
            limit of the x-axis [def: [], current limit of x]

        ylim:
            limit of the y-axis [def: [], current limit of y]

        xticks:
            ticks of x-axis [def: [], current x-ticks]

        yticks:
            ticks of y-axis [def: [], current y-ticks]

        xticklabels:
            label of the x-ticks [def: [], current x-ticklabels]

        yticklabels:
            label of the y-ticks [def: [], current y-ticklabels]

        style:
            style of the plot [def: None]

        dpax:
            List of axis to despine ['left', 'right', 'top', 'bottom']

        rmax:
            Remove axis ['left', 'right', 'top', 'bottom']

    """

    def __init__(self, ax, title='', xlabel='', ylabel='', xlim=[], ylim=[],
                 ytitle=1.02, xticks=[], yticks=[], xticklabels=[], yticklabels=[],
                 style=None, dpax=None, rmax=None):

        if not hasattr(self, '_xType'):
            self._xType = int
        if not hasattr(self, '_yType'):
            self._yType = int
        # Axes ticks :
        if np.array(xticks).size:
            ax.set_xticks(xticks)
        if np.array(yticks).size:
            ax.set_yticks(yticks)
        # Axes ticklabels :
        if xticklabels:
            ax.set_xticklabels(xticklabels)
        if yticklabels:
            ax.set_yticklabels(yticklabels)
        # Axes labels :
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        # Axes limit :
        if np.array(xlim).size:
            ax.set_xlim(xlim)
        if np.array(ylim).size:
            ax.set_ylim(ylim)
        ax.set_title(title, y=ytitle)
        # Style :
        if style:
            plt.style.use(style)
        # Despine :
        if dpax:
            despine(ax, dpax)
        # Remove axis :
        if rmax:
            rmaxis(ax, rmax)


def rmaxis(ax, rmax):
    """Remove ticks and axis of a existing plot

    Args:
        ax: matplotlib axes
            Axes to remove axis

        rmax: list of strings
            List of axis name to be removed. For example, use
            ['left', 'right', 'top', 'bottom']
    """
    for loc, spine in ax.spines.items():
        if loc in rmax:
            spine.set_color('none')  # don't draw spine
            ax.tick_params(**{loc: 'off'})

def despine(ax, dpax, outward=10):
    """Despine axis of a existing plot

    Args:
        ax: matplotlib axes
            Axes to despine axis

        dpax: list of strings
            List of axis name to be despined. For example, use
            ['left', 'right', 'top', 'bottom']

    Kargs:
        outward: int/float, optional, [def: 10]
            Distance of despined axis from the original position.
    """
    for loc, spine in ax.spines.items():
        if loc in dpax:
            spine.set_position(('outward', outward))  # outward by 10 points
            spine.set_smart_bounds(True)


class tilerplot(object):

    """Automatic tiler plot for 1, 2 and 3D data.
    """

    def plot1D(self, fig, y, x=None, maxplot=10, figtitle='', sharex=False,
               sharey=False,  subdim=None, transpose=False, color='b',
               subspace=None, **kwargs):
        """Simple one dimentional plot

        Args:
            y: array
                Data to plot. y can either have one, two or three dimensions.
                If y is a vector, it will be plot in a simple window. If y is
                a matrix, all values inside are going to be superimpose. If y
                is a 3D matrix, the first dimension control the number of subplots.

            x: array, optional, [def: None]
                x vector for plotting data.

        Kargs:
            figtitle: string, optional, [def: '']
                Add a name to your figure

            subdim: tuple, optional, [def: None]
                Force subplots to be subdim=(n_colums, n_rows)

            maxplot: int, optional, [def: 10]
                Control the maximum number of subplot to prevent very large plot.
                By default, maxplot is 10 which mean that only 10 subplot can be
                defined.

            transpose: bool, optional, [def: False]
                Invert subplot (row <-> column)

            color: string, optional, [def: 'b']
                Color of the plot

            subspace: dict, optional, [def: None]
                Control the distance in subplots. Use 'left', 'bottom',
                'right', 'top', 'wspace', 'hspace'.
                Example: {'top':0.85, 'wspace':0.8}

            kwargs:
                Supplementar arguments to control each suplot:
                title, xlabel, ylabel (which can be list for each subplot)
                xlim, ylim, xticks, yticks, xticklabels, yticklabels, style,
                dpax, rmax.
        """
        # Fig properties:
        self._fig = self._figmngmt(fig, figtitle=figtitle, transpose=transpose)
        # Check y shape :
        y = self._checkarray(y)
        if x is None:
            x = np.arange(y.shape[1])
        # Get default for title, xlabel and ylabel:
        kwout, kwargs = self._completeLabels(kwargs, y.shape[0], 'title',
                                             'xlabel', 'ylabel', default='')
        # Plotting function :

        def _fcn(y, k):
            plt.plot(x, y, color=color)
            plt.axis('tight')
            _pltutils(plt.gca(), kwout['title'][k], kwout['xlabel'][k],
                      kwout['ylabel'][k], **kwargs)

        axAll = self._subplotND(y, _fcn, maxplot, subdim, sharex, sharey)
        fig = plt.gcf()
        fig.tight_layout()

        if subspace:
            fig.subplots_adjust(**subspace)

        return fig, axAll

    def plot2D(self, fig, y, xvec=None, yvec=None, cmap='inferno',
               colorbar=True, cbticks='minmax', ycb=-10, cblabel='',
               under=None, over=None, vmin=None, vmax=None, sharex=False,
               sharey=False, textin=False, textcolor='w', textype='%.4f', subdim=None,
               mask=None, interpolation='none', resample=(0, 0), figtitle='',
               transpose=False, maxplot=10, subspace=None, contour=None, pltargs={},
               pltype='pcolor', ncontour=10, polar=False, **kwargs):
        """Plot y as an image

        Args:
            fig: figure
                A matplotlib figure where plotting

            y: array
                Data to plot. y can either have one, two or three dimensions.
                If y is a vector, it will be plot in a simple window. If y is
                a matrix, all values inside are going to be superimpose. If y
                is a 3D matrix, the first dimension control the number of subplots.

        Kargs:
            xvec, yvec: array, optional, [def: None]
                Vectors for y and x axis of each picture

            cmap: string, optional, [def: 'inferno']
                Choice of the colormap

            colorbar: bool/string, optional, [def: True]
                Add or not a colorbar to your plot. Alternatively, use
                'center-max' or 'center-dev' to have a centered colorbar

            cbticks: list/string, optional, [def: 'minmax']
                Control colorbar ticks. Use 'auto' for [min,(min+max)/2,max],
                'minmax' for [min, max] or your own list.

            ycb: int, optional, [def: -10]
                Distance between the colorbar and the label.

            cblabel: string, optional, [def: '']
                Label for the colorbar

            under, over: string, optional, [def: '']
                Color for everything under and over the colorbar limit.

            vmin, vmax: int/float, optional, [def: None]
                Control minimum and maximum of the image

            sharex, sharey: bool, optional, [def: False]
                Define if subplots should share x and y

            textin: bool, optional, [def: False]
                Display values inside the heatmap

            textcolor: string, optional, [def: 'w']
                Color of values inside the heatmap

            textype: string, optional, [def: '%.4f']
                Way of display text inside the heatmap

            subdim: tuple, optional, [def: None]
                Force subplots to be subdim=(n_colums, n_rows)

            interpolation: string, optional, [def: 'none']
                Plot interpolation

            resample: tuple, optional, [def: (0, 0)]
                Interpolate the map for a specific dimension. If (0.5, 0.1),
                this mean that the programme will insert one new point on x-axis,
                and 10 new points on y-axis. Pimp you map and make it sooo smooth.

            figtitle: string, optional, [def: '']
                Add a name to your figure

            maxplot: int, optional, [def: 10]
                Control the maximum number of subplot to prevent very large plot.
                By default, maxplot is 10 which mean that only 10 subplot can be
                defined.

            transpose: bool, optional, [def: False]
                Invert subplot (row <-> column)

            subspace: dict, optional, [def: None]
                Control the distance in subplots. Use 'left', 'bottom',
                'right', 'top', 'wspace', 'hspace'.
                Example: {'top':0.85, 'wspace':0.8}

            contour: dict, optional, [def: None]
                Add a contour to your 2D-plot. In order to use this parameter,
                define contour={'data':yourdata, 'label':[yourlabel], kwargs}
                where yourdata must have the same shape as y, level must float/int
                from smallest to largest. Use kwargs to pass other arguments to the
                contour function

            kwargs:
                Supplementar arguments to control each suplot:
                title, xlabel, ylabel (which can be list for each subplot)
                xlim, ylim, xticks, yticks, xticklabels, yticklabels, style
                dpax, rmax.
        """

        # Fig properties:
        self._fig = self._figmngmt(fig, figtitle=figtitle, transpose=transpose)

        # Share axis:
        if sharex:
            self._fig.subplots_adjust(hspace=0)
        # Mask properties:
        if (mask is not None):
            if not (mask.shape == y.shape):
                warn('The shape of mask '+str(mask.shape)+' must be the same '
                     'of y '+str(y.shape)+'. Mask will be ignored')
                mask = None
            if mask.ndim == 2:
                mask = mask[np.newaxis, ...]
        else:
            mask = []

        # Check y shape :
        y = self._checkarray(y)
        if xvec is None:
            xvec = np.arange(y.shape[-1]+1)
        if yvec is None:
            yvec = np.arange(y.shape[1]+1)
        l0, l1, l2 = y.shape

        if (vmin is None) and (vmax is None):
            if colorbar == 'center-max':
                m, M = y.min(), y.max()
                vmin, vmax = -np.max([np.abs(m), np.abs(M)]), np.max([np.abs(m), np.abs(M)])
                colorbar = True
            if colorbar == 'center-dev':
                m, M = y.mean()-y.std(), y.mean()+y.std()
                vmin, vmax = -np.max([np.abs(m), np.abs(M)]), np.max([np.abs(m), np.abs(M)])
                colorbar = True

        # Resample data:
        if resample != (0, 0):
            yi = []
            maski = []
            for k in range(l0):
                yT, yvec, xvec = mapinterpolation(y[k, ...], x=yvec, y=xvec,
                                                  interpx=resample[0],
                                                  interpy=resample[1])
                yi.append(yT)
                if np.array(mask).size:
                    maskT, _, _ = mapinterpolation(mask[k, ...], x=xvec, y=yvec,
                                                   interpx=resample[0],
                                                   interpy=resample[1])
                    maski.append(maskT)
            y = np.array(yi)
            mask = maski
            del yi, yT
        # Get default for title, xlabel and ylabel:
        kwout, kwargs = self._completeLabels(kwargs, y.shape[0], 'title',
                                             'xlabel', 'ylabel', default='')

        # Plotting function :
        def _fcn(y, k, mask=mask):
            # Get a mask for data:
            if pltype is 'pcolor':
                im = plt.pcolormesh(xvec, yvec, y, cmap=cmap, vmin=vmin, vmax=vmax, **pltargs)
            elif  pltype is 'imshow':
                if np.array(mask).size:
                    mask = np.array(mask)
                    norm = Normalize(vmin, vmax)
                    y = plt.get_cmap(cmap)(norm(y))
                    y[..., 3] = mask[k, ...]
                # Plot picture:
                im = plt.imshow(y, aspect='auto', cmap=cmap, origin='upper',
                                interpolation=interpolation, vmin=vmin, vmax=vmax,
                                extent=[xvec[0], xvec[-1], yvec[-1], yvec[0]], **pltargs)
                plt.gca().invert_yaxis()
            elif pltype is 'contour':
                im = plt.contourf(xvec, yvec, y, ncontour, cmap=cmap, vmin=vmin, vmax=vmax, **pltargs)

            # Manage under and over:
            if (under is not None) and (isinstance(under, str)):
                im.cmap.set_under(color=under)
            if (over is not None) and (isinstance(over, str)):
                im.cmap.set_over(color=over)

            # Manage contour:
            if contour is not None:
                contour_bck = contour.copy()
                # Unpack necessary arguments :
                datac = contour_bck['data']
                level = contour_bck['level']
                # Check data size:
                if len(datac.shape) == 2:
                    datac = datac[np.newaxis, ...]
                contour_bck.pop('data'), contour_bck.pop('level')
                _ = plt.contour(datac[k, ...], extent=[xvec[0], xvec[-1], yvec[0], yvec[-1]],
                                levels=level, **contour_bck)

            # Manage ticks, labek etc:
            plt.axis('tight')
            ax = plt.gca()
            _pltutils(ax, kwout['title'][k], kwout['xlabel'][k],
                      kwout['ylabel'][k], **kwargs)

            # Manage colorbar:
            if colorbar:
                cb = plt.colorbar(im, shrink=0.7, pad=0.01, aspect=10)
                if cbticks == 'auto':
                    clim = im.colorbar.get_clim()
                    cb.set_ticks([clim[0], (clim[0]+clim[1])/2, clim[1]])
                elif cbticks == 'minmax':
                    clim = im.colorbar.get_clim()
                    cb.set_ticks([clim[0], clim[1]])
                elif cbticks is None:
                    pass
                else:
                    cb.set_ticks(cbticks)
                cb.set_label(cblabel, labelpad=ycb)
                cb.outline.set_visible(False)

            # Text inside:
            if textin:
                for k in range(y.shape[0]):
                    for i in range(y.shape[1]):
                        plt.text(i + 0.5, k + 0.5, textype % y[i, k],
                                 color=textcolor,
                                 horizontalalignment='center',
                                 verticalalignment='center')

        axAll = self._subplotND(y, _fcn, maxplot, subdim, sharex, sharey,
                                polar=polar)
        fig = plt.gcf()
        fig.tight_layout()

        if subspace:
            fig.subplots_adjust(**subspace)

        return fig, axAll

    def _figmngmt(self, fig, figtitle='', transpose=False):
        # Change title:
        if figtitle:
            fig.suptitle(figtitle, fontsize=14, fontweight='bold')
        self._transpose = transpose

        return fig

    def _checkarray(self, y):
        """Check input shape
        """
        # Vector :
        if y.ndim == 1:
            y = y[np.newaxis, ..., np.newaxis]
        # 2D array :
        elif y.ndim == 2:
            y = y[np.newaxis]
        # more than 3D array :
        elif y.ndim > 3:
            raise ValueError('array to plot should not have more than '
                             '3 dimensions')
        return y

    def _subplotND(self, y, fcn, maxplot, subdim, sharex, sharey,
                   polar=False):
        """Manage subplots
        """
        axall = []
        L = y.shape[0]
        if L <= maxplot:
            fig = self._fig
            # If force subdim:
            if not subdim:
                if L < 4:
                    ncol, nrow = L, 1
                else:
                    ncol = round(np.sqrt(L)).astype(int)
                    nrow = round(L/ncol).astype(int)
                    while nrow*ncol < L:
                        nrow += 1
            else:
                nrow, ncol = subdim
            # Sublots:
            if self._transpose:
                backup = ncol
                ncol = nrow
                nrow = backup
            self._nrow, self._ncol = nrow, ncol
            for k in range(L):
                fig.add_subplot(nrow, ncol, k+1, polar=polar)
                fcn(y[k, ...], k)
                ax = plt.gca()
                # Share-y axis:
                if sharey and (k % ncol == 0):
                    pass
                else:
                    if sharey:
                        ax.set_yticklabels([])
                        ax.set_ylabel('')
                        rmaxis(ax, ['left'])
                # Share-x axis:
                if sharex and (k < (nrow-1)*ncol):
                    ax.set_xticklabels([])
                    ax.set_xlabel('')
                axall.append(plt.gca())
                if polar:
                    ax.grid(color='gray', lw=0.5, linestyle='-')
                else:
                    plt.gca().grid('off')
            return axall
        else:
            raise ValueError('Warning : the "maxplot" parameter prevent to a'
                             'large number of plot. To increase the number'
                             ' of plot, change "maxplot"')

    def _completeLabels(self, kwargs, L, *arg, default=''):
        """Function for completing title, xlabel, ylabel...
        """
        kwlst = list(kwargs.keys())
        kwval = list(kwargs.values())
        kwout = {}
        # For each arg:
        for k in arg:
            # If empty set to default :
            if k not in kwlst:
                kwout[k] = [default]*L
            else:
                val = kwargs[k]
                # If not empty and is string:
                if isinstance(val, str):
                    kwout[k] = [val]*L
                # If not empty and is string:
                elif isinstance(val, list):
                    # Check size:
                    if len(val) == L:
                        kwout[k] = val
                    else:
                        warn('The length of "'+k+'" must be '+str(L))
                        kwout[k] = [val[0]]*L
                # remove the key:
                kwargs.pop(k, None)

        return kwout, kwargs

# tilerplot.plot1D.__doc__ += _pltutils.__doc__
# tilerplot.plot2D.__doc__ += _pltutils.__doc__


class addLines(object):

    """Add vertical and horizontal lines to an existing plot.

    Args:
        ax: matplotlib axes
            The axes to add lines. USe for example plt.gca()

    Kargs:
        vLines: list, [def: []]
            Define vertical lines. vLines should be a list of int/float

        vColor: list of strings, [def: ['gray']]
            Control the color of the vertical lines. The length of the
            vColor list must be the same as the length of vLines

        vShape: list of strings, [def: ['--']]
            Control the shape of the vertical lines. The length of the
            vShape list must be the same as the length of vLines

        hLines: list, [def: []]
            Define horizontal lines. hLines should be a list of int/float

        hColor: list of strings, [def: ['black']]
            Control the color of the horizontal lines. The length of the
            hColor list must be the same as the length of hLines

        hShape: list of strings, [def: ['-']]
            Control the shape of the horizontal lines. The length of the
            hShape list must be the same as the length of hLines

    Return:
        The current axes

    Example:
        >>> # Create an empty plot:
        >>> plt.plot([])
        >>> plt.ylim([-1, 1]), plt.xlim([-10, 10])
        >>> addLines(plt.gca(), vLines=[0, -5, 5, -7, 7], vColor=['k', 'r', 'g', 'y', 'b'],
        >>>          vWidth=[5, 4, 3, 2, 1], vShape=['-', '-', '--', '-', '--'],
        >>>          hLines=[0, -0.5, 0.7], hColor=['k', 'r', 'g'], hWidth=[5, 4, 3],
        >>>          hShape=['-', '-', '--'])

    """

    def __init__(self, ax,
                 vLines=[], vColor=None, vShape=None, vWidth=None,
                 hLines=[], hColor=None, hWidth=None, hShape=None):
        pass

    def __new__(self, ax,
                vLines=[], vColor=None, vShape=None, vWidth=None,
                hLines=[], hColor=None, hWidth=None, hShape=None):
        # Get the axes limits :
        xlim = list(ax.get_xlim())
        ylim = list(ax.get_ylim())

        # Get the number of vertical and horizontal lines :
        nV = len(vLines)
        nH = len(hLines)

        # Define the color :
        if not vColor:
            vColor = ['gray']*nV
        if not hColor:
            hColor = ['black']*nH

        # Define the width :
        if not vWidth:
            vWidth = [1]*nV
        if not hWidth:
            hWidth = [1]*nH

        # Define the shape :
        if not vShape:
            vShape = ['--']*nV
        if not hShape:
            hShape = ['-']*nH

        # Plot Verticale lines :
        for k in range(0, nV):
            ax.plot((vLines[k], vLines[k]), (ylim[0], ylim[1]), vShape[k],
                    color=vColor[k], linewidth=vWidth[k])
        # Plot Horizontal lines :
        for k in range(0, nH):
            ax.plot((xlim[0], xlim[1]), (hLines[k], hLines[k]), hShape[k],
                    color=hColor[k], linewidth=hWidth[k])

        return plt.gca()


class BorderPlot(_pltutils):

    """Plot a signal with it associated deviation. The function plot the
    mean of the signal, and the deviation (std) or standard error on the mean
    (sem) in transparency.

    Args:
        time: array/limit
            The time vector of the plot (len(time)=N)

        x: numpy array
            The signal to plot. One dimension of x must be the length of time
            N. The other dimension will be consider to define the deviation.
            For example, x.shape = (N, M)

    Kargs:
        y: numpy array, optional, [def: None]
            Label vector to separate the x signal in diffrent classes. The
            length of y must be M. If no y is specified, the deviation will be
            computed for the entire array x. If y is composed with integers
            Example: y = np.array([1,1,1,1,2,2,2,2]), the function will
            geneate as many curve as the number of unique classes in y. In this
            case, two curves are going to be considered.

        kind: string, optional, [def: 'sem']
            Choose between 'std' for standard deviation and 'sem', standard
            error on the mean (wich is: std(x)/sqrt(N-1))

        color: string or list of strings, optional
            Specify the color of each curve. The length of color must be the
            same as the length of unique classes in y.

        alpha: int/float, optional [def: 0.2]
            Control the transparency of the deviation.

        linewidth: int/float, optional, [def: 2]
            Control the width of the mean curve.

        legend: string or list of strings, optional, [def: '']
            Specify the label of each curve and generate a legend. The length
            of legend must be the same as the length of unique classes in y.

        ncol: integer, optional, [def: 1]
            Number of colums for the legend

        kwargs:
            Supplementar arguments to control each suplot:
            title, xlabel, ylabel (which can be list for each subplot)
            xlim, ylim, xticks, yticks, xticklabels, yticklabels, style.
    Return:
        The axes of the plot.
    """
    # __doc__ += _pltutils.__doc__

    def __init__(self, time, x, y=None, kind='sem', color='',
                 alpha=0.2, linewidth=2, legend='', ncol=1, **kwargs):
        pass

    def __new__(self, time, x, y=None, kind='sem', color='', alpha=0.2,
                linewidth=2, legend='', ncol=1, axes=None, **kwargs):

        self.xType = type(time[0])
        self.yType = type(x[0, 0])

        # Check arguments :
        if x.shape[1] == len(time):
            x = x.T
        npts, dev = x.shape
        if y is None:
            y = np.array([0]*dev)
        yClass = np.unique(y)
        nclass = len(yClass)
        if not color:
            color = ['darkblue', 'darkgreen', 'darkred',
                     'darkorange', 'purple', 'gold', 'dimgray', 'black']
        else:
            if type(color) is not list:
                color = [color]
            if len(color) is not nclass:
                color = color*nclass
        if not legend:
            legend = ['']*dev
        else:
            if type(legend) is not list:
                legend = [legend]
            if len(legend) is not nclass:
                legend = legend*nclass

        # For each class :
        for k in yClass:
            _BorderPlot(time, x[:, np.where(y == k)[0]], color[k], kind,
                        alpha, legend[k], linewidth, axes)
        ax = plt.gca()
        plt.axis('tight')

        _pltutils.__init__(self, ax, **kwargs)

        return plt.gca()


def _BorderPlot(time, x, color, kind, alpha, legend, linewidth, axes):
    npts, dev = x.shape
    # Get the deviation/sem :
    xStd = np.std(x, axis=1)
    if kind is 'sem':
        xStd = xStd/np.sqrt(npts-1)
    xMean = np.mean(x, 1)
    xLow, xHigh = xMean-xStd, xMean+xStd

    # Plot :
    if axes is None:
        axes = plt.gca()
    plt.sca(axes)
    ax = plt.plot(time, xMean, color=color, label=legend, linewidth=linewidth)
    plt.fill_between(time, xLow, xHigh, alpha=alpha, color=ax[0].get_color())


def addPval(ax, pval, y=0, x=None, p=0.05, minsucc=1, color='b', shape='-',
            lw=2, **kwargs):
    """Add significants p-value to an existing plot

    Args:
        ax: matplotlib axes
            The axes to add lines. Use for example plt.gca()

        pval: vector
            Vector of pvalues

    Kargs:
        y: int/float
            The y location of your p-values

        x: vector
            x vector of the plot. Must have the same size as pval

        p: float
            p-value threshold to plot

        minsucc: int
            Minimum number of successive significants p-values

        color: string
            Color of th p-value line

        shape: string
            Shape of th p-value line

        lw: int
            Linewidth of th p-value line

        kwargs:
            Any supplementar arguments are passed to the plt.plot()
            function

    Return:
        ax: updated matplotlib axes
    """
    # Check inputs:
    pval = np.ravel(pval)
    N = len(pval)
    if x is None:
        x = np.arange(N)
    if len(x)-N is not 0:
        raise ValueError("The length of pval ("+str(N)+") must be the same as x ("+str(len(x))+")")

    # Find successive points:
    underp = np.where(pval < p)[0]
    pvsplit = np.split(underp, np.where(np.diff(underp) != 1)[0]+1)
    succlst = [[k[0], k[-1]] for k in pvsplit if len(k) >= minsucc ]

    # Plot lines:
    for k in succlst:
        ax.plot((x[k[0]], x[k[1]]), (y, y), lw=lw, color=color, **kwargs)

    return plt.gca()


class continuouscol(_pltutils):

    """Plot signal with continuous color

    Args:
        ax: matplotlib axes
            The axes to add lines. Use for example plt.gca()

        y: vector
            Vector to plot

    Kargs:
        x: vector, optional, [def: None]
            Values on the x-axis. x should have the same length as y.
            By default, x-values are 0, 1, ..., len(y)

        color: vector, optional, [def: None]
            Values to colorize the line. color should have the same length as y.

        cmap: string, optional, [def: 'inferno']
            The name of the colormap to use

        pltargs: dict, optional, [def: {}]
            Arguments to pass to the LineCollection() function of matplotlib

        kwargs:
            Supplementar arguments to control each suplot:
            title, xlabel, ylabel (which can be list for each subplot)
            xlim, ylim, xticks, yticks, xticklabels, yticklabels, style. 
    """

    def __init__(self, ax, y, x=None, color=None, cmap='inferno', pltargs={}, **kwargs):
        pass

    def __new__(self, ax, y, x=None, color=None, cmap='inferno', pltargs={}, **kwargs):
        # Check inputs :
        y = np.ravel(y)
        if x is None:
            x = np.arange(len(y))
        else:
            x = np.ravel(x)
            if len(y) != len(x):
                raise ValueError('x and y must have the same length')
        if color is None:
            color = np.arange(len(y))

        # Create segments:
        xy = np.array([x, y]).T[..., np.newaxis].reshape(-1, 1, 2)
        segments = np.concatenate((xy[0:-1, :], xy[1::]), axis=1)
        lc = LineCollection(segments, cmap=cmap, **pltargs)
        lc.set_array(color)

        # Plot managment:
        ax.add_collection(lc)
        plt.axis('tight')
        _pltutils.__init__(self, ax, **kwargs)
        
        return plt.gca()