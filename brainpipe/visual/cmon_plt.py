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
            spine.set_color(None)  # don't draw spine
            ax.tick_params(**{loc: False})

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


def get_turbo_cmap():
    from matplotlib.colors import ListedColormap
    turbo_colormap_data = [[0.18995,0.07176,0.23217],[0.19483,0.08339,0.26149],[0.19956,0.09498,0.29024],[0.20415,0.10652,0.31844],[0.20860,0.11802,0.34607],[0.21291,0.12947,0.37314],[0.21708,0.14087,0.39964],[0.22111,0.15223,0.42558],[0.22500,0.16354,0.45096],[0.22875,0.17481,0.47578],[0.23236,0.18603,0.50004],[0.23582,0.19720,0.52373],[0.23915,0.20833,0.54686],[0.24234,0.21941,0.56942],[0.24539,0.23044,0.59142],[0.24830,0.24143,0.61286],[0.25107,0.25237,0.63374],[0.25369,0.26327,0.65406],[0.25618,0.27412,0.67381],[0.25853,0.28492,0.69300],[0.26074,0.29568,0.71162],[0.26280,0.30639,0.72968],[0.26473,0.31706,0.74718],[0.26652,0.32768,0.76412],[0.26816,0.33825,0.78050],[0.26967,0.34878,0.79631],[0.27103,0.35926,0.81156],[0.27226,0.36970,0.82624],[0.27334,0.38008,0.84037],[0.27429,0.39043,0.85393],[0.27509,0.40072,0.86692],[0.27576,0.41097,0.87936],[0.27628,0.42118,0.89123],[0.27667,0.43134,0.90254],[0.27691,0.44145,0.91328],[0.27701,0.45152,0.92347],[0.27698,0.46153,0.93309],[0.27680,0.47151,0.94214],[0.27648,0.48144,0.95064],[0.27603,0.49132,0.95857],[0.27543,0.50115,0.96594],[0.27469,0.51094,0.97275],[0.27381,0.52069,0.97899],[0.27273,0.53040,0.98461],[0.27106,0.54015,0.98930],[0.26878,0.54995,0.99303],[0.26592,0.55979,0.99583],[0.26252,0.56967,0.99773],[0.25862,0.57958,0.99876],[0.25425,0.58950,0.99896],[0.24946,0.59943,0.99835],[0.24427,0.60937,0.99697],[0.23874,0.61931,0.99485],[0.23288,0.62923,0.99202],[0.22676,0.63913,0.98851],[0.22039,0.64901,0.98436],[0.21382,0.65886,0.97959],[0.20708,0.66866,0.97423],[0.20021,0.67842,0.96833],[0.19326,0.68812,0.96190],[0.18625,0.69775,0.95498],[0.17923,0.70732,0.94761],[0.17223,0.71680,0.93981],[0.16529,0.72620,0.93161],[0.15844,0.73551,0.92305],[0.15173,0.74472,0.91416],[0.14519,0.75381,0.90496],[0.13886,0.76279,0.89550],[0.13278,0.77165,0.88580],[0.12698,0.78037,0.87590],[0.12151,0.78896,0.86581],[0.11639,0.79740,0.85559],[0.11167,0.80569,0.84525],[0.10738,0.81381,0.83484],[0.10357,0.82177,0.82437],[0.10026,0.82955,0.81389],[0.09750,0.83714,0.80342],[0.09532,0.84455,0.79299],[0.09377,0.85175,0.78264],[0.09287,0.85875,0.77240],[0.09267,0.86554,0.76230],[0.09320,0.87211,0.75237],[0.09451,0.87844,0.74265],[0.09662,0.88454,0.73316],[0.09958,0.89040,0.72393],[0.10342,0.89600,0.71500],[0.10815,0.90142,0.70599],[0.11374,0.90673,0.69651],[0.12014,0.91193,0.68660],[0.12733,0.91701,0.67627],[0.13526,0.92197,0.66556],[0.14391,0.92680,0.65448],[0.15323,0.93151,0.64308],[0.16319,0.93609,0.63137],[0.17377,0.94053,0.61938],[0.18491,0.94484,0.60713],[0.19659,0.94901,0.59466],[0.20877,0.95304,0.58199],[0.22142,0.95692,0.56914],[0.23449,0.96065,0.55614],[0.24797,0.96423,0.54303],[0.26180,0.96765,0.52981],[0.27597,0.97092,0.51653],[0.29042,0.97403,0.50321],[0.30513,0.97697,0.48987],[0.32006,0.97974,0.47654],[0.33517,0.98234,0.46325],[0.35043,0.98477,0.45002],[0.36581,0.98702,0.43688],[0.38127,0.98909,0.42386],[0.39678,0.99098,0.41098],[0.41229,0.99268,0.39826],[0.42778,0.99419,0.38575],[0.44321,0.99551,0.37345],[0.45854,0.99663,0.36140],[0.47375,0.99755,0.34963],[0.48879,0.99828,0.33816],[0.50362,0.99879,0.32701],[0.51822,0.99910,0.31622],[0.53255,0.99919,0.30581],[0.54658,0.99907,0.29581],[0.56026,0.99873,0.28623],[0.57357,0.99817,0.27712],[0.58646,0.99739,0.26849],[0.59891,0.99638,0.26038],[0.61088,0.99514,0.25280],[0.62233,0.99366,0.24579],[0.63323,0.99195,0.23937],[0.64362,0.98999,0.23356],[0.65394,0.98775,0.22835],[0.66428,0.98524,0.22370],[0.67462,0.98246,0.21960],[0.68494,0.97941,0.21602],[0.69525,0.97610,0.21294],[0.70553,0.97255,0.21032],[0.71577,0.96875,0.20815],[0.72596,0.96470,0.20640],[0.73610,0.96043,0.20504],[0.74617,0.95593,0.20406],[0.75617,0.95121,0.20343],[0.76608,0.94627,0.20311],[0.77591,0.94113,0.20310],[0.78563,0.93579,0.20336],[0.79524,0.93025,0.20386],[0.80473,0.92452,0.20459],[0.81410,0.91861,0.20552],[0.82333,0.91253,0.20663],[0.83241,0.90627,0.20788],[0.84133,0.89986,0.20926],[0.85010,0.89328,0.21074],[0.85868,0.88655,0.21230],[0.86709,0.87968,0.21391],[0.87530,0.87267,0.21555],[0.88331,0.86553,0.21719],[0.89112,0.85826,0.21880],[0.89870,0.85087,0.22038],[0.90605,0.84337,0.22188],[0.91317,0.83576,0.22328],[0.92004,0.82806,0.22456],[0.92666,0.82025,0.22570],[0.93301,0.81236,0.22667],[0.93909,0.80439,0.22744],[0.94489,0.79634,0.22800],[0.95039,0.78823,0.22831],[0.95560,0.78005,0.22836],[0.96049,0.77181,0.22811],[0.96507,0.76352,0.22754],[0.96931,0.75519,0.22663],[0.97323,0.74682,0.22536],[0.97679,0.73842,0.22369],[0.98000,0.73000,0.22161],[0.98289,0.72140,0.21918],[0.98549,0.71250,0.21650],[0.98781,0.70330,0.21358],[0.98986,0.69382,0.21043],[0.99163,0.68408,0.20706],[0.99314,0.67408,0.20348],[0.99438,0.66386,0.19971],[0.99535,0.65341,0.19577],[0.99607,0.64277,0.19165],[0.99654,0.63193,0.18738],[0.99675,0.62093,0.18297],[0.99672,0.60977,0.17842],[0.99644,0.59846,0.17376],[0.99593,0.58703,0.16899],[0.99517,0.57549,0.16412],[0.99419,0.56386,0.15918],[0.99297,0.55214,0.15417],[0.99153,0.54036,0.14910],[0.98987,0.52854,0.14398],[0.98799,0.51667,0.13883],[0.98590,0.50479,0.13367],[0.98360,0.49291,0.12849],[0.98108,0.48104,0.12332],[0.97837,0.46920,0.11817],[0.97545,0.45740,0.11305],[0.97234,0.44565,0.10797],[0.96904,0.43399,0.10294],[0.96555,0.42241,0.09798],[0.96187,0.41093,0.09310],[0.95801,0.39958,0.08831],[0.95398,0.38836,0.08362],[0.94977,0.37729,0.07905],[0.94538,0.36638,0.07461],[0.94084,0.35566,0.07031],[0.93612,0.34513,0.06616],[0.93125,0.33482,0.06218],[0.92623,0.32473,0.05837],[0.92105,0.31489,0.05475],[0.91572,0.30530,0.05134],[0.91024,0.29599,0.04814],[0.90463,0.28696,0.04516],[0.89888,0.27824,0.04243],[0.89298,0.26981,0.03993],[0.88691,0.26152,0.03753],[0.88066,0.25334,0.03521],[0.87422,0.24526,0.03297],[0.86760,0.23730,0.03082],[0.86079,0.22945,0.02875],[0.85380,0.22170,0.02677],[0.84662,0.21407,0.02487],[0.83926,0.20654,0.02305],[0.83172,0.19912,0.02131],[0.82399,0.19182,0.01966],[0.81608,0.18462,0.01809],[0.80799,0.17753,0.01660],[0.79971,0.17055,0.01520],[0.79125,0.16368,0.01387],[0.78260,0.15693,0.01264],[0.77377,0.15028,0.01148],[0.76476,0.14374,0.01041],[0.75556,0.13731,0.00942],[0.74617,0.13098,0.00851],[0.73661,0.12477,0.00769],[0.72686,0.11867,0.00695],[0.71692,0.11268,0.00629],[0.70680,0.10680,0.00571],[0.69650,0.10102,0.00522],[0.68602,0.09536,0.00481],[0.67535,0.08980,0.00449],[0.66449,0.08436,0.00424],[0.65345,0.07902,0.00408],[0.64223,0.07380,0.00401],[0.63082,0.06868,0.00401],[0.61923,0.06367,0.00410],[0.60746,0.05878,0.00427],[0.59550,0.05399,0.00453],[0.58336,0.04931,0.00486],[0.57103,0.04474,0.00529],[0.55852,0.04028,0.00579],[0.54583,0.03593,0.00638],[0.53295,0.03169,0.00705],[0.51989,0.02756,0.00780],[0.50664,0.02354,0.00863],[0.49321,0.01963,0.00955],[0.47960,0.01583,0.01055]]
    return ListedColormap(turbo_colormap_data)
