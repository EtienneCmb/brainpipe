import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from warnings import warn
import numpy as np

from brainpipe.visual.cmon_plt import _pltutils, addLines, tilerplot
from brainpipe.statistics import *




class clfplt(object):

    """Classification plot class
    """

    def daplot(self, da, featinfo=True, cmap=None, daperm=None, chance_method=None,
               chance_unique=True, chance_color='firebrick', chance_level = 0.01,
               **kwargs):
        """Plot decoding accuracies

        Args:
            da: array
                Array of shape n_features or (n_repetitions x n_features)

        Kargs:
            featinfo: bool, optional, [def: True]
                Use self.info.featinfo to add informations to your plot

            cmap: string, optional, [def: None]
                Define a colormap like cmap='viridis'

            daperm: array, optional, [def: None]
                Array of permutations decoding

            chance_method: string, optional, [def: None]
                Parameter to display the chance level in the plot. Use:

                    - 'bino': binomial chance level
                    - 'perm': permutation's based chance level

            chance_unique: bool, optional, [def: True]
                Use this parameter to have a unique chance level.

            chance_color: string, optional, [def: 'firebrick']
                Color of you chance level

            chance_level: float, optional, [def: 0.01]
                Level to display

            kwargs:
                Supplementar arguments to control each suplot:
                title, xlabel, ylabel (which can be list for each subplot)
                xlim, ylim, xticks, yticks, xticklabels, yticklabels, style
                dpax, rmax.

        Return:
            ax: matplotlib axes
        """
        da = np.squeeze(da)

        # Barplot :
        if (da.ndim == 1) or (da.shape[1] == 1):
            nda = len(da)
            # Main plot :
            bar = plt.bar(np.arange(nda), da, align='center', width=0.9)
            # Manage colormap:
            if cmap is not None:
                cm = ScalarMappable(cmap=cmap)
                cmap = np.ndarray.tolist(cm.to_rgba(da, alpha=1))
                [bar[k].set_color(cmap[k]) for k in range(len(da))]

        # Boxplot :
        elif (da.ndim == 2):
            nda = da.shape[1]
            bar = plt.boxplot(da, widths=0.9, patch_artist=True, positions=np.arange(nda))
            for whisker in bar['whiskers']:
                whisker.set(color='gray', linewidth=2)
            for cap in bar['caps']:
                cap.set(color='gray', linewidth=2)
            for median in bar['medians']:
                median.set(color='firebrick', linewidth=2)
            for flier in bar['fliers']:
                flier.set(marker='o', color='#ab4642', alpha=0.5)
            # Manage colormap:
            if cmap is not None:
                cm = ScalarMappable(cmap=cmap)
                cmap = np.ndarray.tolist(cm.to_rgba(da.mean(0), alpha=1))
                for k, box in enumerate(bar['boxes']):
                    box.set( color='gray', linewidth=2)
                    box.set( facecolor=cmap[k], alpha=0.9)

        # Not found :
        else:
            raise ValueError("da have a inconsistent shape for plotting")
        ax = plt.gca()
        # Manage label:
        ax.set_xlabel('Features'), ax.set_ylabel('Decoding accuracy (%)')
        ax.set_xticks(np.arange(nda))
        plt.axis('tight')

        # kwargs pass to _pltutils:
        _pltutils(plt.gca(), **kwargs)

        # Use featinfo:
        if featinfo:
            # Chance level:
            chanceth = self.info.statinfo['Chance (theorical, %)'][0]
            addLines(ax, hLines=[chanceth], hWidth=[2], hColor=['k'])
            # Custom title:
            title = 'Classification with a '+self._clf.lgStr+'\nand a '+self._cv.lgStr
            ax.set_title(title, y=1.02)
            # Custom labels:
            level1 = self.info.featinfo.keys()[0][0]
            featname = list(self.info.featinfo[level1]['Group'])
            ax.set_xticklabels(featname, rotation=45)

        # Binomial chance level:
        if chance_method == 'bino':
            if hasattr(self, '_ry'):
                level = bino_p2da(self._ry, chance_level)
            else:
                level = bino_p2da(self._y, chance_level)
            addLines(ax, hLines=[level], hColor=[chance_color], hWidth=[2], hShape=['--'])
        # Permutations chance level:
        elif chance_method == 'perm':
            if daperm is not None:
                # Unique chance level:
                if chance_unique:
                    level = np.max(self.stat.perm_pvalue2da(daperm, p=chance_level, maxst=True))
                    addLines(ax, hLines=[level], hColor=[chance_color], hWidth=[2], hShape=['--'])
                else:
                    level = self.stat.perm_pvalue2da(daperm, p=chance_level)
                    ticks = ax.get_xticks()
                    for k, l in enumerate(level):
                        x1, x2 = ticks[k]-0.5, ticks[k]+0.5
                        ax.plot((x1, x2), (l, l), '--', lw=2, color=chance_color)
            else:
                warn("The display of the chance level using permutations has been ignored "
                     "either because you did'nt compute it using fit() or because you didn't "
                     "specified the daperm argument in the current function")

        return plt.gca()

    def cmplot(self, fig, cm, fignum=0, cmap='Spectral_r', textin=True, **kwargs):
        """Plot one confusion matrix

        Args:
            fig: figure
                Matplotlib figure ex: fig=plt.figure(0, figsize(8, 6))
            cm: array
                Confusion matrix of shape (n_class x n_class)

        Kargs:
            fignum: int, optional, [def: 0]
                Number of the figure

            cmap: string, optional, [def: 'Spectral_r]
                Colormap of the confusion matrix

            textin: bool, optional, [def: True]
                Display or values inside

            kwargs: supplementar arguments
                Any supplementar argument is passed to the plot2D() fonction
                of brainpipe
        """
        y = self._y
        p = tilerplot()
        p.plot2D(fig, cm, pltargs={'edgecolors':'w'},
                 rmax=['top', 'right'], cmap=cmap, xticks=list(set(y+0.5)), xticklabels=set(y),
                 textin=textin, textype='%i', textcolor='k', yticks=list(set(y+0.5)),
                 yticklabels=set(y), dpax=['left', 'bottom'], cblabel='Decoding accuracy (%)',
                 **kwargs)
        ax = plt.gca()
        ax.invert_yaxis()
        return plt.gca()
