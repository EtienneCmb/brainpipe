#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

__all__ = ['PltUtils', 'rmaxis', 'despine']


class PltUtils(object):

    """
    **kwargs:
        title: title of plot [def: '']

        xlabel: label of x-axis [def: '']

        ylabel: label of y-axis [def: '']

        xlim: limit of the x-axis [def: [], current limit of x]

        ylim: limit of the y-axis [def: [], current limit of y]

        xticks: ticks of x-axis [def: [], current x-ticks]

        yticks: ticks of y-axis [def: [], current y-ticks]

        xticklabels: label of the x-ticks [def: [], current x-ticklabels]

        yticklabels: label of the y-ticks [def: [], current y-ticklabels]

        style: style of the plot [def: None]

        dpax: List of axis to despine ['left', 'right', 'top', 'bottom']

        rmax: Remove axis ['left', 'right', 'top', 'bottom']

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
    """Remove ticks and axis of a existing plot.

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
    """Despine axis of a existing plot.

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
