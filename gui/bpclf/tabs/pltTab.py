import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from mne.viz import plot_topomap

class pltTab(object):

    """Docstring for pltTab. """

    def __init__(self):
        """TODO: to be defined1. """
        # Plot type control :
        self.barBut.clicked.connect(self.fcn_pltType)
        self.topoBut.clicked.connect(self.fcn_pltType)
        self.topoW.setVisible(False)

        # Barplot :
        self.updatePlot.clicked.connect(self.fcn_updatePlot)

        # Topoplot :
        # self.vminSpin.valueChanged.connect(self.fcn_topoPlot)
        # self.vmaxSpin.valueChanged.connect(self.fcn_topoPlot)
        # self.cmapCombo.currentIndexChanged.connect(self.fcn_topoPlot)
        self.cmap_lst = [k for k in list(cm.datad.keys()) + list(cm.cmaps_listed.keys()) if not k.find('_r') + 1]
        self.cmap_lst.sort()
        self.cmapCombo.addItems(self.cmap_lst)
        virididx = self.cmap_lst.index('viridis')
        self.cmapCombo.setCurrentIndex(virididx)


    def fcn_pltType(self):
        self.barW.setVisible(self.barBut.isChecked())
        self.topoW.setVisible(self.topoBut.isChecked())
        if self.barBut.isChecked():
            self.fcn_barPlot()
        elif self.topoBut.isChecked():
            self.fcn_topoPlot()


    def fcn_updatePlot(self):
        if self.barBut.isChecked():
            self.fcn_barPlot()
        elif self.topoBut.isChecked():
            self.fcn_topoPlot()


    def fcn_barPlot(self):
        """Plot classification
        """
        # Get da upper to :
        xticks = np.arange(self._nelec)
        daUpper = self._da.mean(0) >= self.daUpper.value()
        xticks = xticks[daUpper]
        self.cleanfig()
        self._fig = plt.figure()
        self._cla.daplot(self._da[:, daUpper], chance_method='bino',
                         chance_level=self._chanceLevel)
        ax = plt.gca()
        ax.set_xticklabels(xticks, rotation=45)
        plt.xlabel('Channels')
        title = 'Eyes open vs Eyes Closed classification using power in '+self._fce[self._selectedFce]+' band\n(Classifier: '+self._clf.lgStr+' / Cross-validation: '+self._cv.lgStr+')'
        plt.title(title, y=1.02)
        # plt.xticklabels()
        self.addmpl(self._fig)


    def fcn_topoPlot(self):
        self.cleanfig()
        self._fig = plt.figure()
        plot_topomap(self._da.mean(0).ravel(), self._chxy, show=False,
                     vmin=self.vminSpin.value(), vmax=self.vmaxSpin.value(),
                     cmap=self.cmapCombo.currentText())
        title = 'Eyes open vs Eyes Closed classification using power in '+self._fce[self._selectedFce]+' band\n(Classifier: '+self._clf.lgStr+' / Cross-validation: '+self._cv.lgStr+'), vmin='+str(self.vminSpin.value())+', vmax='+str(self.vmaxSpin.value())
        plt.title(title)
        plt.axis('square')
        self.addmpl(self._fig)
