import sys
from PyQt4 import QtGui, QtCore
import numpy as np

from PyQt4.uic import loadUiType

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)
import matplotlib.pyplot as plt


from ui import Ui_bpui
from tabs import uiTabs
from brainpipe.visual import *
from brainpipe.gui.subpltClass import subpltClass


class clfInit(QtGui.QMainWindow, Ui_bpui, uiTabs, subpltClass):

    """Load all ui elements from pyqt."""

    def __init__(self):
        # Create the main window :
        super(clfInit, self).__init__(None)
        self.setupUi(self)
        uiTabs.__init__(self)
        subpltClass.__init__(self)
        self._fig = plt.figure()
        self.actionAdvanced.triggered.connect(self.fcn_advanced)

        self.fcn_userMsg('Load power files for eyes open / closed')

    def fcn_advanced(self):
        """"""
        # self.tabWidget.setTabEnabled(1, self.actionAdvanced.isChecked())
        # self.tabWidget.setTabEnabled(2, self.actionAdvanced.isChecked())
        self.clfGrp.setVisible(self.actionAdvanced.isChecked())


class bpclf(clfInit):

    """"""

    def __init__(self, *args, **kwargs):
        # Create the app and initialize all graphical elements :
        self._app = QtGui.QApplication(sys.argv)
        clfInit.__init__(self, bgcolor)


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    main = clfInit()
    main.showMaximized()
    sys.exit(app.exec_())
