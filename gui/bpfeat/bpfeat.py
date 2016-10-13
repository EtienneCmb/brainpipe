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




class featInit(QtGui.QMainWindow, Ui_bpui, uiTabs):

    """Load all ui elements from pyqt
    """

    def __init__(self):
        # Create the main window :
        super(featInit, self).__init__(None)
        self.setupUi(self)
        uiTabs.__init__(self)

        # Quit :
        exitAction = QtGui.QAction(QtGui.QIcon('exit.png'), '&Exit', self)        
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(QtGui.qApp.quit)
        self.menuOption.addAction(exitAction)

        # Display panel settings :
        self.actionTab_settings.triggered.connect(self.fcn_dispTabSettings)

        self.visTo.valueChanged.connect(self.fcn_updateElec)

        # fig1 = Figure()
        # ax1f1 = fig1.add_subplot(111)
        # ax1f1.plot(np.random.rand(5))
     

        # self.addmpl(fig1)

        self._x = np.random.rand(63, 1000, 100)
        self._time = np.arange(1000)
        self._elec = 0
        # self._timeseries()

        # model = QtGui.QFileSystemModel()
        # model.setRootPath( QtCore.QDir.currentPath() )

        # set the model
        # self.treeView.setModel(model)



    def fcn_dispTabSettings(self):
        """
        """
        self.tabWidget.setVisible(self.actionTab_settings.isChecked())

    def fcn_updateElec(self):
        """
        """
        self._elec += 1
        self._timeseries()


    def _timeseries(self):
        """
        """
        fig1 = plt.figure()
        BorderPlot(self._time, self._x[self._elec, ...], color='#ab4642', xlabel='Time (ms)', ylabel='uV')

        try:
            self.rmmpl()
        except:
            pass
        self.addmpl(fig1)

    def addmpl(self, fig):
        self.canvas = FigureCanvas(fig)
        self.mplbox.addWidget(self.canvas)
        self.canvas.draw()
        self.toolbar = NavigationToolbar(self.canvas, 
                self.mplwindow, coordinates=True)
        self.mplbox.addWidget(self.toolbar)

    def rmmpl(self,):
        self.mplbox.removeWidget(self.canvas)
        self.canvas.close()
        self.mplbox.removeWidget(self.toolbar)
        self.toolbar.close()

class bpfeat(featInit):

    """
    """

    def __init__(self, *args, **kwargs):
        # Create the app and initialize all graphical elements :
        self._app = QtGui.QApplication(sys.argv)
        featInit.__init__(self, bgcolor)

        # Run it :


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    main = featInit()
    main.showMaximized()
    sys.exit(app.exec_())

