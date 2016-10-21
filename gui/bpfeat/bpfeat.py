import sys
from PyQt4 import QtGui, QtCore
import numpy as np

from PyQt4.uic import loadUiType

import matplotlib.pyplot as plt


from ui import Ui_bpui
from tabs import uiTabs
from brainpipe.visual import *
from brainpipe.gui.subpltClass import subpltClass




class featInit(QtGui.QMainWindow, Ui_bpui, uiTabs, subpltClass):

    """Load all ui elements from pyqt
    """

    def __init__(self):
        # Create the main window :
        super(featInit, self).__init__(None)
        self.setupUi(self)
        uiTabs.__init__(self)
        subpltClass.__init__(self)

        # Set tabs enable false :
        self.fcn_manageTabs(False)
        self.tabWidget_2.setTabEnabled(2, False)
        self.tabWidget_3.setTabEnabled(1, False)
        self.advResume.setVisible(False)
        self.tfResume.setVisible(False)
        self.fcn_userMsg('Wait for loading data')

        # Quit :
        exitAction = QtGui.QAction(QtGui.QIcon('exit.png'), '&Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(QtGui.qApp.quit)
        self.menuOption.addAction(exitAction)

        # Display panel settings :
        self.actionTab_settings.triggered.connect(self.fcn_dispTabSettings)
        self.actionAdvanced.triggered.connect(self.fcn_advanced)

        # Figure :
        self._fig = plt.figure()
        self._ax = plt.gca()
        self.cleanAx.clicked.connect(self.fcn_cleanAx)
        self.cleanAx.setVisible(False)

        self.getConfig.clicked.connect(self.fcn_userConfig)
        self.getConfig.setVisible(False)



    def fcn_manageTabs(self, setto):
        """
        """
        # Set tabs off :
        self.tabWidget.setTabEnabled(1, setto)
        self.tabWidget.setTabEnabled(2, setto)
        self.tabWidget.setTabEnabled(3, setto)
        # Reset data :
        if not setto:
            self._data = None
            self._data2plot = None
            self._nelec, self._npts, self._ntrials = None, None, None
            self._timevec = None
            self.powSave.setVisible(False)
        # Clear plot :
        # self.rmmpl()


    ################################################################
    # MENU ACTION
    ################################################################
    def fcn_dispTabSettings(self):
        """Display the setting panel
        """
        actionBool = self.actionTab_settings.isChecked()
        self.tabWidget.setVisible(actionBool)
        self.userMsg.setVisible(actionBool)


    def fcn_advanced(self):
        """Display advanced tab
        """
        advBool = self.actionAdvanced.isChecked()
        self.tabWidget_2.setTabEnabled(2, advBool)
        self.tabWidget_3.setTabEnabled(1, advBool)
        self.advResume.setVisible(advBool)
        self.tfResume.setVisible(advBool)
        self.fcn_manageTabs(True)
        if advBool:
            self.fcn_userMsg("Advanced mode activate (big boy :D)")
        else:
            self.fcn_userMsg("Basic mode activate")


    ################################################################
    # USER SYSTEM
    ################################################################
    def fcn_userConfig(self):
        """Get current config
        """
        # Data configuration :
        self.fcn_userMsg('Configuration copied to clipboard !')
        separator = '\n\n--------------------\n\n'
        dataConf = '___ DATA SETUP ___\nFile: '+self._file+'\nAdvanced mode: '+str(self.actionAdvanced.isChecked())+'\nSampling frequency: '+str(self._sf)
        # Power configuration :
        powbsl = str(self._baseline)
        powerConf = '___ POWER SETUP ___\n\n'+str(self._power)+'\n\nFrequencies:\n'+str(self._fce)+'\n\nBaseline: '+powbsl
        # TF configuration :
        tfbsl = str(self._tfArgs['baseline'])
        tfce = 'From: '+str(self.tfFrom.value())+', to: '+str(self.tfTo.value())+', width: '+str(self.tfWidth.value())+', step: '+str(self.tfStep.value())
        tfConf = '___ TF SETUP ___\n\n'+str(self._tf)+'\n\nFrequencies:\n'+tfce+'\n\nBaseline: '+tfbsl

        config = dataConf+separator+powerConf+separator+tfConf
        QtGui.QApplication.clipboard().setText(config)


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

