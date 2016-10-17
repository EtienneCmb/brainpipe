from PyQt4 import QtGui
import os
from scipy.io import loadmat
import numpy as np

class loadTab(object):

    """docstring for loadTab
    """

    def __init__(self):
        self._resetLoading()
        # Load buttons :
        self.loadFile1.clicked.connect(self.fcn_loadFile1)
        self.loadFile2.clicked.connect(self.fcn_loadFile2)
        self.resetData.clicked.connect(self._resetLoading)


    ################################################################
    # MENU ACTION
    ################################################################
    def fcn_loadFile1(self):
        """Load file 1
        """
        self._data1 = self._loadfile(guistr='1')
        self._checkLoadedFiles()


    def fcn_loadFile2(self):
        """Load file 2
        """
        self._data2 = self._loadfile(guistr='2')
        self._checkLoadedFiles()


    def _loadfile(self, guistr=''):
        """Sub-loading function
        """
        loadname = QtGui.QFileDialog.getOpenFileName(self, 'Load file '+guistr, os.getenv('HOME'))
        mat = loadmat(loadname)
        loadstr = str(loadname)
        self.fcn_userMsg('File '+guistr+' loaded :D')
        # Add shape and fce list :
        eval("self.d"+guistr+"Shape.setText(str(mat['x'].shape))")
        eval("self.d"+guistr+"Fce.clear()")
        eval("self.d"+guistr+"Fce.addItems(list(mat['fcename']))")
        eval("self.d"+guistr+"Panel.setVisible(True)")
        eval("self.d"+guistr+"Name.setText('"+loadstr+"')")
        return mat


    def _checkLoadedFiles(self):
        """
        """
        if (self._data1 is not None) and (self._data2 is not None):
            self._checkVariable()


    def _resetLoading(self):
        """Reset loaded files
        """
        self._data1, self._data2 = None, None
        self.tabWidget.setTabEnabled(1, False)
        self.tabWidget.setTabEnabled(2, False)
        self.d1Fce.clear()
        self.d1Panel.setVisible(False)
        self.d1Shape.setText('   ')
        self.d1Name.setText('   ')
        self.d2Fce.clear()
        self.d2Panel.setVisible(False)
        self.d2Name.setText('   ')


    ################################################################
    # VARIABLE MANAGEMENT
    ################################################################
    def _checkVariable(self):
        """Check variables inside data1 and data2
        """
        # Check :
        x1, x2 = self._data1['x'], self._data2['x']
        fce1, fce2 = list(self._data1['fcename']), list(self._data2['fcename'])
        if x1.shape != x2.shape: # Check matrices inside
            self.fcn_userMsg("Les données dans les deux fichiers n'ont pas la même taille...")
        
        elif fce1 != fce2: # Check Frequency inside
            self.fcn_userMsg("Les fréquences extraites dans les deux fichiers ne sont pas les mêmes...")
        else:
            # Build classification matrices :
            self._x = np.concatenate((x1, x2), axis=2)
            self._nfce, self._nelec, self._ntrials = self._x.shape
            self._y = [0]*x1.shape[2] + [1]*x2.shape[2]
            self._fce = fce1
            # Initialize classifier objects :
            self._clfInit()
            self._selectedFce = 0
            # Manage displayed elements :
            self.SFfce.addItems(list(fce1))
            self.tabWidget.setTabEnabled(1, True)
            self.fcn_userMsg('Now you can run the classification. Enjoy')