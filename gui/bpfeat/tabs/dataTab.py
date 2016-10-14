from PyQt4 import QtGui, QtCore
import os
import pickle
from scipy.io import loadmat
import numpy as np
import pandas as pd

class dataTab(object):

    """docstring for dataTab
    """

    def __init__(self):

        # Manage path to data :
        self._cd = '/media/etienne/E438C4AE38C480D2/Users/Etienne Combrisson/Documents/MATLAB/Sujets/C_rev/Données' #QtCore.QDir.currentPath()
        self._file, self._data, self._dataset = None, pd.DataFrame({}), 0
        self.uicd.clicked.connect(self.fcn_changecd)
        self.treeFiles.clicked.connect(self.fcn_varInsideFile)

        # Dataset :
        self.selectData.currentIndexChanged.connect(self.fcn_selectDataVar)
        self.fcn_detectFilesIncd()

    ########################################################################
    # PATH MANAGEMENT
    ########################################################################
    def fcn_changecd(self):
        """Change the current directory
        """
        self._cd = str(QtGui.QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.fcn_detectFilesIncd()

    def fcn_detectFilesIncd(self):
        """Update files in current directory
        """
        filter = ["*.pickle", '*.npy', '*.mat', '*.npz'] 
        model = QtGui.QFileSystemModel()
        model.setFilter(QtCore.QDir.Files)
        model.setNameFilters(filter)
        model.setNameFilterDisables(False) 
        model.setRootPath(self._cd)
        self.treeFiles.setModel(model)
        self.treeFiles.setRootIndex(model.index(self._cd))
        self.model = model

    def fcn_varInsideFile(self):
        """Detect variables inside the file
        """
        # Find the name of the file :
        index = self.treeFiles.selectedIndexes()[0]
        self._file = index.model().itemData(index)[0]
        # Load the file :
        self._loadFile()


    def _loadFile(self):
        """Load the file
        """
        # Join path and filename according to the system :
        file = os.path.join(self._cd, self._file)
        # Find extension :
        fileName, fileExt = os.path.splitext(self._file)
        # Pickle :
        if fileExt == '.pickle':
            with open(file, "rb") as f:
                data = pickle.load(f)
        # Matlab :
        elif fileExt == '.mat':
            data = loadmat(file)
        # Numpy (single array)
        elif fileExt == '.npz':
            data = np.load(file)
        self._temp = data
        # Update variables in files :
        self.selectData.clear()
        self.selectData.addItems(list(data.keys()))
        self.fcn_selectDataVar()
        self.fcn_userMsg('File loaded. Specify data variable')

    ########################################################################
    # SELECT VARIABLE
    ########################################################################
    def fcn_selectDataVar(self):
        """Select data
        """
        self.cleanfig()
        # Current variable name :
        curDataVar = self.selectData.currentText()
        var = self._temp[curDataVar]
        # Check variable :
        if not isinstance(var, np.ndarray):
            self.dataShape.setText('DATA MUST A MATRIX')
            self._data = None
            self.fcn_manageTabs(False)
            self.fcn_userMsg('Data not compatible')
        else:
            if var.ndim != 3:
                self.dataShape.setText('DATA MUST HAVE 3 DIMENSIONS\n(N_elec x N_pts x N_trials)')
                self._data = None
                self.fcn_manageTabs(False)
                self.fcn_userMsg('Data not compatible')
            else:
                self._sf = self.selectSf.value()
                self.dataShape.setText('Shape: '+str(var.shape))
                self._data = var
                self._nelec, self._npts, self._ntrials = self._data.shape
                self._timevec = (np.arange(1, self._npts+1)/self._sf).ravel()
                self.fcn_userMsg('Data loaded !!')
                self.fcn_manageTabs(True)
                # Initialize most basics objects :
                self.fcn_powerInit()
                self.fcn_tfInit()
                self.getConfig.setVisible(True)
                self.cleanAx.setVisible(True)
                # self.fcn_rawChan()
                # self._updatePowerObject()
                print('-> Data loaded')

