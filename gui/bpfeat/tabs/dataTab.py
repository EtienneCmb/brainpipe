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
        self.addDataW.setVisible(False)

        # Manage path to data :
        self._cd = '/media/etienne/E438C4AE38C480D2/Users/Etienne Combrisson/Documents/MATLAB/Sujets/C_rev/Données' #QtCore.QDir.currentPath()
        self._file, self._data, self._dataset = None, pd.DataFrame({}), 0
        self.uicd.clicked.connect(self.fcn_changecd)
        self.treeFiles.clicked.connect(self.fcn_varInsideFile)

        # Dataset :
        self.selectData.currentIndexChanged.connect(self.fcn_selectDataVar)
        self.selectName.editingFinished.connect(self._checkVisibleInfo)
        self.addNewData.clicked.connect(self.fcn_addDataset)
        self.rmDataset.clicked.connect(self.fcn_rmDataset)

        self.fcn_updateTreeFiles()

    ########################################################################
    # PATH MANAGEMENT
    ########################################################################
    def fcn_changecd(self):
        """Change the current directory
        """
        self._cd = str(QtGui.QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.fcn_updateTreeFiles()

    def fcn_updateTreeFiles(self):
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

    ########################################################################
    # SELECT VARIABLE
    ########################################################################
    def fcn_selectDataVar(self):
        """Select data
        """
        # Current variable name :
        curDataVar = self.selectData.currentText()
        var = self._temp[curDataVar]
        # Check variable :
        if not isinstance(var, np.ndarray):
            self.dataShape.setText('DATA MUST A MATRIX')
            self._var = None
            self.addDataW.setVisible(False)
        else:
            if var.ndim != 3:
                self.dataShape.setText('DATA MUST HAVE 3 DIMENSIONS\n(N_elec x N_pts x N_trials)')
                self._var = None
                self.addDataW.setVisible(False)
            else:
                self.dataShape.setText('Shape: '+str(var.shape))
                self._var = var
                self.addDataW.setVisible(True)
        self._checkVisibleInfo()


    def _checkVisibleInfo(self):
        """
        """
        check = []
        # Check data :
        check.append(self._var is not None)
        # Check name :
        check.append(self.selectName.text() is not '')
        # Display push buttons :
        self.addNewData.setVisible(np.all(check))


    def fcn_addDataset(self):
        """
        """
        datasetName = self.selectName.text()
        if datasetName != '':
            if self._var is not None:
                self._data[datasetName] = [{'data':self._var, 'sf':self.selectSf.value()}]
                self.datasetValid.setText('DATA ADDED TO SUMMARIZE :D')
                self.fcn_updateSummarize()
            else:
                self.datasetValid.setText('NO DATA DETECTED')
        else:
            self.datasetValid.setText('PLEASE ENTER A NAME FOR THE DATASET')


    ########################################################################
    # SUMMARIZE
    ########################################################################
    def fcn_updateSummarize(self):
        """
        """
        self.sumTable.clear()
        # print(self._data)
        for k, name in enumerate(self._data.keys()):
            self.sumTable.insertRow(k)
            curdata = self._data[name][0]
            self.sumTable.setItem(k , 0, QtGui.QTableWidgetItem(name))
            self.sumTable.setItem(k , 1, QtGui.QTableWidgetItem(str(curdata['data'].shape)))
            self.sumTable.setItem(k , 2, QtGui.QTableWidgetItem(str(curdata['sf'])))

    def fcn_rmDataset(self):
        """
        """
        indices = self.sumTable.selectionModel().selectedRows()
        for index in indices:
            name = self.sumTable.item(index.row(), 0).text()
            # self.sumTable.removeRow(index.row())
            del self._data[name]
        self.fcn_updateSummarize()

