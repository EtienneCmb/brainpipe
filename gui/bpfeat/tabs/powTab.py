import sys
from PyQt4 import QtGui
from ui import Ui_bpui

import numpy as np
from brainpipe.feature import *
import pandas as pd


class powTab(object):

    """docstring for powTab
    """

    def __init__(self, ):
        self._sf = 1000
        self._npts = 4000

        # Manadge advanced panel :
        self.powAdvW.setVisible(False)
        self.powAdv.clicked.connect(self.fcn_advancedVisible)

        # Table control :
        self._fce = pd.DataFrame({})
        self.powTable.currentItemChanged.connect(self._getTableFcy)
        self.powFceAdd.clicked.connect(self.fcn_addLigne)
        self.powFceRm.clicked.connect(self.fcn_rmLigne)

        # -----------------------------------------------------
        # ADVANCED
        # -----------------------------------------------------
        # Baseline :
        self.powBslW.setVisible(False)
        self.powBsl.clicked.connect(self.fcn_bslVisible)
        self.powBslFrom.valueChanged.connect(self.fcn_bslVisible)
        self.powBslTo.valueChanged.connect(self.fcn_bslVisible)
        # Norm :
        self.powNorm.valueChanged.connect(self._updatePowerObject)
        self.powBut.currentIndexChanged.connect(self._updatePowerObject)
        self.powOrder.valueChanged.connect(self._updatePowerObject)
        self.powCycle.valueChanged.connect(self._updatePowerObject)
        self.powTrans.currentIndexChanged.connect(self._updatePowerObject)
        self.powWidth.valueChanged.connect(self._updatePowerObject)
        self.powDetrend.clicked.connect(self._updatePowerObject)

        self.fcn_bslVisible()
        self._getTableFcy()


    ##############################################################
    # TABLE
    ##############################################################
    def fcn_addLigne(self):
        """Add a ligne to the table
        """
        # Add row :
        rowPosition = self.powTable.rowCount()
        self.powTable.insertRow(rowPosition)
        # Update data :
        self._getTableFcy()

    def fcn_rmLigne(self):
        """Remove ligne from table
        """
        # Remove row(s)
        indices = self.powTable.selectionModel().selectedRows()
        for index in indices:
            self.powTable.removeRow(index.row())
        # Update data :
        self._getTableFcy()

    def _getTableFcy(self):
        """Get table name and frequency
        """
        model = self.powTable.model()
        data = []
        bandnames = self._fce.keys()
        self._fce = pd.DataFrame({})
        for row in range(self.powTable.rowCount()):
            # Get row name :
            name = model.data(model.index(row, 0))
            fce_sta = model.data(model.index(row, 1))
            fce_end = model.data(model.index(row, 2))
            # Check if name exist :
            if name is None:
                name = 'band'
                q = 1
                while name in list(bandnames):
                    name = 'band' + str(q)
                    q +=1
            # Starting frequency :
            if fce_sta is None:
                fce_sta = 1.0
            else:
                fce_sta = float(fce_sta)
            # Ending frequency
            if fce_end is None:
                fce_end = fce_sta+1.0
            if not isinstance(fce_end, float):
                fce_end = float(fce_end)
            if fce_end <= fce_sta:
                fce_end = fce_sta+1.0
            # Update fce :
            self._fce[name] = [fce_sta, fce_end]
            # Set elements :
            self.powTable.setItem(row, 0, QtGui.QTableWidgetItem(name))
            self.powTable.setItem(row, 1, QtGui.QTableWidgetItem(str(float(fce_sta))))
            self.powTable.setItem(row, 2, QtGui.QTableWidgetItem(str(float(fce_end))))
        # Update power object :
        self._updatePowerObject()


    ##############################################################
    # BASELINE
    ##############################################################
    def fcn_bslVisible(self):
        """Baseline panel
        """
        self.powBslW.setVisible(self.powBsl.isChecked())
        if self.powBsl.isChecked():
            bsl_sta = self.powBslFrom.value()
            bsl_end = self.powBslTo.value()
            if bsl_end <= bsl_sta:
                bsl_end += 1
                self.powBslTo.setValue(bsl_end)
            self._baseline = [bsl_sta, bsl_end]
        else:
            self._baseline = None


    ##############################################################
    # ADVANCED
    ##############################################################
    def fcn_advancedVisible(self):
        """Advanced panel
        """
        self.powAdvW.setVisible(self.powAdv.isChecked())
        self._updatePowerObject()



    def _updatePowerObject(self):
        """Update power and set string
        """
        # Build frequency vector :
        f = [list(self._fce[k]) for k in self._fce.keys()]
        # Get methods :
        self.fcn_bslVisible() # Get baseline
        # Get norm :
        self._norm = self.powNorm.value()
        # Get filter settings :
        self._filtname = self.powBut.currentText()
        self._order = self.powOrder.value()
        self._cycle = self.powCycle.value()
        # Transformation :
        self._method = self.powTrans.currentText()
        self._wwidth = self.powWidth.value()
        # Detrend:
        self._detrend = self.powDetrend.isChecked()
        # Update power object :
        self._power = power(self._sf, self._npts, f=f, baseline=self._baseline, norm=self._norm,
                            method=self._method, filtname=self._filtname, order=self._order,
                            cycle=self._cycle, wltWidth=self._wwidth, dtrd=self._detrend)
        self.powResume.setText(str(self._power))

