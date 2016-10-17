import sys
from PyQt4 import QtGui
from ui import Ui_bpui
import os
from time import sleep

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.io import loadmat, savemat

from brainpipe.feature import power
from brainpipe.visual import BorderPlot


class powTab(object):

    """docstring for powTab
    """

    def __init__(self, ):
        # Table control :
        self._fce = pd.DataFrame({})
        self.powTable.currentItemChanged.connect(self.fcn_tablechanged)
        self.powFceAdd.clicked.connect(self.fcn_addLigne)
        self.powFceRm.clicked.connect(self.fcn_rmLigne)
        self.powFceUp.clicked.connect(self._getTableFcy)
        self.powFceUp.setVisible(False)
        self.powRun.setVisible(False)

        # Channel selection :
        self.powChanNb.setVisible(False)
        self.powAllChan.clicked.connect(self.fcn_selectChan)
        self.powSelectChan.clicked.connect(self.fcn_selectChan)
        self.powChanNb.valueChanged.connect(self.fcn_selectChan)
        self.fcn_selectChan()

        # Baseline :
        self.powBslW.setVisible(False)
        self.powBsl.clicked.connect(self.fcn_bslVisible)
        self.powBslFrom.valueChanged.connect(self.fcn_bslVisible)
        self.powBslTo.valueChanged.connect(self.fcn_bslVisible)
        self.fcn_bslVisible()

        # Run
        self.powRun.clicked.connect(self.fcn_powRun)
        self._getTableFcy()

        # Plotting :
        self.powChan.setVisible(False)
        self.powFce2Plt.currentIndexChanged.connect(self.fcn_powPlot)
        self.powPer1D.clicked.connect(self.fcn_powPlot)
        self.powPer2D.clicked.connect(self.fcn_powPlot)
        self.powAcross.clicked.connect(self.fcn_powPlot)
        self.powChan.valueChanged.connect(self.fcn_powPlot)

        # Load/save :
        self.actionLoadPow.triggered.connect(self.fcn_powLoad)
        self.powSave.clicked.connect(self.fcn_powSave)
        self.powLoad.clicked.connect(self.fcn_powLoad)
        self.powSave.setVisible(False)
        self.powLoad.setVisible(False)

        # -----------------------------------------------------------------
        # Advanced tab :
        # -----------------------------------------------------------------
        # Advanced filtering :
        self._PowArgs = {}
        self.powNorm.valueChanged.connect(self._AdvSetup)
        self.powBut.currentIndexChanged.connect(self._AdvSetup)
        self.powOrder.valueChanged.connect(self._AdvSetup)
        self.powCycle.valueChanged.connect(self._AdvSetup)
        self.powTrans.currentIndexChanged.connect(self._AdvSetup)
        self.powWidthW.valueChanged.connect(self._AdvSetup)
        self.powDetrend.clicked.connect(self._AdvSetup)
        self.powRunCustom.clicked.connect(self.fcn_powRun)
        # Advanced vizualization :
        self._pow2DPlt = {}
        self.powVmin.valueChanged.connect(self._AdvSetup)
        self.powVmax.valueChanged.connect(self._AdvSetup)
        self.powCmap.currentIndexChanged.connect(self._AdvSetup)
        self.powPltUpd.clicked.connect(self.fcn_powPlot)
        self.cmap_lst = [k for k in list(cm.datad.keys()) + list(cm.cmaps_listed.keys()) if not k.find('_r') + 1]
        self.cmap_lst.sort()
        self.powCmap.addItems(self.cmap_lst)



    def fcn_powerInit(self):
        """Initialize the most basic power object
        """
        self._power = power(self._sf, self._npts)
        print('-> Power object created')

    ##############################################################
    # TABLE
    ##############################################################
    def fcn_tablechanged(self):
        """
        """
        self.powFceUp.setVisible(True)
        self.powRun.setVisible(False)


    def fcn_addLigne(self):
        """Add a ligne to the table
        """
        # Add row :
        rowPosition = self.powTable.rowCount()
        self.powTable.insertRow(rowPosition)
        self.powFceUp.setVisible(True)
        self.powRun.setVisible(False)
        # Manage row name :
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
            self._fce[name] = [[fce_sta, fce_end]]
            # Set elements :
            self.powTable.setItem(row, 0, QtGui.QTableWidgetItem(name))
            self.powTable.setItem(row, 1, QtGui.QTableWidgetItem(str(float(fce_sta))))
            self.powTable.setItem(row, 2, QtGui.QTableWidgetItem(str(float(fce_end))))


    def fcn_rmLigne(self):
        """Remove ligne from table
        """
        # Remove row(s)
        indices = self.powTable.selectionModel().selectedRows()
        for index in indices:
            self.powTable.removeRow(index.row())
        self.powFceUp.setVisible(True)
        self.powRun.setVisible(False)


    def _getTableFcy(self):
        """Get table name and frequency
        """
        self.cleanfig()
        model = self.powTable.model()
        self._fce = pd.DataFrame({})
        for row in range(self.powTable.rowCount()):
            # Get row name :
            name = model.data(model.index(row, 0))
            fce_sta = model.data(model.index(row, 1))
            fce_end = model.data(model.index(row, 2))
            # Update fce :
            self._fce[name] = [[float(fce_sta), float(fce_end)]]
        self._f2extract = [self._fce[k][0] for k in self._fce.keys()]
        print('-> Frequency bands updated to:\n', self._fce)
        self.powFceUp.setVisible(False)
        self.powRun.setVisible(True)
        # Update plotable frequency :
        self.powFce2Plt.clear()
        self.powFce2Plt.addItems(list(self._fce.keys()))



    ##############################################################
    # CHANNEL
    ##############################################################
    def fcn_selectChan(self):
        """Select channel
        """
        try:
            if self.powAllChan.isChecked():
                self._powSelect = np.arange(self._nelec)
                self.powChanNb.setVisible(False)
                self.powChan.setEnabled(True)
                self.powAcross.setEnabled(True)
            elif self.powSelectChan.isChecked():
                self.powChanNb.setVisible(True)
                self.powChan.setValue(0)
                self.powChan.setEnabled(False)
                self.powAcross.setEnabled(False)
                self._powSelect = np.array([self.powChanNb.value()])
            self.powChan.setMaximum(self._powSelect.max())
        except:
            self.fcn_userMsg('Oups :d (power channel)')


    ##############################################################
    # BASELINE
    ##############################################################
    def fcn_bslVisible(self):
        """Baseline panel
        """
        self.powBslW.setVisible(self.powBsl.isChecked())
        if self.powBsl.isChecked():
            self._baseline = [self.powBslFrom.value(), self.powBslTo.value()]
        else:
            self._baseline = None
        try:
            self._AdvSetup()
        except:
            self.fcn_userMsg('Oups :d (power baseline)')



    ##############################################################
    # PLOTTING 
    ##############################################################
    def fcn_powPlot(self):
        """Plot power
        """
        # Get current frequency to plot :
        index = self.powFce2Plt.currentIndex()
        name = self.powFce2Plt.currentText()
        try:
            # Across or per channel :
            if self.powAcross.isChecked():
                self.powChan.setVisible(False)
                data2plot = self._fce2plt[index, ...].mean(0)
                title = 'Power in '+name+' band across channels'
            elif self.powPer1D.isChecked():
                self.powChan.setVisible(True)
                data2plot = self._fce2plt[index, self.powChan.value(), :, :]
                title = '1D Power in '+name+' band for channel'+str(self._powSelect[self.powChan.value()])
            elif self.powPer2D.isChecked():
                self.powChan.setVisible(True)
                data2plot = self._fce2plt[index, self.powChan.value(), :, :]
                title = '2D Power in '+name+' band for channel'+str(self._powSelect[self.powChan.value()])
            self.fcn_userMsg('Plotting: '+title)
            # Send the figure :
            self.cleanfig()
            self._fig = plt.figure()
            if self.powAcross.isChecked() or self.powPer1D.isChecked():
                BorderPlot(self._timevec, data2plot, color='#ab4642', kind='sem')
                self._setplot(data2plot, title, 'Time (ms)', 'uV²/hz')
            else:
                self._power.plot2D(self._fig, data2plot.T, cblabel=name+' power modulations',
                                   xvec=self._timevec, **self._pow2DPlt)
                self._setplot(data2plot, title, 'Time (ms)', 'Trials')
        except:
            self.fcn_userMsg('Oups :d (power plot)')



    ##############################################################
    # LOAD / SAVE 
    ##############################################################
    def fcn_powSave(self):
        """Save current power
        """
        # Define savename :
        savename = QtGui.QFileDialog.getSaveFileName(self, 'Save power configuration', os.getenv('HOME'))
        savename = os.path.splitext(savename)[0] + '.mat'
        # Define data to save :
        data = {}
        data['x'] = self._fce2plt.mean(2)
        data['powObj'] = self._power
        data['sf'] = self._sf
        data['conf'] = str(self._power)
        data['fcename'] = list(self._fce.keys())
        data['fcemat'] = np.array(list(self._fce.values[0]))
        data['chanselect'] = self._powSelect
        data['chanval'] = self.powChan.value()
        # Save then clean data :
        savemat(savename, data)
        del data
        self.fcn_userMsg('Power saved !')


    def fcn_powLoad(self):
        """Load power
        """
        self.cleanfig()
        self.rmmpl()
        self.tabWidget.setTabEnabled(2, True)
        # Get loadname :
        loadname = QtGui.QFileDialog.getOpenFileName(self, 'Load power configuration', os.getenv('HOME'))
        # Load file :
        mat = loadmat(loadname)
        # Set parameters :
        self._fce2plt = mat['x']
        self._power = mat['powObj']
        self._sf = mat['sf']
        self._nelec, self._npts, self._ntrials = list(self._fce2plt.shape)[1::]
        self._timevec = (np.arange(1, self._npts+1)/self._sf).ravel()
        self._powSelect = mat['chanselect']
        self.powChan.setValue(mat['chanval'])
        # Reconstruct fce :
        self._fce = pd.DataFrame({})
        for name, val in zip(mat['fcename'], mat['fcemat']):
            self._fce[name] = val
        self._getTableFcy()
        self.fcn_powPlot()
        self.fcn_userMsg('Power loaded (might be buggy...)')



    ##############################################################
    # RUN POWER COMPUTATION
    ##############################################################
    def fcn_powRun(self):
        """Compute power
        """
        # Get basic or advanced setup :
        boolSetup = self.actionAdvanced.isChecked()
        self.fcn_userMsg('Wait while power is computing (basics setup)')
        print('-> Running power computation')
        # Get selected channel :
        self.fcn_selectChan()
        # Use either basic or advanced setup :
        if boolSetup:
            self._AdvSetup()
        else:
            self._BasicSetup()
        # Run power computation :
        self._fce2plt = self._power.get(self._data[self._powSelect, ...])[0]
        self._fce2plt /= np.abs(self._fce2plt).max()
        if self._fce2plt.ndim == 3:
            n1, n2, n3 = self._fce2plt.shape
            self._fce2plt = self._fce2plt.reshape(n1, 1, n2, n3)
        # Make save visible :
        self.powSave.setVisible(True)
        # Plot :
        self.fcn_powPlot()
        if boolSetup:
            self.fcn_userMsg('Power computed (with advanced setup) !')
        else:
            self.fcn_userMsg('Power computed (with basic setup) !')



    ##############################################################
    # BASIC / ADVANCED
    ##############################################################
    def _BasicSetup(self):
        """Basic setup
        """
        try:
            # -------------------------------------------
            # BASIC FILTERING
            # -------------------------------------------
            # Build frequency vector :
            self._PowArgs['f'] = self._f2extract
            # Get norm :
            self._PowArgs['norm'] = 0
            # Get filter settings :
            self._PowArgs['filtname'] = 'butter'
            self._PowArgs['order'] = 3
            self._PowArgs['cycle'] = 3
            # Transformation :
            self._PowArgs['method'] = 'wavelet'
            self._PowArgs['wltWidth'] = 7
            # Detrend:
            self._PowArgs['dtrd'] = False
            # Update power object :
            self._power = power(self._sf, self._npts, baseline=self._baseline,
                                **self._PowArgs)
            self.advResume.setText(str(self._power))
            # Print updated power :
            self.fcn_userMsg('Basic setup for power object')

            # -------------------------------------------
            # BASIC PLOT
            # -------------------------------------------
            self._pow2DPlt['vmin'] = None
            self._pow2DPlt['vmax'] = None
            self._pow2DPlt['cmap'] = 'inferno'
        except:
            self.fcn_userMsg('Oups :d (Power basic setup)')


    def _AdvSetup(self):
        """Advanced setup
        """
        try:
            # -------------------------------------------
            # BASIC FILTERING
            # -------------------------------------------
            # Build frequency vector :
            self._PowArgs['f'] = self._f2extract
            # Get norm :
            self._PowArgs['norm'] = self.powNorm.value()
            # Get filter settings :
            self._PowArgs['filtname'] = self.powBut.currentText()
            self._PowArgs['order'] = self.powOrder.value()
            self._PowArgs['cycle'] = self.powCycle.value()
            # Transformation :
            self._PowArgs['method'] = self.powTrans.currentText()
            self._PowArgs['wltWidth'] = self.powWidthW.value()
            # Detrend:
            self._PowArgs['dtrd'] = self.powDetrend.isChecked()
            # Update power object :
            self._power = power(self._sf, self._npts, baseline=self._baseline,
                                **self._PowArgs)
            self.advResume.setText(str(self._power))
            # Print updated power :
            self.fcn_userMsg('Advanced setup for power object')

            # -------------------------------------------
            # BASIC PLOT
            # -------------------------------------------

            # Get values :
            self._pow2DPlt['vmin'] = self.powVmin.value()
            self._pow2DPlt['vmax'] = self.powVmax.value()
            self._pow2DPlt['cmap'] = self.powCmap.currentText()
        except:
            self.fcn_userMsg('Oups :d (Power advanced setup)')