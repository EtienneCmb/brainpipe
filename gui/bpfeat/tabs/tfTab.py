import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

from brainpipe.feature import TF
from brainpipe.tools import binarize



class tfTab(object):

    """docstring for tfTab
    """

    def __init__(self):
        # Main settings
        self.tfFrom.valueChanged.connect(self.fcn_mainSettings)
        self.tfTo.valueChanged.connect(self.fcn_mainSettings)
        self.tfWidth.valueChanged.connect(self.fcn_mainSettings)
        self.tfStep.valueChanged.connect(self.fcn_mainSettings)
        self.tfChanNb.valueChanged.connect(self.fcn_mainSettings)

        # Run :
        self.tfRun.clicked.connect(self.fcn_tfRun)
        self.tfRunCustom.clicked.connect(self.fcn_tfRun)

        # Baseline :
        self.tfBslW.setVisible(False)
        self.tfBsl.clicked.connect(self._tfAdvSetup)

        # -----------------------------------------------------------------
        # Advanced tab :
        # -----------------------------------------------------------------
        # Advanced filtering :
        self._tfArgs = {}
        self.tfNorm.valueChanged.connect(self._tfAdvSetup)
        self.tfBut.currentIndexChanged.connect(self._tfAdvSetup)
        self.tfOrder.valueChanged.connect(self._tfAdvSetup)
        self.tfCycle.valueChanged.connect(self._tfAdvSetup)
        self.tfTrans.currentIndexChanged.connect(self._tfAdvSetup)
        self.tfWidthW.valueChanged.connect(self._tfAdvSetup)
        self.tfDetrend.clicked.connect(self._tfAdvSetup)
        # Advanced vizualization :
        self._tf2DPlt = {}
        self.tfVmin.valueChanged.connect(self._tfAdvSetup)
        self.tfVmax.valueChanged.connect(self._tfAdvSetup)
        self.tfCmap.currentIndexChanged.connect(self._tfAdvSetup)
        self.tfPltUpd.clicked.connect(self.fcn_tfPlt)
        self.cmap_lst = [k for k in list(cm.datad.keys()) + list(cm.cmaps_listed.keys()) if not k.find('_r') + 1]
        self.cmap_lst.sort()
        self.tfCmap.addItems(self.cmap_lst)


    def fcn_tfInit(self):
        """Initialize the most basic power object
        """
        self._tf = TF(self._sf, self._npts)
        # Get and build frequency vector :
        self._tfvec = binarize(self.tfFrom.value(), self.tfTo.value(),
                               self.tfWidth.value(), self.tfStep.value(), kind='list')
        self._tfvecM = [np.array(k).mean() for k in self._tfvec]
        self._tfBasicSetup()
        print('-> TF object created')


    ##############################################################
    # GET MAIN SETTINGS
    ##############################################################
    def fcn_mainSettings(self):
        """
        """
        # Get and build frequency vector :
        self._tfvec = binarize(self.tfFrom.value(), self.tfTo.value(),
                               self.tfWidth.value(), self.tfStep.value(), kind='list')
        self._tfvecM = [np.array(k).mean() for k in self._tfvec]
        # Get the channel :
        self._tfChan = self.tfChanNb.value()
        # Get TF settings :
        if self.actionAdvanced.isChecked():
            self._tfAdvSetup()
        else:
            self._tfBasicSetup()
        # Create tf object with current setup :
        try:
            self._tf = TF(self._sf, self._npts, **self._tfArgs)
        except:
            self.fcn_userMsg('Oups :d (building tf object)')


    ##############################################################
    # BASIC / ADVANCED SETUP
    ##############################################################
    def fcn_tfRun(self):
        """Run tf computation
        """
        try:
            # Be sure to get last updated variables :
            self.fcn_mainSettings()
            # Run tf :
            self._tf2plot = np.squeeze(self._tf.get(self._data[self._tfChan, ...])[0])
            self._tf2plot /= np.abs(self._tf2plot).max()
            self.fcn_userMsg('Time-Fequency map computed !!')
        except:
            self.fcn_userMsg('Oups :d (tf computing)')
        # Go to plotting function :
        self.fcn_tfPlt()



    ##############################################################
    # PLOT
    ##############################################################
    def fcn_tfPlt(self):
        """Plot tf
        """
        try:
            title = 'Time-frequency map of channel '+str(self._tfChan)
            self.fcn_userMsg('Plotting: '+title)
            # Send the figure :
            self.cleanfig()
            self._fig = plt.figure()
            self._tf.plot2D(self._fig, self._tf2plot, cblabel='Power modulations', title=title,
                               xvec=self._timevec, yvec=self._tfvecM, xlabel='Time (ms)', ylabel='Frequency (Hz)',
                               **self._tf2DPlt)
            self.addmpl(self._fig)
        except:
            self.fcn_userMsg('Oups :d (plot tf)')


    ##############################################################
    # BASIC / ADVANCED SETUP
    ##############################################################
    def _tfBasicSetup(self):
        """Basic setup
        """
        try:
            # -------------------------------------------
            # BASIC FILTERING
            # -------------------------------------------
            self._tfArgs['f'] = self._tfvec
            self._tfArgs['baseline'] = None
            self._tfArgs['norm'] = 0
            self._tfArgs['filtname'] = 'butter'
            self._tfArgs['order'] = 3
            self._tfArgs['cycle'] = 3
            self._tfArgs['method'] = 'wavelet'
            self._tfArgs['wltWidth'] = 7
            self._tfArgs['dtrd'] = False
            self.tfResume.setText(str(self._tf))

            # -------------------------------------------
            # BASIC PLOT
            # -------------------------------------------
            self._tf2DPlt['vmin'] = None
            self._tf2DPlt['vmax'] = None
            self._tf2DPlt['cmap'] = 'viridis'
            self.fcn_userMsg('Basic setup for tf object')
        except:
            self.fcn_userMsg('Oups :d (tf basic setup)')


    def _tfAdvSetup(self):
        """Advanced setup
        """
        try:
            # -------------------------------------------
            # BASIC FILTERING
            # -------------------------------------------
            self._tfArgs['f'] = self._tfvec
            self._tfArgs['norm'] = self.tfNorm.value()
            self._tfArgs['filtname'] = self.tfBut.currentText()
            self._tfArgs['order'] = self.tfOrder.value()
            self._tfArgs['cycle'] = self.tfCycle.value()
            self._tfArgs['method'] = self.tfTrans.currentText()
            self._tfArgs['wltWidth'] = self.tfWidthW.value()
            self._tfArgs['dtrd'] = self.tfDetrend.isChecked()
            self.tfBslW.setVisible(self.tfBsl.isChecked())
            if self.tfBsl.isChecked():
                self._tfArgs['baseline'] = [self.tfBslFrom.value(), self.tfBslTo.value()]
            else:
                self._tfArgs['baseline'] = None

            self._tf = TF(self._sf, self._npts, **self._tfArgs)
            self.tfResume.setText(str(self._tf))

            # -------------------------------------------
            # BASIC PLOT
            # -------------------------------------------
            # Get values :
            self._tf2DPlt['vmin'] = self.tfVmin.value()
            self._tf2DPlt['vmax'] = self.tfVmax.value()
            self._tf2DPlt['cmap'] = self.tfCmap.currentText()
            self.fcn_userMsg('Advanced setup for tf object')
        except:
            self.fcn_userMsg('Oups :d (tf advanced setup)')
