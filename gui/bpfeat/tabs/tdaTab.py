import matplotlib.pyplot as plt
import numpy as np

from brainpipe.feature import sigfilt
from brainpipe.visual import BorderPlot

class tdaTab(object):

    """docstring for tdaTab
    """

    def __init__(self):
        # Panel management :
        self.rawPanelBox.clicked.connect(self.fcn_panelMngmt)
        self.erpPanelBox.clicked.connect(self.fcn_panelMngmt)
        self.rawPanel.setVisible(True)
        self.erpPanel.setVisible(False)

        # Raw data :
        self.rawAcross.clicked.connect(self.fcn_rawChan)
        self.rawChan.valueChanged.connect(self.fcn_rawChan)
        self.rawPer.clicked.connect(self.fcn_rawChan)
        self.fcn_rawChan()

        # ERP :
        self._data2plot = None
        self.erpAcross.clicked.connect(self.fcn_erpChan)
        self.erpChan.valueChanged.connect(self.fcn_erpChan)
        self.erpPer.clicked.connect(self.fcn_erpChan)
        self.erpSTD.clicked.connect(self.fcn_erpPlot)
        self.erpSEM.clicked.connect(self.fcn_erpPlot)
        self.fcn_erpChan()



    def fcn_panelMngmt(self):
        """Manage panel visibility
        """
        # Show raw data :
        if self.rawPanelBox.isChecked():
            self.rawPanel.setVisible(True)
            self.erpPanel.setVisible(False)
            self.fcn_rawChan()
        # Show ERP data :
        elif self.erpPanelBox.isChecked():
            self.erpPanel.setVisible(True)
            self.rawPanel.setVisible(False)
            # Filt the signal :
            if self._data2plot is None:
                print('-> Filt every trials under 10hz')
                sfobj = sigfilt(self._sf, self._npts, f=[0.1, 10], filtname='butter')
                self._data2plot = np.squeeze(sfobj.get(self._data)[0])
            self.fcn_erpChan()

    ################################################################
    # FCN RAW DATA 
    ################################################################
    def fcn_rawChan(self):
        """Display channel selection for raw data
        """
        self.rawChanW.setVisible(self.rawPer.isChecked())
        try:
            self.fcn_rawPlot()
        except:
            self.fcn_userMsg('Oups :d (power chanel selection)')


    def fcn_rawPlot(self):
        """Plot raw data
        """
        if self.rawAcross.isChecked():
            data2plot = self._data.mean(0)
            title = 'Raw data across channels'
            self.fcn_userMsg('Plotting raw data across channels')
        else:
            data2plot = self._data[self.rawChan.value(), :, :]
            title = 'Raw data for channel '+str(self.rawChan.value())
            self.fcn_userMsg('Plotting raw data per channel')
        # Send the figure :
        self.cleanfig()
        self._fig = plt.figure()
        plt.plot(self._timevec, data2plot)
        self._setplot(data2plot, title)


    ################################################################
    # FCN ERP
    ################################################################
    def fcn_erpChan(self):
        """Display channel selection for ERP
        """
        self.erpChanW.setVisible(self.erpPer.isChecked())
        try:
            self.fcn_erpPlot()
        except:
            self.fcn_userMsg('Oups :d')

    def fcn_erpPlot(self):
        """Plot erp
        """
        # Across or per channel :
        if self.erpAcross.isChecked():
            data2plot = self._data2plot.mean(0)
            title = 'ERP across channels'
            self.fcn_userMsg('Plotting ERP across channels')
        else:
            data2plot = self._data2plot[self.erpChan.value(), :, :]
            title = 'ERP for channel '+str(self.erpChan.value())
            self.fcn_userMsg('Plotting ERP per channel')
        # STD/SEM :
        if self.erpSTD.isChecked():
            kind = 'std'
        else:
            kind = 'sem'
        # Send the figure :
        self.cleanfig()
        self._fig = plt.figure()
        BorderPlot(self._timevec, data2plot, color='#ab4642', kind=kind)
        self._setplot(data2plot, title)

