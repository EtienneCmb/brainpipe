from brainpipe.classification import *
import matplotlib.pyplot as plt
import numpy as np

class sfTab(object):

    """docstring for sfTab
    """

    def __init__(self):
        self.clfGrp.setVisible(False)

        # Classification settings :
        self._clfVisible()
        self.SFfce.currentIndexChanged.connect(self.fcn_clfSettings)
        self.SFclf.currentIndexChanged.connect(self.fcn_clfSettings)
        self.svmRBF.clicked.connect(self.fcn_clfSettings)
        self.svmLin.clicked.connect(self.fcn_clfSettings)
        self.knnN.valueChanged.connect(self.fcn_clfSettings)
        self.rfNtree.valueChanged.connect(self.fcn_clfSettings)
        self.cvType.currentIndexChanged.connect(self.fcn_clfSettings)
        self.cvRep.valueChanged.connect(self.fcn_clfSettings)
        self.cvNfold.valueChanged.connect(self.fcn_clfSettings)

        # Run classification :
        self.runSF.clicked.connect(self.fcn_runClf)


    ################################################################
    # CLF OBJECTS MANAGEMENT
    ################################################################
    def _clfInit(self):
        self._da = None
        self._clfArg = {'clf':'lda', 'kern':'rbf', 'n_knn':10, 'n_tree':100}
        self._cvArg = {'cvtype':'kfold', 'rep':1}
        self._chanceLevel = 0.05/self._nelec
        self._updateAll()


    def _updateAll(self):
        """Update all classification elements
        """
        self._updateClf()
        self._updateCv()
        self._updateCla()


    def _updateClf(self):
        """Update classifier object
        """
        self._clf = defClf(self._y, **self._clfArg)
        self.clfConf.setText(str(self._clf))


    def _updateCv(self):
        """Update cross-validation object
        """
        self._cv = defCv(self._y, **self._cvArg)
        self.cvConf.setText(str(self._cv))


    def _updateCla(self):
        """Update classify
        """
        self._cla = classify(self._y, clf=self._clf, cvtype=self._cv)


    ################################################################
    # CLF SETTINGS
    ################################################################
    def fcn_clfSettings(self):
        """Get clf settings
        """
        # Get selected frequency :
        self._selectedFce = self.SFfce.currentIndex()
        # Manage displayed elements :
        currentClf = self.SFclf.currentText()
        if currentClf == 'lda':
            self._clfVisible()
            self._center = False
        elif currentClf == 'svm':
            self._clfVisible(svm=True)
            self._center = True
        elif currentClf == 'knn':
            self._clfVisible(knn=True)
            self._center = False
        elif currentClf == 'nb':
            self._clfVisible()
            self._center = False
        elif currentClf == 'rf':
            self._clfVisible(rf=True)
            self._center = False
        self._clfArg['clf'] = currentClf
        # Get settings :
        if self.svmRBF.isChecked():
            self._clfArg['kern'] = 'rbf'
        else:
            self._clfArg['kern'] = 'linear'
        self._clfArg['n_knn'] = self.knnN.value()
        self._clfArg['n_tree'] = self.rfNtree.value()
        # Cross-validation :
        self._cvArg['cvtype'] = self.cvType.currentText()
        self._cvArg['rep'] = self.cvRep.value()
        self._cvArg['n_folds'] = self.cvNfold.value()
        # Finally update cla :
        self._updateAll()
        self._da = None


    def _clfVisible(self, svm=False, knn=False, rf=False):
        """Manage displayed sub-settings
        """
        self.svmW.setVisible(svm)
        self.knnW.setVisible(knn)
        self.rfW.setVisible(rf)


    def fcn_runClf(self):
        """Run classification
        """
        self._da = self._cla.fit(self._x[self._selectedFce, ...].T, center=self._center)[0]
        self.tabWidget.setTabEnabled(2, True)
        self.vminSpin.setValue(self._da.mean(0).min())
        self.vmaxSpin.setValue(self._da.mean(0).max())
        self.fcn_barPlot()

