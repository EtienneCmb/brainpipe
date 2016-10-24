from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)
import matplotlib.pyplot as plt


class subpltClass(object):

    """Class for plotting management
    """

    def addmpl(self, fig):
        # Remove plot
        self.rmmpl()
        # Create canvas :
        self.canvas = FigureCanvas(fig)
        # Add toolbar :
        self.toolbar = NavigationToolbar(self.canvas, 
                self.mplwindow, coordinates=True)
        self.mplbox.addWidget(self.toolbar)
        # Add plot :
        self.mplbox.addWidget(self.canvas)
        self.canvas.draw()


    def rmmpl(self,):
        try:
            self.mplbox.removeWidget(self.canvas)
            self.canvas.close()
            self.mplbox.removeWidget(self.toolbar)
            self.toolbar.close()
        except:
            pass

    def cleanfig(self):
        plt.cla()
        plt.clf()
        plt.close()
        self.rmmpl()

    def _setplot(self, data2plot, title='', xlabel='Time (ms)', ylabel='uV'):
        """Plot data
        """
        plt.xlabel(xlabel), plt.ylabel(ylabel)
        plt.title(title), plt.axis('tight')
        self.addmpl(self._fig)


    def fcn_cleanAx(self):
        """
        """
        self.cleanfig()
        self.rmmpl()


    def fcn_userMsg(self, msg='   '):
        """Send a user message
        """
        self.userMsg.setText(msg)