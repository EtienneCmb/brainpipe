from .powTab import powTab
from .loadTab import loadTab
from .dataTab import dataTab

class uiTabs(dataTab, powTab, loadTab):

    """
    Initialize tabs
    """
    def __init__(self):
        dataTab.__init__(self)
        powTab.__init__(self)
        loadTab.__init__(self)