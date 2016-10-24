from .dataTab import dataTab
from .tdaTab import tdaTab
from .powTab import powTab
from .tfTab import tfTab

class uiTabs(dataTab, powTab, tdaTab, tfTab):

    """
    Initialize tabs
    """
    def __init__(self):
        dataTab.__init__(self)
        tdaTab.__init__(self)
        powTab.__init__(self)
        tfTab.__init__(self)