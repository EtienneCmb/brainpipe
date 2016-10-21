from .loadTab import loadTab
from .sfTab import sfTab
from .mfTab import mfTab
from .pltTab import pltTab

class uiTabs(loadTab, sfTab, mfTab, pltTab):

	"""docstring for uiTab
	"""

	def __init__(self):
            loadTab.__init__(self)
            sfTab.__init__(self)
            mfTab.__init__(self)
            pltTab.__init__(self)
