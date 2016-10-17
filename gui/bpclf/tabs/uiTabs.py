from .loadTab import loadTab
from .sfTab import sfTab
from .mfTab import mfTab


class uiTabs(loadTab, sfTab, mfTab):

	"""docstring for uiTab
	"""

	def __init__(self):
		loadTab.__init__(self)
		sfTab.__init__(self)
		mfTab.__init__(self)
