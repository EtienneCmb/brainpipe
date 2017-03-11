from .utils._physio import loadatlas, pos2label
from brainpipe.sys.dataframe import pdTools

__all__ = ['xyz2phy']


class xyz2phy(pdTools):

    """Transform coordinates to physiological informations.

    Args:
        nearest: bool, optional, [def: True]
            If no physiological is found, use this parameter to force
            to search in sphere of interest.

        r: integer, optional, [def: 5]
            Find physiological informations inside a sphere of
            interest with a radius of r.

        rm_unfound: bool, optional, [def: False]
            Remove un-found structures

    .. automethod:: search
    .. automethod:: keep
    .. automethod:: remove
    """

    def __init__(self, nearest=True, r=5, rm_unfound=False):
        self._r = r
        self._nearest = nearest
        self._rm_unfound = rm_unfound
        (self._hdr, self._mask, self._gray, self._label) = loadatlas(r=self._r)
        pdTools.__init__(self)

    def get(self, xyz, channel=[]):
        """Get physiological informations from (x,y,z) coordinates.

        Args:
            xyz: array
                Array of coordinates. The shape of xyz must be
                (n_electrodes x 3).

            channel: list, optional, [def: []]
                List of channels name.

        Return:
            A pandas DataFrame with physiological informations;
        """
        pdPhy = pos2label(xyz, self._mask, self._hdr, self._gray,
                          self._label, r=self._r, nearest=self._nearest)
        if not channel:
            channel = ['channel'+str(k) for k in range(len(pdPhy))]
        pdPhy['channel'] = channel
        return pdPhy
