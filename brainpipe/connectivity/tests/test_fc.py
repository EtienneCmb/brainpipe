"""Test fonctional connectivity related functions."""
import numpy as np

from brainpipe.connectivity import (sfc, directional_sfc, dfc, directional_dfc,
                                    fc_summarize)



class TestFc(object):  # noqa

    sfcm = ['corr', 'mtd', 'cmi']

    def _generate_arrays(self, n, phi=0):
        ts_1 = np.random.rand(10, n, 20)
        ts_1[0, :, 4] = self._generate_sine(n, 0)
        ts_2 = np.random.rand(10, n, 20)
        ts_2[0, :, 4] = self._generate_sine(n, phi)
        return ts_1, ts_2

    def _generate_sine(self, n, phi, noise=0.):
        sf = 256
        f = 2.
        t = np.arange(n) / sf
        return np.sin(f * np.pi * 2 * t + phi) + noise * np.random.rand(n)

    def test_sfc(self):  # noqa
        ts_1, ts_2 = self._generate_arrays(100)
        _sfc = [sfc(ts_1, ts_2, axis=1, measure=k)[0] for k in self.sfcm]
        for k in _sfc:
            max_ = np.r_[np.where(k == k.max())]
            assert np.all(max_ == np.array([0, 4]))

    def test_directional_sfc(self):  # noqa
        ts_1, ts_2 = self._generate_arrays(200, -np.pi / 2)
        lags = np.arange(0, 2 * np.pi, np.pi / 2)
        lags = (256. * lags / (2 * np.pi * 2.)).astype(int)
        for m in self.sfcm:
            _sfc = np.zeros((len(lags), 10, 20))
            for k, l in enumerate(lags):
                _sfc[k, ...] = directional_sfc(ts_1, ts_2, l, axis=1,
                                               measure=m)[0]
            # Find the maximum :
            max_ = np.where(_sfc[:, 0, 4] == _sfc[:, 0, 4].max())[0]
            assert float(max_) == 1.

    def test_dfc(self):  # noqa
        ts_1, ts_2 = self._generate_arrays(100)
        _sfc = [dfc(ts_1, ts_2, 10, axis=1, measure=k)[0] for k in self.sfcm]
        for k in _sfc:
            max_ = np.r_[np.where(k.mean(1) == k.mean(1).max())]
            assert np.all(max_ == np.array([0, 4]))

    def test_directional_dfc(self):  # noqa
        ts_1, ts_2 = self._generate_arrays(100, -np.pi / 2)
        for m in self.sfcm:
            directional_dfc(ts_1, ts_2, 50, 10, axis=1, measure=m)

    def test_fc_summarize(self):  # noqa
        ts_1, ts_2 = self._generate_arrays(1000)
        _dfc, _pvals, _ = dfc(ts_1, ts_2, 100, axis=1, measure='corr')
        idx = []
        # STD :
        std = fc_summarize(_dfc, 1, 'std')
        idx += [np.r_[np.where(std == std.min())]]
        # MEAN :
        mean = fc_summarize(_dfc, 1, 'mean')
        idx += [np.r_[np.where(mean == mean.max())]]
        # COEFVAR :
        coefvar = fc_summarize(_pvals, 1, 'coefvar')
        idx += [np.r_[np.where(coefvar == coefvar.min())]]
        for k in idx:
            assert np.all(k == np.array([0, 4]))
