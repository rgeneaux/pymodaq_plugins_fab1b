"""Shutter TA: dR/R and ΔOD from an on/off pair, and the params wiring."""
import unittest

import numpy as np

from pymodaq_plugins_fab1b.daq_viewer_plugins.plugins_2D.daq_2Dviewer_MaranaX import (
    DAQ_2DViewer_MaranaX,
)
from pymodaq_plugins_fab1b.daq_viewer_plugins.plugins_2D.daq_2Dviewer_MaranaXShutter import (
    SHUTTER_TA_MODE,
    DAQ_2DViewer_MaranaXShutter,
    shutter_observable,
)


def _param(cls, name):
    return next(p for p in cls.params if p.get("name") == name)


class ObservableTest(unittest.TestCase):
    def test_dr_over_r(self):
        name, data = shutter_observable([110.0, 90.0, 100.0], [100.0, 100.0, 100.0], "dR/R")
        self.assertEqual(name, "Differential Reflectivity")
        np.testing.assert_allclose(data, [0.1, -0.1, 0.0])

    def test_delta_od_is_minus_log10_of_ratio(self):
        name, data = shutter_observable([50.0, 100.0, 200.0], [100.0, 100.0, 100.0], "ΔOD")
        self.assertEqual(name, "Differential Optical Density")
        np.testing.assert_allclose(data, [np.log10(2), 0.0, -np.log10(2)])

    def test_zero_in_pump_off_stays_finite(self):
        _, data = shutter_observable([10.0, 10.0], [0.0, 10.0], "dR/R")
        self.assertTrue(np.all(np.isfinite(data)))
        self.assertEqual(data[1], 0.0)

    def test_negative_ratio_in_delta_od_becomes_zero(self):
        _, data = shutter_observable([-5.0, 10.0], [10.0, 10.0], "ΔOD")
        self.assertEqual(list(data), [0.0, 0.0])

    def test_inputs_are_not_modified(self):
        pon, poff = np.array([1.0, 2.0]), np.array([0.0, 2.0])
        shutter_observable(pon, poff, "dR/R")
        np.testing.assert_array_equal(poff, [0.0, 2.0])


class ParamsWiringTest(unittest.TestCase):
    def test_shutter_ta_mode_does_not_leak_into_the_base_plugin(self):
        self.assertIn(SHUTTER_TA_MODE, _param(DAQ_2DViewer_MaranaXShutter, "acquisition_mode")["limits"])
        self.assertNotIn(SHUTTER_TA_MODE, _param(DAQ_2DViewer_MaranaX, "acquisition_mode")["limits"])
        self.assertEqual(
            _param(DAQ_2DViewer_MaranaX, "acquisition_mode")["limits"], ["Preview", "Chunked"]
        )

    def test_base_modes_are_kept(self):
        limits = _param(DAQ_2DViewer_MaranaXShutter, "acquisition_mode")["limits"]
        self.assertEqual(limits, ["Preview", "Chunked", SHUTTER_TA_MODE])

    def test_shutter_and_ta_groups_exist_only_in_the_shutter_plugin(self):
        for group in ("shutter", "shutter_ta"):
            self.assertTrue(any(p.get("name") == group for p in DAQ_2DViewer_MaranaXShutter.params))
            self.assertFalse(any(p.get("name") == group for p in DAQ_2DViewer_MaranaX.params))

    def test_is_a_2d_plugin(self):
        self.assertIn("2D", DAQ_2DViewer_MaranaXShutter.__name__)
        self.assertNotIn("1D", str(DAQ_2DViewer_MaranaXShutter))


if __name__ == "__main__":
    unittest.main()
