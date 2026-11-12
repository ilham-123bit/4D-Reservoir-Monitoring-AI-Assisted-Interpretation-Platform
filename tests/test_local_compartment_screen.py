"""Numerical, missing-data and interpretation tests for local reservoir screens."""
import unittest
import numpy as np

from reservoir_time_machine.analysis.local_compartment_screen import (
    screen_local_compartments, development_hypothesis,
)


class LocalCompartmentScreenTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(96)
        wave = np.sin(np.linspace(0, 8*np.pi, 45))
        base = wave[None, None, :] + rng.normal(0, 0.02, (31, 31, 45))
        self.base = np.asarray(base)
        self.monitor = self.base.copy()
        self.missing = np.zeros((31, 31), bool)

    def test_center_responds_more_than_nearby_traces(self):
        self.monitor[13:18, 13:18, 10:35] *= 0.5
        result = screen_local_compartments(self.base, self.monitor,
                                           self.missing, [15, 15, 22])
        self.assertEqual(result["status"], "measured_local_spatial_contrast")
        self.assertEqual(result["pattern"], "selected_region_more_responsive_than_neighborhood")
        self.assertGreater(result["core_median_nrms_percent"], 50)
        self.assertLess(result["surrounding_median_nrms_percent"], 1)
        self.assertEqual(len(result["nrms_percent_map"].shape), 2)
        self.assertGreater(result["evaluated_trace_count"], 400)
        decision = development_hypothesis(result)
        self.assertIn("production-affected", decision["headline"])
        self.assertEqual(len(decision["discipline_actions"]), 6)
        self.assertEqual(decision["mechanisms"][2]["status"], "not_established")

    def test_quiet_center_with_responsive_surroundings(self):
        # Change neighboring traces while preserving a quiet inner core.
        self.monitor[:, :, 10:35] *= 0.60
        self.monitor[13:18, 13:18, :] = self.base[13:18, 13:18, :]
        result = screen_local_compartments(self.base, self.monitor,
                                           self.missing, [15, 15, 22])
        self.assertEqual(result["pattern"], "selected_region_less_responsive_than_neighborhood")
        decision = development_hypothesis(result)
        self.assertIn("Relatively quiet", decision["headline"])
        self.assertIn("not a proven unswept", decision["poor_sweep"])
        self.assertTrue(decision["development_options"])

    def test_missing_trace_not_reclassified_as_quiet(self):
        self.missing[:,:] = True
        self.missing[15, 15] = False
        result = screen_local_compartments(self.base, self.monitor,
                                           self.missing, [15, 15, 22])
        self.assertEqual(result["status"], "insufficient_spatial_comparisons")
        self.assertIsNone(result["quiet_candidate"])
        self.assertIsNone(result["change_candidate"])
        self.assertEqual(development_hypothesis(result)["evidence_level"], "unsupported")

    def test_zero_amplitude_traces_are_not_unswept_hypotheses(self):
        zero = np.zeros_like(self.base)
        result = screen_local_compartments(zero, zero, self.missing, [15, 15, 22])
        self.assertEqual(result["status"], "insufficient_spatial_comparisons")
        self.assertIsNone(result["quiet_candidate"])

    def test_unsupported_inputs_fail_closed(self):
        with self.assertRaises(ValueError):
            screen_local_compartments(self.base, self.monitor,
                                      self.missing, [31, 15, 22])
        with self.assertRaises(ValueError):
            screen_local_compartments(self.base, self.monitor,
                                      np.ones((12,12)), [15,15,22])
        with self.assertRaises(ValueError):
            screen_local_compartments(self.base, self.monitor,
                                      self.missing, [15,15,22], spatial_radius=500)

    def test_masked_nonfinite_values_do_not_crash(self):
        self.base[3, 3, :] = np.nan
        self.monitor[4, 4, :] = np.inf
        self.monitor[13:18, 13:18, 10:35] *= 0.5
        result = screen_local_compartments(self.base, self.monitor, self.missing, [15,15,22])
        self.assertEqual(result["status"], "measured_local_spatial_contrast")
        # Crop starts at IL/XL array index 3, so map indices start at zero.
        self.assertTrue(np.isnan(result["nrms_percent_map"][0,0]))
        self.assertTrue(np.isnan(result["nrms_percent_map"][1,1]))


if __name__ == "__main__":
    unittest.main()
