"""Self-contained numerical/negative-control tests for conditional I-100 screening."""
import unittest

import numpy as np

from reservoir_time_machine.analysis.conditional_reservoir import (
    conditional_i100_locations,
    evaluate_conditional_window,
)


class ConditionalReservoirTests(unittest.TestCase):
    def _wells(self):
        return [{
            "well_id": "well-5",
            "marker_checkshot_relationship": {
                "status": "source_marker_to_checkshot_interpolation_only",
                "markers": [{"marker": "I-100",
                             "interpolated_source_checkshot_twt_ms": 1702.0}],
            },
        }]

    def _projection(self):
        return {"wells": [{
            "well_id": "well-5",
            "status": "conditional_inside_header_axis_extent",
            "conditional_inline_id": 102.2,
            "conditional_crossline_id": 203.1,
        }]}

    def test_conditional_sample_mapping_and_source_ids(self):
        rows = conditional_i100_locations(
            self._wells(), self._projection(),
            np.arange(100, 110), np.arange(200, 210),
            1301)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["array_location"], [2, 3, 851])
        self.assertEqual((row["segy_inline_id"], row["segy_crossline_id"]), (102, 203))
        self.assertIn("NOT_A_VERIFIED", row["status"])

    def test_origin_sensitivity_and_invalid_grid_exclusion(self):
        moved = conditional_i100_locations(
            self._wells(), self._projection(),
            np.arange(100, 110), np.arange(200, 210),
            1301, origin_ms=20)
        self.assertEqual(moved[0]["array_location"][2], 841)
        out = self._projection()
        out["wells"][0]["conditional_inline_id"] = 150.0
        self.assertEqual(conditional_i100_locations(
            self._wells(), out, np.arange(100, 110), np.arange(200, 210),
            1301), [])
        with self.assertRaises(ValueError):
            conditional_i100_locations(self._wells(), self._projection(),
                                       [100, 100], [200, 201], 1301)

    def test_signal_change_is_measured_against_off_window_controls(self):
        rng = np.random.default_rng(7)
        b = rng.normal(0, 1, (5, 5, 256))
        m = b.copy()
        m[:, :, 115:146] += rng.normal(0, 4, (5, 5, 31))
        item = {
            "id": "conditional-i100-well-5", "well_id": "well-5",
            "array_location": [2, 2, 130],
            "segy_inline_id": 102, "segy_crossline_id": 203,
            "source_twt_ms": 260, "assumed_time_origin_ms": 0,
        }
        result = evaluate_conditional_window(b, m, np.zeros((5, 5), bool), item,
                                             half_window=5, control_distance_samples=60)
        self.assertEqual(result["response_vs_off_window_controls"],
                         "larger_change_than_local_off_window_comparisons")
        self.assertAlmostEqual(result["observed"]["support_fraction"], 1.0)
        self.assertGreater(result["observed"]["nrms_percent"], 0)
        self.assertEqual(result["eligibility"],
                         "conditional_seismic_comparison_not_recovery_target")

    def test_lag_diagnostic_is_not_registration(self):
        tt = np.arange(256, dtype=float)
        trace = np.sin(tt / 3) + np.cos(tt / 7)
        b = np.broadcast_to(trace, (5, 5, 256)).copy()
        m = np.roll(b, shift=2, axis=2)
        item = {
            "id": "conditional-i100-well-5", "well_id": "well-5",
            "array_location": [2, 2, 120],
            "segy_inline_id": 102, "segy_crossline_id": 203,
            "source_twt_ms": 240, "assumed_time_origin_ms": 0,
        }
        result = evaluate_conditional_window(b, m, np.zeros((5, 5), bool), item)
        self.assertEqual(result["best_local_lag_samples"], 2)
        self.assertLess(result["best_lag_nrms_percent"],
                        result["observed"]["nrms_percent"])
        self.assertIn("assumptions", result["interpretation"])

    def test_missing_traces_are_excluded(self):
        b = np.ones((5, 5, 256))
        m = b + 1
        missing = np.zeros((5, 5), bool)
        missing[0:2, :] = True
        item = {
            "id": "conditional-i100-well-5", "well_id": "well-5",
            "array_location": [2, 2, 120],
            "segy_inline_id": 102, "segy_crossline_id": 203,
            "source_twt_ms": 240, "assumed_time_origin_ms": 0,
        }
        result = evaluate_conditional_window(b, m, missing, item)
        self.assertAlmostEqual(result["observed"]["support_fraction"], 0.6)
        self.assertEqual(result["observed"]["valid_pairs"], 15 * 21)
        with self.assertRaises(ValueError):
            evaluate_conditional_window(b, m, np.zeros((4, 5), bool), item)


if __name__ == "__main__":
    unittest.main()
