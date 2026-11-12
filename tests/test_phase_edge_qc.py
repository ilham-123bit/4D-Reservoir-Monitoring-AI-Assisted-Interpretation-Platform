import unittest

import numpy as np

from reservoir_time_machine.analysis.edge_qc import (
    gradient_magnitude_chunked,
    summarize_bounded_coverage_edges,
    summarize_candidate_edges,
    validity_aware_gradient,
)


class EdgeQualityTests(unittest.TestCase):
    def test_bounded_edge_report_separates_survey_crop_missing_monitor_and_chunk_edges(self):
        shape = (20, 20, 20)
        support = np.full(shape, 9, dtype=np.uint8)
        missing = np.zeros(shape[:2], dtype=bool)
        missing[9, 10] = True
        flags = np.zeros((10, 10, 10), dtype=np.uint8)
        flags[3, 5, 3] = 1  # global (8,10,8), adjacent to unoccupied/Monitor-only footprint
        flags[6, 6, 6] = 1
        result = summarize_bounded_coverage_edges(
            support=support, missing_trace_mask=missing,
            scoring_bounds=(5, 15, 5, 15, 5, 15),
            inline_ids=np.arange(100, 120), crossline_ids=np.arange(200, 220),
            monitor_only_keys=np.asarray([[109, 210], [999, 999]]),
            detector_flags_by_chunk={"4x4x4": {"mahalanobis": flags}},
            window_samples=9, min_support_fraction=0.8)
        self.assertFalse(result["scoring_crop_edges_are_survey_edges"])
        self.assertEqual(result["crop_to_survey_edge_clearance_cells"], [5, 5, 5, 5])
        self.assertEqual(result["monitor_only_keys_mapped_to_baseline_axes_for_coverage_only"], 1)
        self.assertEqual(result["monitor_only_keys_unmapped_to_baseline_axes"], 1)
        self.assertFalse(result["monitor_only_used_in_paired_calculation"])
        model = result["detectors"]["mahalanobis"]["4x4x4"]
        self.assertFalse(model["high_confidence_eligible"])
        self.assertTrue(model["flagged_adjacent_to_unoccupied_grid"]["fraction"] > 0)
        self.assertTrue(model["flagged_adjacent_to_monitor_only_footprint"]["fraction"] > 0)
        self.assertTrue(result["processing_chunk_seams"]["4x4x4"]["mahalanobis"][
            "seams_are_processing_boundaries_not_survey_edges"])

    def test_common_survey_step_is_not_a_4d_change_edge(self):
        baseline = np.zeros((9, 10, 11), dtype=np.float32)
        baseline[4:, :, :] = 10
        monitor = baseline.copy()
        valid = np.ones(baseline.shape, dtype=bool)
        base_grad = validity_aware_gradient(baseline, valid)
        diff_grad = validity_aware_gradient(monitor - baseline, valid)
        self.assertGreater(float(np.nanmax(np.abs(base_grad["inline_index"]))), 0)
        self.assertEqual(float(np.nanmax(np.abs(diff_grad["inline_index"]))), 0)

    def test_invalid_gap_does_not_create_nan_to_valid_bridge(self):
        values = np.zeros((9, 9, 9), dtype=np.float32)
        values[5:, :, :] = 1000
        valid = np.ones(values.shape, dtype=bool)
        valid[4, :, :] = False
        gradients = validity_aware_gradient(values, valid)
        self.assertTrue(np.isnan(gradients["inline_index"][3, 4, 4]))
        self.assertTrue(np.isnan(gradients["inline_index"][5, 4, 4]))

    def test_chunk_halos_match_whole_gradient_across_chunk_shapes(self):
        rng = np.random.default_rng(2046)
        data = rng.normal(size=(13, 11, 17)).astype(np.float32)
        valid = np.ones(data.shape, dtype=bool)
        valid[5, 3:8, 4:9] = False
        data[~valid] = np.nan
        expected = validity_aware_gradient(data, valid)
        expected_magnitude = np.sqrt(sum(expected[name].astype(np.float64) ** 2
                                         for name in expected))
        expected_valid = np.logical_and.reduce([np.isfinite(x) for x in expected.values()])
        expected_magnitude[~expected_valid] = np.nan
        for chunks in ((4, 4, 5), (7, 3, 8), (13, 11, 17)):
            actual = gradient_magnitude_chunked(data, valid, chunk_shape=chunks)
            np.testing.assert_allclose(actual, expected_magnitude, rtol=1e-6, atol=1e-6,
                                       equal_nan=True)

    def test_summary_reports_boundary_and_unoccupied_support(self):
        shape = (8, 8, 9)
        base = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        mon = base + 2
        diff = mon - base
        nrms = np.full(shape, 20, dtype=np.float32)
        support = np.full(shape, 9, dtype=np.uint8)
        unoccupied = np.zeros(shape[:2], dtype=bool)
        unoccupied[2, 2] = True
        support[2, 2] = 0
        diff[2, 2] = np.nan
        nrms[2, 2] = np.nan
        member = np.ones((3, 3, 3), dtype=bool)
        result = summarize_candidate_edges(
            baseline=base, monitor=mon, signed_difference=diff,
            absolute_difference=np.abs(diff), nrms=nrms,
            support=support, rms_baseline=np.ones(shape, dtype=np.float32),
            rms_monitor=np.ones(shape, dtype=np.float32),
            unoccupied_mask=unoccupied, member_mask=member,
            bounds=(slice(0, 3), slice(1, 4), slice(3, 6)),
            min_support_by_sample=np.full(shape[2], 8, dtype=np.uint8),
        )
        self.assertEqual(result["status"], "computed")
        self.assertGreater(result["survey_boundary_member_fraction"], 0)
        self.assertGreater(result["unoccupied_grid_member_fraction"], 0)
        self.assertGreaterEqual(result["gradient_coverage_fraction"], 0)
        self.assertEqual(result["nrms_normalization_denominator"]["median_rms_sum"], 2.0)
        self.assertIn("not a geological fault", result["interpretation"])


if __name__ == "__main__":
    unittest.main()
