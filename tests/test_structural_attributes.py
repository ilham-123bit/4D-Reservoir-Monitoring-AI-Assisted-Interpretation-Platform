import unittest

import numpy as np

from reservoir_time_machine.attributes.structural import local_waveform_similarity


class LocalWaveformSimilarityTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(412)
        base = rng.normal(size=41)
        left = base + rng.normal(scale=0.015, size=(12, 8, 41))
        right = rng.normal(size=(12, 8, 41))
        self.volume = np.concatenate((left, right), axis=1).astype(np.float32)

    def test_detects_local_continuity_change_and_keeps_source_axes(self):
        result = local_waveform_similarity(self.volume, 5, 4, 20, radius=7)
        similarity = result["similarity"]
        self.assertEqual(result["array_bounds"], ((0, 12), (0, 12)))
        self.assertEqual(result["sample_bounds"], (16, 25))
        self.assertGreater(float(np.nanmean(similarity[:, :3])), 0.9)
        self.assertLess(float(np.nanmean(similarity[:, 9:])), 0.6)
        self.assertEqual(result["neighbor_pair_count"].shape, similarity.shape)
        self.assertIn("not calibrated", result["interpretation_limit"])

    def test_missing_trace_is_excluded_from_every_neighbor_pair(self):
        missing = np.zeros(self.volume.shape[:2], dtype=bool)
        missing[5, 4] = True
        result = local_waveform_similarity(
            self.volume, 5, 4, 0, radius=2, excluded_trace_mask=missing,
        )
        ci, cx = result["center_local_index"]
        self.assertEqual(result["neighbor_pair_count"][ci, cx], 0)
        self.assertTrue(np.isnan(result["similarity"][ci, cx]))
        self.assertEqual(result["sample_bounds"], (0, 5))

    def test_invalid_geometry_or_support_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cube"):
            local_waveform_similarity(np.ones((4, 4)), 1, 1, 1)
        with self.assertRaisesRegex(ValueError, "integer"):
            local_waveform_similarity(self.volume, 1.5, 1, 1)
        with self.assertRaisesRegex(ValueError, "match"):
            local_waveform_similarity(self.volume, 1, 1, 1,
                                      excluded_trace_mask=np.zeros((2, 2), bool))


if __name__ == "__main__":
    unittest.main()
