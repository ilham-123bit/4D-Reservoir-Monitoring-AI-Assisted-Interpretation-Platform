from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from reservoir_time_machine.attributes.difference import calculate_attributes
from reservoir_time_machine.processing.full_field import FIELDS, run_full_field_attributes


class FullFieldAttributeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "analysis"
        rng = np.random.default_rng(37)
        self.baseline = rng.normal(size=(5, 4, 13)).astype(np.float32)
        self.monitor = self.baseline + rng.normal(scale=0.1, size=self.baseline.shape).astype(np.float32)
        self.missing = np.zeros(self.baseline.shape[:2], dtype=bool)
        self.missing[1, 2] = True
        self.missing[4, 0] = True
        self.baseline[0, 0, 4] = np.nan
        self.monitor[0, 0, 4] = np.nan
        self.identity = hashlib.sha256(b"synthetic-paired-input-v1").hexdigest()

    def tearDown(self):
        self.temp.cleanup()

    def run_engine(self, *, max_chunks=None):
        return run_full_field_attributes(
            self.baseline, self.monitor, self.missing, self.root,
            source_identity=self.identity, chunk_shape=(2, 3, 4),
            window_samples=5, min_support_fraction=0.6,
            protected_cache=Path(self.temp.name) / "protected-cache",
            max_chunks=max_chunks,
        )

    def test_chunked_outputs_match_full_array_across_all_boundaries(self):
        result = self.run_engine()
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["completed_chunks"], result["total_chunks"])
        masked_b = np.where(self.missing[..., None], np.nan, self.baseline)
        masked_m = np.where(self.missing[..., None], np.nan, self.monitor)
        with np.errstate(invalid="ignore"):
            expected = calculate_attributes(masked_b, masked_m, window_samples=5)
        centers = np.arange(self.baseline.shape[2])
        available = np.minimum(self.baseline.shape[2], centers + 3) - np.maximum(0, centers - 2)
        unsupported = expected.paired_sample_count < np.ceil(0.6 * available)[None, None, :]
        for name in ("rms_baseline", "rms_monitor", "nrms"):
            getattr(expected, name)[unsupported] = np.nan
        run_dir = Path(result["run_directory"])
        for field in FIELDS:
            actual = np.load(run_dir / f"{field}.npy", mmap_mode="r")
            truth = getattr(expected, field)
            np.testing.assert_allclose(actual, truth, rtol=2e-6, atol=2e-6, equal_nan=True)
        support = np.load(run_dir / "paired_sample_count.npy", mmap_mode="r")
        self.assertTrue(np.all(support[self.missing] == 0))
        self.assertTrue(np.isnan(np.load(run_dir / "rms_baseline.npy", mmap_mode="r")[self.missing]).all())
        self.assertEqual(result["source_voxels_scanned"], self.baseline.size)
        self.assertEqual(result["finite_paired_sample_positions"], int((~self.missing[..., None] * np.isfinite(self.baseline) * np.isfinite(self.monitor)).sum()))

    def test_interruption_resumes_and_verified_artifact_is_cached(self):
        partial = self.run_engine(max_chunks=1)
        self.assertEqual(partial["status"], "incomplete")
        self.assertEqual(partial["completed_chunks"], 1)
        resumed = self.run_engine()
        self.assertEqual(resumed["status"], "complete")
        self.assertFalse(resumed["cache_hit"])
        cached = self.run_engine()
        self.assertTrue(cached["cache_hit"])

    def test_changed_source_is_rejected_on_resume(self):
        self.run_engine(max_chunks=1)
        self.monitor[0, 0, 0] += 1
        with self.assertRaisesRegex(ValueError, "source data changed"):
            self.run_engine()

    def test_completed_cache_hit_revalidates_source_chunks(self):
        self.run_engine()
        self.monitor[0, 0, 0] += 0.25
        with self.assertRaisesRegex(ValueError, "source data changed"):
            self.run_engine()

    def test_invalid_mask_and_unsupported_count_width_are_rejected(self):
        with self.assertRaises(ValueError):
            run_full_field_attributes(self.baseline, self.monitor, self.missing.astype(np.uint8), self.root,
                                      source_identity=self.identity)
        with self.assertRaises(ValueError):
            run_full_field_attributes(self.baseline, self.monitor, self.missing, self.root,
                                      source_identity=self.identity, window_samples=256)


if __name__ == "__main__":
    unittest.main()
