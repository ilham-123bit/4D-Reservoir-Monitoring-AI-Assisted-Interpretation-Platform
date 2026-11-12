from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from reservoir_time_machine.analysis.full_field_models import (
    fit_classical_models, sample_spatial_population, score_full_field_models,
    validate_spatial_calibration_design, propose_spatial_regions_from_occupancy,
)
from reservoir_time_machine.attributes.difference import calculate_attributes


class FullFieldClassicalModelTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(66)
        self.baseline = rng.normal(size=(12, 12, 15)).astype(np.float32)
        self.monitor = self.baseline + rng.normal(scale=0.08, size=self.baseline.shape).astype(np.float32)
        self.missing = np.zeros(self.baseline.shape[:2], dtype=bool)
        self.missing[3, 3] = True
        self.identity = hashlib.sha256(b"synthetic-full-field-attributes").hexdigest()
        b = np.where(self.missing[..., None], np.nan, self.baseline)
        m = np.where(self.missing[..., None], np.nan, self.monitor)
        attrs = calculate_attributes(b, m, window_samples=3)
        self.attributes = {name: getattr(attrs, name).astype(np.float32 if name != "paired_sample_count" else np.uint8)
                           for name in ("signed_difference", "absolute_difference", "normalized_difference",
                                        "rms_baseline", "rms_monitor", "nrms", "paired_sample_count")}

    @staticmethod
    def _review(reference, calibration, review_id="fixture-review"):
        return {"review_id": review_id, "reviewed_by": "test",
                "selection_basis": "grid_partition_selected_without_inspecting_anomaly_scores",
                "ground_truth_status": "unlabeled_screening_only",
                "physical_registration_status": "unverified",
                "occupancy_proposal": {"schema": "rtm-calibration-region-proposal-1",
                    "status": "requires_human_review", "reference_bounds": list(reference),
                    "calibration_bounds": list(calibration),
                    "selection_basis": "Baseline-grid occupancy only; no amplitude, NRMS, attribute, detector, or finding scores inspected"}}

    def test_four_models_fit_separate_regions_and_field_scoring_resumes(self):
        reference = sample_spatial_population(self.attributes, self.missing, bounds=(0, 6, 0, 12),
            population_id="synthetic-reference", source_identity=self.identity, max_rows=500, chunk_shape=(3, 4, 5), window_samples=3)
        calibration = sample_spatial_population(self.attributes, self.missing, bounds=(6, 12, 0, 12),
            population_id="synthetic-calibration", source_identity=self.identity, max_rows=500, chunk_shape=(3, 4, 5), window_samples=3)
        bundle = fit_classical_models(reference, calibration, quantile=0.98, random_state=13)
        self.assertEqual(set(bundle["models"]), {"mahalanobis", "pca", "isolation_forest", "gmm"})
        self.assertTrue(all(v["score_is_probability"] is False for v in bundle["fit_record"]["models"].values()))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "models"
            partial = score_full_field_models(self.attributes, self.missing, root, bundle=bundle,
                source_identity=self.identity, window_samples=3, chunk_shape=(4, 4, 5), max_chunks=1,
                workers=2)
            self.assertEqual(partial["status"], "incomplete")
            done = score_full_field_models(self.attributes, self.missing, root, bundle=bundle,
                source_identity=self.identity, window_samples=3, chunk_shape=(4, 4, 5), workers=4)
            self.assertEqual(done["status"], "complete")
            self.assertEqual(done["execution"]["worker_count_peak"], 4)
            self.assertEqual(set(done["flagged_voxels"]), set(bundle["models"]))
            for name in bundle["models"]:
                score = np.load(Path(done["run_directory"]) / f"{name}-score.npy", mmap_mode="r")
                flags = np.load(Path(done["run_directory"]) / f"{name}-flag.npy", mmap_mode="r")
                self.assertEqual(score.shape, self.baseline.shape)
                self.assertTrue(np.isnan(score[self.missing]).all())
                self.assertFalse(flags[self.missing].any())
                self.assertTrue(np.isfinite(score[~self.missing]).all())
            cached = score_full_field_models(self.attributes, self.missing, root, bundle=bundle,
                source_identity=self.identity, window_samples=3, chunk_shape=(4, 4, 5), workers=1)
            self.assertTrue(cached["cache_hit"])
            serial = score_full_field_models(self.attributes, self.missing, Path(temp) / "serial",
                bundle=bundle, source_identity=self.identity, window_samples=3,
                chunk_shape=(4, 4, 5), workers=1)
            self.assertEqual(serial["status"], "complete")
            alternate = score_full_field_models(self.attributes, self.missing, Path(temp) / "alternate",
                bundle=bundle, source_identity=self.identity, window_samples=3,
                chunk_shape=(3, 6, 5), workers=4)
            self.assertEqual(alternate["status"], "complete")
            for name in bundle["models"]:
                parallel_score = np.load(Path(done["run_directory"]) / f"{name}-score.npy", mmap_mode="r")
                serial_score = np.load(Path(serial["run_directory"]) / f"{name}-score.npy", mmap_mode="r")
                alternate_score = np.load(Path(alternate["run_directory"]) / f"{name}-score.npy", mmap_mode="r")
                parallel_flag = np.load(Path(done["run_directory"]) / f"{name}-flag.npy", mmap_mode="r")
                serial_flag = np.load(Path(serial["run_directory"]) / f"{name}-flag.npy", mmap_mode="r")
                alternate_flag = np.load(Path(alternate["run_directory"]) / f"{name}-flag.npy", mmap_mode="r")
                np.testing.assert_array_equal(parallel_score, serial_score)
                np.testing.assert_array_equal(parallel_flag, serial_flag)
                np.testing.assert_array_equal(alternate_score, serial_score)
                np.testing.assert_array_equal(alternate_flag, serial_flag)

    def test_overlapping_or_same_population_fails_closed(self):
        pop_a = sample_spatial_population(self.attributes, self.missing, bounds=(0, 8, 0, 12),
            population_id="a", source_identity=self.identity, max_rows=200, window_samples=3)
        pop_b = sample_spatial_population(self.attributes, self.missing, bounds=(6, 12, 0, 12),
            population_id="b", source_identity=self.identity, max_rows=200, window_samples=3)
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            fit_classical_models(pop_a, pop_b)

    def test_calibration_requires_review_and_spatial_buffer(self):
        review = self._review((0, 4, 0, 12), (6, 10, 0, 12), "fixture-review-1")
        design = validate_spatial_calibration_design((0, 4, 0, 12), (6, 10, 0, 12),
            shape=self.baseline.shape, missing_trace_mask=self.missing,
            buffer_grid_cells=2, minimum_occupied_fraction=0.9, review=review)
        self.assertEqual(design["measured_region_separation_grid_cells"], 2.0)
        self.assertIn("not unchanged-ground-truth", design["interpretation"])
        adjacent_review = self._review((0, 6, 0, 12), (6, 12, 0, 12), "adjacent-review")
        with self.assertRaisesRegex(ValueError, "buffer"):
            validate_spatial_calibration_design((0, 6, 0, 12), (6, 12, 0, 12),
                shape=self.baseline.shape, missing_trace_mask=self.missing,
                buffer_grid_cells=2, minimum_occupied_fraction=0.9, review=adjacent_review)

    def test_unlabeled_exploratory_design_does_not_require_a_fake_reviewer(self):
        reference, calibration = (0, 4, 0, 12), (6, 10, 0, 12)
        design = {
            "execution_mode": "unlabeled_exploratory_screening",
            "record_status": "not_a_geological_approval",
            "selection_basis": "grid_partition_selected_without_inspecting_anomaly_scores",
            "ground_truth_status": "unlabeled_screening_only",
            "physical_registration_status": "unverified",
            "limitations": ["physical_4d_registration_unverified", "sample_origin_unverified",
                "geological_comparability_uncertain", "acquisition_processing_comparability_unverified"],
            "occupancy_proposal": {"schema": "rtm-calibration-region-proposal-1",
                "status": "requires_human_review", "reference_bounds": list(reference),
                "calibration_bounds": list(calibration),
                "selection_basis": "Baseline-grid occupancy only; no anomaly scores inspected"}}
        result = validate_spatial_calibration_design(reference, calibration,
            shape=self.baseline.shape, missing_trace_mask=self.missing,
            buffer_grid_cells=2, minimum_occupied_fraction=0.9, review=design)
        self.assertEqual(result["review"], design)
        self.assertIn("not unchanged-ground-truth", result["interpretation"])

    def test_calibration_rejects_score_selected_or_low_occupancy_regions(self):
        review = self._review((0, 4, 0, 12), (6, 10, 0, 12), "fixture-review-2")
        review["selection_basis"] = "chosen_because_low_NRMS"
        with self.assertRaisesRegex(ValueError, "without inspecting"):
            validate_spatial_calibration_design((0, 4, 0, 12), (6, 10, 0, 12),
                shape=self.baseline.shape, missing_trace_mask=self.missing,
                buffer_grid_cells=2, minimum_occupied_fraction=0.9, review=review)
        review["selection_basis"] = "grid_partition_selected_without_inspecting_anomaly_scores"
        sparse = self.missing.copy(); sparse[6:10, 0:12] = True
        with self.assertRaisesRegex(ValueError, "occupied-trace fraction"):
            validate_spatial_calibration_design((0, 4, 0, 12), (6, 10, 0, 12),
                shape=self.baseline.shape, missing_trace_mask=sparse,
                buffer_grid_cells=2, minimum_occupied_fraction=0.9, review=review)

    def test_occupancy_proposal_is_deterministic_and_requires_review(self):
        mask=np.zeros((256,320),dtype=bool)
        mask[32:160,32:160]=True
        first=propose_spatial_regions_from_occupancy(mask,region_shape=(64,64),stride=(32,32),
            buffer_grid_cells=32,minimum_occupied_fraction=0.95)
        second=propose_spatial_regions_from_occupancy(mask,region_shape=(64,64),stride=(32,32),
            buffer_grid_cells=32,minimum_occupied_fraction=0.95)
        self.assertEqual(first,second)
        self.assertEqual(first["status"],"requires_human_review")
        self.assertGreaterEqual(first["measured_separation_grid_cells"],32)
        self.assertIn("no amplitude",first["selection_basis"])


if __name__ == "__main__":
    unittest.main()
