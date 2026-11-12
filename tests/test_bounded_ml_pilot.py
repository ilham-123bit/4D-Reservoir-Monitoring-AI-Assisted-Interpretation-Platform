from pathlib import Path
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from reservoir_time_machine.analysis.bounded_ml_pilot import (
    pilot_execution_blockers, run_bounded_ml_pilot, validate_crop_plan)
from reservoir_time_machine.attributes.difference import calculate_attributes
from scripts.hpc import cleanup_disposable_ml_run


class BoundedMlPilotTests(unittest.TestCase):
    def setUp(self):
        self.shape = (32, 64, 32)
        rng = np.random.default_rng(78)
        baseline = rng.normal(size=self.shape).astype(np.float32)
        monitor = baseline + rng.normal(scale=0.12, size=self.shape).astype(np.float32)
        self.missing = np.zeros(self.shape[:2], dtype=bool)
        self.missing[18, 26] = True
        baseline[self.missing] = np.nan
        monitor[self.missing] = np.nan
        calculated = calculate_attributes(baseline, monitor, window_samples=3)
        self.attributes = {
            key: np.asarray(getattr(calculated, key), dtype=np.uint8 if key == "paired_sample_count" else np.float32)
            for key in ("signed_difference", "absolute_difference", "normalized_difference",
                        "rms_baseline", "rms_monitor", "nrms", "paired_sample_count")
        }
        self.source_identity = "a" * 64
        self.ref = [0, 8, 0, 8]
        self.cal = [0, 8, 48, 56]
        self.score = [16, 24, 24, 32, 5, 25]

    def plan(self, **kwargs):
        values = dict(shape=self.shape, reference_bounds=self.ref,
            calibration_bounds=self.cal, approved_reference_bounds=self.ref,
            approved_calibration_bounds=self.cal, sample_bounds=[5, 25],
            scoring_bounds=self.score, max_samples=128, max_score_voxels=2000,
            max_output_bytes=1_000_000, minimum_separation=8, window_samples=3)
        values.update(kwargs)
        return validate_crop_plan(**values)

    def test_bounds_and_separation_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            self.plan(scoring_bounds=[28, 36, 24, 32, 5, 25])
        with self.assertRaisesRegex(ValueError, "separation"):
            self.plan(calibration_bounds=[0, 8, 6, 14],
                      approved_calibration_bounds=[0, 8, 0, 16])
        with self.assertRaisesRegex(ValueError, "subsets"):
            self.plan(reference_bounds=[0, 9, 0, 8])

    def test_sample_and_output_budgets_are_enforced(self):
        with self.assertRaisesRegex(ValueError, "max_samples"):
            self.plan(max_samples=50_001)
        with self.assertRaisesRegex(ValueError, "max_score_voxels"):
            self.plan(max_score_voxels=100)
        with self.assertRaisesRegex(ValueError, "max_output_bytes"):
            self.plan(max_output_bytes=100)

    def test_static_config_and_slurm_allocation_are_execution_gates(self):
        self.assertEqual(len(pilot_execution_blockers(
            config_valid=False, in_slurm_allocation=False)), 2)
        self.assertEqual(pilot_execution_blockers(
            config_valid=True, in_slurm_allocation=False),
            ["real-data pilot must run inside an approved Slurm allocation"])
        self.assertEqual(pilot_execution_blockers(
            config_valid=True, in_slurm_allocation=True), [])

    def test_synthetic_models_score_masked_crop_and_refuse_overwrite(self):
        plan = self.plan()
        input_before = {key: value.copy() for key, value in self.attributes.items()}
        mask_before = self.missing.copy()
        with tempfile.TemporaryDirectory() as td:
            parent = Path(td)
            (parent / "scratch").mkdir(mode=0o700)
            out1 = parent / "scratch" / "trial-a"
            report = run_bounded_ml_pilot(attributes=self.attributes,
                missing_trace_mask=self.missing, source_identity=self.source_identity,
                plan=plan, output_dir=out1, max_samples=128, random_seed=2026,
                chunk_shape=(4, 4, 8), min_support_fraction=0.8,
                window_samples=3, threshold_quantile=0.99,
                inline_ids=np.arange(100, 132), crossline_ids=np.arange(200, 264),
                monitor_only_keys=np.asarray([[118, 226]], dtype=np.int32))
            self.assertEqual(report["status"], "complete")
            self.assertEqual(sum(report["paired_sample_support_distribution"].values()),
                             plan["scoring_voxels"])
            self.assertEqual(set(report["model_status"]),
                             {"mahalanobis", "pca", "isolation_forest", "gmm"})
            self.assertLessEqual(report["reference_population_rows"], 128)
            self.assertLessEqual(report["calibration_population_rows"], 128)
            self.assertGreater(report["unoccupied_grid_sample_positions"], 0)
            run_dir = out1 / report["model_score_run_directory"]
            bad_score = np.load(run_dir / "mahalanobis-score.npy", mmap_mode="r")
            bad_flag = np.load(run_dir / "mahalanobis-flag.npy", mmap_mode="r")
            self.assertTrue(np.isnan(bad_score[2, 2]).all())
            self.assertFalse(bad_flag[2, 2].any())
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                run_bounded_ml_pilot(attributes=self.attributes,
                    missing_trace_mask=self.missing, source_identity=self.source_identity,
                    plan=plan, output_dir=out1, max_samples=128, random_seed=2026,
                    chunk_shape=(4, 4, 8), min_support_fraction=0.8,
                    window_samples=3, threshold_quantile=0.99)

            out2 = parent / "scratch" / "trial-b"
            second = run_bounded_ml_pilot(attributes=self.attributes,
                missing_trace_mask=self.missing, source_identity=self.source_identity,
                plan=plan, output_dir=out2, max_samples=128, random_seed=2026,
                chunk_shape=(4, 4, 8), min_support_fraction=0.8,
                window_samples=3, threshold_quantile=0.99)
            run_dir2 = out2 / second["model_score_run_directory"]
            score2 = np.load(run_dir2 / "mahalanobis-score.npy", mmap_mode="r")
            np.testing.assert_array_equal(bad_score, score2)
            self.assertLessEqual(report["output_bytes"], plan["expected_score_output_bytes"] + 100_000)
            self.assertGreater(report["peak_process_rss_mib"], 0)
            self.assertGreaterEqual(report["input_read_seconds"], 0)
            self.assertGreaterEqual(report["feature_preprocessing_seconds"], 0)
            self.assertIn("mahalanobis", report["per_model_fit_seconds"])
            self.assertIn("gmm", report["per_model_scoring_seconds"])
            self.assertEqual(report["edge_and_coverage_qc"]["status"], "computed")
            self.assertFalse(report["edge_and_coverage_qc"]["monitor_only_used_in_paired_calculation"])
        for key, value in self.attributes.items():
            np.testing.assert_equal(value, input_before[key])
        np.testing.assert_array_equal(self.missing, mask_before)

    def test_shared_fit_multi_chunk_benchmark_is_consistent(self):
        plan = self.plan(scoring_bounds=[16, 20, 24, 28, 5, 21],
                         max_score_voxels=1024, max_output_bytes=1_000_000)
        with tempfile.TemporaryDirectory() as td:
            parent = Path(td)
            out = parent / "scratch" / "multi"
            (parent / "scratch").mkdir(mode=0o700)
            report = run_bounded_ml_pilot(
                attributes=self.attributes, missing_trace_mask=self.missing,
                source_identity=self.source_identity, plan=plan, output_dir=out,
                max_samples=128, random_seed=19, min_support_fraction=0.8,
                window_samples=3, threshold_quantile=0.99,
                chunk_shapes=[(2, 2, 4), (4, 4, 8)],
                experiment_id="well2-coal-control-fixture")
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["experiment_id"], "well2-coal-control-fixture")
            self.assertEqual([x["chunk_shape"] for x in report["chunk_benchmarks"]],
                             [[2, 2, 4], [4, 4, 8]])
            self.assertTrue(report["chunk_score_consistency"]["all_configurations_within_tolerance"])
            self.assertEqual(report["reference_population_rows"],
                             report["calibration_population_rows"])
            self.assertFalse(report["feature_distribution_comparison"]["sample_intervals_match"])
            self.assertEqual(report["feature_distribution_comparison"]["scoring_sample_bounds"],
                             [5, 21])
            self.assertEqual(report["mad_screening"]["status"], "screened")
            self.assertLessEqual(report["mad_screening"]["flagged_positions"],
                                 report["mad_screening"]["valid_positions"])
            self.assertEqual(report["chunk_benchmarks"][0]["score_distributions"]["gmm"]["finite_score_count"],
                             report["chunk_benchmarks"][0]["valid_positions"])
            self.assertGreater(report["chunk_benchmarks"][0]["output_bytes"], 0)

    def test_cleanup_dry_run_is_read_only_and_requires_terminal_job(self):
        plan = self.plan()
        # cleanup_disposable_ml_run rejects symlinked ancestors by design.
        # This environment's tempfile default (/usr/tmp) is a symlink to
        # /var/tmp, so use the canonical private temp root for this test.
        with tempfile.TemporaryDirectory(dir="/var/tmp") as td:
            parent = Path(td)
            root = parent / "ml-runs"
            root.mkdir(mode=0o700)
            protected = parent / "protected"
            protected.mkdir(mode=0o700)
            protected_cache = parent / "protected-cache"
            protected_cache.mkdir(mode=0o700)
            protected_attributes = parent / "protected-attributes"
            protected_attributes.mkdir(mode=0o700)
            protected_paths = [protected, protected_cache, protected_attributes]
            run = root / "pilot-123"
            os.environ["SLURM_JOB_ID"] = "123"
            try:
                run_bounded_ml_pilot(attributes=self.attributes,
                    missing_trace_mask=self.missing, source_identity=self.source_identity,
                    plan=plan, output_dir=run, max_samples=128, random_seed=2026,
                    chunk_shape=(4, 4, 8), min_support_fraction=0.8,
                    window_samples=3, threshold_quantile=0.99)
            finally:
                os.environ.pop("SLURM_JOB_ID", None)
            with mock.patch.object(cleanup_disposable_ml_run, "_scheduler_state", return_value="COMPLETED"):
                preview = cleanup_disposable_ml_run.cleanup(
                    output_root=root, run_id="pilot-123", protected_paths=protected_paths)
            self.assertTrue(preview["dry_run"])
            self.assertTrue(run.exists())
            self.assertGreater(preview["bytes_to_remove"], 0)
            with mock.patch.object(cleanup_disposable_ml_run, "_scheduler_state", return_value="RUNNING"):
                with self.assertRaisesRegex(RuntimeError, "RUNNING"):
                    cleanup_disposable_ml_run.cleanup(
                        output_root=root, run_id="pilot-123", protected_paths=protected_paths,
                        confirm_id="pilot-123", storage_authorization_reference="storage-ref-1")
            self.assertTrue(run.exists())


if __name__ == "__main__":
    unittest.main()
