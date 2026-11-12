import importlib.util
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/hpc/validate_full_field_ml_review.py"
SPEC = importlib.util.spec_from_file_location("review_gate", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SubmissionGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = {
            "schema": "rtm-full-field-ml-config-1",
            "status": "APPROVED_FOR_SCREENING",
            "reference_bounds": [64, 192, 1088, 1216],
            "calibration_bounds": [869, 997, 64, 192],
            "attribute_run_directory": "full-field-attributes/run-a",
            "analysis_output_directory": "full-field-ml-runs/run-b",
            "expected_shape": [8, 8, 8],
            "minimum_free_output_bytes": 1000000,
            "calibration_review": {
                "reviewed_by": "Authorized Reviewer",
                "review_id": "review-2026-001",
                "storage_authorization_reference": "storage-review-2026-003",
                "storage_authorization_evidence": "Synthetic fixture capacity approval for validator tests only.",
                "authorized_capacity_bytes": 134217728,
                "decision": "approved_with_conditions",
                "approved_scope": ["design_a_reference_calibration_fit", "well2_coal_control_scoring"],
                "review_notes": "Approved for unlabeled exploratory screening with structural and acquisition limitations accepted.",
                "decision_scope": "unlabeled_exploratory_4d_anomaly_screening_only",
                "accepted_limitations": ["neither_region_verified_unchanged",
                    "geological_comparability_uncertain",
                    "acquisition_processing_comparability_unverified",
                    "physical_4d_registration_unverified", "sample_domain_unverified"],
                "selection_basis": "grid_partition_selected_without_inspecting_anomaly_scores",
                "ground_truth_status": "unlabeled_screening_only",
                "physical_registration_status": "unverified",
                "minimum_buffer_grid_cells": 32,
                "minimum_occupied_fraction": 0.95,
                "occupancy_proposal": {
                    "schema": "rtm-calibration-region-proposal-1",
                    "status": "requires_human_review",
                    "reference_bounds": [64, 192, 1088, 1216],
                    "calibration_bounds": [869, 997, 64, 192],
                    "reference_occupied_fraction": 1.0,
                    "calibration_occupied_fraction": 1.0,
                    "measured_separation_grid_cells": 1123.0071237530062,
                    "minimum_buffer_grid_cells": 32,
                    "minimum_occupied_fraction": 0.95,
                    "selection_basis": "Baseline-grid occupancy only; no amplitude, NRMS, attribute, detector, or finding scores inspected",
                },
            },
        }
        self.output = Path(self.temp.name) / "output"
        self.output.mkdir(mode=0o700)
        self.cache = Path(self.temp.name) / "cache"
        self.data = Path(self.temp.name) / "data"
        self.data.mkdir(mode=0o700)

    def test_reviewed_preferred_configuration_passes_without_running_models(self):
        result = MODULE.validate_config(self.config, self.output, self.cache, self.data)
        self.assertEqual(result["status"], "review_config_valid")
        self.assertEqual(result["calibration_bounds"], [869, 997, 64, 192])
        self.assertAlmostEqual(result["separation_grid_cells"], 1123.0071237530062)

    def test_pending_status_and_placeholder_reviewer_fail_closed(self):
        self.config["status"] = "REVIEW_REQUIRED_BEFORE_EXECUTION"
        with self.assertRaisesRegex(ValueError, "APPROVED_FOR_SCREENING"):
            MODULE.validate_config(self.config, self.output, self.cache, self.data)
        self.config["status"] = "APPROVED_FOR_SCREENING"
        self.config["calibration_review"]["reviewed_by"] = "REPLACE_WITH_REVIEWER"
        with self.assertRaisesRegex(ValueError, "actual human review"):
            MODULE.validate_config(self.config, self.output, self.cache, self.data)

    def test_bounds_and_separation_metadata_must_match(self):
        self.config["calibration_bounds"] = [896, 1024, 64, 192]
        with self.assertRaisesRegex(ValueError, "bounds"):
            MODULE.validate_config(self.config, self.output, self.cache, self.data)
        self.config["calibration_bounds"] = [869, 997, 64, 192]
        self.config["calibration_review"]["occupancy_proposal"]["measured_separation_grid_cells"] = 1139.4876
        with self.assertRaisesRegex(ValueError, "separation"):
            MODULE.validate_config(self.config, self.output, self.cache, self.data)

    def test_proposal_thresholds_must_match_review_thresholds(self):
        self.config["calibration_review"]["occupancy_proposal"]["minimum_buffer_grid_cells"] = 64
        with self.assertRaisesRegex(ValueError, "thresholds disagree"):
            MODULE.validate_config(self.config, self.output, self.cache, self.data)

    def test_approved_scope_decision_and_capacity_evidence_are_required(self):
        self.config["calibration_review"].pop("approved_scope")
        with self.assertRaisesRegex(ValueError, "approved_scope"):
            MODULE.validate_config(self.config, self.output, self.cache, self.data)
        self.config["calibration_review"]["approved_scope"] = [
            "design_a_reference_calibration_fit", "well2_coal_control_scoring"]
        self.config["calibration_review"]["authorized_capacity_bytes"] = 1
        with self.assertRaisesRegex(ValueError, "128 MiB"):
            MODULE.validate_config(self.config, self.output, self.cache, self.data)

    def test_output_paths_cannot_overlap_paired_cache_or_attributes(self):
        with self.assertRaisesRegex(ValueError, "overlap paired cache"):
            MODULE.validate_config(self.config, self.output, self.output / "cache", self.data)
        self.config["analysis_output_directory"] = "full-field-attributes/run-a/subdir"
        with self.assertRaisesRegex(ValueError, "separate"):
            MODULE.validate_config(self.config, self.output, self.cache, self.data)

    def test_preferred_regions_are_required_and_output_cannot_overlap_source(self):
        self.config["calibration_bounds"] = [869, 997, 65, 193]
        with self.assertRaisesRegex(ValueError, "preferred"):
            MODULE.validate_config(self.config, self.output, self.cache, self.data)
        self.config["calibration_bounds"] = [869, 997, 64, 192]
        with self.assertRaisesRegex(ValueError, "original data"):
            MODULE.validate_config(self.config, self.data, self.cache, self.data)

    def test_operator_authorized_exploratory_mode_does_not_claim_scientific_approval(self):
        config = {
            "schema": "rtm-full-field-ml-config-1",
            "status": "UNLABELED_EXPLORATORY_SCREENING",
            "execution_scope": "operator_authorized_unlabeled_full_field_screening",
            "reference_bounds": [64, 192, 1088, 1216],
            "calibration_bounds": [869, 997, 64, 192],
            "fit_sample_bounds": [0, 8],
            "expected_shape": [1029, 1248, 1301],
            "minimum_free_output_bytes": 65000000000,
            "attribute_run_directory": "full-field-attributes/run-a",
            "analysis_output_directory": "full-field-ml-runs/run-b",
            "calibration_design": {
                "execution_mode": "unlabeled_exploratory_screening",
                "record_status": "not_a_geological_approval",
                "selection_basis": "grid_partition_selected_without_inspecting_anomaly_scores",
                "ground_truth_status": "unlabeled_screening_only",
                "physical_registration_status": "unverified",
                "minimum_buffer_grid_cells": 32,
                "minimum_occupied_fraction": 0.95,
                "limitations": ["physical_4d_registration_unverified", "sample_origin_unverified",
                    "geological_comparability_uncertain", "acquisition_processing_comparability_unverified",
                    "well_to_seismic_tie_unverified", "thresholds_are_not_probabilities"],
                "occupancy_proposal": {
                    "schema": "rtm-calibration-region-proposal-1",
                    "status": "requires_human_review",
                    "reference_bounds": [64, 192, 1088, 1216],
                    "calibration_bounds": [869, 997, 64, 192],
                    "reference_occupied_fraction": 1.0,
                    "calibration_occupied_fraction": 1.0,
                    "measured_separation_grid_cells": 1123.0071237530062,
                    "minimum_buffer_grid_cells": 32,
                    "minimum_occupied_fraction": 0.95,
                    "selection_basis": "Baseline-grid occupancy only; no amplitude or anomaly scores inspected"}},
        }
        with patch.object(MODULE.os, "statvfs", return_value=SimpleNamespace(
                f_bavail=100_000_000_000, f_frsize=1)):
            result = MODULE.validate_config(config, self.output, self.cache, self.data)
        self.assertEqual(result["status"], "exploratory_config_valid")
        self.assertNotIn("review_id", result)

    def test_exploratory_mode_rejects_old_calibration_and_missing_limitations(self):
        # Derive the complete fixture from the test above without invoking ML.
        config = {
            "schema": "rtm-full-field-ml-config-1", "status": "UNLABELED_EXPLORATORY_SCREENING",
            "execution_scope": "operator_authorized_unlabeled_full_field_screening",
            "reference_bounds": [64, 192, 1088, 1216], "calibration_bounds": [896, 1024, 64, 192],
            "fit_sample_bounds": [0, 1301], "expected_shape": [1029, 1248, 1301],
            "minimum_free_output_bytes": 65000000000,
            "attribute_run_directory": "full-field-attributes/run-a", "analysis_output_directory": "full-field-ml-runs/run-b",
            "calibration_design": {"execution_mode": "unlabeled_exploratory_screening",
                "record_status": "not_a_geological_approval", "selection_basis": "grid_partition_selected_without_inspecting_anomaly_scores",
                "ground_truth_status": "unlabeled_screening_only", "physical_registration_status": "unverified",
                "minimum_buffer_grid_cells": 32, "minimum_occupied_fraction": 0.95,
                "limitations": ["physical_4d_registration_unverified", "sample_origin_unverified", "geological_comparability_uncertain",
                    "acquisition_processing_comparability_unverified", "well_to_seismic_tie_unverified", "thresholds_are_not_probabilities"],
                "occupancy_proposal": {"schema": "rtm-calibration-region-proposal-1", "status": "requires_human_review",
                    "reference_bounds": [64,192,1088,1216], "calibration_bounds": [869,997,64,192],
                    "reference_occupied_fraction": 1.0, "calibration_occupied_fraction": 1.0,
                    "measured_separation_grid_cells": 1123.0071237530062, "minimum_buffer_grid_cells": 32,
                    "minimum_occupied_fraction": 0.95, "selection_basis": "Baseline-grid occupancy only; no anomaly scores inspected"}}}
        with patch.object(MODULE.os, "statvfs", return_value=SimpleNamespace(
                f_bavail=100_000_000_000, f_frsize=1)):
            with self.assertRaisesRegex(ValueError, "preferred"):
                MODULE.validate_config(config, self.output, self.cache, self.data)
            config["calibration_bounds"] = [869, 997, 64, 192]
            config["calibration_design"]["limitations"] = []
            with self.assertRaisesRegex(ValueError, "omits a required scientific limitation"):
                MODULE.validate_config(config, self.output, self.cache, self.data)


if __name__ == "__main__":
    unittest.main()
