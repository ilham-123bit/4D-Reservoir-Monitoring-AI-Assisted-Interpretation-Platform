from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/hpc/record_full_field_ml_approval.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("record_full_field_ml_approval", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
build_candidate = MODULE.build_candidate


class ApprovalRecordTests(unittest.TestCase):
    def setUp(self):
        self.current = {
            "schema": "rtm-full-field-ml-config-1",
            "status": "REVIEW_REQUIRED_BEFORE_EXECUTION",
            "reference_bounds": [64, 192, 1088, 1216],
            "calibration_bounds": [896, 1024, 64, 192],
            "well2_scoring_bounds": [600, 728, 131, 259, 850, 901],
            "calibration_review": {
                "reviewed_by": "REPLACE_WITH_REVIEWER",
                "review_id": "REPLACE_WITH_REVIEW_ID",
                "ground_truth_status": "unlabeled_screening_only",
                "physical_registration_status": "unverified",
                "selection_basis": "grid_partition_selected_without_inspecting_anomaly_scores",
                "occupancy_proposal": {
                    "reference_bounds": [64, 192, 1088, 1216],
                    "calibration_bounds": [896, 1024, 64, 192],
                    "measured_separation_grid_cells": 1139.4876041449509,
                },
            },
        }
        self.answers = {
            "reviewed_by": "Authorized Reviewer",
            "review_id": "review-unique-2026-11-11",
            "decision_confirmation": "APPROVED WITH CONDITIONS",
            "scope_confirmation": "YES",
            "limitations_confirmation": "YES",
            "review_notes": "Approved Design A fitting and independent Well-2 scoring with stated limits.",
            "storage_authorization_reference": "storage-auth-actual-2026-11-11",
            "storage_authorization_evidence": "Project storage allocation approved for private pilot output.",
            "authorized_capacity_bytes": "134217728",
        }

    def test_builds_candidate_with_consistent_bounds_and_retains_scoring_region(self):
        original = copy.deepcopy(self.current)
        candidate = build_candidate(self.current, self.answers)
        self.assertEqual(self.current, original)
        self.assertEqual(candidate["status"], "APPROVED_FOR_SCREENING")
        self.assertEqual(candidate["reference_bounds"], [64, 192, 1088, 1216])
        self.assertEqual(candidate["calibration_bounds"], [869, 997, 64, 192])
        review = candidate["calibration_review"]
        proposal = review["occupancy_proposal"]
        self.assertEqual(proposal["calibration_bounds"], candidate["calibration_bounds"])
        self.assertEqual(proposal["measured_separation_grid_cells"], 1123.0071237530062)
        self.assertEqual(candidate["well2_scoring_bounds"], [600, 728, 131, 259, 850, 901])
        self.assertEqual(review["approved_scope"], [
            "design_a_reference_calibration_fit", "well2_coal_control_scoring"])
        self.assertIn("well_to_seismic_tie_unverified", review["accepted_limitations"])

    def test_decline_or_insufficient_capacity_is_rejected(self):
        declined = dict(self.answers, decision_confirmation="NO")
        with self.assertRaisesRegex(ValueError, "did not confirm"):
            build_candidate(self.current, declined)
        too_small = dict(self.answers, authorized_capacity_bytes="128")
        with self.assertRaisesRegex(ValueError, "at least 134217728"):
            build_candidate(self.current, too_small)


if __name__ == "__main__":
    unittest.main()
