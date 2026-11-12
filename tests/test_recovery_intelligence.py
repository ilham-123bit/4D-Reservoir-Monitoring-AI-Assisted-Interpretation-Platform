import unittest

from reservoir_time_machine.analysis.recovery_intelligence import (
    default_investigation_id,
    investigation_records,
    select_investigation,
)


class RecoveryIntelligenceTests(unittest.TestCase):
    def setUp(self):
        self.registry = {
            "id": "assessment-test-identity",
            "historical_findings": [
                {"id": "historical-1", "rank": 1, "array_location": [3, 4, 5],
                 "inline_id": 103, "crossline_id": 204, "sample_index": 5,
                 "extent": [[2, 4], [3, 5], [4, 6]], "component_voxels": 20,
                 "signed_difference_mean": -2.5, "absolute_difference_mean": 3.0,
                 "nrms_mean_percent": 74.55, "min_paired_support": 9,
                 "detector_flagged_voxels": {"mahalanobis": 0, "pca": 0,
                     "isolation_forest": 0, "gmm": 0}, "mad_only": True,
                 "data_support": {"MAD_member_fraction": 1.0},
                 "plausible_alternatives": ["registration sensitivity"],
                 "observation_confidence": "screening only",
                 "geological_interpretation_confidence": "unverified"},
                {"id": "historical-2", "rank": 2, "array_location": [7, 8, 9],
                 "extent": [[6, 8], [7, 9], [8, 10]], "component_voxels": 12},
            ],
            "qc_follow_up_actions": [{
                "finding": {"id": "historical-2", "array_location": [7, 8, 9]},
                "basis": "repeatability QC",
            }],
        }

    def test_qc_action_is_default_and_precedes_historical_catalogue(self):
        records = investigation_records(self.registry)
        self.assertEqual(default_investigation_id(records), "historical-2")
        self.assertEqual([item["id"] for item in records], ["historical-2", "historical-1"])
        self.assertEqual(records[0]["kind"], "qc_follow_up")
        self.assertEqual(records[1]["kind"], "historical_screening_record")

    def test_selection_preserves_original_location_and_nonprobabilistic_status(self):
        record = select_investigation(self.registry, "historical-1")
        self.assertEqual(record["finding"]["array_location"], [3, 4, 5])
        self.assertEqual(record["recovery_opportunity"], "unverified")
        self.assertIn("Not yet quantifiable", record["recovery_chance_of_success"])

    def test_structured_interpretation_preserves_observations_and_limits_claims(self):
        record = select_investigation(self.registry, "historical-1")["interpretation_record"]
        self.assertEqual(record["schema"], "rtm-investigation-interpretation-1")
        self.assertEqual(record["source_assessment_id"], "assessment-test-identity")
        self.assertEqual(record["location"]["inline_id"], 103)
        self.assertEqual(record["location"]["crossline_id"], 204)
        self.assertEqual(record["location"]["sample_index"], 5)
        self.assertEqual(record["observed"]["signed_monitor_minus_baseline_mean_source_units"], -2.5)
        self.assertEqual(record["observed"]["nrms_mean_percent"], 74.55)
        self.assertEqual(record["model_evidence"]["detector_threshold_flagged_voxels"]["gmm"], 0)
        self.assertEqual(record["data_quality"]["minimum_paired_sample_support"], 9)
        self.assertEqual(record["reservoir_relevance"], "Unverified; no validated reservoir interval or well-to-seismic association is available.")
        self.assertIn("no recovery opportunity", record["recovery_status"])
        self.assertEqual(record["alternatives"], ["registration sensitivity"])

    def test_unknown_id_is_rejected(self):
        with self.assertRaises(KeyError):
            select_investigation(self.registry, "made-up")


if __name__ == "__main__":
    unittest.main()
