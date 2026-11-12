import unittest

from reservoir_time_machine.analysis.recovery_intelligence import (
    default_investigation_id,
    investigation_records,
    select_investigation,
)


class RecoveryIntelligenceTests(unittest.TestCase):
    def setUp(self):
        self.registry = {
            "historical_findings": [
                {"id": "historical-1", "rank": 1, "array_location": [3, 4, 5],
                 "extent": [[2, 4], [3, 5], [4, 6]], "component_voxels": 20},
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

    def test_unknown_id_is_rejected(self):
        with self.assertRaises(KeyError):
            select_investigation(self.registry, "made-up")


if __name__ == "__main__":
    unittest.main()
