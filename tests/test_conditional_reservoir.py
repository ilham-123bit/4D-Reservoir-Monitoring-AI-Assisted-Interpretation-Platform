"""Self-contained numerical/negative-control tests for conditional I-100 screening."""
import unittest
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np

from reservoir_time_machine.analysis.conditional_reservoir import (
    conditional_i100_locations,
    conditional_follow_up_actions,
    conditional_scenario_key,
    evaluate_conditional_window,
    load_conditional_i100_evidence,
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
        self.assertIn("origin-p0", row["scenario_id"])

    def test_origin_scenarios_have_distinct_state_keys(self):
        rows = [conditional_i100_locations(
            self._wells(), self._projection(), np.arange(100, 110),
            np.arange(200, 210), 1301, origin_ms=origin)[0]
            for origin in (-20, 0, 20)]
        self.assertEqual(len({conditional_scenario_key(row) for row in rows}), 3)
        self.assertEqual(len({row["scenario_id"] for row in rows}), 3)

    def test_saved_evidence_checksum_identity_and_geometry_are_validated(self):
        candidates = [conditional_i100_locations(
            self._wells(), self._projection(), np.arange(100, 110),
            np.arange(200, 210), 1301, origin_ms=origin)[0]
            for origin in (-20, 0, 20)]
        body = {
            "schema": "rtm-conditional-i100-evidence-1",
            "status": "bounded_conditional_scientific_test_completed",
            "data_identity": "paired-cache-test-id",
            "source_shape": [10, 10, 1301],
            "source_axes": {"inline_ids": [100, 109], "crossline_ids": [200, 209]},
            "measurements": [{
                "candidate": case,
                "measurement": {"array_location": case["array_location"],
                                "observed": {"support_fraction": 1.0, "valid_pairs": 525}},
            } for case in candidates],
        }
        digest = hashlib.sha256(json.dumps(
            body, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.json"
            path.write_text(json.dumps({"body": body, "sha256": digest}), encoding="utf-8")
            loaded = load_conditional_i100_evidence(
                path, expected_data_identity="paired-cache-test-id",
                expected_shape=(10, 10, 1301), inline_ids=np.arange(100, 110),
                crossline_ids=np.arange(200, 210))
            self.assertEqual(len(loaded["records"]), 3)
            self.assertEqual(len({r["scenario_key"] for r in loaded["records"]}), 3)
            with self.assertRaisesRegex(ValueError, "different paired cache"):
                load_conditional_i100_evidence(path, expected_data_identity="wrong-cache")
            body["measurements"][0]["measurement"]["observed"]["support_fraction"] = 1.1
            path.write_text(json.dumps({"body": body, "sha256": digest}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_conditional_i100_evidence(path)

    def test_follow_up_ranking_responds_to_origin_and_exception_without_probability(self):
        rows = []
        for well, origin, nrms, signed, comparison in (
            ("well-5", -20, 14.0, -2.0, "smaller_change_than_local_off_window_comparisons"),
            ("well-5", 0, 26.0, 3.0, "change_similar_to_local_off_window_comparisons"),
            ("well-6", 20, 40.0, 1.0, "larger_change_than_local_off_window_comparisons"),
        ):
            rows.append(({"well_id": well, "assumed_time_origin_ms": origin}, {
                "observed": {"nrms_percent": nrms, "signed_difference_median": signed},
                "response_vs_off_window_controls": comparison,
            }))
        actions = conditional_follow_up_actions(rows)
        self.assertEqual([item["rank"] for item in actions], [1, 2, 3])
  