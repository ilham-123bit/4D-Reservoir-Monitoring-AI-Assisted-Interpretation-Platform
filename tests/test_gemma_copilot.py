"""No network, private file or Gemma service is required for these tests."""
from __future__ import annotations

import json
import unittest

from reservoir_time_machine.ai.gemma_copilot import (
    REGISTRATION_WARNING, ask_gemma, build_copilot_context,
    context_identity, offline_answer, validate_gemma_reply, validate_question,
)
from reservoir_time_machine.settings import LLMSettings


class GemmaCopilotTests(unittest.TestCase):
    def setUp(self):
        self.records = [{
            "candidate": {
                "well_id": "well-6", "assumed_time_origin_ms": 20.0,
                "array_location": [405, 660, 841],
            },
            "measurement": {
                "observed": {
                    "nrms_percent": 40.09553923,
                    "signed_difference_median": 16.45,
                    "zero_lag_correlation": 0.91868,
                    "support_fraction": 1.0, "valid_pairs": 525,
                },
                "best_local_lag_samples": 0,
                "off_window_control_nrms_median_percent": 26.50,
                "response_vs_off_window_controls": "larger_change_than_local_off_window_comparisons",
            },
            "local_spatial": {
                "status": "measured_local_spatial_contrast",
                "core_median_nrms_percent": 40.05,
                "surrounding_median_nrms_percent": 34.34,
                "spatial_nrms_contrast_percentage_points": 5.71,
                "pattern": "selected_region_similar_to_neighborhood",
            },
        }]
        self.context = build_copilot_context(
            selected={"id": "archived-finding-1", "kind": "qc_follow_up",
                      "finding": {"array_location": [12, 34, 56]}},
            scenario_records=self.records,
            evidence_sha256="a" * 64,
        )

    def test_bounded_context_and_provenance_without_private_files(self):
        assert len(self.context["evidence"]) == 5
        assert "source-limitations" in {x["id"] for x in self.context["evidence"]}
        assert REGISTRATION_WARNING in json.dumps(self.context)
        assert "/data/epic20" not in json.dumps(self.context)
        assert self.context["evidence"][-1]["paired_samples"] == 525
        assert self.context["evidence"][-1]["nrms_percent"] > 40
        assert self.context["evidence"][-1]["local_spatial_pattern"] == "selected_region_similar_to_neighborhood"
        assert self.context["evidence"][-1]["local_core_minus_ring_percentage_points"] == 5.71
        assert context_identity(self.context) == context_identity(self.context)

    def test_all_origin_scenarios_get_separate_provenance_ids(self):
        cases = [dict(self.records[0]) for _ in range(3)]
        for row, origin in zip(cases, (-20.0, 0.0, 20.0)):
            row["candidate"] = dict(row["candidate"], assumed_time_origin_ms=origin)
        context = build_copilot_context(scenario_records=cases)
        ids = [x["id"] for x in context["evidence"]]
        assert len(ids) == len(set(ids))

    def test_local_answer_is_explicitly_not_gemma_or_recovery(self):
        answer = offline_answer("Where is the bypassed oil?", self.context)
        assert answer["mode"] == "deterministic_local"
        assert "No recovery location" in answer["answer"]
        assert "40.10%" in answer["answer"]
        assert "source-limitations" in answer["evidence_ids"]

    def test_rejects_unverified_recovery_and_fabricated_evidence(self):
        item = {"answer": "Review Well-6 as a possible investigation.",
                "evidence_ids": ["invented"], "uncertainty": "Unknown registration"}
        with self.assertRaises(ValueError):
            validate_gemma_reply(json.dumps(item), self.context)
        item["evidence_ids"] = ["i100-well-6-origin-+20"]
        item["answer"] = "Proven bypassed oil exists at Well-6"
        with self.assertRaises(ValueError):
            validate_gemma_reply(json.dumps(item), self.context)
        item["answer"] = "Check the time-origin sensitivity at Well-6."
        self.assertEqual(
            validate_gemma_reply(json.dumps(item), self.context)["mode"],
            "gemma_evidence_grounded_unverified",
        )

    def test_private_evidence_is_never_sent_without_authorization(self):
        settings = LLMSettings(base_url="http://127.0.0.1:9999/v1")
        with self.assertRaises(PermissionError):
            ask_gemma("What changed?", self.context, settings, consent=False)
        with self.assertRaises(PermissionError):
            ask_gemma("What changed?", self.context, settings, consent=True)
        with self.assertRaises(ValueError):
            validate_question(" " * 4)
        with self.assertRaises(ValueError):
            validate_question("a" * 1201)

    def test_empty_context_still_yields_transparent_local_answer(self):
        context = build_copilot_context()
        answer = offline_answer("What happened?", context)
        assert "No checksummed I-100 scenario measurements" in answer["answer"]
        assert "not proven recovery targets" in answer["answer"]


if __name__ == "__main__":
    unittest.main()
