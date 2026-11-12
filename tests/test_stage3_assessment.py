from __future__ import annotations

import unittest

from reservoir_time_machine.analysis.stage3_assessment import (
    capped_stage3_priority,
    normalized_support_fraction,
)


class Stage3AssessmentTests(unittest.TestCase):
    def test_support_fraction_is_capped_when_support_exceeds_requirement(self):
        self.assertEqual(normalized_support_fraction(9, 8), 1.0)
        self.assertEqual(normalized_support_fraction(6, 8), 0.75)

    def test_capped_priority_is_a_separate_score_and_preserves_qc_multiplier(self):
        finding = {
            "priority_components": {
                "change_strength": 1.0,
                "component_extent": 0.5,
                "detector_vote_fraction": 0.0,
                "support_fraction": 1.125,
            },
            "survey_edge_factor": 1.0,
            "unoccupied_grid_factor": 1.0,
            "monitor_only_coverage_factor": 1.0,
            "edge_gradient_factor": 1.0,
        }
        self.assertAlmostEqual(capped_stage3_priority(finding), 0.625)
        self.assertEqual(finding["priority_components"]["support_fraction"], 1.125)

    def test_invalid_support_and_qc_factors_are_rejected(self):
        with self.assertRaises(ValueError):
            normalized_support_fraction(8, 0)
        finding = {
            "priority_components": {
                "change_strength": 0.2,
                "component_extent": 0.2,
                "detector_vote_fraction": 0.2,
                "support_fraction": 1.0,
            },
            "survey_edge_factor": 1.2,
        }
        with self.assertRaises(ValueError):
            capped_stage3_priority(finding)


if __name__ == "__main__":
    unittest.main()
