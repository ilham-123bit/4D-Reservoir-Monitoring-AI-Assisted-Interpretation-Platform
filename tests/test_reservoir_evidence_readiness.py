import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from reservoir_time_machine.analysis.recovery_intelligence import reservoir_evolution_readiness
from reservoir_time_machine.data_ingest.well_evidence import (
    build_well_evidence_matrix,
    marker_checkshot_correlations,
    marker_deviation_correlations,
)


class WellEvidenceMatrixTests(unittest.TestCase):
    def test_marker_twt_is_interpolated_only_inside_same_well_checkshot_range(self):
        group = {"tables": [
            {"category": "checkshots", "source": "Well-1_Checkshot.txt",
             "quality": {"status": "structurally_valid_declared_checkshot_pairs"},
             "declared_columns": ["TVDSS", "TWT"], "declared_units": ["m", "ms"],
             "raw_records": [[100, 120], [200, 220]]},
            {"category": "markers", "source": "Well-1_Markers.txt", "ambiguous_rows": 0,
             "declared_columns": ["NAME", "TVDSS"], "declared_units": [None, "m"],
             "raw_records": [["Top_A", 150], ["Outside", 250]]},
        ]}
        result = marker_checkshot_correlations(group)
        self.assertEqual(result["status"], "source_marker_to_checkshot_interpolation_only")
        self.assertEqual(len(result["markers"]), 1)  # no extrapolation
        self.assertEqual(result["markers"][0]["marker"], "Top_A")
        self.assertEqual(result["markers"][0]["interpolated_source_checkshot_twt_ms"], 170.0)
        self.assertIn("Not a seismic tie", result["limitation"])

    def test_matrix_keeps_curve_semantics_and_coordinate_reference_unresolved(self):
        report = {"by_well": {"well-2": {
            "workbook_records": [{"coordinate_evidence": {"x": 1.0, "y": 2.0,
                "units": None, "crs": None, "datum": None}}],
            "logs": [{"source": "Well-2.las", "provenance": "standard", "samples": 10,
                "depth_mnemonic": "DEPT", "depth_unit": "METRES", "depth_start": 100,
                "depth_stop": 109, "curves": [
                    {"mnemonic": "GR_P", "unit": "gAPI"},
                    {"mnemonic": "SWE", "unit": "fraction"},
                ]}],
            "tables": [],
        }}}
        row = build_well_evidence_matrix(report)[0]
        self.assertEqual(row["curve_families"]["gamma_ray"], ["GR_P"])
        self.assertEqual(row["curve_families"]["saturation_or_volume_interpreted"], ["SWE"])
        self.assertIn("not compared to SEG-Y", row["xy_reference"])
        self.assertEqual(row["seismic_association"], "unverified")

    def test_interpreted_fraction_and_sentinel_like_values_are_flagged_not_relabelled(self):
        from reservoir_time_machine.data_ingest.well_evidence import well_evidence_summary
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "well_data" / "las" / "Well-1.las"
            path.parent.mkdir(parents=True)
            path.write_text("fixture")
            parsed = {"curves": [
                {"mnemonic": "SWE", "unit": "frac", "description": "interpreted", "sample_count": 2, "null_count": 0},
                {"mnemonic": "VWATER", "unit": "frac", "description": "interpreted", "sample_count": 2, "null_count": 0},
            ], "data": {"SWE": [0.2, 1.2], "VWATER": [9999.0, 0.2]},
                "provenance": "standard", "well_id": "Well-1", "sample_count": 2,
                "depth_mnemonic": "DEPT", "depth_unit": "m", "depth_start": 0, "depth_stop": 1}
            with patch("reservoir_time_machine.data_ingest.well_evidence.inspect_las", return_value=parsed):
                report = well_evidence_summary(temp)
            curves = {c["mnemonic"]: c for c in report["logs"][0]["curves"]}
            self.assertIn("declared_fraction_outside_0_to_1_review_scaling_or_sentinels",
                          curves["SWE"]["value_quality_flags"])
            self.assertIn("extreme_finite_interpreted_value_review_source_encoding_or_sentinels",
                          curves["VWATER"]["value_quality_flags"])
            self.assertEqual(curves["SWE"]["evidence_kind"], "derived_or_interpreted_petrophysical_curve")

    def test_mdkb_marker_maps_to_source_tvd_but_not_to_checkshot_tvdss(self):
        group = {"tables": [
            {"category": "deviations", "source": "Well-2_Deviation.txt",
             "quality": {"status": "depth_survey_structurally_valid_orientation_units_missing"},
             "declared_columns": ["MDKB", "TVD", "INCL", "AZIM"], "declared_units": ["m", "m", None, None],
             "raw_records": [[0, 0, 0, 0], [1000, 999, 0, 0], [2000, 1998, 0, 0]]},
            {"category": "markers", "source": "Well-2_Markers.txt", "ambiguous_rows": 0,
             "declared_columns": ["NAME", "MDKB"], "declared_units": [None, "m"],
             "raw_records": [["I-110_Coal", 1919]]},
        ]}
        result = marker_deviation_correlations(group)
        marker = result["markers"][0]
        self.assertEqual(marker["marker"], "I-110_Coal")
        self.assertAlmostEqual(marker["interpolated_source_deviation_tvd_m"], 1917.081)
        self.assertIn("not establish TWT", result["limitation"])


class ReservoirEvolutionReadinessTests(unittest.TestCase):
    def test_neither_change_nor_low_sweep_is_promoted_without_ties(self):
        result = reservoir_evolution_readiness(
            {"association_assessment": {"spatial_well_to_seismic": False,
                "vertical_well_to_sample": False}, "well_evidence_matrix": [{}], "by_well": {}},
            {"provenance": {"sample_domain": "unknown", "sample_origin": None,
                "physical_4d_registration": "unverified"},
             "candidate_interpretation": {"qualified_recovery_opportunity_count": 0}},
        )
        self.assertEqual(result["statuses"]["production_response"], "withheld_insufficient_tie_or_registration")
        self.assertEqual(result["statuses"]["limited_response_or_poor_sweep"],
            "withheld_insufficient_reservoir_mask_and_production_context")
        self.assertEqual(result["statuses"]["development_opportunity_count"], 0)
        self.assertEqual(len(result["discipline_actions"]), 6)
        self.assertIn("not evidence of remaining hydrocarbons", result["interpretation_rule"])


if __name__ == "__main__":
    unittest.main()
