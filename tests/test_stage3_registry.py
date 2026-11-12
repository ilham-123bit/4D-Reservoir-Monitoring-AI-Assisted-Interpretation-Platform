import hashlib
import json
import tempfile
from pathlib import Path

import unittest
import numpy as np

from reservoir_time_machine.analysis.stage3_registry import (
    detector_threshold_flag_count,
    load_stage3_registry,
)


def _write_package(tmp_path, shape=(4, 5, 6)):
    shape = list(shape)
    records = []
    for rank in range(1, 11):
        il, xl, sample = ((1, 2, 3) if rank == 1 else
                          (rank % shape[0], (rank + 1) % shape[1], rank % shape[2]))
        records.append({
            "rank": rank, "id": f"field-{rank:020d}", "array_location": [il, xl, sample],
            "inline_id": 100 + il, "crossline_id": 200 + xl, "sample_index": sample,
            "extent": [[il, il], [xl, xl], [sample, sample]], "component_voxels": 1,
            "signed_difference_mean": -1.0, "absolute_difference_mean": 1.0,
            "nrms_mean_percent": 20.0, "min_paired_support": 9,
            "detector_flagged_voxels": {"mahalanobis": 0, "pca": 0, "isolation_forest": 0, "gmm": 0},
            "mad_only": True, "edge_risk": 0.0, "edge_qc": {}, "local_repeatability": {},
            "geophysics_next_action": "Check repeatability.",
        })
    ids = [item["id"] for item in records]
    corrected = [{"id": ident, "corrected_index": 0.5} for ident in ids]
    document = {
        "schema": "rtm-stage3-evidence-assessment-1",
        "status": "separate_scientific_assessment_not_published",
        "provenance": {"source_shape": shape, "source_inline_ids": [100, 100 + shape[0] - 1],
                       "source_crossline_ids": [200, 200 + shape[1] - 1]},
        "limitations": ["registration unverified"],
        "consensus": {"voxel_count": 0},
        "stage2_top_ten_review": {
            "all_ten_mad_only": True, "all_ten_four_detector_threshold_flags_zero": True,
            "mad_only_voxels_in_candidate_union": 99,
            "three_detector_consensus_voxels_in_candidate_union": 1,
            "finding_records_detailed": records,
            "corrected_priority_order_without_geological_reweighting": corrected,
        },
        "candidate_interpretation": {"qualified_recovery_opportunity_count": 0,
            "follow_up_qc_queue": ids[:2], "queue_basis": "QC follow-up only"},
    }
    payload = json.dumps(document, sort_keys=True).encode()
    (tmp_path / "stage3_assessment.json").write_bytes(payload)
    companion = b"small fixture"
    (tmp_path / "local_repeatability.csv").write_bytes(companion)
    entries = [f"{hashlib.sha256(payload).hexdigest()}  stage3_assessment.json",
               f"{hashlib.sha256(companion).hexdigest()}  local_repeatability.csv"]
    (tmp_path / "stage3_manifest.sha256").write_text("\n".join(entries) + "\n")
    return tmp_path / "stage3_assessment.json"


class Stage3RegistryTests(unittest.TestCase):
  def test_detector_flag_count_accepts_both_stage3_mapping_fields(self):
    self.assertEqual(detector_threshold_flag_count({
        "detector_threshold_flagged_voxels": {"mahalanobis": 2, "pca": 1},
        "detector_flagged_voxels": {"mahalanobis": 99},
    }), 3)
    self.assertEqual(detector_threshold_flag_count({
        "detector_flagged_voxels": {"mahalanobis": 2, "gmm": 1},
    }), 3)
    self.assertEqual(detector_threshold_flag_count({
        "detector_threshold_flagged_voxels": 4,
    }), 4)

  def test_detector_flag_count_rejects_invalid_counts(self):
    with self.assertRaisesRegex(ValueError, "non-negative integers"):
      detector_threshold_flag_count({"detector_flagged_voxels": {"gmm": -1}})

  def test_stage3_grounding_packet_is_deterministic_and_nonprobabilistic(self):
    from reservoir_time_machine.ai.grounding import interpret, stage3_finding_packet
    item = {"id": "field-test-grounding", "absolute_difference_mean": 2.0,
        "signed_difference_mean": -1.0, "nrms_mean_percent": 30.0,
        "min_paired_support": 9, "component_voxels": 20, "mad_only": True,
        "inline_id": 410, "crossline_id": 2283, "sample_index": 61,
        "detector_threshold_flagged_voxels": {"mahalanobis": 0, "pca": 0,
            "isolation_forest": 0, "gmm": 0}, "edge_qc": {}}
    packet = stage3_finding_packet(item, assessment_id="stage3-test")
    result = interpret(packet)
    self.assertEqual(result["provenance"]["source"], "deterministic_local")
    self.assertFalse(result["provenance"]["external_transmission"])
    self.assertTrue(any("probabilit" in value.lower() or "recovery" in value.lower()
                        for value in result["limitations"]))
    self.assertEqual(len(result["measurements"]), 4)

  def test_stage3_adapter_preserves_coordinates_and_keeps_actions_separate(self):
    with tempfile.TemporaryDirectory() as temp:
      path = _write_package(Path(temp))
      result = load_stage3_registry(path, expected_shape=(4, 5, 6),
                                    inline_ids=range(100, 104), crossline_ids=range(200, 205))
      self.assertEqual(result["schema"], "rtm-stage3-interpretation-registry-1")
      self.assertEqual(len(result["historical_findings"]), 10)
      self.assertEqual(len(result["qc_follow_up_actions"]), 2)
      self.assertEqual(result["recovery_opportunities"], [])
      self.assertEqual(result["historical_findings"][0]["inline_id"], 101)
      self.assertEqual(result["historical_findings"][0]["crossline_id"], 202)


  def test_stage3_adapter_rejects_coordinate_mismatch(self):
    with tempfile.TemporaryDirectory() as temp:
      path = _write_package(Path(temp))
      with self.assertRaisesRegex(ValueError, "SEG-Y axes"):
        load_stage3_registry(path, expected_shape=(4, 5, 6),
                             inline_ids=range(300, 304), crossline_ids=range(200, 205))


  def test_stage3_adapter_rejects_modified_package_artifact(self):
    with tempfile.TemporaryDirectory() as temp:
      root = Path(temp)
      path = _write_package(root)
      (root / "local_repeatability.csv").write_text("tampered")
      with self.assertRaisesRegex(ValueError, "checksum mismatch"):
        load_stage3_registry(path, expected_shape=(4, 5, 6),
                             inline_ids=range(100, 104), crossline_ids=range(200, 205))

  def test_recovery_intelligence_page_renders_saved_assessment_without_analysis(self):
    try:
      import streamlit as st
      from streamlit.testing.v1 import AppTest
    except ImportError:
      self.skipTest("Streamlit testing is unavailable")
    from unittest.mock import patch
    from reservoir_time_machine.models import SeismicVolume, SurveyMetadata
    with tempfile.TemporaryDirectory() as temp:
      root = Path(temp)
      stage3 = root / "stage3"
      stage3.mkdir()
      _write_package(stage3, shape=(16, 18, 20))
      rng = np.random.default_rng(27)
      baseline = rng.normal(size=(16, 18, 20)).astype(np.float32)
      monitor = baseline.copy()
      monitor[1:4, 1:5, 3:5] += 0.2
      missing = np.zeros((16, 18), dtype=bool)
      metadata = {"paired_grid_available": True, "trace_key_pairing_verified": True,
                  "physical_4d_registration_verified": False, "inline_ids": list(range(100, 116)),
                  "crossline_ids": list(range(200, 218)), "paired_trace_count": 288,
                  "baseline_trace_count": 288, "monitor_only_trace_count": 0}
      volumes = tuple(SeismicVolume(array, SurveyMetadata(name, name, array.shape,
                        sample_axis_domain="unknown", source_metadata=metadata), missing)
                      for name, array in (("baseline", baseline), ("monitor", monitor)))
      st.cache_resource.clear()
      st.cache_data.clear()
      with patch.dict("os.environ", {"RTM_CHALLENGE3_ROOT": str(root), "RTM_CACHE_DIR": str(root),
          "RTM_FINDINGS_REGISTRY": str(root / "missing-registry.json"),
          "RTM_STAGE3_EVIDENCE_DIR": str(stage3), "RTM_LLM_BASE_URL": ""}), \
          patch("reservoir_time_machine.data_ingest.manifest.challenge3_availability",
                return_value={"available": True, "well_inventory_available": False}), \
          patch("reservoir_time_machine.data_ingest.cache_identity.paired_cache_identity", return_value="fixture-v1"), \
          patch("reservoir_time_machine.data_ingest.segy_loader.load_challenge3_paired_volumes", return_value=volumes), \
          patch("reservoir_time_machine.analysis.preliminary.run_preliminary",
                side_effect=AssertionError("Streamlit navigation must not fit models")):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30).run()
        self.assertFalse(list(app.exception), str([e.message for e in app.exception]))
        self.assertEqual(app.sidebar.radio[0].value, "Reservoir Command Center")
        self.assertEqual(len(app.slider), 0, "Judge-facing startup must not expose manual coordinate sliders")
        self.assertIn("WHAT IS HAPPENING IN THE RESERVOIR?",
                      " ".join(element.value for element in app.markdown))
        self.assertTrue(any(
            element.label == "Selected finding · structured scientific evidence"
            for element in app.get("expander")
        ))
        self.assertEqual(app.session_state["selected_finding_id"],
                         app.session_state["stage3_finding_selection"])
        app.sidebar.radio[0].set_value("3D discovery map").run()
        self.assertFalse(list(app.exception), str([e.message for e in app.exception]))
        self.assertEqual(len(app.slider), 0)
        app.sidebar.radio[0].set_value("Recovery intelligence").run()
        self.assertFalse(list(app.exception), str([e.message for e in app.exception]))
        text = "\n".join(element.value for element in app.markdown)
        self.assertIn("Recovery intelligence", text)
        self.assertIn("Recovery Chance of Success: Not yet quantifiable", " ".join(x.value for x in app.info))
        self.assertTrue(any("SEG-Y IL 101" in value or "IL 101" in value for value in
                            [element.value for element in app.markdown]))
        self.assertEqual([app.session_state[k] for k in ("shared_inline", "shared_crossline", "shared_sample")], [1, 2, 3])
        next_id = "field-00000000000000000002"
        app.selectbox[0].set_value(next_id).run()
        self.assertFalse(list(app.exception), str([e.message for e in app.exception]))
        self.assertEqual(app.session_state["selected_finding_id"], next_id)
        self.assertEqual(app.session_state["stage3_finding_selection"], next_id)
        self.assertEqual([app.session_state[k] for k in ("shared_inline", "shared_crossline", "shared_sample")], [2, 3, 2])
        app.sidebar.radio[0].set_value("Well inventory").run()
        self.assertFalse(list(app.exception), str([e.message for e in app.exception]))
        self.assertTrue(any("Eight-well evidence matrix" in value for value in
                            [element.value for element in app.markdown]))
        self.assertEqual(len(app.slider), 0)
        app.sidebar.radio[0].set_value("4D comparison").run()
        self.assertFalse(list(app.exception), str([e.message for e in app.exception]))
        self.assertEqual([app.session_state[k] for k in ("shared_inline", "shared_crossline", "shared_sample")], [2, 3, 2])
        app.sidebar.radio[0].set_value("Evidence explanation").run()
        self.assertFalse(list(app.exception), str([e.message for e in app.exception]))
        self.assertTrue(any("No external request was made" in value for value in
                            [item.value for item in app.info]))
      st.cache_resource.clear()
      st.cache_data.clear()


if __name__ == "__main__":
    unittest.main()
