"""Zero-input, evidence-contract and provider tests; fixtures contain no reservoir data."""

import json
from pathlib import Path

import pytest

from reservoir_time_machine.ai.interfaces import EvidenceItem, EvidencePacket, EvidenceProvider, GemmaEvidenceProvider
from reservoir_time_machine.visualization.dashboard_state import empty_dataset_state


ROOT = Path(__file__).resolve().parents[1]


def test_contract_schemas_are_well_formed_json_and_require_unknown_safe_fields():
    for name in ("dataset-manifest", "analysis-run", "validation-report", "evidence-packet"):
        schema = json.loads((ROOT / "schemas" / f"{name}.schema.json").read_text(encoding="utf-8"))
        assert schema["$schema"].endswith("2020-12/schema")
        assert "required" in schema
    manifest = json.loads((ROOT / "schemas/dataset-manifest.schema.json").read_text(encoding="utf-8"))
    assert manifest["properties"]["synthetic"] == {"const": False}
    assert manifest["$defs"]["survey"]["properties"]["geometry_status"]["enum"]


def test_math_only_empty_dashboard_state_has_no_analysis_or_findings(tmp_path):
    state = empty_dataset_state(tmp_path, ["baseline.sgy", "monitor.sgy"])
    assert state["state"] == "no_input_data"
    assert state["analysis_enabled"] is False
    assert state["geological_findings"] == []
    assert state["missing_files"] == ["baseline.sgy", "monitor.sgy"]


def test_evidence_packet_separates_observation_detection_hypothesis_and_unknowns():
    packet = EvidencePacket(
        run_id="unit-test-run",
        data_classification="public",
        observations=(EvidenceItem("Known scalar fixture value", "test fixture", "measured"),),
        detections=(EvidenceItem("Expected threshold result", "unit test", "algorithmic_detection", 2.0, "uncalibrated_strength"),),
        alternatives=("No geological alternative is evaluated by this unit test.",),
        uncertainty=("No reservoir data are present in this test.",),
        missing_evidence=("Actual seismic input",),
    )
    payload = json.loads(packet.to_json())
    assert payload["observations"][0]["status"] == "measured"
    assert payload["detections"][0]["score_semantics"] == "uncalibrated_strength"
    assert payload["data_classification"] == "public"


def test_evidence_rejects_nonfinite_scores_and_payload_overflow():
    with pytest.raises(ValueError, match="finite"):
        EvidenceItem("bad", "test", "algorithmic_detection", float("nan"))
    packet = EvidencePacket("unit-test", data_classification="public", uncertainty=("x" * 200,))
    with pytest.raises(ValueError, match="exceeds"):
        packet.to_json(max_bytes=100)


def test_probability_label_requires_calibration_and_independent_validation_reference():
    with pytest.raises(ValueError, match="calibrated probability"):
        EvidenceItem("candidate", "detector", "algorithmic_detection", 0.8, "calibrated_probability")
    item = EvidenceItem(
        "validated estimate", "calibration report", "algorithmic_detection", 0.8,
        "calibrated_probability", calibration_method="isotonic", validation_reference="held-out labels",
    )
    assert item.to_dict()["score_semantics"] == "calibrated_probability"


def test_provider_protocol_supports_local_fake_without_network():
    class TestProvider:
        provider_name = "test-only"

        def explain(self, packet):
            return f"Explanation for {packet.run_id}"

    provider: EvidenceProvider = TestProvider()
    assert provider.explain(EvidencePacket("math-only")) == "Explanation for math-only"


def test_gemma_provider_refuses_unknown_or_restricted_data_without_explicit_opt_in():
    provider = GemmaEvidenceProvider()
    with pytest.raises(PermissionError, match="explicitly authorized"):
        provider.explain(EvidencePacket("unit-test", data_classification="unknown"))
