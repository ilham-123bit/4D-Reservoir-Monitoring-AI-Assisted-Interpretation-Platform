"""Adapter tests use mocked headers and temporary files, never competition inputs."""

import sys
from types import ModuleType
from types import SimpleNamespace

import numpy as np

from reservoir_time_machine.data_audit import audit as audit_module
from reservoir_time_machine.data_audit.audit import _inspect_segy, audit_challenge3
from reservoir_time_machine.data_ingest.segy_loader import load_segy_volume, validate_pair_files
from reservoir_time_machine.data_ingest import segy_loader as segy_loader_module
from reservoir_time_machine.data.validation import ValidationFinding, ValidationReport
from reservoir_time_machine.data_ingest.las_loader import inspect_las, load_las
from reservoir_time_machine.data_ingest.well_information import inspect_well_information, load_well_information
from reservoir_time_machine.settings import LLMSettings
from reservoir_time_machine.llm.gemma_client import explain_evidence
from reservoir_time_machine.data_ingest.well_models import well_record_from_source


class _FakeSegy:
    def __init__(self):
        class BinField:
            Format, Interval, Samples, SEGYRevision = "format", "interval", "samples", "revision"

        class TraceField:
            INLINE_3D, CROSSLINE_3D, TRACE_SEQUENCE_FILE = "il", "xl", "seq"
            SourceGroupScalar, CoordinateUnits = "scalar", "units"
            SourceX, SourceY, GroupX, GroupY = "sx", "sy", "gx", "gy"

        self.BinField, self.TraceField = BinField, TraceField
        self.bin = {"format": 5, "interval": 2000, "samples": 3, "revision": 1}
        self.samples = np.arange(3)
        self.headers = [
            {"il": il, "xl": xl, "seq": index+1, "scalar": 1, "units": 1,
             "sx": 100+index, "sy": 200+index, "gx": 110+index, "gy": 210+index}
            for index, (il, xl) in enumerate(((10, 20), (10, 21), (11, 20), (11, 21)))
        ]
        self.header = self.headers
        self.trace = [np.asarray([index, index+1, index+2], dtype=np.float32) for index in range(4)]
        self.tracecount = 4
        self.text = [["REVISION 1"]]

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def _install_fake_segyio(monkeypatch):
    module = ModuleType("segyio")
    fake = _FakeSegy()
    module.BinField, module.TraceField = fake.BinField, fake.TraceField
    module.open = lambda *args, **kwargs: _FakeSegy()
    monkeypatch.setitem(sys.modules, "segyio", module)
    monkeypatch.setattr(audit_module, "_optional_dependency", lambda _name: True)
    monkeypatch.setattr(segy_loader_module, "inspect_segy_integrity", lambda *_args, **_kwargs: ValidationReport((
        ValidationFinding("test_integrity", "pass", "Mathematical mock only.", ("fixture",)),
    )))


def test_segy_metadata_audit_uses_mock_headers_and_omits_coordinate_values(tmp_path, monkeypatch):
    _install_fake_segyio(monkeypatch)
    path = tmp_path / "mock.sgy"
    path.write_bytes(b"mock")
    report = _inspect_segy(path)
    assert report["status"] == "inspected"
    assert report["trace_count"] == 4
    assert report["samples_per_trace"] == 3
    assert report["sample_interval_microseconds"] == 2000
    assert report["inline_count"] == report["crossline_count"] == 2
    assert report["duplicate_inline_crossline_traces"] == 0
    assert report["missing_inline_crossline_positions_estimate"] == 0
    assert "_coordinate_sequence_fingerprint" in report  # internal pairing aid only
    assert "source_coordinate_values" not in report


def test_segy_loader_standardizes_mock_traces_to_inline_crossline_sample(tmp_path, monkeypatch):
    _install_fake_segyio(monkeypatch)
    path = tmp_path / "mock.sgy"
    path.write_bytes(b"mock")
    volume = load_segy_volume(path)
    assert volume.data.shape == (2, 2, 3)
    assert volume.metadata.axis_labels == ("inline", "crossline", "sample")
    assert volume.metadata.synthetic is False
    np.testing.assert_array_equal(volume.data[0, 0], [0, 1, 2])
    assert not volume.missing_trace_mask.any()


def test_segy_loader_reads_explicit_inline_crossline_and_sample_subset(tmp_path, monkeypatch):
    _install_fake_segyio(monkeypatch)
    path = tmp_path / "mock.sgy"
    path.write_bytes(b"mock")
    volume = load_segy_volume(path, inline_ids={11}, crossline_ids={21}, sample_range=(1, 3))
    assert volume.data.shape == (1, 1, 2)
    np.testing.assert_array_equal(volume.data[0, 0], [4, 5])
    assert volume.metadata.source_metadata["sample_range_selected"] == [1, 3]


def test_segy_pair_validation_compares_order_and_coordinates(tmp_path, monkeypatch):
    _install_fake_segyio(monkeypatch)
    a, b = tmp_path / "a.sgy", tmp_path / "b.sgy"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    result = validate_pair_files(
        a, b,
        sample_axis_domain="two_way_time",
        sample_origin=0.0,
        sample_origin_units="ms",
        geometry_verified=True,
        geometry_verification_reference="math fixture geometry evidence",
        coordinate_reference="LOCAL:MATHEMATICAL-FIXTURE",
    )
    assert result["trace_count_matches"]
    assert result["trace_order_matches"]
    assert result["coordinate_headers_present"]
    assert result["coordinate_headers_match"]
    assert result["spatial_geometry_compatible"]


def test_audit_on_empty_root_reports_missing_inputs_without_fabricated_findings(tmp_path):
    report = audit_challenge3(tmp_path)
    assert report["root_accessible"]
    assert report["segy"] == {}
    assert report["manifest_contains_raw_samples_or_coordinates"] is False
    assert report["well_file_availability"]["las"]["present"] == 0
    assert not report["unexpected_files"]


def test_public_manifest_omits_internal_coordinate_fingerprint(tmp_path, monkeypatch):
    _install_fake_segyio(monkeypatch)
    volume_dir = tmp_path / "volumes"
    volume_dir.mkdir()
    (volume_dir / "Challenge3_Baseline_FilteredScaled_FullStack.sgy").write_bytes(b"a")
    (volume_dir / "Challenge3_Monitor_FilteredScaled_FullStack.sgy").write_bytes(b"b")
    report = audit_challenge3(tmp_path)
    serialized = str(report)
    assert report["segy_pair_compatibility"]["trace_order_and_coordinate_headers_match"]
    assert "_coordinate_sequence_fingerprint" not in serialized
    assert "source_coordinate_values" not in serialized


def test_workbook_inspection_and_explicit_source_column_mapping(monkeypatch, tmp_path):
    rows = [("source_key", "source_coord"), ("row-a", 3.0), ("row-b", None)]

    class Sheet:
        title = "source"
        def iter_rows(self, values_only=True):
            return iter(rows)

    class Workbook:
        active = Sheet()
        sheetnames = ["source"]
        worksheets = [active]
        def __getitem__(self, name):
            assert name == "source"
            return self.active
        def close(self):
            pass

    module = ModuleType("openpyxl")
    module.load_workbook = lambda *args, **kwargs: Workbook()
    monkeypatch.setitem(sys.modules, "openpyxl", module)
    path = tmp_path / "mock.xlsx"
    path.write_bytes(b"not an actual workbook")
    report = inspect_well_information(path)
    assert report["sheets"][0]["columns"] == ["source_key", "source_coord"]
    loaded = load_well_information(path, sheet_name="source", column_mapping={"well_id": "source_key", "x": "source_coord"})
    assert loaded[0]["well_id"] == "row-a"
    assert loaded[0]["source_values"]["source_coord"] == 3.0


def test_las_reader_keeps_curve_provenance_and_nulls(monkeypatch, tmp_path):
    curve = SimpleNamespace(mnemonic="CURVE_A", unit="unit_x", descr="mock curve")
    class FakeLas:
        curves = [curve]
        index = np.array([1.0, 2.0])
        well = SimpleNamespace(WELL=SimpleNamespace(value="mock_source_identifier"),
                               NULL=SimpleNamespace(value=-999.25))
        version = SimpleNamespace(VERS=SimpleNamespace(value=2.0))
        def __getitem__(self, key):
            return np.array([1.0, -999.25])
    las = FakeLas()
    module = ModuleType("lasio")
    module.read = lambda *args, **kwargs: las
    monkeypatch.setitem(sys.modules, "lasio", module)
    report = inspect_las(tmp_path / "mock_Composite_Interp.las")
    assert report["provenance"] == "composite_interpreted"
    assert report["curves"][0]["mnemonic"] == "CURVE_A"
    assert report["curves"][0]["null_count"] == 1
    log = load_las(tmp_path / "mock_Composite_Interp.las")
    assert log.provenance == "composite_interpreted"
    assert log.curves[0].values.shape == (2,)


def test_well_record_preserves_identifier_and_marks_unknown_references():
    record = well_record_from_source({"well_id": "source_identifier", "source_values": {"raw_id": "source_identifier"}})
    assert record.original_identifier == "source_identifier"
    assert record.x is None and record.y is None
    assert "coordinate_reference" in record.missing_metadata
    assert record.source_fields["raw_id"] == "source_identifier"


def test_gemma_payload_rejects_full_arrays_before_any_network_request():
    try:
        explain_evidence({"cube": np.zeros((2, 2, 2))}, LLMSettings())
    except TypeError:
        pass
    else:
        raise AssertionError("NumPy seismic arrays must not serialize into LLM evidence payloads")


def test_gemma_missing_endpoint_is_reported_without_network_request(monkeypatch):
    from reservoir_time_machine.llm.gemma_client import test_connection

    monkeypatch.delenv("RTM_LLM_BASE_URL", raising=False)
    settings = LLMSettings.from_environment()
    assert settings.base_url is None
    result = test_connection(settings)
    assert result["connected"] is False
    assert "No LLM service endpoint is configured" in result["error"]


def test_gemma_explanation_requires_endpoint_after_validating_payload():
    try:
        explain_evidence({"summary": "mathematical test evidence"}, LLMSettings())
    except RuntimeError as exc:
        assert "Set RTM_LLM_BASE_URL" in str(exc)
    else:
        raise AssertionError("A missing LLM endpoint must fail clearly without a network request")
