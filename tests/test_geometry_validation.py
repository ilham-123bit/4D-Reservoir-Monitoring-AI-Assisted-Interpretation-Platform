"""Geometry/integrity/QC tests use only small mathematical fixtures and handcrafted headers."""

import hashlib
import json
import struct

import numpy as np
import pytest

from reservoir_time_machine.data.validation import (
    compare_coordinate_headers,
    compare_trace_keys,
    compare_volume_geometry,
    inspect_segy_integrity,
    validate_pair,
)
from reservoir_time_machine.models import SeismicVolume, SurveyMetadata
from reservoir_time_machine.preprocessing.qc import quality_report
from reservoir_time_machine.preprocessing.repeatability import repeatability_report
from reservoir_time_machine.registration.temporal import estimate_temporal_registration


def _write_math_segy(path, trace_count=2, *, extended_headers=0, partial_samples=0):
    """Write tiny structural SEG-Y bytes containing no geological or seismic fixture."""
    text = bytes(3200)
    binary = bytearray(400)
    struct.pack_into(">H", binary, 16, 2000)
    struct.pack_into(">H", binary, 20, 4)
    struct.pack_into(">H", binary, 24, 5)  # IEEE float32
    struct.pack_into(">H", binary, 300, 0x0100)
    struct.pack_into(">H", binary, 302, 1)
    struct.pack_into(">h", binary, 304, extended_headers)
    payload = bytearray(text + binary + bytes(3200 * extended_headers))
    record_length = 240 + 4 * 4
    for index in range(trace_count):
        header = bytearray(240)
        struct.pack_into(">H", header, 114, 4)
        struct.pack_into(">H", header, 116, 2000)
        payload.extend(header)
        payload.extend(struct.pack(">4f", 1.0, -1.0, 1.0, -1.0))
    if partial_samples:
        header = bytearray(240)
        struct.pack_into(">H", header, 114, 4)
        struct.pack_into(">H", header, 116, 2000)
        payload.extend(header)
        payload.extend(bytes(4 * partial_samples))
    path.write_bytes(payload)
    return len(payload), record_length


def _volume(data, *, inline=(100, 101), crossline=(200, 201), domain="two_way_time",
            interval=2.0, origin=0.0, origin_units="ms", coordinate_reference=None,
            geometry_status="verified"):
    metadata = SurveyMetadata(
        survey_id="math-baseline", label="mathematical fixture", dimensions=data.shape,
        sample_interval=interval, sample_interval_units="ms", sample_units="ms",
        sample_axis_domain=domain,
        source_metadata={
            "inline_ids_selected": inline,
            "crossline_ids_selected": crossline,
            "sample_origin": origin,
            "sample_origin_units": origin_units,
            "coordinate_reference": coordinate_reference,
            "geometry_status": geometry_status,
            "geometry_verification_reference": "math fixture geometry provenance" if geometry_status == "verified" else None,
        },
    )
    return SeismicVolume(data, metadata)


def test_segy_integrity_checks_fixed_record_boundaries_and_expected_file_hash(tmp_path):
    path = tmp_path / "complete_math_fixture.sgy"
    size, _ = _write_math_segy(path, trace_count=2)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    result = inspect_segy_integrity(path, expected_trace_count=2, expected_size_bytes=size, expected_sha256=digest)
    assert result.status == "pass"
    assert result.can_proceed
    assert result.metadata["complete_trace_records"] == 2
    without_trusted_hash = inspect_segy_integrity(path, expected_trace_count=2, require_sha256=True)
    assert not without_trusted_hash.can_proceed
    assert any(item.check_id == "manifest_sha256" and item.blocking for item in without_trusted_hash.findings)


def test_segy_integrity_reports_exact_partial_sample_range_and_expected_count(tmp_path):
    path = tmp_path / "partial_math_fixture.sgy"
    _, record_bytes = _write_math_segy(path, trace_count=2, partial_samples=2)
    result = inspect_segy_integrity(path, expected_trace_count=3)
    assert result.status == "fail"
    finding = next(item for item in result.findings if item.check_id == "trace_record_boundary")
    assert finding.metadata["partial_trace_sample_bytes"] == 8
    assert finding.metadata["bytes_to_finish_partial_trace"] == record_bytes - 240 - 8
    assert not result.can_proceed


def test_segy_integrity_respects_extended_text_header_offset_and_rejects_hash_mismatch(tmp_path):
    path = tmp_path / "extended_math_fixture.sgy"
    size, _ = _write_math_segy(path, trace_count=1, extended_headers=1)
    correct = hashlib.sha256(path.read_bytes()).hexdigest()
    result = inspect_segy_integrity(path, expected_trace_count=1, expected_sha256=correct)
    assert result.metadata["first_trace_offset_bytes"] == 6800
    assert result.can_proceed
    bad_hash = inspect_segy_integrity(path, expected_trace_count=1, expected_sha256="0" * 64)
    assert bad_hash.status == "fail"
    assert any(item.check_id == "manifest_sha256" for item in bad_hash.findings)


def test_trace_key_correspondence_reindexes_different_file_order():
    result = compare_trace_keys(((10, 20), (10, 21), (11, 20)), ((11, 20), (10, 20), (10, 21)))
    assert result.report.status == "warning"
    assert result.report.can_proceed
    assert result.monitor_index_for_baseline == (1, 2, 0)


def test_trace_key_correspondence_fails_on_duplicates_or_missing_bins():
    duplicate = compare_trace_keys(((10, 20), (10, 20)), ((10, 20),))
    assert not duplicate.report.can_proceed
    assert any(item.check_id == "trace_key_uniqueness" for item in duplicate.report.findings)
    missing = compare_trace_keys(((10, 20), (10, 21)), ((10, 20),))
    assert not missing.report.can_proceed
    assert any(item.check_id == "trace_key_coverage" and item.status == "fail" for item in missing.report.findings)


def test_coordinate_header_comparison_detects_coordinate_and_unit_mismatches():
    keys = ((10, 20), (10, 21))
    base = {(10, 20): (100.0, 200.0, 1), (10, 21): (110.0, 200.0, 1)}
    same = dict(base)
    assert compare_coordinate_headers(base, same, keys).status == "pass"
    shifted = {(10, 20): (100.0, 200.0, 1), (10, 21): (111.0, 200.0, 1)}
    assert compare_coordinate_headers(base, shifted, keys).status == "fail"
    wrong_units = {(10, 20): (100.0, 200.0, 2), (10, 21): (110.0, 200.0, 2)}
    assert compare_coordinate_headers(base, wrong_units, keys).status == "fail"


def test_volume_geometry_requires_known_domain_origin_and_verified_axis_ids():
    data = np.ones((2, 2, 8))
    a, b = _volume(data), _volume(data.copy())
    report = compare_volume_geometry(a, b)
    assert report.can_proceed
    assert report.status == "warning"  # missing CRS is non-blocking for verified bin-index work
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["findings"][0]["affected_input"] == ["math-baseline", "math-baseline"]
    assert all({"check_id", "status", "explanation", "affected_input", "metadata", "recommended_action", "blocking"}
               <= finding.keys() for finding in payload["findings"])
    validate_pair(a, b)
    unknown = _volume(data, domain="unknown", origin=None, origin_units=None, geometry_status="unverified")
    report = compare_volume_geometry(unknown, b)
    assert not report.can_proceed
    assert {item.check_id for item in report.findings if item.blocking} >= {"sample_domain", "sample_origin", "survey_geometry_status"}


def test_volume_geometry_rejects_sampling_dimensions_axis_and_crs_mismatches():
    data = np.ones((2, 2, 8))
    a = _volume(data)
    wrong_interval = _volume(data, interval=4.0)
    with pytest.raises(ValueError, match="sample_interval"):
        validate_pair(a, wrong_interval)
    wrong_inline = _volume(data, inline=(101, 102))
    assert not compare_volume_geometry(a, wrong_inline).can_proceed
    known_crs_a = _volume(data, coordinate_reference="EPSG:1000")
    known_crs_b = _volume(data, coordinate_reference="EPSG:1001")
    assert any(item.check_id == "coordinate_reference" and item.status == "fail"
               for item in compare_volume_geometry(known_crs_a, known_crs_b).findings)
    different_shape = _volume(np.ones((2, 3, 8)), crossline=(200, 201, 202))
    assert not compare_volume_geometry(a, different_shape).can_proceed


def test_quality_report_counts_nan_infinity_missing_and_zero_energy_traces():
    values = np.ones((2, 2, 4))
    values[0, 0, 0] = np.nan
    values[0, 1, 0] = np.inf
    values[1, 0] = 0
    report = quality_report(values)
    assert report["nan_sample_count"] == 1
    assert report["positive_infinity_sample_count"] == 1
    assert report["zero_energy_trace_count"] == 1
    assert report["validation"]["status"] == "warning"


def test_repeatability_known_correlation_nrms_and_constant_zero_energy():
    baseline = np.tile(np.array([-1.0, 1.0, -1.0, 1.0]), (2, 2, 2))
    identical = repeatability_report(baseline, baseline.copy())
    assert identical["metrics"]["global_waveform_correlation"] == pytest.approx(1.0)
    assert identical["metrics"]["global_nrms_percent"] == pytest.approx(0.0)
    inverse = repeatability_report(baseline, -baseline)
    assert inverse["metrics"]["global_waveform_correlation"] == pytest.approx(-1.0)
    assert inverse["metrics"]["global_nrms_percent"] == pytest.approx(200.0)
    zeros = repeatability_report(np.zeros((1, 1, 8)), np.zeros((1, 1, 8)))
    assert zeros["metrics"]["global_nrms_percent"] == 0.0
    assert zeros["metrics"]["global_waveform_correlation"] is None
    assert zeros["validation"]["status"] == "warning"
    high_offset = 1.0e8 + np.tile(np.array([-1.0, 1.0, -1.0, 1.0]), (1, 1, 2)).astype(np.float64)
    stable = repeatability_report(high_offset, high_offset.copy())
    assert stable["metrics"]["global_waveform_correlation"] == pytest.approx(1.0)


def test_repeatability_pairwise_finite_mask_and_no_valid_pair_failure():
    baseline = np.ones((1, 2, 8))
    monitor = np.ones_like(baseline)
    baseline[0, 0, 0] = np.nan
    monitor[0, 1, 1] = np.inf
    result = repeatability_report(baseline, monitor, inline_chunk=1, inline_tile=1, crossline_tile=1)
    assert result["metrics"]["paired_finite_samples"] == 14
    assert result["metrics"]["paired_finite_fraction"] == pytest.approx(14 / 16)
    assert result["validation"]["status"] == "warning"
    none = repeatability_report(np.full((1, 1, 8), np.nan), np.ones((1, 1, 8)))
    assert none["validation"]["status"] == "fail"
    assert none["metrics"] == {}


def test_registration_rejects_zero_energy_and_below_threshold_traces():
    zeros = np.zeros((2, 2, 32))
    result = estimate_temporal_registration(zeros, zeros, max_lag=4)
    assert not result.accepted.any()
    assert np.isnan(result.correlation).all()
    assert len(result.failure_reason) == 4
    noisy = np.random.default_rng(3).normal(size=(1, 1, 64))
    unrelated = np.random.default_rng(21).normal(size=(1, 1, 64))
    rejected = estimate_temporal_registration(noisy, unrelated, max_lag=3, min_correlation=0.99)
    assert not rejected.accepted[0, 0]
    assert "below minimum" in rejected.failure_reason["0,0"]


def test_registration_known_shift_uses_nan_edge_padding_without_wraparound():
    baseline = np.random.default_rng(12).normal(size=(1, 1, 64))
    monitor = np.concatenate((np.zeros((1, 1, 2)), baseline[:, :, :-2]), axis=2)
    result = estimate_temporal_registration(baseline, monitor, max_lag=4, apply=True)
    assert result.accepted[0, 0]
    assert result.shifts[0, 0] == 2
    assert np.isnan(result.registered_monitor[0, 0, -2:]).all()
    np.testing.assert_allclose(result.registered_monitor[0, 0, :-2], baseline[0, 0, :-2])
