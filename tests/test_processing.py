"""Mathematical unit tests using small in-memory arrays only (not reservoir fixtures)."""

import numpy as np
import pytest

from reservoir_time_machine.attributes.difference import calculate_attributes, calculate_sample_attributes
from reservoir_time_machine.data.validation import validate_pair
from reservoir_time_machine.data_audit.audit import audit_challenge3
from reservoir_time_machine.data_ingest.manifest import challenge3_availability
from reservoir_time_machine.data_ingest.well_tables import read_well_table
from reservoir_time_machine.detection.clustering import connected_zones
from reservoir_time_machine.detection.statistical import statistical_screen
from reservoir_time_machine.evaluation.metrics import evaluate_detection
from reservoir_time_machine.models import SeismicVolume, SurveyMetadata
from reservoir_time_machine.preprocessing.normalization import normalize_pair
from reservoir_time_machine.preprocessing.qc import quality_report
from reservoir_time_machine.registration.temporal import estimate_temporal_registration


def test_difference_sign_and_nrms_formula_on_constant_arrays():
    baseline = np.ones((4, 3, 12))
    monitor = np.full_like(baseline, 2.0)
    attrs = calculate_attributes(baseline, monitor, window_samples=5)
    np.testing.assert_allclose(attrs.signed_difference, 1)
    np.testing.assert_allclose(attrs.absolute_difference, 1)
    np.testing.assert_allclose(attrs.nrms, 200/3, rtol=1e-6)


def test_sample_attribute_window_matches_full_volume_slice():
    rng = np.random.default_rng(4)
    baseline = rng.normal(size=(5, 4, 20))
    monitor = baseline.copy()
    monitor[:, :, 9:12] += 0.4
    full = calculate_attributes(baseline, monitor, window_samples=5)
    one = calculate_sample_attributes(baseline, monitor, 10, window_samples=5)
    np.testing.assert_allclose(one.nrms, full.nrms[:, :, 10], rtol=1e-6)
    np.testing.assert_allclose(one.signed_difference, full.signed_difference[:, :, 10])


def test_zero_energy_and_missing_values_are_handled():
    zeros = np.zeros((3, 3, 8))
    result = calculate_attributes(zeros, zeros, window_samples=3)
    assert np.all(result.nrms == 0)
    baseline = zeros.copy()
    baseline[1, 1] = np.nan
    result = calculate_sample_attributes(baseline, zeros, 4, window_samples=3)
    assert np.isnan(result.nrms[1, 1])


def test_qc_and_normalization_preserve_input_arrays():
    baseline = np.ones((3, 3, 8))
    monitor = np.full_like(baseline, 2.0)
    monitor[0, 0, 0] = np.nan
    report = quality_report(monitor)
    assert report["missing_trace_count"] == 0
    assert report["availability_fraction"] < 1
    b, m, diagnostics = normalize_pair(baseline, monitor, "global_rms")
    np.testing.assert_array_equal(baseline, 1)
    assert not np.shares_memory(m, monitor)
    assert "warning" in diagnostics


def test_bounded_registration_recovers_known_integer_lag():
    baseline = np.random.default_rng(9).normal(size=(4, 4, 64))
    monitor = np.roll(baseline, 2, axis=2)
    result = estimate_temporal_registration(baseline, monitor, max_lag=4)
    assert np.all(result.shifts == 2)
    assert np.nanmean(result.correlation) > 0.95


def test_statistical_screen_and_connected_components_on_2d_math_array():
    scores = np.zeros((20, 20))
    scores[4:7, 5:8] = 3
    result = statistical_screen(scores)
    labels, zones = connected_zones(result.mask, result.score, min_size=4)
    assert zones and zones[0].voxel_count >= 4
    assert labels.max() == 1


def test_evaluation_metrics_on_binary_vectors():
    truth = np.array([1, 1, 0, 0], dtype=bool)
    pred = np.array([1, 0, 1, 0], dtype=bool)
    metrics = evaluate_detection(pred, truth)
    assert (metrics.precision, metrics.recall, metrics.f1) == (0.5, 0.5, 0.5)
    assert metrics.false_positive_count == metrics.false_negative_count == 1


def test_pair_validation_rejects_sampling_mismatch():
    data = np.zeros((16, 16, 32))
    a = SeismicVolume(data, SurveyMetadata("A", "Baseline", data.shape))
    b = SeismicVolume(data.copy(), SurveyMetadata("B", "Monitor", data.shape, sample_interval=0.004))
    with pytest.raises(ValueError, match="sampling geometry"):
        validate_pair(a, b)


def test_missing_dataset_is_reported_without_substitute_data(tmp_path):
    status = challenge3_availability(tmp_path)
    assert not status["available"]
    assert len(status["missing"]) >= 2
    report = audit_challenge3(tmp_path)
    assert report["root_accessible"]
    assert report["manifest_contains_raw_samples_or_coordinates"] is False
    assert report["segy"] == {}


def test_generic_well_table_preserves_source_rows_without_unit_guessing(tmp_path):
    path = tmp_path / "in_memory_table.txt"
    path.write_text("# unit metadata must remain uninterpreted\nfield_a\tfield_b\nalpha\tbeta\n", encoding="utf-8")
    table = read_well_table(path)
    assert table.columns == ("field_a", "field_b")
    assert table.rows == (("alpha", "beta"),)
    assert "uninterpreted" in table.comment_lines[0]


def test_legacy_volume_math_stays_out_of_main_imports():
    import reservoir_time_machine as package
    assert not hasattr(package, "generate_scenario")
