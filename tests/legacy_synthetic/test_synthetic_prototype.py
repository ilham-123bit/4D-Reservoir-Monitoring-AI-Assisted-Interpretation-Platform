import inspect
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from reservoir_time_machine.attributes.difference import calculate_attributes
from reservoir_time_machine.config import SyntheticConfig
from reservoir_time_machine.data.validation import validate_pair
from reservoir_time_machine.detection.clustering import connected_zones
from reservoir_time_machine.detection.statistical import statistical_detector
from reservoir_time_machine.evaluation.metrics import evaluate_detection
from reservoir_time_machine.models import ARRAY_AXES, SeismicVolume, SurveyMetadata
from reservoir_time_machine.preprocessing.normalization import normalize_pair
from reservoir_time_machine.preprocessing.qc import quality_report
from reservoir_time_machine.registration.temporal import estimate_temporal_registration
from reservoir_time_machine.synthetic import generate_scenario
from reservoir_time_machine.visualization.slices import display_volume, slice_triplet


def small_config(**kwargs):
    return SyntheticConfig(dimensions=(24, 24, 48), **kwargs)


def test_default_generator_is_genuine_3d_with_documented_axis_order():
    scenario = generate_scenario()
    assert scenario.baseline.data.shape == (48, 48, 96)
    assert scenario.baseline.metadata.axis_labels == ARRAY_AXES == ("inline", "crossline", "sample")
    assert scenario.monitor.metadata.dimensions == scenario.monitor.data.shape
    assert scenario.baseline.metadata.sample_interval == 0.002
    assert scenario.baseline.metadata.synthetic


def test_generator_reproduces_data_and_separate_labels():
    first, again = generate_scenario(small_config(seed=41)), generate_scenario(small_config(seed=41))
    np.testing.assert_array_equal(first.baseline.data, again.baseline.data)
    np.testing.assert_array_equal(first.monitor.data, again.monitor.data)
    np.testing.assert_array_equal(first.ground_truth.amplitude_change, again.ground_truth.amplitude_change)
    assert not np.shares_memory(first.ground_truth.amplitude_change, first.monitor.data)


def test_reflectors_are_oscillatory_and_structural_feature_changes_horizons():
    scenario = generate_scenario(small_config())
    volume = scenario.baseline.data
    # A reflector trace has both polarities around a strong event.
    trace = volume[12, 12]
    assert np.any(trace > 0) and np.any(trace < 0)
    assert np.std(volume) > 0.05
    assert scenario.ground_truth.structural_fault.any()
    assert scenario.ground_truth.structural_fault.shape == volume.shape


@pytest.mark.parametrize("scenario_name", ["amplitude_increase", "amplitude_decrease", "weak_amplitude", "time_shift",
                                              "fluid_front", "unchanged", "acquisition_artifact", "noise_only"])
def test_required_scenarios_have_independent_truth_fields(scenario_name):
    scenario = generate_scenario(small_config(scenario=scenario_name))
    assert scenario.ground_truth.scenario_label == scenario_name
    assert scenario.baseline.data.shape == scenario.monitor.data.shape
    assert scenario.ground_truth.acquisition_artifact.shape == scenario.baseline.data.shape
    if scenario_name == "amplitude_increase":
        assert scenario.ground_truth.amplitude_change.any()
    if scenario_name == "time_shift":
        assert scenario.ground_truth.time_shift.any()
    if scenario_name == "fluid_front":
        assert scenario.ground_truth.fluid_front.any()
    if scenario_name == "unchanged":
        assert scenario.ground_truth.unchanged.all()


def test_detector_inputs_do_not_accept_or_read_ground_truth():
    assert "ground_truth" not in inspect.signature(statistical_detector).parameters
    scenario = generate_scenario(small_config())
    first = statistical_detector(scenario.baseline.data, scenario.monitor.data)
    changed_labels = np.zeros_like(scenario.ground_truth.amplitude_change)
    assert changed_labels.shape == first.mask.shape
    second = statistical_detector(scenario.baseline.data, scenario.monitor.data)
    np.testing.assert_array_equal(first.mask, second.mask)


def test_difference_and_nrms_match_conventional_formula():
    baseline = np.ones((4, 4, 12))
    monitor = np.full_like(baseline, 2.0)
    attrs = calculate_attributes(baseline, monitor, window_samples=5)
    np.testing.assert_allclose(attrs.signed_difference, 1.0)
    np.testing.assert_allclose(attrs.absolute_difference, 1.0)
    np.testing.assert_allclose(attrs.nrms, 200 / 3, rtol=1e-6)


def test_nrms_zero_energy_and_missing_window_behavior():
    zeros = np.zeros((2, 2, 8))
    attrs = calculate_attributes(zeros, zeros, window_samples=3)
    assert np.all(attrs.nrms == 0)
    one_missing = zeros.copy()
    one_missing[0, 0, :] = np.nan
    attrs = calculate_attributes(one_missing, zeros, window_samples=3)
    assert np.isnan(attrs.nrms[0, 0]).all()


def test_quality_control_reports_missing_traces_and_rms():
    volume = np.ones((3, 4, 8))
    volume[1, 2] = np.nan
    report = quality_report(volume)
    assert report["missing_trace_count"] == 1
    assert report["availability_fraction"] < 1
    assert report["amplitude_rms"] == 1


def test_normalization_is_copying_and_warns_about_amplitude_erasure():
    baseline = np.ones((3, 3, 8))
    monitor = np.full_like(baseline, 2.0)
    b, m, diagnostic = normalize_pair(baseline, monitor, "global_rms")
    np.testing.assert_array_equal(baseline, np.ones_like(baseline))
    assert not np.shares_memory(m, monitor)
    assert diagnostic["warning"]
    np.testing.assert_allclose(np.sqrt(np.mean(b**2)), np.sqrt(np.mean(m**2)))


def test_registration_recovers_known_trace_level_lag_without_wraparound():
    rng = np.random.default_rng(2)
    baseline = rng.normal(size=(6, 6, 64))
    monitor = np.roll(baseline, 2, axis=2)
    result = estimate_temporal_registration(baseline, monitor, max_lag=4, apply=True)
    assert np.all(result.shifts == 2)
    assert np.nanmean(result.correlation) > 0.95
    assert np.mean((result.registered_monitor[:, :, 4:-4] - baseline[:, :, 4:-4]) ** 2) < 0.1


def test_statistical_detection_and_spatial_components():
    baseline = np.zeros((12, 12, 16))
    monitor = baseline.copy()
    monitor[2:5, 2:5, 4:7] = 2.0
    result = statistical_detector(baseline, monitor)
    labels, zones = connected_zones(result.mask, result.score, min_size=4)
    assert result.mask.sum() > 0
    assert labels.max() >= 1
    assert zones[0].voxel_count >= 4


def test_metrics_compute_precision_recall_iou_and_error_counts():
    truth = np.array([1, 1, 0, 0], dtype=bool)
    pred = np.array([1, 0, 1, 0], dtype=bool)
    result = evaluate_detection(pred, truth, runtime_seconds=0.25)
    assert result.precision == 0.5
    assert result.recall == 0.5
    assert result.f1 == 0.5
    assert result.iou == 1/3
    assert result.false_positive_count == result.false_negative_count == 1
    assert result.runtime_seconds == 0.25


def test_display_downsampling_is_copy_and_slice_indices_follow_axis_order():
    volume = np.arange(24*32*40).reshape(24, 32, 40)
    shown = display_volume(volume, max_axis=12)
    assert max(shown.shape) <= 12
    assert not np.shares_memory(shown, volume)
    inline, crossline, sample = slice_triplet(volume, 2, 3, 4)
    assert inline.shape == (32, 40)
    assert crossline.shape == (24, 40)
    assert sample.shape == (24, 32)


def test_configuration_and_volume_geometry_are_validated():
    with pytest.raises(ValueError):
        SyntheticConfig(dimensions=(4, 24, 48))
    with pytest.raises(ValueError):
        SyntheticConfig(scenario="not-a-scenario")
    with pytest.raises(ValueError):
        SeismicVolume(np.zeros((2, 2)), SurveyMetadata("x", "x", (2, 2, 2)))


def test_pair_geometry_validation_rejects_interval_mismatch():
    a = SeismicVolume(np.zeros((16, 16, 32)), SurveyMetadata("a", "a", (16, 16, 32)))
    b = SeismicVolume(np.zeros((16, 16, 32)), SurveyMetadata("b", "b", (16, 16, 32), sample_interval=0.004))
    with pytest.raises(ValueError):
        validate_pair(a, b)


def test_cli_persists_volume_and_separate_ground_truth(tmp_path):
    out = tmp_path / "tiny.npz"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(os.path.abspath("src"))
    subprocess.run([sys.executable, "-m", "reservoir_time_machine.cli", "--dimensions", "16", "16", "32",
                    "--seed", "13", "--output", str(out)], check=True, env=env, capture_output=True, text=True)
    with np.load(out) as data:
        assert data["baseline"].shape == (16, 16, 32)
        metadata = json.loads(str(data["metadata"]))
        assert metadata["baseline"]["axis_labels"] == ["inline", "crossline", "sample"]
    with np.load(tmp_path / "tiny.ground_truth.npz") as truth:
        assert truth["amplitude_change"].shape == (16, 16, 32)
