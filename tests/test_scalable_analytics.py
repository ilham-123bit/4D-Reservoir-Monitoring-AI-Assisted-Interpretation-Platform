"""Small mathematical tests; these arrays are not seismic/reservoir fixtures."""

import json

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from reservoir_time_machine.attributes.chunked import calculate_attributes_chunked
from reservoir_time_machine.attributes.difference import calculate_attributes
from reservoir_time_machine.detection.clustering import connected_zones
from reservoir_time_machine.detection.multivariate import fuse_attributes, local_robust_screen
from reservoir_time_machine.preprocessing.normalization import normalize_pair_to_memmap
from reservoir_time_machine.processing.chunks import map_chunks
from reservoir_time_machine.processing.provenance import configuration_fingerprint, write_analysis_manifest
from reservoir_time_machine.registration.spatial import reindex_volume_to_axes
from reservoir_time_machine.evaluation.performance import benchmark_chunked_map
from reservoir_time_machine.models import SeismicVolume, SurveyMetadata


def _volume(data, inline, crossline, missing=None):
    metadata = SurveyMetadata(
        survey_id="mathematical-array", label="mathematical fixture", dimensions=data.shape,
        sample_interval=1.0, sample_interval_units="index", sample_units="index", sample_axis_domain="index",
        source_metadata={"inline_ids_selected": inline, "crossline_ids_selected": crossline},
    )
    return SeismicVolume(data, metadata, missing)


def test_halo_chunk_processing_matches_full_array_and_edges(tmp_path):
    values = np.arange(11 * 9 * 7, dtype=np.float32).reshape(11, 9, 7)
    expected = gaussian_filter(values, sigma=(1.1, 0.8, 0.7), mode="reflect")
    result = map_chunks(values, lambda block: gaussian_filter(block, sigma=(1.1, 0.8, 0.7), mode="reflect"),
                        tmp_path / "filtered.npy", chunk_shape=(4, 3, 4), halo=(5, 5, 5), output_dtype="float32")
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=2e-5)
    np.testing.assert_array_equal(values, np.arange(11 * 9 * 7, dtype=np.float32).reshape(11, 9, 7))


def test_chunk_processing_refuses_overwrite_and_bad_processor_shape(tmp_path):
    values = np.ones((3, 4, 5), dtype=np.float32)
    target = tmp_path / "out.npy"
    target.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        map_chunks(values, lambda block: block, target, chunk_shape=(2, 2, 2))
    with pytest.raises(ValueError, match="changed block shape"):
        map_chunks(values, lambda block: block[..., :-1], tmp_path / "bad.npy", chunk_shape=(2, 2, 2))


def test_chunked_attributes_match_full_array_with_nan_and_window_edges(tmp_path):
    baseline = np.arange(7 * 5 * 13, dtype=np.float32).reshape(7, 5, 13) / 19
    monitor = baseline * 1.15
    monitor[2, 1, 0] = np.nan
    expected = calculate_attributes(baseline, monitor, window_samples=5)
    actual = calculate_attributes_chunked(baseline, monitor, tmp_path / "attributes",
                                          inline_chunk=3, crossline_chunk=2, window_samples=5)
    for name in ("signed_difference", "absolute_difference", "normalized_difference", "rms_baseline",
                 "rms_monitor", "nrms"):
        np.testing.assert_allclose(getattr(actual, name), getattr(expected, name), rtol=2e-6, atol=1e-6,
                                   equal_nan=True)
    np.testing.assert_array_equal(baseline[2, 1, 0], np.float32((2 * 5 * 13 + 13) / 19))


def test_explicit_spatial_reindexing_reorders_axes_and_preserves_missing_traces():
    values = np.zeros((2, 2, 3), dtype=np.float32)
    values[0, 0] = 11
    values[0, 1] = 12
    values[1, 0] = 21
    values[1, 1] = 22
    missing = np.zeros((2, 2), dtype=bool)
    missing[1, 0] = True
    volume = _volume(values, (10, 20), (100, 200), missing)
    aligned, report = reindex_volume_to_axes(volume, (20, 10), (200, 100, 300))
    assert report.status == "warning" and not report.can_proceed
    np.testing.assert_array_equal(aligned.data[1, 1], values[0, 0])
    assert aligned.missing_trace_mask[0, 1]
    assert np.isnan(aligned.data[0, 1]).all()
    assert aligned.missing_trace_mask[1, 2]


def test_out_of_core_global_normalization_preserves_input_and_matches_rms(tmp_path):
    baseline = np.array([[[1.0, -1.0, 2.0], [0.0, 0.0, 0.0]], [[3.0, 1.0, np.nan], [2.0, -2.0, 1.0]]])
    monitor = baseline * 2
    original = monitor.copy()
    _, equalized, diagnostics = normalize_pair_to_memmap(baseline, monitor, tmp_path / "monitor.npy",
                                                         method="global_rms", inline_chunk=1)
    np.testing.assert_array_equal(monitor, original)
    valid = np.isfinite(equalized)
    assert np.sqrt(np.mean(equalized[valid] ** 2)) == pytest.approx(np.sqrt(np.mean(baseline[np.isfinite(baseline)] ** 2)))
    assert diagnostics["source_arrays_modified"] is False


def test_local_mad_and_multi_attribute_fusion_find_math_fixture_outlier():
    first = np.zeros((7, 7, 7), dtype=np.float32)
    second = np.zeros_like(first)
    first[3, 3, 3] = 10
    second[3, 3, 3] = 5
    local = local_robust_screen(first, footprint=(3, 3, 3), threshold_sigma=3)
    assert local.mask[3, 3, 3]
    fused = fuse_attributes({"absolute_difference": first, "nrms": second}, threshold_sigma=3)
    assert fused.mask[3, 3, 3]
    assert fused.diagnostics["score_is_probability"] is False
    assert np.isfinite(fused.diagnostics["uncertainty"][3, 3, 3])


def test_connected_components_and_run_manifest_are_deterministic(tmp_path):
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[0:2, 0:2, 0:2] = True
    mask[3, 3, 3] = True
    labels, zones = connected_zones(mask, min_size=2)
    assert len(zones) == 1
    assert zones[0].voxel_count == 8
    assert labels[0, 0, 0] == labels[1, 1, 1] > 0
    assert configuration_fingerprint({"z": 2, "a": [1, 3]}) == configuration_fingerprint({"a": [1, 3], "z": 2})
    manifest_path = tmp_path / "run.json"
    manifest = write_analysis_manifest(manifest_path, operation="math test", inputs=[{"id": "fixture"}],
        parameters={"window": 5}, outputs=[{"path": "output.npy", "kind": "attribute"}], created_utc="fixed-time")
    assert json.loads(manifest_path.read_text()) == manifest
    assert "not a geological conclusion" in manifest["interpretation_status"]
    with pytest.raises(FileExistsError):
        write_analysis_manifest(manifest_path, operation="duplicate", inputs=[], parameters={}, outputs=[])


def test_mathematical_performance_benchmark_reports_reference_and_memory_bounds():
    report = benchmark_chunked_map(shape=(8, 7, 6), chunk_shape=(3, 4, 2))
    assert report["output_matches_full_array_reference"]
    assert report["benchmark_kind"] == "mathematical_array_arithmetic_not_seismic"
    assert report["maximum_input_block_bytes"] < report["input_bytes"]
    assert report["elapsed_seconds"] >= 0
