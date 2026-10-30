"""Small mathematical fixtures for chunking, recovery, and coordinate transforms."""

import json

import numpy as np
import pytest
from scipy.ndimage import label, shift

from reservoir_time_machine.detection.clustering import connected_zones, connected_zones_chunked
from reservoir_time_machine.detection.multivariate import local_robust_screen, local_robust_screen_chunked
from reservoir_time_machine.processing.resumable import run_resumable_chunks
from reservoir_time_machine.registration.temporal import estimate_fractional_temporal_registration
from reservoir_time_machine.registration.transform import AffineGridTransform, resample_volume_to_target_grid


def test_chunked_local_mad_matches_full_array_across_seams_and_volume_edges(tmp_path):
    rng = np.random.default_rng(18)
    values = rng.normal(size=(9, 8, 11)).astype(np.float32)
    values[2, 3, 4] += 8
    values[0, 0, 0] = np.nan
    expected = local_robust_screen(values, footprint=(3, 3, 5), threshold_sigma=2.5)
    actual = local_robust_screen_chunked(values, tmp_path / "local", chunk_shape=(4, 3, 5),
                                         footprint=(3, 3, 5), threshold_sigma=2.5)
    np.testing.assert_allclose(actual.score, expected.score, rtol=1e-6, atol=1e-6, equal_nan=True)
    np.testing.assert_array_equal(actual.mask, expected.mask)
    np.testing.assert_allclose(actual.diagnostics["uncertainty"], expected.diagnostics["uncertainty"], atol=1e-6, equal_nan=True)


@pytest.mark.parametrize("connectivity,offset", [(1, (1, 0, 0)), (2, (1, 1, 0)), (3, (1, 1, 1))])
def test_chunked_components_reconcile_cross_chunk_face_edge_and_corner(connectivity, offset, tmp_path):
    mask = np.zeros((6, 6, 6), dtype=bool)
    mask[1, 1, 1] = True
    mask[1+offset[0], 1+offset[1], 1+offset[2]] = True
    score = mask.astype(np.float32) * 4
    labels, zones = connected_zones_chunked(mask, tmp_path / f"cc-{connectivity}",
                                            chunk_shape=(2, 2, 2), score=score,
                                            min_size=2, connectivity=connectivity)
    assert len(zones) == 1
    assert zones[0].voxel_count == 2
    assert labels[1, 1, 1] == labels[1+offset[0], 1+offset[1], 1+offset[2]] > 0
    # Independent full-array reference uses the same structuring-element convention.
    reference, count = label(mask, structure=np.ones((3, 3, 3), dtype=bool) if connectivity == 3 else
                             (None if connectivity == 1 else np.array([[[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                                                                      [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                                                                      [[0, 1, 0], [1, 1, 1], [0, 1, 0]]], dtype=bool)))
    assert count == 1
    assert reference[1, 1, 1] == reference[1+offset[0], 1+offset[1], 1+offset[2]]


def test_chunked_components_match_full_labels_for_many_components_and_filtering(tmp_path):
    rng = np.random.default_rng(3)
    mask = rng.random((9, 8, 7)) > 0.88
    full_labels, full_zones = connected_zones(mask, min_size=2, connectivity=1)
    chunk_labels, chunk_zones = connected_zones_chunked(mask, tmp_path / "random-cc",
        chunk_shape=(3, 3, 2), min_size=2, connectivity=1)
    assert sorted(zone.voxel_count for zone in chunk_zones) == sorted(zone.voxel_count for zone in full_zones)
    assert int(np.count_nonzero(chunk_labels)) == int(np.count_nonzero(full_labels[full_labels > 0]))


def test_interrupted_chunk_run_resumes_verified_completed_chunks(tmp_path):
    source = np.arange(6 * 4 * 4, dtype=np.float32).reshape(6, 4, 4)
    output_path = tmp_path / "resumable.npy"
    calls = {"count": 0}

    def interrupted(block):
        calls["count"] += 1
        if calls["count"] == 3:
            raise RuntimeError("controlled mathematical interruption")
        return block * 2

    with pytest.raises(RuntimeError, match="controlled mathematical interruption"):
        run_resumable_chunks(source, interrupted, output_path, chunk_shape=(2, 2, 2),
                             processor_name="multiply-two", parameters={"factor": 2})
    journal_path = output_path.with_name(output_path.name + ".journal.jsonl")
    state = [json.loads(line) for line in journal_path.read_text().splitlines()]
    assert state[0]["event"] == "start"
    completed_before = sum(item["event"] == "chunk_complete" for item in state)
    with journal_path.open("ab") as stream:
        stream.write(b'{"event":"torn')  # controlled interrupted append, no terminator
    calls_before = calls["count"]
    result = run_resumable_chunks(source, lambda block: block * 2, output_path,
        chunk_shape=(2, 2, 2), processor_name="multiply-two", parameters={"factor": 2})
    np.testing.assert_array_equal(result, source * 2)
    assert sum(json.loads(line)["event"] == "chunk_complete" for line in journal_path.read_text().splitlines()) == 12
    assert calls_before == 3 and completed_before == 2
    assert output_path.with_name(output_path.name + ".complete").is_file()
    assert b"torn" not in journal_path.read_bytes()
    rerun = run_resumable_chunks(source, lambda block: pytest.fail("complete run should be reused"), output_path,
        chunk_shape=(2, 2, 2), processor_name="multiply-two", parameters={"factor": 2})
    np.testing.assert_array_equal(rerun, source * 2)
    changed_source = source.copy()
    changed_source[0, 0, 0] += 1
    with pytest.raises(ValueError, match="input checksum mismatch"):
        run_resumable_chunks(changed_source, lambda block: block * 2, output_path,
            chunk_shape=(2, 2, 2), processor_name="multiply-two", parameters={"factor": 2})


def test_affine_coordinate_transform_maps_known_grid_and_preserves_nan_support(tmp_path):
    source = np.zeros((4, 5, 2), dtype=np.float32)
    for i in range(4):
        for x in range(5):
            source[i, x] = i * 10 + x
    source[2, 2, 0] = np.nan
    x, y = np.meshgrid(np.arange(3), np.arange(3), indexing="ij")
    target_xy = np.stack((x, y), axis=-1).astype(float)
    transform = AffineGridTransform(((1, 0, 1), (0, 1, 1)), "LOCAL:SOURCE", "LOCAL:TARGET", "math-affine-map-v1")
    result, report = resample_volume_to_target_grid(source, target_xy, transform, tmp_path / "mapped.npy",
                                                     sample_chunk=1, interpolation_order=0)
    assert result.shape == (3, 3, 2)
    assert result[0, 0, 0] == source[1, 1, 0]
    assert np.isnan(result[1, 1, 0])
    assert report.status == "warning"  # spatial resampling is always explicitly QC-gated
    np.testing.assert_array_equal(source[1, 1], [11, 11])


def test_fractional_lag_estimation_recovers_shift_and_nan_pads_corrected_edges():
    n = 192
    t = np.arange(n, dtype=float)
    baseline_trace = np.exp(-((t-55)/9)**2) - 0.65*np.exp(-((t-117)/15)**2) + 0.2*np.sin(t*0.27)
    delayed_trace = shift(baseline_trace, 1.35, order=3, mode="constant", cval=0.0)
    baseline = baseline_trace[None, None, :]
    monitor = delayed_trace[None, None, :]
    result = estimate_fractional_temporal_registration(baseline, monitor, max_lag=4,
                                                       min_correlation=0.5, apply=True)
    assert result.accepted[0, 0]
    assert result.shifts_samples[0, 0] == pytest.approx(1.35, abs=0.25)
    assert np.isnan(result.registered_monitor[0, 0, -1])
    assert np.corrcoef(baseline_trace[5:-5], result.registered_monitor[0, 0, 5:-5])[0, 1] > 0.99
    np.testing.assert_array_equal(monitor[0, 0], delayed_trace)
