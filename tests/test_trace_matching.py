"""Mathematical key fixtures; these are not seismic or survey data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest

from reservoir_time_machine.data.validation import ValidationReport
from reservoir_time_machine.data_ingest import trace_matching
from reservoir_time_machine.data_ingest.trace_matching import (
    inspect_trace_coverage_files,
    iter_common_segy_trace_pairs,
    iter_monitor_only_segy_traces,
    match_trace_key_sequences,
    write_trace_coverage_outputs,
)


def test_common_pair_mapping_uses_keys_and_preserves_baseline_order():
    baseline = [(10, 100), (10, 101), (11, 100), (11, 101)]
    monitor = [(11, 101), (10, 101), (12, 200), (10, 100), (11, 100)]

    match = match_trace_key_sequences(baseline, monitor)

    np.testing.assert_array_equal(match.common_keys, baseline)
    np.testing.assert_array_equal(match.common_baseline_indices, [0, 1, 2, 3])
    np.testing.assert_array_equal(match.common_monitor_indices, [3, 1, 4, 0])
    np.testing.assert_array_equal(match.monitor_index_for_baseline, [3, 1, 4, 0])
    assert match.baseline_only_count == 0
    assert match.baseline_duplicate_keys.size == match.monitor_duplicate_keys.size == 0
    assert match.can_pair_common_coverage
    assert match.report.metadata["physical_registration_verified"] is False


def test_monitor_only_keys_are_retained_and_grouped_by_inline():
    baseline = [(20, 1), (20, 2), (21, 1), (21, 2)]
    monitor = [(20, 4), (20, 1), (21, 6), (21, 1), (20, 5), (20, 2), (21, 7), (21, 2)]

    match = match_trace_key_sequences(baseline, monitor)

    assert match.baseline_trace_count == 4
    assert match.monitor_trace_count == 8
    assert match.common_keys.shape[0] == 4
    assert match.monitor_only_count == 4
    np.testing.assert_array_equal(match.monitor_only_keys, [[20, 4], [21, 6], [20, 5], [21, 7]])
    np.testing.assert_array_equal(match.monitor_only_trace_indices, [0, 2, 4, 6])
    assert match.monitor_only_by_inline() == {20: (4, 5), 21: (6, 7)}
    assert {20: 2, 21: 2} == {inline: len(crosslines) for inline, crosslines in match.monitor_only_by_inline().items()}
    all_monitor_indices = set(match.common_monitor_indices) | set(match.monitor_only_trace_indices)
    assert all_monitor_indices == set(range(len(monitor)))


def test_fixture_with_three_monitor_only_keys_per_inline():
    baseline = [(inline, crossline) for inline in (30, 31, 32)
                for crossline in (100, 101, 102, 103)]
    monitor = [(inline, crossline) for inline in (30, 31, 32)
               for crossline in (100, 101, 102, 103, 200 + inline, 300 + inline, 400 + inline)]

    match = match_trace_key_sequences(baseline, monitor)

    assert match.common_keys.shape[0] == 12
    assert match.monitor_only_count == 9
    assert len(match.monitor_only_by_inline()) == 3
    assert {inline: len(crosslines) for inline, crosslines in match.monitor_only_by_inline().items()} == {
        30: 3, 31: 3, 32: 3
    }


def test_baseline_only_keys_are_retained_and_unpaired_indices_use_minus_one():
    match = match_trace_key_sequences([(40, 1), (40, 2), (40, 3)], [(40, 1), (40, 3)])

    assert match.baseline_only_count == 1
    np.testing.assert_array_equal(match.baseline_only_keys, [[40, 2]])
    np.testing.assert_array_equal(match.baseline_only_trace_indices, [1])
    np.testing.assert_array_equal(match.monitor_index_for_baseline, [0, -1, 1])
    assert match.can_pair_common_coverage


def test_duplicate_keys_are_reported_and_never_selected_arbitrarily():
    match = match_trace_key_sequences([(50, 1), (50, 1), (50, 2)],
                                      [(50, 1), (50, 1), (50, 2), (50, 3)])

    assert match.baseline_duplicate_keys.tolist() == [[50, 1]]
    assert match.monitor_duplicate_keys.tolist() == [[50, 1]]
    assert match.baseline_duplicate_trace_indices == ((0, 1),)
    assert match.monitor_duplicate_trace_indices == ((0, 1),)
    assert match.ambiguous_common_keys.tolist() == [[50, 1]]
    assert match.monitor_only_count == 1
    assert not match.can_pair_common_coverage
    assert match.monitor_only_trace_indices.tolist() == [3]
    assert match.report.metadata["monitor_duplicate_key_count"] == 1


def test_input_index_mismatch_and_missing_key_are_never_silently_paired():
    match = match_trace_key_sequences([(60, 1), (60, 2)], [(60, 2), (61, 1)])

    assert match.monitor_index_for_baseline.tolist() == [-1, 0]
    assert match.baseline_only_keys.tolist() == [[60, 1]]
    assert match.monitor_only_keys.tolist() == [[61, 1]]
    assert match.monitor_only_trace_indices.tolist() == [1]
    assert match.common_keys.tolist() == [[60, 2]]


def test_zero_identifier_is_retained_but_blocks_pairing():
    match = match_trace_key_sequences([(0, 7), (61, 2)], [(0, 7), (61, 2)])

    assert match.common_keys.tolist() == [[61, 2]]
    assert match.monitor_index_for_baseline.tolist() == [-1, 1]
    assert not match.can_pair_common_coverage
    assert match.report.status == "fail"


def test_empty_key_sequences_fail_without_fabricating_pairs():
    match = match_trace_key_sequences([], [])

    assert match.baseline_trace_count == 0
    assert match.monitor_trace_count == 0
    assert match.common_keys.shape == (0, 2)
    assert not match.can_pair_common_coverage
    assert match.report.status == "fail"


def test_writer_emits_mapping_and_monitor_only_csv_without_overwriting(tmp_path):
    match = match_trace_key_sequences([(70, 1), (70, 2)], [(70, 2), (71, 4), (70, 1)])
    report = {"created_utc": "mathematical-fixture", "software_version": "test",
              "input_files": {"baseline": {"path": "fixture-baseline"},
                              "monitor": {"path": "fixture-monitor"}},
              "counts": {"baseline_trace_count": 2, "monitor_trace_count": 3,
                         "shared_trace_count": 2, "baseline_only_key_count": 0,
                         "baseline_only_trace_count": 0, "monitor_only_key_count": 1,
                         "monitor_only_trace_count": 1, "baseline_duplicate_key_count": 0,
                         "baseline_duplicate_trace_count": 0, "monitor_duplicate_key_count": 0,
                         "monitor_duplicate_trace_count": 0, "ambiguous_common_key_count": 0}}

    outputs = write_trace_coverage_outputs(match, report, tmp_path / "output")

    assert set(outputs) == {"coverage_report.json", "coverage_report.md", "trace_match_map.csv",
                            "monitor_only_traces.csv", "baseline_only_traces.csv"}
    report_json = json.loads(Path(outputs["coverage_report.json"]).read_text(encoding="utf-8"))
    assert report_json["monitor_only_traces"] == [{"trace_index": 1, "inline": 71, "crossline": 4}]
    mapping_rows = Path(outputs["trace_match_map.csv"]).read_text(encoding="utf-8").splitlines()
    assert mapping_rows[1:] == ["70,1,0,2", "70,2,1,0"]
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_trace_coverage_outputs(match, report, tmp_path / "output")


def test_header_file_inspection_is_read_only_and_returns_provenance(tmp_path, monkeypatch):
    baseline = tmp_path / "baseline.fixture.sgy"
    monitor = tmp_path / "monitor.fixture.sgy"
    baseline.write_bytes(b"baseline mathematical fixture")
    monitor.write_bytes(b"monitor mathematical fixture")
    original_hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in (baseline, monitor)}

    class FakeSegy:
        def __init__(self, keys):
            self.tracecount = len(keys)
            self.header = [{"inline": il, "crossline": xl} for il, xl in keys]
        def __enter__(self): return self
        def __exit__(self, *_): return False

    key_sets = {
        str(baseline): [(80, 1), (80, 2)],
        str(monitor): [(80, 2), (80, 3), (80, 1)],
    }
    fake = ModuleType("segyio")
    fake.TraceField = type("TraceField", (), {"INLINE_3D": "inline", "CROSSLINE_3D": "crossline"})
    fake.open = lambda path, *_args, **_kwargs: FakeSegy(key_sets[path])
    monkeypatch.setitem(sys.modules, "segyio", fake)
    monkeypatch.setattr(trace_matching, "inspect_segy_integrity", lambda *_args, **_kwargs: ValidationReport(()))

    match, report = inspect_trace_coverage_files(baseline, monitor, include_sha256=True)

    assert match.common_keys.shape[0] == 2
    assert match.monitor_only_keys.tolist() == [[80, 3]]
    assert report["input_files"]["baseline"]["file_size_bytes"] == baseline.stat().st_size
    assert report["input_files"]["baseline"]["sha256"] == original_hashes[baseline]
    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in (baseline, monitor)} == original_hashes


def test_streaming_iterators_keep_paired_and_monitor_only_traces_separate(monkeypatch, tmp_path):
    baseline_path = tmp_path / "baseline.sgy"
    monitor_path = tmp_path / "monitor.sgy"
    baseline_path.write_bytes(b"baseline fixture")
    monitor_path.write_bytes(b"monitor fixture")
    trace_sets = {
        str(baseline_path): [(90, 1), (90, 2)],
        str(monitor_path): [(90, 2), (90, 3), (90, 1)],
    }

    class FakeSegy:
        def __init__(self, path):
            self.keys = trace_sets[path]
            self.tracecount = len(self.keys)
            self.header = [{"inline": il, "crossline": xl} for il, xl in self.keys]
            self.samples = np.arange(3)
            self.trace = [np.array([index, index + 1, index + 2]) for index in range(self.tracecount)]
        def __enter__(self): return self
        def __exit__(self, *_): return False

    fake = ModuleType("segyio")
    fake.TraceField = type("TraceField", (), {"INLINE_3D": "inline", "CROSSLINE_3D": "crossline"})
    fake.open = lambda path, *_args, **_kwargs: FakeSegy(path)
    monkeypatch.setitem(sys.modules, "segyio", fake)
    monkeypatch.setattr(trace_matching, "inspect_segy_integrity", lambda *_args, **_kwargs: ValidationReport(()))
    match, _ = inspect_trace_coverage_files(baseline_path, monitor_path, include_sha256=False)

    pairs = list(iter_common_segy_trace_pairs(baseline_path, monitor_path, match))
    monitor_only = list(iter_monitor_only_segy_traces(monitor_path, match))

    assert [key for key, _, _ in pairs] == [(90, 1), (90, 2)]
    np.testing.assert_array_equal(pairs[0][1], [0, 1, 2])
    np.testing.assert_array_equal(pairs[0][2], [2, 3, 4])
    assert [item[:2] for item in monitor_only] == [((90, 3), 1)]
    np.testing.assert_array_equal(monitor_only[0][2], [1, 2, 3])
    with pytest.raises(ValueError, match="source path differs"):
        list(iter_monitor_only_segy_traces(tmp_path / "different.sgy", match))
