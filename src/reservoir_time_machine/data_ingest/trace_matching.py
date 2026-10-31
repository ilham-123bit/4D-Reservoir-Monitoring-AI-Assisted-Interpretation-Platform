"""Read-only key-based baseline/monitor trace coverage matching.

This module reads trace headers only. Matching inline/crossline keys establish an
index correspondence, not identical bin centres, physical registration, or waveform
comparability.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import csv
import importlib.metadata
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from ..data.validation import ValidationFinding, ValidationReport, inspect_segy_integrity
from ..processing.provenance import sha256_file


TraceKey = tuple[int, int]


def _readonly_int_array(values: Any, *, columns: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.int64)
    if columns is not None:
        array = array.reshape((-1, columns))
    else:
        array = array.reshape(-1)
    array.setflags(write=False)
    return array


def _occurrences(keys: Iterable[tuple[int, int]]) -> dict[TraceKey, int | list[int]]:
    positions: dict[TraceKey, int | list[int]] = {}
    for index, raw_key in enumerate(keys):
        try:
            if len(raw_key) != 2:
                raise ValueError
            inline, crossline = int(raw_key[0]), int(raw_key[1])
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError(f"Trace key at index {index} must contain integer inline and crossline IDs") from exc
        key = (inline, crossline)
        prior = positions.get(key)
        if prior is None:
            positions[key] = index
        elif isinstance(prior, int):
            positions[key] = [prior, index]
        else:
            prior.append(index)
    return positions


def _indices_for_keys(positions: dict[TraceKey, int | list[int]], keys: set[TraceKey]) -> tuple[int, ...]:
    indices: list[int] = []
    for key in keys:
        occurrence = positions[key]
        indices.extend([occurrence] if isinstance(occurrence, int) else occurrence)
    return tuple(sorted(indices))


def _trace_records_for_keys(positions: dict[TraceKey, int | list[int]],
                            keys: Iterable[TraceKey]) -> list[tuple[TraceKey, int]]:
    records: list[tuple[TraceKey, int]] = []
    for key in keys:
        occurrence = positions[key]
        records.extend((key, index) for index in ([occurrence] if isinstance(occurrence, int) else occurrence))
    return sorted(records, key=lambda item: item[1])


@dataclass(frozen=True)
class TraceCoverageMatch:
    """Index mapping and disjoint coverage details for two trace-key sequences.

    `monitor_index_for_baseline` is baseline-order aligned and uses -1 when a
    baseline trace cannot be uniquely paired. `common_*` arrays contain only unique
    one-to-one pairs, also in Baseline order. Unmatched and duplicate trace indices
    are retained in separate arrays; no input trace is deleted or rewritten.
    """

    baseline_trace_count: int
    monitor_trace_count: int
    monitor_index_for_baseline: np.ndarray
    common_keys: np.ndarray
    common_baseline_indices: np.ndarray
    common_monitor_indices: np.ndarray
    baseline_only_keys: np.ndarray
    baseline_only_trace_indices: np.ndarray
    monitor_only_keys: np.ndarray
    monitor_only_trace_indices: np.ndarray
    baseline_duplicate_keys: np.ndarray
    baseline_duplicate_trace_indices: tuple[tuple[int, ...], ...]
    monitor_duplicate_keys: np.ndarray
    monitor_duplicate_trace_indices: tuple[tuple[int, ...], ...]
    ambiguous_common_keys: np.ndarray
    ambiguous_common_trace_indices: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    ambiguous_baseline_trace_indices: np.ndarray
    ambiguous_monitor_trace_indices: np.ndarray
    report: ValidationReport
    source_file_identity: tuple[tuple[str, str, int, int], ...] = ()
    source_integrity_verified: bool = False

    @property
    def can_pair_common_coverage(self) -> bool:
        return bool(self.common_keys.shape[0]) and bool(
            self.report.metadata.get("pairable_common_coverage", False)
        )

    @property
    def baseline_only_count(self) -> int:
        return len({(int(row[0]), int(row[1])) for row in self.baseline_only_keys})

    @property
    def monitor_only_count(self) -> int:
        return len({(int(row[0]), int(row[1])) for row in self.monitor_only_keys})

    def monitor_only_by_inline(self) -> dict[int, tuple[int, ...]]:
        """Return monitor-only crossline identifiers grouped by inline."""
        grouped: dict[int, list[int]] = {}
        for inline, crossline in self.monitor_only_keys:
            grouped.setdefault(int(inline), []).append(int(crossline))
        return {inline: tuple(sorted(crosslines)) for inline, crosslines in sorted(grouped.items())}

    def duplicate_summary(self, side: str) -> list[dict[str, Any]]:
        if side == "baseline":
            keys, positions = self.baseline_duplicate_keys, self.baseline_duplicate_trace_indices
        elif side == "monitor":
            keys, positions = self.monitor_duplicate_keys, self.monitor_duplicate_trace_indices
        else:
            raise ValueError("side must be 'baseline' or 'monitor'")
        return [{"inline": int(key[0]), "crossline": int(key[1]), "trace_indices": list(indices)}
                for key, indices in zip(keys, positions)]


def match_trace_key_sequences(baseline_keys: Iterable[tuple[int, int]],
                              monitor_keys: Iterable[tuple[int, int]], *,
                              baseline_name: str = "baseline", monitor_name: str = "monitor") -> TraceCoverageMatch:
    """Match unique `(inline,crossline)` keys, retaining all unmatched indices.

    Common pairs are ordered by the Baseline sequence. Duplicate keys are never
    selected arbitrarily. Key-set asymmetry is reported as a warning when unique
    common coverage exists; physical geometry remains a separate validation step.
    """
    baseline_positions = _occurrences(baseline_keys)
    monitor_positions = _occurrences(monitor_keys)
    baseline_keys_set, monitor_keys_set = set(baseline_positions), set(monitor_positions)

    baseline_duplicates = {key: value for key, value in baseline_positions.items() if isinstance(value, list)}
    monitor_duplicates = {key: value for key, value in monitor_positions.items() if isinstance(value, list)}
    baseline_unique = {key: value for key, value in baseline_positions.items() if isinstance(value, int)}
    monitor_unique = {key: value for key, value in monitor_positions.items() if isinstance(value, int)}

    baseline_only = baseline_keys_set - monitor_keys_set
    monitor_only = monitor_keys_set - baseline_keys_set
    common_key_set = baseline_keys_set & monitor_keys_set
    invalid_common = {key for key in common_key_set if key[0] == 0 or key[1] == 0}
    ambiguous = common_key_set & (set(baseline_duplicates) | set(monitor_duplicates) | invalid_common)
    pairable = common_key_set - ambiguous
    ordered_common = sorted(pairable, key=baseline_unique.__getitem__)

    # The number of baseline traces includes repeated keys; derive it from maximum
    # observed index so duplicates do not shrink the baseline-order mapping.
    baseline_count = max((max(v) if isinstance(v, list) else v for v in baseline_positions.values()), default=-1) + 1
    monitor_count = max((max(v) if isinstance(v, list) else v for v in monitor_positions.values()), default=-1) + 1
    mapping = np.full(baseline_count, -1, dtype=np.int64)
    common_baseline_indices = [baseline_unique[key] for key in ordered_common]
    common_monitor_indices = [monitor_unique[key] for key in ordered_common]
    for baseline_index, monitor_index in zip(common_baseline_indices, common_monitor_indices):
        mapping[baseline_index] = monitor_index

    baseline_only_ordered = sorted(baseline_only, key=lambda key: min(
        baseline_positions[key] if isinstance(baseline_positions[key], list) else [baseline_positions[key]]))
    monitor_only_ordered = sorted(monitor_only, key=lambda key: min(
        monitor_positions[key] if isinstance(monitor_positions[key], list) else [monitor_positions[key]]))
    ambiguous_ordered = sorted(ambiguous, key=lambda key: min(
        baseline_positions[key] if isinstance(baseline_positions[key], list) else [baseline_positions[key]]))

    baseline_only_records = _trace_records_for_keys(baseline_positions, baseline_only_ordered)
    monitor_only_records = _trace_records_for_keys(monitor_positions, monitor_only_ordered)
    baseline_only_keys_by_trace = [record[0] for record in baseline_only_records]
    baseline_only_indices = [record[1] for record in baseline_only_records]
    monitor_only_keys_by_trace = [record[0] for record in monitor_only_records]
    monitor_only_indices = [record[1] for record in monitor_only_records]
    ambiguous_baseline_indices = _indices_for_keys(baseline_positions, ambiguous)
    ambiguous_monitor_indices = _indices_for_keys(monitor_positions, ambiguous)
    baseline_duplicate_ordered = sorted(baseline_duplicates, key=lambda key: baseline_duplicates[key][0])
    monitor_duplicate_ordered = sorted(monitor_duplicates, key=lambda key: monitor_duplicates[key][0])

    findings: list[ValidationFinding] = []
    if baseline_count == 0 or monitor_count == 0:
        findings.append(ValidationFinding("trace_key_presence", "fail",
            "One or both surveys contain no trace keys.", (baseline_name, monitor_name),
            {"baseline_trace_count": baseline_count, "monitor_trace_count": monitor_count},
            "Inspect trace headers; a common-coverage mapping cannot be formed from an empty survey.", True))
    zero_ids_a = sum(key[0] == 0 or key[1] == 0 for key in baseline_keys_set)
    zero_ids_b = sum(key[0] == 0 or key[1] == 0 for key in monitor_keys_set)
    if zero_ids_a or zero_ids_b:
        findings.append(ValidationFinding("trace_key_values", "fail",
            "Zero inline/crossline IDs are retained as unmatched metadata but are not accepted for pairing.",
            (baseline_name, monitor_name), {"baseline_invalid_key_count": zero_ids_a,
            "monitor_invalid_key_count": zero_ids_b},
            "Verify header fields and resolve invalid identifiers before using this mapping.", True))
        for key in ordered_common:
            if key[0] == 0 or key[1] == 0:
                mapping[baseline_unique[key]] = -1
        valid = [(key, bi, mi) for key, bi, mi in zip(ordered_common, common_baseline_indices, common_monitor_indices)
                 if key[0] != 0 and key[1] != 0]
        ordered_common = [item[0] for item in valid]
        common_baseline_indices = [item[1] for item in valid]
        common_monitor_indices = [item[2] for item in valid]
    duplicate_a = sum(len(indices) - 1 for indices in baseline_duplicates.values())
    duplicate_b = sum(len(indices) - 1 for indices in monitor_duplicates.values())
    if baseline_duplicates or monitor_duplicates:
        findings.append(ValidationFinding("trace_key_uniqueness", "fail",
            "Duplicate keys are recorded with every source index and excluded from one-to-one pairing.",
            (baseline_name, monitor_name), {"baseline_duplicate_key_count": len(baseline_duplicates),
            "baseline_duplicate_trace_count": duplicate_a, "monitor_duplicate_key_count": len(monitor_duplicates),
            "monitor_duplicate_trace_count": duplicate_b},
            "Resolve duplicates from source headers; no occurrence is selected arbitrarily.", True))
    if baseline_only or monitor_only:
        findings.append(ValidationFinding("trace_key_coverage", "warning",
            "Trace-key coverage differs; unique common keys are paired and unmatched traces remain separately indexed.",
            (baseline_name, monitor_name), {"baseline_only_key_count": len(baseline_only),
            "monitor_only_key_count": len(monitor_only), "ambiguous_common_key_count": len(ambiguous)},
            "Use only the returned unique common mapping for paired calculations; inspect unmatched coverage independently."))
    if not findings:
        findings.append(ValidationFinding("trace_key_coverage", "pass",
            "Trace-key sets are identical and every key has a unique one-to-one mapping.",
            (baseline_name, monitor_name), {"shared_key_count": len(ordered_common)}))
    if not ordered_common and baseline_count and monitor_count:
        findings.append(ValidationFinding("trace_key_common_coverage", "fail",
            "No uniquely matched non-zero trace keys are available.", (baseline_name, monitor_name),
            {"shared_unique_key_count": 0}, "Do not run paired calculations until correspondence is established.", True))

    report = ValidationReport(tuple(findings), {
        "baseline_trace_count": baseline_count, "monitor_trace_count": monitor_count,
        "shared_trace_count": len(ordered_common), "baseline_only_key_count": len(baseline_only),
        "monitor_only_key_count": len(monitor_only), "ambiguous_common_key_count": len(ambiguous),
        "baseline_duplicate_key_count": len(baseline_duplicates),
        "monitor_duplicate_key_count": len(monitor_duplicates),
        "pairable_common_coverage": bool(ordered_common) and not baseline_duplicates and not monitor_duplicates
                                    and not zero_ids_a and not zero_ids_b,
        "physical_registration_verified": False,
    })
    return TraceCoverageMatch(
        baseline_trace_count=baseline_count, monitor_trace_count=monitor_count,
        monitor_index_for_baseline=_readonly_int_array(mapping),
        common_keys=_readonly_int_array(ordered_common, columns=2),
        common_baseline_indices=_readonly_int_array(common_baseline_indices),
        common_monitor_indices=_readonly_int_array(common_monitor_indices),
        baseline_only_keys=_readonly_int_array(baseline_only_keys_by_trace, columns=2),
        baseline_only_trace_indices=_readonly_int_array(baseline_only_indices),
        monitor_only_keys=_readonly_int_array(monitor_only_keys_by_trace, columns=2),
        monitor_only_trace_indices=_readonly_int_array(monitor_only_indices),
        baseline_duplicate_keys=_readonly_int_array(baseline_duplicate_ordered, columns=2),
        baseline_duplicate_trace_indices=tuple(tuple(baseline_duplicates[key]) for key in baseline_duplicate_ordered),
        monitor_duplicate_keys=_readonly_int_array(monitor_duplicate_ordered, columns=2),
        monitor_duplicate_trace_indices=tuple(tuple(monitor_duplicates[key]) for key in monitor_duplicate_ordered),
        ambiguous_common_keys=_readonly_int_array(ambiguous_ordered, columns=2),
        ambiguous_common_trace_indices=tuple((
            tuple(baseline_positions[key] if isinstance(baseline_positions[key], list) else [baseline_positions[key]]),
            tuple(monitor_positions[key] if isinstance(monitor_positions[key], list) else [monitor_positions[key]]),
        ) for key in ambiguous_ordered),
        ambiguous_baseline_trace_indices=_readonly_int_array(ambiguous_baseline_indices),
        ambiguous_monitor_trace_indices=_readonly_int_array(ambiguous_monitor_indices),
        report=report,
    )


def iter_segy_trace_keys(path: str | Path, *, endian: str = "big") -> Iterable[TraceKey]:
    """Yield `(inline,crossline)` IDs using trace headers only; samples are not read."""
    try:
        import segyio
    except ImportError as exc:
        raise RuntimeError("SEG-Y header inspection requires the competition-data extra: segyio") from exc
    with segyio.open(str(path), "r", ignore_geometry=True, endian=endian) as segy:
        inline_field = segyio.TraceField.INLINE_3D
        crossline_field = segyio.TraceField.CROSSLINE_3D
        for index in range(int(segy.tracecount)):
            header = segy.header[index]
            yield int(header[inline_field]), int(header[crossline_field])


def iter_common_segy_trace_pairs(baseline_path: str | Path, monitor_path: str | Path,
                                 match: TraceCoverageMatch, *, endian: str = "big") -> Iterator[tuple[TraceKey, np.ndarray, np.ndarray]]:
    """Yield `(key, baseline_samples, monitor_samples)` for common unique keys only.

    The iterator preserves Baseline order and never yields Monitor-only or ambiguous
    traces. It reads one trace pair at a time; callers can perform bounded/chunked
    calculations without loading either full volume.
    """
    if not match.can_pair_common_coverage:
        raise ValueError("A unique common-coverage mapping is required before reading paired traces")
    _verify_bound_source(match, "baseline", baseline_path)
    _verify_bound_source(match, "monitor", monitor_path)
    try:
        import segyio
    except ImportError as exc:
        raise RuntimeError("SEG-Y reading requires the competition-data extra: segyio") from exc
    with segyio.open(str(baseline_path), "r", ignore_geometry=True, endian=endian) as baseline, \
            segyio.open(str(monitor_path), "r", ignore_geometry=True, endian=endian) as monitor:
        if int(baseline.tracecount) != match.baseline_trace_count or int(monitor.tracecount) != match.monitor_trace_count:
            raise ValueError("SEG-Y trace counts changed since the key mapping was built; rebuild the mapping")
        if len(baseline.samples) != len(monitor.samples):
            raise ValueError("Baseline and Monitor samples-per-trace differ; paired trace iteration is unsafe")
        for key, baseline_index, monitor_index in zip(match.common_keys, match.common_baseline_indices,
                                                       match.common_monitor_indices):
            baseline_trace = np.asarray(baseline.trace[int(baseline_index)])
            monitor_trace = np.asarray(monitor.trace[int(monitor_index)])
            if baseline_trace.shape != monitor_trace.shape:
                raise ValueError("A matched trace pair has incompatible sample shapes")
            yield (int(key[0]), int(key[1])), baseline_trace, monitor_trace


def iter_monitor_only_segy_traces(monitor_path: str | Path, match: TraceCoverageMatch, *,
                                  endian: str = "big") -> Iterator[tuple[TraceKey, int, np.ndarray]]:
    """Yield `(key, monitor_trace_index, samples)` for Monitor-only traces alone."""
    _verify_bound_source(match, "monitor", monitor_path)
    try:
        import segyio
    except ImportError as exc:
        raise RuntimeError("SEG-Y reading requires the competition-data extra: segyio") from exc
    with segyio.open(str(monitor_path), "r", ignore_geometry=True, endian=endian) as monitor:
        if int(monitor.tracecount) != match.monitor_trace_count:
            raise ValueError("Monitor trace count changed since the key mapping was built; rebuild the mapping")
        for key, index in zip(match.monitor_only_keys, match.monitor_only_trace_indices):
            yield (int(key[0]), int(key[1])), int(index), np.asarray(monitor.trace[int(index)])


def _verify_bound_source(match: TraceCoverageMatch, side: str, path: str | Path) -> None:
    if not match.source_integrity_verified:
        raise ValueError("Paired sample access requires successful checks against trusted source-integrity metadata")
    if not match.source_file_identity:
        raise ValueError("Sample iterators require a mapping bound to inspected source files")
    identities = {name: (source_path, size, modified_ns)
                  for name, source_path, size, modified_ns in match.source_file_identity}
    if side not in identities:
        raise ValueError(f"The mapping has no bound {side} source identity")
    source_path, expected_size, expected_modified_ns = identities[side]
    candidate = Path(path).expanduser().resolve()
    if str(candidate) != source_path:
        raise ValueError(f"{side} source path differs from the file used to build the trace mapping")
    stat = candidate.stat()
    if stat.st_size != expected_size or stat.st_mtime_ns != expected_modified_ns:
        raise ValueError(f"{side} source file changed since the trace mapping was built")


def inspect_trace_coverage_files(baseline_path: str | Path, monitor_path: str | Path, *,
                                 endian: str = "big", include_sha256: bool = True,
                                 expected_baseline_trace_count: int | None = None,
                                 expected_monitor_trace_count: int | None = None,
                                 expected_baseline_size_bytes: int | None = None,
                                 expected_monitor_size_bytes: int | None = None,
                                 expected_baseline_sha256: str | None = None,
                                 expected_monitor_sha256: str | None = None) -> tuple[TraceCoverageMatch, dict[str, Any]]:
    """Inspect file metadata and trace headers without loading seismic samples."""
    paths = {"baseline": Path(baseline_path).expanduser().resolve(),
             "monitor": Path(monitor_path).expanduser().resolve()}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"SEG-Y input does not exist: {path}")
    initial_stats = {label: (path.stat().st_size, path.stat().st_mtime_ns)
                     for label, path in paths.items()}
    integrity: dict[str, Any] = {}
    expected_counts = {"baseline": expected_baseline_trace_count,
                       "monitor": expected_monitor_trace_count}
    expected_sizes = {"baseline": expected_baseline_size_bytes,
                      "monitor": expected_monitor_size_bytes}
    expected_hashes = {"baseline": expected_baseline_sha256,
                       "monitor": expected_monitor_sha256}
    for label, path in paths.items():
        result = inspect_segy_integrity(path, endian=endian,
                                        expected_trace_count=expected_counts[label],
                                        expected_size_bytes=expected_sizes[label],
                                        expected_sha256=expected_hashes[label],
                                        require_sha256=True)
        integrity[label] = result.to_dict()
        failures = [item for item in result.findings if item.status == "fail"]
        if failures:
            summary = "; ".join(f"{item.check_id}: {item.explanation}" for item in failures)
            raise ValueError(f"{label} SEG-Y integrity check failed; refusing trace matching: {summary}")

    match = match_trace_key_sequences(iter_segy_trace_keys(paths["baseline"], endian=endian),
                                      iter_segy_trace_keys(paths["monitor"], endian=endian))
    input_metadata = {}
    for label, path in paths.items():
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != initial_stats[label]:
            raise ValueError(f"{label} source changed during coverage inspection; discard the mapping and retry")
        input_metadata[label] = {"path": str(path), "file_size_bytes": stat.st_size,
                                 "modified_time_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                                 "sha256": sha256_file(path) if include_sha256 else None,
                                 "integrity_report": integrity[label]}
    try:
        version = importlib.metadata.version("reservoir-time-machine")
    except importlib.metadata.PackageNotFoundError:
        version = "source-checkout"
    identities = tuple((label, str(paths[label]), initial_stats[label][0], initial_stats[label][1])
                       for label in ("baseline", "monitor"))
    integrity_verified = all(integrity[label]["can_proceed"] for label in ("baseline", "monitor"))
    match = replace(match, source_file_identity=identities,
                    source_integrity_verified=integrity_verified)
    return match, {"report_version": "1.0", "created_utc": datetime.now(timezone.utc).isoformat(),
                   "software_version": version, "input_files": input_metadata,
                   "source_integrity_verified": integrity_verified,
                   "matching": match.report.to_dict(),
                   "counts": {"baseline_trace_count": match.baseline_trace_count,
                              "monitor_trace_count": match.monitor_trace_count,
                              "shared_trace_count": int(match.common_keys.shape[0]),
                              "baseline_only_key_count": match.baseline_only_count,
                              "baseline_only_trace_count": int(match.baseline_only_trace_indices.size),
                              "monitor_only_key_count": match.monitor_only_count,
                              "monitor_only_trace_count": int(match.monitor_only_trace_indices.size),
                              "baseline_duplicate_key_count": int(match.baseline_duplicate_keys.shape[0]),
                              "baseline_duplicate_trace_count": sum(len(indices) - 1 for indices in match.baseline_duplicate_trace_indices),
                              "monitor_duplicate_key_count": int(match.monitor_duplicate_keys.shape[0]),
                              "monitor_duplicate_trace_count": sum(len(indices) - 1 for indices in match.monitor_duplicate_trace_indices),
                              "ambiguous_common_key_count": int(match.ambiguous_common_keys.shape[0])},
                   "monitor_only_by_inline": {str(il): list(xls) for il, xls in match.monitor_only_by_inline().items()},
                   "monitor_only_affected_inline_count": len(match.monitor_only_by_inline()),
                   "monitor_only_extra_traces_per_inline": {
                       str(il): len(xls) for il, xls in match.monitor_only_by_inline().items()},
                   "monitor_only_traces": [{"trace_index": int(index), "inline": int(key[0]), "crossline": int(key[1])}
                                           for key, index in zip(match.monitor_only_keys,
                                                                 match.monitor_only_trace_indices)],
                   "baseline_only_traces": [{"trace_index": int(index), "inline": int(key[0]), "crossline": int(key[1])}
                                            for key, index in zip(match.baseline_only_keys,
                                                                  match.baseline_only_trace_indices)],
                   "duplicate_keys": {"baseline": match.duplicate_summary("baseline"),
                                      "monitor": match.duplicate_summary("monitor")},
                   "ambiguous_common_keys": [{
                       "inline": int(key[0]), "crossline": int(key[1]),
                       "baseline_trace_indices": list(indices[0]), "monitor_trace_indices": list(indices[1]),
                   } for key, indices in zip(match.ambiguous_common_keys,
                                             match.ambiguous_common_trace_indices)],
                   "assumptions_and_unresolved": [
                       "Trace correspondence is based only on the supplied inline/crossline header IDs.",
                       "Matching IDs do not prove identical physical bin centres, CRS equivalence, or spatial registration.",
                       "Waveform phase, polarity, amplitude, and time alignment are not established by this report.",
                       "Input completeness requires an authoritative expected trace count or trusted manifest beyond record-boundary checks.",
                   ],
                   "common_mapping_csv": "trace_match_map.csv",
                   "monitor_only_csv": "monitor_only_traces.csv"}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Read-only trace-key coverage audit; reads headers, not seismic samples.")
    parser.add_argument("--baseline", required=True, help="Authorized Baseline SEG-Y path.")
    parser.add_argument("--monitor", required=True, help="Authorized Monitor SEG-Y path.")
    parser.add_argument("--output-dir", required=True,
                        help="Authorized output directory; existing report files are never overwritten.")
    parser.add_argument("--endian", choices=("big", "little"), default="big")
    parser.add_argument("--expected-baseline-traces", type=int,
                        help="Optional trusted-manifest Baseline trace count for completeness validation.")
    parser.add_argument("--expected-monitor-traces", type=int,
                        help="Optional trusted-manifest Monitor trace count for completeness validation.")
    parser.add_argument("--expected-baseline-size", type=int,
                        help="Optional trusted-manifest Baseline size in bytes.")
    parser.add_argument("--expected-monitor-size", type=int,
                        help="Optional trusted-manifest Monitor size in bytes.")
    parser.add_argument("--expected-baseline-sha256", help="Optional trusted Baseline SHA-256.")
    parser.add_argument("--expected-monitor-sha256", help="Optional trusted Monitor SHA-256.")
    parser.add_argument("--skip-sha256", action="store_true",
                        help="Skip recording calculated source hashes; sizes and modification times are still recorded.")
    args = parser.parse_args()
    match, report = inspect_trace_coverage_files(
        args.baseline, args.monitor, endian=args.endian, include_sha256=not args.skip_sha256,
        expected_baseline_trace_count=args.expected_baseline_traces,
        expected_monitor_trace_count=args.expected_monitor_traces,
        expected_baseline_size_bytes=args.expected_baseline_size,
        expected_monitor_size_bytes=args.expected_monitor_size,
        expected_baseline_sha256=args.expected_baseline_sha256,
        expected_monitor_sha256=args.expected_monitor_sha256)
    outputs = write_trace_coverage_outputs(match, report, args.output_dir)
    print(json.dumps({"can_pair_common_coverage": match.can_pair_common_coverage,
                      "source_integrity_verified": match.source_integrity_verified,
                      "counts": report["counts"], "outputs": outputs}, indent=2))


def _write_csv_exclusive(path: Path, header: tuple[str, ...], rows: Iterable[tuple[Any, ...]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def write_trace_coverage_outputs(match: TraceCoverageMatch, report: dict[str, Any],
                                 output_directory: str | Path) -> dict[str, str]:
    """Write JSON, Markdown, and CSV coverage products without overwriting files.

    Reports contain actual trace IDs and source paths when run on competition files;
    write them only to authorized storage. All target files use exclusive creation.
    """
    output = Path(output_directory).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    targets = {name: output / name for name in
               ("coverage_report.json", "coverage_report.md", "trace_match_map.csv",
                "monitor_only_traces.csv", "baseline_only_traces.csv")}
    existing = [str(path) for path in targets.values() if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite trace coverage outputs: " + ", ".join(existing))

    machine_report = dict(report)
    machine_report.update({
        "monitor_only_traces": [{"trace_index": int(index), "inline": int(key[0]), "crossline": int(key[1])}
                                for key, index in zip(match.monitor_only_keys, match.monitor_only_trace_indices)],
        "baseline_only_traces": [{"trace_index": int(index), "inline": int(key[0]), "crossline": int(key[1])}
                                 for key, index in zip(match.baseline_only_keys, match.baseline_only_trace_indices)],
        "duplicate_keys": {"baseline": match.duplicate_summary("baseline"),
                           "monitor": match.duplicate_summary("monitor")},
        "ambiguous_common_keys": [{
            "inline": int(key[0]), "crossline": int(key[1]),
            "baseline_trace_indices": list(indices[0]), "monitor_trace_indices": list(indices[1]),
        } for key, indices in zip(match.ambiguous_common_keys, match.ambiguous_common_trace_indices)],
    })

    mapping_rows = ((int(key[0]), int(key[1]), int(bi), int(mi))
                    for key, bi, mi in zip(match.common_keys, match.common_baseline_indices,
                                           match.common_monitor_indices))
    _write_csv_exclusive(targets["trace_match_map.csv"],
                         ("inline", "crossline", "baseline_trace_index", "monitor_trace_index"), mapping_rows)
    monitor_rows = ((int(key[0]), int(key[1]), int(index))
                    for key, index in zip(match.monitor_only_keys, match.monitor_only_trace_indices))
    _write_csv_exclusive(targets["monitor_only_traces.csv"],
                         ("inline", "crossline", "monitor_trace_index"), monitor_rows)
    baseline_rows = ((int(key[0]), int(key[1]), int(index))
                     for key, index in zip(match.baseline_only_keys, match.baseline_only_trace_indices))
    _write_csv_exclusive(targets["baseline_only_traces.csv"],
                         ("inline", "crossline", "baseline_trace_index"), baseline_rows)

    targets["coverage_report.json"].open("x", encoding="utf-8").write(
        json.dumps(machine_report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    counts = machine_report["counts"]
    lines = ["# Baseline–Monitor trace coverage audit", "",
             f"- Created (UTC): `{machine_report['created_utc']}`",
             f"- Software version: `{machine_report['software_version']}`",
             f"- Source integrity verified against trusted metadata: `{machine_report.get('source_integrity_verified', False)}`",
             f"- Baseline input: `{machine_report['input_files']['baseline']['path']}`",
             f"- Monitor input: `{machine_report['input_files']['monitor']['path']}`", "",
             "## Coverage counts", "",
             "| Measurement | Count |", "|---|---:|",
             f"| Baseline traces | {counts['baseline_trace_count']:,} |",
             f"| Monitor traces | {counts['monitor_trace_count']:,} |",
             f"| Unique common pairs | {counts['shared_trace_count']:,} |",
             f"| Baseline-only keys / traces | {counts['baseline_only_key_count']:,} / {counts['baseline_only_trace_count']:,} |",
             f"| Monitor-only keys / traces | {counts['monitor_only_key_count']:,} / {counts['monitor_only_trace_count']:,} |",
             f"| Baseline duplicate keys / repeated traces | {counts['baseline_duplicate_key_count']:,} / {counts['baseline_duplicate_trace_count']:,} |",
             f"| Monitor duplicate keys / repeated traces | {counts['monitor_duplicate_key_count']:,} / {counts['monitor_duplicate_trace_count']:,} |",
             f"| Ambiguous common keys | {counts['ambiguous_common_key_count']:,} |", "",
             "## Monitor-only coverage", "",
             f"Affected inlines: {len(match.monitor_only_by_inline()):,}.", "",
             "| Inline | Monitor-only crossline IDs |", "|---:|---|"]
    lines.extend(f"| {inline} | {', '.join(map(str, crosslines))} |"
                 for inline, crosslines in match.monitor_only_by_inline().items())
    lines += ["", "The complete per-trace indices are in `monitor_only_traces.csv`. The paired mapping in `trace_match_map.csv` is ordered by Baseline trace sequence.",
              "", "## Interpretation limits", "",
              "This report establishes trace-key correspondence only. It does not establish physical bin-centre equivalence, spatial registration, waveform phase/polarity/amplitude comparability, or time alignment.",
              "Monitor-only traces are retained separately for subsequent QC and are not part of any paired calculation.",
              "They are not automatically classified as edge effects.", "",
              "These reports contain source paths and actual trace coordinates when generated from competition files. Keep them in authorized storage; do not publish them.", ""]
    targets["coverage_report.md"].open("x", encoding="utf-8", newline="\n").write("\n".join(lines))
    return {name: str(path) for name, path in targets.items()}
