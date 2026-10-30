"""Machine-readable SEG-Y integrity, trace correspondence, and pair geometry gates.

This module uses the standard library and small metadata structures; it never alters inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
import hashlib
import struct

import numpy as np

from ..models import ARRAY_AXES, SeismicVolume


@dataclass(frozen=True)
class ValidationFinding:
    check_id: str
    status: str
    explanation: str
    affected_input: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    recommended_action: str = ""
    blocking: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "status": self.status,
            "explanation": self.explanation,
            "affected_input": list(self.affected_input),
            "metadata": self.metadata,
            "recommended_action": self.recommended_action,
            "blocking": self.blocking,
        }


@dataclass(frozen=True)
class ValidationReport:
    findings: tuple[ValidationFinding, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        statuses = {item.status for item in self.findings}
        if "fail" in statuses:
            return "fail"
        if "warning" in statuses:
            return "warning"
        return "pass"

    @property
    def can_proceed(self) -> bool:
        return not any(item.status == "fail" or item.blocking for item in self.findings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "can_proceed": self.can_proceed,
            "metadata": self.metadata,
            "findings": [item.to_dict() for item in self.findings],
        }


def _finding(check_id: str, status: str, explanation: str, inputs: Iterable[str], *,
             metadata: dict[str, Any] | None = None, action: str = "", blocking: bool = False) -> ValidationFinding:
    return ValidationFinding(check_id, status, explanation, tuple(inputs), metadata or {}, action, blocking)


_FORMAT_BYTES = {1: 4, 2: 4, 3: 2, 4: 4, 5: 4, 6: 8, 7: 3, 8: 1, 9: 8, 10: 4, 11: 2, 12: 8, 15: 3, 16: 1}


def inspect_segy_integrity(path: str | Path, *, endian: str = "big", expected_size_bytes: int | None = None,
                           expected_sha256: str | None = None, expected_trace_count: int | None = None,
                           compute_sha256: bool = False, require_sha256: bool = False,
                           chunk_bytes: int = 8 * 1024 * 1024) -> ValidationReport:
    """Inspect SEG-Y fixed-record boundaries and optional manifest size/hash expectations.

    Without an externally supplied trace count this can prove record-boundary integrity, not that
    the file contains every intended trace. Unknown extended-header layouts and variable-length
    traces are reported as blocking rather than guessed.
    """
    source = Path(path)
    name = str(source)
    findings: list[ValidationFinding] = []
    if chunk_bytes < 1:
        raise ValueError("chunk_bytes must be positive")
    if expected_size_bytes is not None and expected_size_bytes < 0:
        raise ValueError("expected_size_bytes must be non-negative")
    if expected_trace_count is not None and expected_trace_count < 1:
        raise ValueError("expected_trace_count must be positive")
    if endian not in {"big", "little"}:
        findings.append(_finding("endianness", "fail", "Unsupported byte order selection.", [name],
                                 metadata={"endianness": endian}, action="Select big- or little-endian from source metadata.", blocking=True))
        return ValidationReport(tuple(findings), {"path": name})
    if not source.is_file():
        findings.append(_finding("file_presence", "fail", "SEG-Y input file does not exist.", [name],
                                 action="Provide the authorized source file and confirm its path.", blocking=True))
        return ValidationReport(tuple(findings), {"path": name})

    initial_stat = source.stat()
    size = initial_stat.st_size
    metadata: dict[str, Any] = {"path": name, "file_size_bytes": size,
                                "last_modified_ns": initial_stat.st_mtime_ns, "endianness": endian}
    prefix = ">" if endian == "big" else "<"
    if size < 3600:
        findings.append(_finding("minimum_header_size", "fail", "File is shorter than the 3200-byte textual plus 400-byte binary header.",
                                 [name], metadata={"file_size_bytes": size, "minimum_bytes": 3600},
                                 action="Replace or re-export the incomplete SEG-Y file.", blocking=True))
        return ValidationReport(tuple(findings), metadata)

    with source.open("rb") as stream:
        textual_header = stream.read(3200)
        binary = stream.read(400)
        interval_us = struct.unpack(prefix + "H", binary[16:18])[0]
        samples = struct.unpack(prefix + "H", binary[20:22])[0]
        sample_format = struct.unpack(prefix + "H", binary[24:26])[0]
        revision_word = struct.unpack(prefix + "H", binary[300:302])[0]
        fixed_length_flag = struct.unpack(prefix + "H", binary[302:304])[0]
        extended_count = struct.unpack(prefix + "h", binary[304:306])[0]
        metadata.update({
            "textual_header_bytes": 3200,
            "binary_header_bytes": 400,
            "binary_sample_interval_us": interval_us,
            "binary_samples_per_trace": samples,
            "sample_format_code": sample_format,
            "revision_word": revision_word,
            "revision": f"{revision_word >> 8}.{revision_word & 255}" if revision_word else "unspecified",
            "fixed_length_trace_flag": fixed_length_flag,
            "extended_text_header_count": extended_count,
            "textual_header_encoding": "not decoded by integrity scanner",
        })
        if extended_count < 0:
            findings.append(_finding("extended_text_headers", "warning", "Extended textual header count is variable/unknown; trace start cannot be proven by simple fixed offset.",
                                     [name], metadata={"extended_text_header_count": extended_count},
                                     action="Use a SEG-Y reader that locates the end-text marker, then record the resolved trace offset.", blocking=True))
            return ValidationReport(tuple(findings), metadata)
        first_trace_offset = 3600 + 3200 * extended_count
        metadata["first_trace_offset_bytes"] = first_trace_offset
        if first_trace_offset > size:
            findings.append(_finding("extended_text_headers", "fail", "Declared extended textual headers exceed the available file bytes.",
                                     [name], metadata={"first_trace_offset_bytes": first_trace_offset, "file_size_bytes": size},
                                     action="Obtain an intact SEG-Y export.", blocking=True))
            return ValidationReport(tuple(findings), metadata)
        if extended_count:
            stream.seek(3600)
            metadata["extended_text_headers_bytes_available"] = len(stream.read(3200 * extended_count))

    if not samples or not interval_us or sample_format not in _FORMAT_BYTES:
        findings.append(_finding("binary_header", "fail", "Binary header has zero/unsupported sample geometry or an unsupported sample format.",
                                 [name], metadata={"samples": samples, "sample_interval_us": interval_us, "sample_format_code": sample_format},
                                 action="Confirm byte order and obtain a supported, valid SEG-Y export.", blocking=True))
        return ValidationReport(tuple(findings), metadata)
    if fixed_length_flag != 1:
        findings.append(_finding("trace_length_mode", "warning", "Variable-length trace records are not validated by the fixed-record scanner.",
                                 [name], metadata={"fixed_length_trace_flag": fixed_length_flag},
                                 action="Validate every trace header and sample payload with a variable-length-aware SEG-Y reader.", blocking=True))
        return ValidationReport(tuple(findings), metadata)

    sample_bytes = samples * _FORMAT_BYTES[sample_format]
    record_bytes = 240 + sample_bytes
    available_trace_bytes = size - first_trace_offset
    complete_count, remainder = divmod(available_trace_bytes, record_bytes)
    partial_header_bytes = min(remainder, 240)
    partial_sample_bytes = max(0, remainder - 240)
    metadata.update({
        "bytes_per_sample": _FORMAT_BYTES[sample_format],
        "trace_header_bytes": 240,
        "trace_sample_bytes": sample_bytes,
        "fixed_trace_record_bytes": record_bytes,
        "complete_trace_records": complete_count,
        "trailing_record_bytes": remainder,
        "partial_trace_header_bytes": partial_header_bytes,
        "partial_trace_sample_bytes": partial_sample_bytes,
        "first_incomplete_offset_bytes": first_trace_offset + complete_count * record_bytes,
        "bytes_to_finish_partial_trace": record_bytes - remainder if remainder else 0,
    })
    if remainder:
        findings.append(_finding("trace_record_boundary", "fail", "File ends within a fixed-length trace record.", [name],
                                 metadata={"complete_trace_records": complete_count, "trailing_record_bytes": remainder,
                                           "partial_trace_header_bytes": partial_header_bytes,
                                           "partial_trace_sample_bytes": partial_sample_bytes,
                                           "first_incomplete_offset_bytes": metadata["first_incomplete_offset_bytes"],
                                           "bytes_to_finish_partial_trace": record_bytes-remainder},
                                 action="Replace/re-export from an intact source; do not pad or process the partial trace.", blocking=True))
    else:
        findings.append(_finding("trace_record_boundary", "pass", "All observed fixed-length trace records end on complete record boundaries.",
                                 [name], metadata={"complete_trace_records": complete_count, "record_bytes": record_bytes}))
    if expected_trace_count is None:
        completeness_has_trusted_reference = expected_sha256 is not None or expected_size_bytes is not None
        findings.append(_finding("total_trace_count", "warning", "SEG-Y metadata supplied no authoritative total trace count; complete records do not prove dataset completeness.",
                                 [name], metadata={"observed_complete_trace_records": complete_count},
                                 action="Compare against a trusted source manifest/export trace count, expected size, and checksum.",
                                 blocking=not completeness_has_trusted_reference))
    else:
        expected_length = first_trace_offset + expected_trace_count * record_bytes
        metadata["expected_trace_count"] = expected_trace_count
        metadata["expected_file_size_bytes_from_trace_count"] = expected_length
        if size != expected_length:
            delta = expected_length - size
            findings.append(_finding("declared_trace_count", "fail", "File byte length disagrees with the externally supplied expected trace count.",
                                     [name], metadata={"expected_size_bytes": expected_length, "actual_size_bytes": size,
                                                       "missing_bytes" if delta > 0 else "extra_bytes": abs(delta)},
                                     action="Obtain the correct complete file and verify the manifest count/hash.", blocking=True))
        else:
            findings.append(_finding("declared_trace_count", "pass", "File length matches the externally supplied fixed-record trace count.",
                                     [name], metadata={"expected_trace_count": expected_trace_count, "expected_size_bytes": expected_length}))
    if expected_size_bytes is not None and size != expected_size_bytes:
        findings.append(_finding("manifest_size", "fail", "Actual file size differs from the manifest size.", [name],
                                 metadata={"expected_size_bytes": expected_size_bytes, "actual_size_bytes": size},
                                 action="Verify source provenance and reacquire if the source size is not expected.", blocking=True))
    elif expected_size_bytes is not None:
        findings.append(_finding("manifest_size", "pass", "Actual file size matches the manifest.", [name],
                                 metadata={"size_bytes": size}))
    actual_hash = None
    if compute_sha256 or expected_sha256:
        hasher = hashlib.sha256()
        with source.open("rb") as stream:
            while block := stream.read(chunk_bytes):
                hasher.update(block)
        actual_hash = hasher.hexdigest()
        metadata["sha256"] = actual_hash
    if expected_sha256:
        if actual_hash is None:
            findings.append(_finding("manifest_sha256", "fail", "Expected SHA-256 could not be calculated.", [name],
                                     action="Retry the read-only hash scan.", blocking=True))
        elif actual_hash.lower() != expected_sha256.lower():
            findings.append(_finding("manifest_sha256", "fail", "Actual SHA-256 differs from the manifest.", [name],
                                     metadata={"expected_sha256": expected_sha256.lower(), "actual_sha256": actual_hash},
                                     action="Reject this input and verify the source copy.", blocking=True))
        else:
            findings.append(_finding("manifest_sha256", "pass", "Actual SHA-256 matches the manifest.", [name],
                                     metadata={"sha256": actual_hash}))
    elif compute_sha256:
        findings.append(_finding("source_sha256", "warning", "SHA-256 was recorded but not checked against a trusted expected digest.", [name],
                                 metadata={"sha256": actual_hash}, action="Compare with a trusted provider checksum before asserting provenance."))
    elif require_sha256:
        findings.append(_finding("manifest_sha256", "warning", "No trusted expected SHA-256 was supplied for verification.", [name],
                                 action="Obtain an authoritative digest and verify the source file before analysis.", blocking=True))
    final_stat = source.stat()
    if final_stat.st_size != size or final_stat.st_mtime_ns != initial_stat.st_mtime_ns:
        findings.append(_finding("file_stability", "fail", "File size or modification time changed during inspection.", [name],
                                 metadata={"initial_size_bytes": size, "final_size_bytes": final_stat.st_size,
                                           "initial_mtime_ns": initial_stat.st_mtime_ns,
                                           "final_mtime_ns": final_stat.st_mtime_ns},
                                 action="Ensure the source is closed to writes and repeat read-only validation.", blocking=True))
    else:
        findings.append(_finding("file_stability", "pass", "File size and modification time remained stable during inspection.", [name]))
    return ValidationReport(tuple(findings), metadata)


@dataclass(frozen=True)
class TraceKeyComparison:
    report: ValidationReport
    monitor_index_for_baseline: tuple[int | None, ...]

    def to_dict(self) -> dict[str, Any]:
        return {**self.report.to_dict(), "monitor_index_for_baseline": list(self.monitor_index_for_baseline)}


def compare_trace_keys(baseline_keys: Iterable[tuple[int, int]], monitor_keys: Iterable[tuple[int, int]], *,
                       baseline_name: str = "baseline", monitor_name: str = "monitor") -> TraceKeyComparison:
    """Compare explicit (inline,crossline) keys and return a reorder map; duplicates/missing keys fail."""
    baseline = tuple((int(il), int(xl)) for il, xl in baseline_keys)
    monitor = tuple((int(il), int(xl)) for il, xl in monitor_keys)
    findings: list[ValidationFinding] = []
    if not baseline or not monitor:
        findings.append(_finding("trace_key_presence", "fail", "One or both surveys contain no usable inline/crossline keys.",
                                 [baseline_name, monitor_name], metadata={"baseline_count": len(baseline), "monitor_count": len(monitor)},
                                 action="Inspect trace headers and axis mapping; no pairing is permitted without non-empty trace keys.", blocking=True))
    zero_id_count_a = sum(il == 0 or xl == 0 for il, xl in baseline)
    zero_id_count_b = sum(il == 0 or xl == 0 for il, xl in monitor)
    if zero_id_count_a or zero_id_count_b:
        findings.append(_finding("trace_key_values", "fail", "Zero inline or crossline identifiers are not accepted as valid survey keys.",
                                 [baseline_name, monitor_name], metadata={"baseline_zero_id_count": zero_id_count_a,
                                                                          "monitor_zero_id_count": zero_id_count_b},
                                 action="Verify the trace-header mapping and obtain authoritative bin identifiers.", blocking=True))
    duplicate_a = len(baseline) - len(set(baseline))
    duplicate_b = len(monitor) - len(set(monitor))
    if duplicate_a or duplicate_b:
        findings.append(_finding("trace_key_uniqueness", "fail", "Duplicate inline/crossline keys prevent unambiguous trace correspondence.",
                                 [baseline_name, monitor_name], metadata={"baseline_duplicate_count": duplicate_a,
                                                                          "monitor_duplicate_count": duplicate_b},
                                 action="Resolve duplicate trace keys from source headers; do not pair by file order.", blocking=True))
    map_b = {key: index for index, key in enumerate(monitor)}
    missing_monitor = sorted(set(baseline) - set(monitor))
    missing_baseline = sorted(set(monitor) - set(baseline))
    if missing_monitor or missing_baseline:
        findings.append(_finding("trace_key_coverage", "fail", "Baseline and monitor do not contain identical trace-key sets.",
                                 [baseline_name, monitor_name], metadata={"baseline_only_count": len(missing_monitor),
                                                                          "monitor_only_count": len(missing_baseline),
                                                                          "baseline_only_examples": missing_monitor[:10],
                                                                          "monitor_only_examples": missing_baseline[:10]},
                                 action="Resolve coverage/grid differences explicitly and construct a documented common-grid mask.", blocking=True))
    elif not duplicate_a and not duplicate_b and baseline == monitor:
        findings.append(_finding("trace_key_order", "pass", "Trace-key sets and file ordering match exactly.",
                                 [baseline_name, monitor_name], metadata={"trace_count": len(baseline)}))
    elif not duplicate_a and not duplicate_b:
        findings.append(_finding("trace_key_order", "warning", "Trace-key sets match but file order differs; reindex by inline/crossline keys before comparing samples.",
                                 [baseline_name, monitor_name], metadata={"trace_count": len(baseline)},
                                 action="Apply the returned monitor index map, then verify axis and coordinate geometry."))
    mapping = tuple(map_b.get(key) for key in baseline)
    if not findings:
        findings.append(_finding("trace_key_coverage", "pass", "Trace-key correspondence is complete and unique.", [baseline_name, monitor_name]))
    return TraceKeyComparison(ValidationReport(tuple(findings), {
        "baseline_trace_count": len(baseline), "monitor_trace_count": len(monitor),
        "matched_trace_count": sum(value is not None for value in mapping),
    }), mapping)


def compare_coordinate_headers(baseline: dict[tuple[int, int], tuple[float, float, int | None]],
                              monitor: dict[tuple[int, int], tuple[float, float, int | None]],
                              matched_keys: Iterable[tuple[int, int]], *,
                              baseline_name: str = "baseline", monitor_name: str = "monitor") -> ValidationFinding:
    """Compare scalar-normalized X/Y and coordinate-unit codes at explicit common bin keys.

    CRS identity is not inferred from SEG-Y coordinate-unit codes; this check only detects
    internal/header coordinate disagreement. A separate authoritative CRS remains necessary for
    map-space/well joins.
    """
    keys = tuple(matched_keys)
    inputs = (baseline_name, monitor_name)
    if not baseline and not monitor:
        return _finding("coordinate_headers", "warning", "Neither file has usable coordinate headers.", inputs,
                        action="Use only externally verified bin geometry; map-space joins remain disabled.")
    if not baseline or not monitor:
        return _finding("coordinate_headers", "fail", "Coordinate headers are present for only one survey.", inputs,
                        metadata={"baseline_coordinate_count": len(baseline), "monitor_coordinate_count": len(monitor)},
                        action="Resolve the missing survey navigation or provide a verified transform.", blocking=True)
    missing_a = [key for key in keys if key not in baseline]
    missing_b = [key for key in keys if key not in monitor]
    if missing_a or missing_b:
        return _finding("coordinate_headers", "warning", "Coordinate headers are incomplete at some matched trace keys.", inputs,
                        metadata={"baseline_missing_count": len(missing_a), "monitor_missing_count": len(missing_b)},
                        action="Validate navigation coverage; do not extrapolate coordinates for absent bins.")
    units_a = {baseline[key][2] for key in keys}
    units_b = {monitor[key][2] for key in keys}
    if units_a != units_b:
        return _finding("coordinate_units", "fail", "Coordinate-unit codes differ between surveys.", inputs,
                        metadata={"baseline_unit_codes": sorted(str(x) for x in units_a),
                                  "monitor_unit_codes": sorted(str(x) for x in units_b)},
                        action="Resolve coordinate units and transformations from authoritative metadata.", blocking=True)
    if any(unit in (None, 0) for unit in units_a):
        return _finding("coordinate_units", "warning", "Coordinate values are present but their units are undefined.", inputs,
                        metadata={"unit_codes": sorted(str(x) for x in units_a)},
                        action="Obtain coordinate units/CRS; do not use these values for spatial joins.", blocking=True)
    mismatches = [key for key in keys if baseline[key] != monitor[key]]
    if mismatches:
        return _finding("coordinate_headers", "fail", "Scalar-normalized coordinates disagree at matched trace keys.", inputs,
                        metadata={"mismatch_count": len(mismatches), "mismatch_examples": mismatches[:10]},
                        action="Verify scalars, coordinate units, trace correspondence, and survey transform before pairing.", blocking=True)
    return _finding("coordinate_headers", "pass", "Scalar-normalized coordinate headers agree at all matched trace keys.", inputs,
                    metadata={"matched_coordinate_key_count": len(keys), "coordinate_unit_codes": sorted(str(x) for x in units_a)})


def compare_volume_geometry(baseline: SeismicVolume, monitor: SeismicVolume) -> ValidationReport:
    """Return a QC report; warnings can block direct comparison when metadata is unknown."""
    findings: list[ValidationFinding] = []
    a, b = baseline.metadata, monitor.metadata
    inputs = (a.survey_id, b.survey_id)
    if baseline.data.ndim != 3 or monitor.data.ndim != 3 or a.axis_labels != ARRAY_AXES or b.axis_labels != ARRAY_AXES:
        findings.append(_finding("array_axes", "fail", "Inputs must be 3D arrays with (inline,crossline,sample) axis labels.", inputs,
                                 action="Load or transpose through an explicit axis mapping.", blocking=True))
    else:
        findings.append(_finding("array_axes", "pass", "Both arrays use the canonical inline/crossline/sample convention.", inputs))
    metadata_shape_mismatches = {
        "baseline_array_vs_metadata": {"array": tuple(baseline.data.shape), "metadata": tuple(a.dimensions)},
        "monitor_array_vs_metadata": {"array": tuple(monitor.data.shape), "metadata": tuple(b.dimensions)},
    }
    inconsistent_shapes = {
        name: values for name, values in metadata_shape_mismatches.items()
        if values["array"] != tuple(values["metadata"])
    }
    if inconsistent_shapes:
        findings.append(_finding("metadata_dimensions", "fail", "Declared volume dimensions disagree with the loaded array shapes.", inputs,
                                 metadata=inconsistent_shapes,
                                 action="Correct the ingestion metadata or reload the volume; do not compare inconsistent axis declarations.", blocking=True))
    else:
        findings.append(_finding("metadata_dimensions", "pass", "Declared dimensions match the loaded array shapes for both surveys.", inputs,
                                 metadata={"baseline_shape": tuple(baseline.data.shape), "monitor_shape": tuple(monitor.data.shape)}))
    if baseline.data.shape != monitor.data.shape or a.dimensions != b.dimensions:
        findings.append(_finding("dimensions", "fail", "Baseline and monitor dimensions differ.", inputs,
                                 metadata={"baseline_shape": baseline.data.shape, "monitor_shape": monitor.data.shape},
                                 action="Establish a documented common grid; do not subtract arrays by ordinal position.", blocking=True))
    else:
        findings.append(_finding("dimensions", "pass", "Baseline and monitor array dimensions match.", inputs,
                                 metadata={"dimensions": a.dimensions}))
    if (a.sample_interval_units in {"", "unknown", "unverified"}
            or b.sample_interval_units in {"", "unknown", "unverified"}):
        findings.append(_finding("sample_interval_units", "warning", "Sample-interval units are unknown or unverified.", inputs,
                                 metadata={"baseline_units": a.sample_interval_units, "monitor_units": b.sample_interval_units},
                                 action="Obtain sampling units from source metadata before comparing sample axes.", blocking=True))
    elif a.sample_interval != b.sample_interval or a.sample_interval_units != b.sample_interval_units:
        findings.append(_finding("sample_interval", "fail", "Sample intervals or their units differ.", inputs,
                                 metadata={"baseline_interval": a.sample_interval, "baseline_units": a.sample_interval_units,
                                           "monitor_interval": b.sample_interval, "monitor_units": b.sample_interval_units},
                                 action="Verify sample geometry and explicitly resample only through a documented workflow.", blocking=True))
    else:
        findings.append(_finding("sample_interval", "pass", "Sample intervals and units match.", inputs,
                                 metadata={"interval": a.sample_interval, "units": a.sample_interval_units}))
    domains_known = a.sample_axis_domain not in {"unknown", "unverified", ""} and b.sample_axis_domain not in {"unknown", "unverified", ""}
    if not domains_known:
        findings.append(_finding("sample_domain", "warning", "One or both sample-axis domains are unknown/unverified.", inputs,
                                 metadata={"baseline_domain": a.sample_axis_domain, "monitor_domain": b.sample_axis_domain},
                                 action="Obtain source documentation and record time/depth convention and sample origin before pairing.", blocking=True))
    elif (a.sample_axis_domain != b.sample_axis_domain or a.sample_units != b.sample_units
          or a.sample_units in {"unknown", "unverified", ""}):
        findings.append(_finding("sample_domain", "fail", "Sample domains or sample units differ.", inputs,
                                 metadata={"baseline_domain": a.sample_axis_domain, "baseline_units": a.sample_units,
                                           "monitor_domain": b.sample_axis_domain, "monitor_units": b.sample_units},
                                 action="Do not compare until a supported and documented conversion is applied.", blocking=True))
    else:
        findings.append(_finding("sample_domain", "pass", "Sample domains and units match.", inputs,
                                 metadata={"domain": a.sample_axis_domain, "units": a.sample_units}))
    a_origin, b_origin = a.source_metadata.get("sample_origin"), b.source_metadata.get("sample_origin")
    a_origin_units = a.source_metadata.get("sample_origin_units")
    b_origin_units = b.source_metadata.get("sample_origin_units")
    if a_origin is None or b_origin is None or not a_origin_units or not b_origin_units:
        findings.append(_finding("sample_origin", "warning", "Sample origin is missing for one or both surveys.", inputs,
                                 metadata={"baseline_origin": a_origin, "baseline_units": a_origin_units,
                                           "monitor_origin": b_origin, "monitor_units": b_origin_units},
                                 action="Supply the documented sample origin and unit; do not infer absolute time/depth from index.", blocking=True))
    elif a_origin != b_origin or a_origin_units != b_origin_units:
        findings.append(_finding("sample_origin", "fail", "Sample origins differ.", inputs,
                                 metadata={"baseline_origin": a_origin, "monitor_origin": b_origin,
                                           "baseline_units": a_origin_units,
                                           "monitor_units": b_origin_units},
                                 action="Resolve the origin mismatch through a documented sample-axis transform.", blocking=True))
    elif a_origin_units != a.sample_units or b_origin_units != b.sample_units:
        findings.append(_finding("sample_origin_units", "fail", "Sample-origin units do not match the declared sample-axis units.", inputs,
                                 metadata={"baseline_origin_units": a_origin_units, "baseline_axis_units": a.sample_units,
                                           "monitor_origin_units": b_origin_units, "monitor_axis_units": b.sample_units},
                                 action="Normalize origins to each declared sample-axis unit and document the conversion.", blocking=True))
    else:
        findings.append(_finding("sample_origin", "pass", "Sample origins match.", inputs, metadata={"sample_origin": a_origin}))

    for axis_name, attr in (("inline", "inline_ids_selected"), ("crossline", "crossline_ids_selected")):
        ids_a, ids_b = a.source_metadata.get(attr), b.source_metadata.get(attr)
        if ids_a is None or ids_b is None:
            findings.append(_finding(f"{axis_name}_identifiers", "warning", f"Explicit {axis_name} identifiers are missing.", inputs,
                                     action="Retain header-derived axis IDs and verify correspondence before pairing.", blocking=True))
        elif tuple(ids_a) != tuple(ids_b):
            findings.append(_finding(f"{axis_name}_identifiers", "fail", f"{axis_name} identifiers or array ordering differ.", inputs,
                                     metadata={"baseline_count": len(ids_a), "monitor_count": len(ids_b),
                                               "baseline_examples": list(ids_a[:5]), "monitor_examples": list(ids_b[:5])},
                                     action="Reindex using verified identifier correspondence; reject incompatible grids.", blocking=True))
        else:
            findings.append(_finding(f"{axis_name}_identifiers", "pass", f"{axis_name} identifier arrays match exactly.", inputs,
                                     metadata={"count": len(ids_a)}))

    geometry_status_a = a.source_metadata.get("geometry_status", "unverified")
    geometry_status_b = b.source_metadata.get("geometry_status", "unverified")
    geometry_ref_a = a.source_metadata.get("geometry_verification_reference")
    geometry_ref_b = b.source_metadata.get("geometry_verification_reference")
    if (geometry_status_a != "verified" or geometry_status_b != "verified"
            or not geometry_ref_a or not geometry_ref_b):
        findings.append(_finding("survey_geometry_status", "warning", "Survey geometry has not been explicitly verified for both inputs.", inputs,
                                 metadata={"baseline_status": geometry_status_a, "monitor_status": geometry_status_b,
                                           "baseline_verification_reference": geometry_ref_a,
                                           "monitor_verification_reference": geometry_ref_b},
                                 action="Validate the pair against authoritative survey/DUG metadata and record a durable evidence reference.", blocking=True))
    else:
        findings.append(_finding("survey_geometry_status", "pass", "Both surveys carry explicit verified geometry statuses and provenance references.", inputs,
                                 metadata={"baseline_reference": geometry_ref_a, "monitor_reference": geometry_ref_b}))

    crs_a, crs_b = a.source_metadata.get("coordinate_reference"), b.source_metadata.get("coordinate_reference")
    def crs_is_known(value: Any) -> bool:
        return isinstance(value, str) and value.strip().casefold() not in {"", "unknown", "unverified", "not supplied"}

    if crs_is_known(crs_a) and crs_is_known(crs_b) and crs_a != crs_b:
        findings.append(_finding("coordinate_reference", "fail", "Survey coordinate references differ.", inputs,
                                 metadata={"baseline_coordinate_reference": crs_a, "monitor_coordinate_reference": crs_b},
                                 action="Apply and record an authorized coordinate transform before spatial comparison.", blocking=True))
    elif not crs_is_known(crs_a) or not crs_is_known(crs_b):
        findings.append(_finding("coordinate_reference", "warning", "Coordinate reference is unknown for at least one survey; map-space and well joins are disabled.", inputs,
                                 metadata={"baseline_coordinate_reference": crs_a, "monitor_coordinate_reference": crs_b},
                                 action="Obtain CRS/units from authoritative survey metadata. Bin-index analysis may proceed only if all other blocking geometry gates pass.", blocking=False))
    else:
        findings.append(_finding("coordinate_reference", "pass", "Coordinate references match.", inputs,
                                 metadata={"coordinate_reference": crs_a}))

    for name, volume in (("baseline", baseline), ("monitor", monitor)):
        invalid_count = int(np.count_nonzero(~np.isfinite(volume.data)))
        if invalid_count:
            findings.append(_finding(f"{name}_invalid_samples", "warning", f"{invalid_count} non-finite samples require an explicit paired validity mask.",
                                     [getattr(volume.metadata, "survey_id", name)], metadata={"nonfinite_sample_count": invalid_count},
                                     action="Propagate the joint finite-data mask and report its coverage; do not fill invalid samples with zero."))
        if volume.missing_trace_mask is not None and bool(np.any(volume.missing_trace_mask)):
            findings.append(_finding(f"{name}_missing_traces", "warning", "Missing traces are represented in the volume mask.",
                                     [volume.metadata.survey_id], metadata={"missing_trace_count": int(np.count_nonzero(volume.missing_trace_mask))},
                                     action="Compare only valid matched traces and report coverage."))
    return ValidationReport(tuple(findings), {"baseline_id": a.survey_id, "monitor_id": b.survey_id})


def validate_pair(baseline: SeismicVolume, monitor: SeismicVolume) -> None:
    """Raise rather than silently compare when any required pairing gate is unresolved."""
    report = compare_volume_geometry(baseline, monitor)
    blockers = [item for item in report.findings if item.status == "fail" or item.blocking]
    if blockers:
        detail = "; ".join(f"{item.check_id}: {item.explanation}" for item in blockers)
        raise ValueError(f"Baseline/monitor sampling geometry validation rejected ({report.status}): {detail}")
