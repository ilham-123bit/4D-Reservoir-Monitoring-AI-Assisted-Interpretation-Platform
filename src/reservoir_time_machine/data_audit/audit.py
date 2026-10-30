"""Read-only Challenge 3 inventory and metadata audit.

The manifest deliberately excludes sample values, well coordinates, and raw headers.
Run this tool on the authorized HPC filesystem where the files reside.
"""

from __future__ import annotations

import hashlib
import importlib.util
import re
from pathlib import Path
from typing import Any

import numpy as np

from ..data.validation import inspect_segy_integrity

BASELINE = "volumes/Challenge3_Baseline_FilteredScaled_FullStack.sgy"
MONITOR = "volumes/Challenge3_Monitor_FilteredScaled_FullStack.sgy"
WELLS = "well_data/Challenge3_WellList_Information.xlsx"
WELL_FILES = {
    "checkshots": [f"well_data/checkshots/Well-{well}_Checkshot.txt" for well in (1, 2, 4, 5, 6, 7, 8)],
    "deviations": [f"well_data/deviations/Well-{well}_Deviation.txt" for well in (2, 4, 5, 6)],
    "las": [f"well_data/las/Well-{well}.las" for well in (1, 2, 3, 4, 5, 6, 7, 8)] + [
        f"well_data/las/Well-{well}_Composite_Interp.las" for well in (4, 6, 7)],
    "markers": [f"well_data/markers/Well-{well}_Markers.txt" for well in (1, 2, 4, 5, 6, 7, 8)],
}


def _optional_dependency(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _safe_inspect(function, path: Path) -> dict[str, Any]:
    try:
        return function(path)
    except Exception as exc:
        return {"status": "error", "reason": f"{type(exc).__name__}: metadata inspection failed; review the source file on the HPC."}


def _inspect_segy(path: Path) -> dict[str, Any]:
    if not _optional_dependency("segyio"):
        return {"status": "not_inspected", "reason": "Install the competition-data extra (segyio) on the HPC."}
    import segyio

    valid_formats = {1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16}
    candidates = []
    for endian in ("big", "little"):
        try:
            with segyio.open(str(path), "r", ignore_geometry=True, endian=endian) as f:
                fmt = int(f.bin[segyio.BinField.Format])
                dt = int(f.bin[segyio.BinField.Interval])
                ns = int(f.bin[segyio.BinField.Samples]) or len(f.samples)
                revision_field = getattr(segyio.BinField, "SEGYRevision", None)
                rev = int(f.bin[revision_field]) if revision_field is not None else -1
                score = int(fmt in valid_formats) + int(0 < dt < 1_000_000) + int(0 < ns < 1_000_000) + int(0 <= rev <= 2)
                candidates.append((score, endian, fmt, dt, ns, rev))
        except Exception:
            continue
    if not candidates:
        return {"status": "error", "reason": "Could not open as big- or little-endian SEG-Y."}
    selected = max(candidates, key=lambda candidate: candidate[0])
    _, endian, fmt, interval, binary_ns, revision = selected
    trace_field = segyio.TraceField

    def field_value(header, name: str) -> int | None:
        field = getattr(trace_field, name, None)
        try:
            return int(header[field]) if field is not None else None
        except Exception:
            return None

    inline: list[int] = []
    crossline: list[int] = []
    sequence: list[int] = []
    key_coordinates: list[tuple[int, int, float, float, int | None, int | None]] = []
    coord_scalars: set[int] = set()
    coord_units: set[int] = set()
    trace_order_hasher = hashlib.sha256()
    finite_count = 0
    sample_count = 0
    nonfinite_trace_count = 0
    dead_trace_count = 0
    rms_values: list[float] = []
    amplitude_samples: list[float] = []
    source_nonzero_count = receiver_nonzero_count = 0
    crs_tokens_detected = False
    amplitude_min, amplitude_max = float("inf"), float("-inf")
    with segyio.open(str(path), "r", ignore_geometry=True, endian=endian) as f:
        trace_count = int(f.tracecount)
        sample_axis = len(f.samples)
        for trace_index in range(trace_count):
            header = f.header[trace_index]
            il = field_value(header, "INLINE_3D")
            xl = field_value(header, "CROSSLINE_3D")
            inline.append(il or 0)
            crossline.append(xl or 0)
            sequence.append(field_value(header, "TRACE_SEQUENCE_FILE") or 0)
            for field_name, destination in (("SourceGroupScalar", coord_scalars), ("CoordinateUnits", coord_units)):
                value = field_value(header, field_name)
                if value is not None:
                    destination.add(value)
            cdp_x = field_value(header, "CDP_X")
            cdp_y = field_value(header, "CDP_Y")
            coordinate_fields = "CDP_X/CDP_Y"
            if cdp_x is None or cdp_y is None:
                cdp_x, cdp_y = field_value(header, "SourceX"), field_value(header, "SourceY")
                coordinate_fields = "SourceX/SourceY fallback"
            scalar = field_value(header, "SourceGroupScalar")
            units = field_value(header, "CoordinateUnits")
            if cdp_x is not None and cdp_y is not None:
                factor = 1.0 if scalar in (None, 0, 1) else (float(scalar) if scalar > 0 else 1.0 / abs(scalar))
                scaled_x, scaled_y = float(cdp_x * factor), float(cdp_y * factor)
                source_nonzero_count += int(bool(cdp_x or cdp_y))
                key_coordinates.append((il or 0, xl or 0, scaled_x, scaled_y, units, scalar))
            samples = f.trace[trace_index]
            finite = samples[np.isfinite(samples)]
            sample_count += int(samples.size)
            finite_count += int(finite.size)
            if finite.size != samples.size:
                nonfinite_trace_count += 1
            if finite.size == 0 or not finite.size or float(abs(finite).max(initial=0.0)) == 0.0:
                dead_trace_count += 1
                rms_values.append(0.0)
            else:
                amplitude_min = min(amplitude_min, float(finite.min()))
                amplitude_max = max(amplitude_max, float(finite.max()))
                rms_values.append(float(np.sqrt(np.mean(finite.astype("float64") ** 2))))
                if len(amplitude_samples) < 100_000:
                    step = max(1, finite.size // 8)
                    amplitude_samples.extend(float(value) for value in finite[::step][:8])
        try:
            textual = " ".join(str(line) for line in f.text[0])
            crs_tokens_detected = bool(re.search(r"\b(EPSG|CRS|UTM|PROJ|DATUM)\b", textual, re.IGNORECASE))
        except Exception:
            pass
    pairs = list(zip(inline, crossline))
    nonzero_pairs = [(il, xl) for il, xl in pairs if il != 0 or xl != 0]
    unique_pairs = set(nonzero_pairs)
    inline_ids = set(inline) - {0}
    crossline_ids = set(crossline) - {0}
    sorted_inline, sorted_crossline = sorted(inline_ids), sorted(crossline_ids)
    expected_grid_traces = len(inline_ids) * len(crossline_ids)
    duplicates = len(nonzero_pairs) - len(unique_pairs)
    ordered_pairs = nonzero_pairs == sorted(nonzero_pairs)
    trace_order_hasher.update(repr(nonzero_pairs).encode("ascii"))
    trace_key_set_fingerprint = hashlib.sha256(repr(sorted(unique_pairs)).encode("ascii")).hexdigest()
    coordinate_hasher = hashlib.sha256(repr(sorted(key_coordinates)).encode("ascii")).hexdigest()
    coordinate_values_present = any(item[2] or item[3] for item in key_coordinates)
    median_trace_rms = float(np.median(rms_values)) if rms_values else 0.0
    degraded_count = sum(0 < rms < median_trace_rms * 0.1 for rms in rms_values) if median_trace_rms else 0
    samples_per_trace = sample_axis if trace_count else binary_ns
    return {
        "status": "inspected", "file_size_bytes": path.stat().st_size, "endianness_candidate": endian,
        "revision_code": revision, "sample_format_code": fmt, "trace_count": trace_count,
        "samples_per_trace": samples_per_trace, "binary_header_samples_per_trace": binary_ns,
        "sample_interval_microseconds": interval, "inline_count": len(inline_ids),
        "inline_identifier_min": min(inline_ids) if inline_ids else None,
        "inline_identifier_max": max(inline_ids) if inline_ids else None,
        "inline_identifier_gaps": sum(b-a != 1 for a, b in zip(sorted_inline, sorted_inline[1:])),
        "crossline_count": len(crossline_ids),
        "crossline_identifier_min": min(crossline_ids) if crossline_ids else None,
        "crossline_identifier_max": max(crossline_ids) if crossline_ids else None,
        "crossline_identifier_gaps": sum(b-a != 1 for a, b in zip(sorted_crossline, sorted_crossline[1:])),
        "duplicate_inline_crossline_traces": duplicates,
        "missing_inline_crossline_positions_estimate": max(0, expected_grid_traces - len(unique_pairs)),
        "zero_inline_or_crossline_id_trace_count": len(pairs) - len(nonzero_pairs),
        "trace_order_lexicographically_inline_crossline": ordered_pairs,
        "trace_key_set_fingerprint": trace_key_set_fingerprint,
        "trace_order_fingerprint": trace_order_hasher.hexdigest(),
        "trace_sequence_unique": len(set(sequence)) == len(sequence),
        "trace_sequence_strictly_increasing": all(b > a for a, b in zip(sequence, sequence[1:])),
        "coordinate_scalar_codes": sorted(coord_scalars), "coordinate_unit_codes": sorted(coord_units),
        "traces_with_nonzero_coordinate_values": source_nonzero_count,
        "traces_with_nonzero_receiver_coordinates": receiver_nonzero_count,
        "coordinate_header_fields_used": coordinate_fields if trace_count else None,
        "coordinate_values_present": coordinate_values_present,
        "coordinate_trace_count": len(key_coordinates),
        "coordinate_geometry_fingerprint": coordinate_hasher,
        "textual_header_mentions_crs_or_datum": crs_tokens_detected,
        "coordinate_reference_system": "not decoded automatically; verify textual header/project documentation",
        "_coordinate_sequence_fingerprint": coordinate_hasher,
        "finite_sample_fraction": finite_count / sample_count if sample_count else 0.0,
        "invalid_or_nonfinite_sample_count": sample_count-finite_count,
        "traces_with_nonfinite_samples": nonfinite_trace_count, "dead_trace_count": dead_trace_count,
        "degraded_trace_count_below_10pct_median_rms": degraded_count,
        "amplitude_min": amplitude_min if finite_count else None,
        "amplitude_max": amplitude_max if finite_count else None,
        "amplitude_percentiles_sampled_approx": {str(p): float(np.percentile(amplitude_samples, p))
                                                  for p in (1, 25, 50, 75, 99)} if amplitude_samples else {},
        "amplitude_rms_mean_of_trace_rms": sum(rms_values) / len(rms_values) if rms_values else None,
    }


def _inspect_xlsx(path: Path) -> dict[str, Any]:
    if not _optional_dependency("openpyxl"):
        return {"status": "not_inspected", "reason": "Install the competition-data extra (openpyxl) on the HPC."}
    from openpyxl import load_workbook

    workbook = load_workbook(path, read_only=True, data_only=True)
    sheets = []
    for sheet in workbook.worksheets:
        rows = sheet.iter_rows(values_only=True)
        first = next(rows, ())
        columns = [str(value).strip() if value is not None else "" for value in first]
        nonempty_counts = [0] * len(columns)
        row_count = 0
        for row in rows:
            if any(value is not None for value in row):
                row_count += 1
            for index, value in enumerate(row[:len(columns)]):
                nonempty_counts[index] += int(value is not None and str(value).strip() != "")
        normalized = [column.casefold() for column in columns]
        sheets.append({"name": sheet.title, "columns": columns, "data_row_count": row_count,
                       "nonempty_value_counts_by_column": dict(zip(columns, nonempty_counts)),
                       "possible_well_id_columns": [columns[i] for i, name in enumerate(normalized)
                           if any(token in name for token in ("well", "uwi", "name", "id"))],
                       "possible_coordinate_columns": [columns[i] for i, name in enumerate(normalized)
                           if any(token in name for token in ("x", "y", "north", "east", "coord", "elev", "datum", "crs"))]})
    workbook.close()
    return {"status": "inspected", "sheet_count": len(sheets), "sheets": sheets,
            "source_values_included": False}


def _inspect_text_table(path: Path) -> dict[str, Any]:
    lines = [line.strip() for line in path.read_text(errors="replace").splitlines() if line.strip()]
    comments = [line for line in lines if line.startswith(("#", "!", ";"))]
    content = [line for line in lines if line not in comments]
    if not content:
        return {"status": "inspected", "row_count": 0, "columns": [], "header_detected": False}
    delimiter = "\t" if "\t" in content[0] else ("," if "," in content[0] else None)
    split = (lambda line: [part.strip() for part in line.split(delimiter)]) if delimiter else (lambda line: re.split(r"\s+", line))
    first = split(content[0])
    header_detected = any(re.search(r"[A-Za-z]", value) for value in first)
    columns = first if header_detected else [f"column_{i+1}" for i in range(len(first))]
    rows = [split(line) for line in (content[1:] if header_detected else content)]
    metadata_terms = sorted({term.lower() for line in comments for term in
        re.findall(r"\b(MD|TVD|TWT|OWT|two[- ]way|one[- ]way|depth|time|unit|datum|reference|ms|msec|s|sec|ft|feet|m|metres?|degrees?|azimuth|inclination|northing|easting)\b", line, re.IGNORECASE)})
    return {"status": "inspected", "delimiter": "whitespace" if delimiter is None else delimiter,
            "header_detected": header_detected, "columns": columns,
            "data_row_count": len(rows), "row_width_mismatch_count": sum(len(row) != len(columns) for row in rows),
            "missing_value_token_count": sum(value.casefold() in {"", "na", "nan", "null", "-999.25"}
                                               for row in rows for value in row),
            "comment_metadata_terms": metadata_terms, "sample_values_included": False}


def _inspect_las(path: Path) -> dict[str, Any]:
    if not _optional_dependency("lasio"):
        return {"status": "not_inspected", "reason": "Install the competition-data extra (lasio) on the HPC."}
    import lasio

    las = lasio.read(str(path), mnemonic_case="preserve")
    depth = las.index
    curves = []
    for curve in las.curves:
        values = las[curve.mnemonic]
        null = las.well.NULL.value if las.well.NULL.value is not None else -999.25
        curves.append({"mnemonic": curve.mnemonic, "unit": curve.unit, "description": curve.descr,
                       "sample_count": int(len(values)),
                       "null_count": int((~np.isfinite(values) | (values == null)).sum())})
    return {"status": "inspected", "las_version": str(las.version.VERS.value),
            "well_id_present": bool(las.well.WELL.value), "depth_curve_mnemonic": las.curves[0].mnemonic if las.curves else None,
            "depth_unit": las.curves[0].unit if las.curves else None,
            "depth_start": float(depth[0]) if len(depth) else None,
            "depth_stop": float(depth[-1]) if len(depth) else None,
            "depth_sample_count": int(len(depth)), "curves": curves, "well_value_included": False}


def audit_challenge3(root: str | Path) -> dict[str, Any]:
    """Inspect file availability and available metadata only; never serializes raw values/coordinates."""
    root = Path(root).expanduser().resolve()
    inventory: list[dict[str, Any]] = []

    def record(relative: str, category: str) -> Path | None:
        path = root / relative
        entry = {"path": relative, "category": category, "exists": path.is_file(),
                 "file_size_bytes": path.stat().st_size if path.is_file() else None}
        inventory.append(entry)
        return path if path.is_file() else None

    baseline_path, monitor_path = record(BASELINE, "segy"), record(MONITOR, "segy")
    well_path = record(WELLS, "well_inventory")
    for category, paths in WELL_FILES.items():
        for relative in paths:
            record(relative, category)
    expected_paths = {BASELINE, MONITOR, WELLS, *(item for values in WELL_FILES.values() for item in values)}
    directory_names = {"volumes": "segy", "well_data": "well_data",
                       "well_data/checkshots": "checkshot", "well_data/deviations": "deviation",
                       "well_data/las": "las", "well_data/markers": "marker"}
    unexpected_files = []
    for relative_dir, category in directory_names.items():
        directory = root / relative_dir
        if directory.is_dir():
            for path in sorted(directory.iterdir()):
                relative = path.relative_to(root).as_posix()
                if path.is_file() and relative not in expected_paths:
                    unexpected_files.append({"path": relative, "category": category,
                                             "file_size_bytes": path.stat().st_size})
    findings: dict[str, Any] = {
        "root_accessible": root.is_dir(), "dataset_root_configured": str(root),
        "inventory": inventory, "unexpected_files": unexpected_files,
        "well_file_availability": {
            category: {"expected": len(paths), "present": sum((root / item).is_file() for item in paths),
                       "missing_paths": [item for item in paths if not (root / item).is_file()]}
            for category, paths in WELL_FILES.items()},
        "segy": {}, "segy_integrity": {}, "well_information": {}, "text_tables": {}, "las": [],
        "manifest_contains_raw_samples_or_coordinates": False,
    }
    if baseline_path:
        findings["segy_integrity"]["baseline"] = inspect_segy_integrity(baseline_path).to_dict()
        findings["segy"]["baseline"] = _safe_inspect(_inspect_segy, baseline_path)
    if monitor_path:
        findings["segy_integrity"]["monitor"] = inspect_segy_integrity(monitor_path).to_dict()
        findings["segy"]["monitor"] = _safe_inspect(_inspect_segy, monitor_path)
    if baseline_path and monitor_path and all(item.get("status") == "inspected" for item in findings["segy"].values()):
        a, b = findings["segy"]["baseline"], findings["segy"]["monitor"]
        from ..data.validation import ValidationFinding, ValidationReport

        pair_inputs = (BASELINE, MONITOR)
        pair_findings = []
        keys_match = a["trace_key_set_fingerprint"] == b["trace_key_set_fingerprint"]
        pair_findings.append(ValidationFinding("trace_key_coverage", "pass" if keys_match else "fail",
            "Inline/crossline key sets match." if keys_match else "Inline/crossline key sets differ.", pair_inputs,
            {"baseline_trace_count": a["trace_count"], "monitor_trace_count": b["trace_count"]},
            "Resolve trace coverage and reindex by keys before sample comparison." if not keys_match else "", not keys_match))
        order_matches = a["trace_order_fingerprint"] == b["trace_order_fingerprint"]
        pair_findings.append(ValidationFinding("trace_file_order", "pass" if order_matches else "warning",
            "Trace orders match." if order_matches else "Trace order differs; key-based reindexing is required.", pair_inputs,
            {}, "Reindex traces using inline/crossline identifiers; never pair by trace sequence." if not order_matches else ""))
        for check_id, predicate, explanation, metadata in (
            ("sample_count", a["samples_per_trace"] == b["samples_per_trace"], "Samples per trace differ.",
             {"baseline": a["samples_per_trace"], "monitor": b["samples_per_trace"]}),
            ("sample_interval", a["sample_interval_microseconds"] == b["sample_interval_microseconds"], "Binary-header sample intervals differ.",
             {"baseline_us": a["sample_interval_microseconds"], "monitor_us": b["sample_interval_microseconds"]}),
            ("sample_format", a["sample_format_code"] == b["sample_format_code"], "Sample format codes differ.",
             {"baseline": a["sample_format_code"], "monitor": b["sample_format_code"]}),
        ):
            pair_findings.append(ValidationFinding(check_id, "pass" if predicate else "fail",
                f"{check_id} headers match." if predicate else explanation, pair_inputs, metadata,
                "Confirm and explicitly normalize the sample geometry before pairing." if not predicate else "", not predicate))
        coords_a = a["coordinate_values_present"] and a["coordinate_trace_count"] == a["trace_count"]
        coords_b = b["coordinate_values_present"] and b["coordinate_trace_count"] == b["trace_count"]
        coordinates_match = (a["coordinate_geometry_fingerprint"] == b["coordinate_geometry_fingerprint"]
                             if coords_a and coords_b else None)
        if coordinates_match is False:
            pair_findings.append(ValidationFinding("coordinate_headers", "fail", "Scalar-normalized coordinate header fingerprints differ.", pair_inputs,
                {"baseline_coordinate_fields": a["coordinate_header_fields_used"], "monitor_coordinate_fields": b["coordinate_header_fields_used"]},
                "Resolve coordinate scalars, units, trace correspondence, or authoritative survey transforms before spatial comparison.", True))
        else:
            pair_findings.append(ValidationFinding("coordinate_headers", "pass" if coordinates_match else "warning",
                "Scalar-normalized coordinate header fingerprints match." if coordinates_match else "Usable coordinates are missing or incomplete.",
                pair_inputs, {"baseline_available": coords_a, "monitor_available": coords_b},
                "Provide authoritative navigation/CRS before map-space or well integration." if not coordinates_match else ""))
        pair_findings.append(ValidationFinding("sample_domain", "warning", "SEG-Y headers alone do not establish the seismic time/depth convention or absolute sample origin.",
            pair_inputs, {}, "Supply verified sample-domain and origin metadata before paired scientific calculations.", True))
        findings["segy_pair_compatibility"] = {
            "trace_and_sample_dimensions_match": (a["trace_count"], a["samples_per_trace"]) == (b["trace_count"], b["samples_per_trace"]),
            "sample_interval_matches": a["sample_interval_microseconds"] == b["sample_interval_microseconds"],
            "inline_crossline_counts_match": (a["inline_count"], a["crossline_count"]) == (b["inline_count"], b["crossline_count"]),
            "trace_key_set_matches": keys_match,
            "trace_order_matches": order_matches,
            "coordinate_headers_present_for_both": bool(coords_a and coords_b),
            "coordinate_headers_match": coordinates_match,
            "spatial_geometry_compatible": bool(keys_match and coordinates_match is True),
            "pair_processing_eligible": False,
            "trace_order_and_coordinate_headers_match": bool(order_matches and coordinates_match is True),
        }
        findings["segy_pair_validation_report"] = ValidationReport(tuple(pair_findings)).to_dict()
    if well_path:
        findings["well_information"] = _safe_inspect(_inspect_xlsx, well_path)
    for category in ("checkshots", "deviations", "markers"):
        findings["text_tables"][category] = [
            {"path": relative, **_safe_inspect(_inspect_text_table, root / relative)}
            for relative in WELL_FILES[category] if (root / relative).is_file()]
    findings["las"] = [{"path": relative, "composite_interpreted": "Composite_Interp" in relative,
                        **_safe_inspect(_inspect_las, root / relative)}
                       for relative in WELL_FILES["las"] if (root / relative).is_file()]
    for survey in findings["segy"].values():
        survey.pop("_coordinate_sequence_fingerprint", None)
        survey.pop("trace_key_set_fingerprint", None)
        survey.pop("trace_order_fingerprint", None)
    return findings
