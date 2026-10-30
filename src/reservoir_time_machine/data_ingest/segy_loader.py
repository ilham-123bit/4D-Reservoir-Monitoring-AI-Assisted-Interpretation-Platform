"""Memory-bounded SEG-Y loading with explicit inline/crossline/sample standardization."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
from dataclasses import replace

from ..models import SeismicVolume, SurveyMetadata
from ..data.validation import inspect_segy_integrity


def iter_segy_traces(path: str | Path, *, endian: str = "big") -> Iterator[tuple[int, int, int, np.ndarray]]:
    """Yield (file trace index, inline id, crossline id, samples) one trace at a time."""
    try:
        import segyio
    except ImportError as exc:
        raise RuntimeError("SEG-Y reading requires the competition-data extra: pip install .[competition-data]") from exc
    with segyio.open(str(path), "r", ignore_geometry=True, endian=endian) as segy:
        for index in range(int(segy.tracecount)):
            header = segy.header[index]
            il = int(header[segyio.TraceField.INLINE_3D])
            xl = int(header[segyio.TraceField.CROSSLINE_3D])
            yield index, il, xl, np.asarray(segy.trace[index])


def load_segy_volume(path: str | Path, *, endian: str = "big", max_memory_bytes: int = 1_500_000_000,
                     memmap_path: str | Path | None = None, label: str | None = None,
                     inline_ids: set[int] | None = None, crossline_ids: set[int] | None = None,
                     sample_range: tuple[int, int] | None = None, sample_axis_domain: str = "unknown",
                     sample_units: str = "unknown", sample_origin: float | None = None,
                     sample_origin_units: str | None = None,
                     coordinate_reference: str | None = None, geometry_verified: bool = False,
                     geometry_verification_reference: str | None = None,
                     expected_size_bytes: int | None = None, expected_sha256: str | None = None) -> SeismicVolume:
    """Load one cube as (inline,crossline,sample); optionally store output as an explicit memmap.

    The input SEG-Y is opened read-only. A memmap output contains restricted derived data and
    must be placed in an authorized HPC scratch/cache directory chosen by the caller.
    """
    integrity = inspect_segy_integrity(path, endian=endian, expected_size_bytes=expected_size_bytes,
                                       expected_sha256=expected_sha256, require_sha256=True)
    if not integrity.can_proceed:
        blocking = [finding for finding in integrity.findings if finding.status == "fail" or finding.blocking]
        summary = "; ".join(f"{finding.check_id}: {finding.explanation}" for finding in blocking)
        raise ValueError(f"SEG-Y integrity gate rejected {path}: {summary}")
    try:
        import segyio
    except ImportError as exc:
        raise RuntimeError("SEG-Y reading requires the competition-data extra: pip install .[competition-data]") from exc
    path = Path(path)
    with segyio.open(str(path), "r", ignore_geometry=True, endian=endian) as segy:
        trace_count = int(segy.tracecount)
        samples_per_trace = len(segy.samples)
        interval_us = int(segy.bin[segyio.BinField.Interval])
        format_code = int(segy.bin[segyio.BinField.Format])
        revision_field = getattr(segyio.BinField, "SEGYRevision", None)
        revision_code = int(segy.bin[revision_field]) if revision_field is not None else None
        measurement_field = getattr(segyio.BinField, "MeasurementSystem", None)
        measurement_system = int(segy.bin[measurement_field]) if measurement_field is not None else None
        headers = []
        coordinate_scalar_codes, coordinate_unit_codes, sequence_numbers = set(), set(), []
        nonzero_coordinate_traces = 0
        for i in range(trace_count):
            header = segy.header[i]
            headers.append((int(header[segyio.TraceField.INLINE_3D]),
                            int(header[segyio.TraceField.CROSSLINE_3D]), i))
            for name, target in (("SourceGroupScalar", coordinate_scalar_codes), ("CoordinateUnits", coordinate_unit_codes)):
                field = getattr(segyio.TraceField, name, None)
                if field is not None:
                    target.add(int(header[field]))
            coord_fields = [getattr(segyio.TraceField, name, None) for name in ("SourceX", "SourceY", "GroupX", "GroupY")]
            nonzero_coordinate_traces += int(any(field is not None and int(header[field]) != 0 for field in coord_fields))
            seq_field = getattr(segyio.TraceField, "TRACE_SEQUENCE_FILE", None)
            if seq_field is not None:
                sequence_numbers.append(int(header[seq_field]))
    if any(il == 0 or xl == 0 for il, xl, _ in headers):
        raise ValueError("Inline/crossline identifiers are absent or zero; inspect geometry before loading as a cube")
    selected_headers = [(il, xl, trace_index) for il, xl, trace_index in headers
                        if (inline_ids is None or il in inline_ids) and (crossline_ids is None or xl in crossline_ids)]
    selected_inline = sorted({il for il, _, _ in selected_headers})
    selected_crossline = sorted({xl for _, xl, _ in selected_headers})
    if not selected_headers:
        raise ValueError("Selected inline/crossline subset contains no traces")
    sample_start, sample_stop = sample_range or (0, samples_per_trace)
    if not 0 <= sample_start < sample_stop <= samples_per_trace:
        raise ValueError(f"sample_range must satisfy 0 <= start < stop <= {samples_per_trace}")
    samples_per_trace = sample_stop - sample_start
    inline_index, crossline_index = {v: i for i, v in enumerate(selected_inline)}, {v: i for i, v in enumerate(selected_crossline)}
    shape = (len(selected_inline), len(selected_crossline), samples_per_trace)
    required_bytes = int(np.prod(shape, dtype=np.int64)) * np.dtype(np.float32).itemsize
    if memmap_path is None and required_bytes > max_memory_bytes:
        raise MemoryError(f"Volume needs {required_bytes:,} bytes, above the configured {max_memory_bytes:,}-byte limit. Use a caller-managed HPC scratch memmap or subset workflow.")
    if memmap_path is None:
        cube = np.full(shape, np.nan, dtype=np.float32)
    else:
        cache = Path(memmap_path)
        cache.parent.mkdir(parents=True, exist_ok=True)
        if cache.exists():
            raise FileExistsError(f"Refusing to overwrite existing cache file: {cache}")
        cube = np.lib.format.open_memmap(cache, mode="w+", dtype=np.float32, shape=shape)
        cube[:] = np.nan
    occupied: set[tuple[int, int]] = set()
    missing = np.ones(shape[:2], dtype=bool)
    with segyio.open(str(path), "r", ignore_geometry=True, endian=endian) as segy:
        for il, xl, index in selected_headers:
            loc = (inline_index[il], crossline_index[xl])
            if loc in occupied:
                raise ValueError(f"Duplicate trace position encountered at inline/crossline array indices {loc}")
            occupied.add(loc)
            trace = np.asarray(segy.trace[index][sample_start:sample_stop], dtype=np.float32)
            if trace.size != samples_per_trace:
                raise ValueError(f"Trace {index} has {trace.size} samples; expected {samples_per_trace}")
            cube[loc[0], loc[1], :] = trace
            missing[loc] = False
    metadata = SurveyMetadata(survey_id=path.stem, label=label or path.stem,
        dimensions=shape, inline_units="SEG-Y inline identifier", crossline_units="SEG-Y crossline identifier",
        sample_interval=interval_us, sample_interval_units="microseconds",
        sample_units=sample_units, sample_axis_domain=sample_axis_domain, synthetic=False,
        source_metadata={"source_format": "SEG-Y", "source_file_name": path.name,
                         "source_file_size_bytes": path.stat().st_size,
                         "sample_origin": sample_origin,
                         "sample_origin_units": sample_origin_units,
                         "coordinate_reference": coordinate_reference,
                         "geometry_status": "verified" if geometry_verified else "unverified",
                         "geometry_verification_reference": geometry_verification_reference,
                         "endianness": endian, "trace_count": trace_count,
                         "segy_revision_code": revision_code, "sample_format_code": format_code,
                         "measurement_system_code": measurement_system,
                         "coordinate_scalar_codes": sorted(coordinate_scalar_codes),
                         "coordinate_unit_codes": sorted(coordinate_unit_codes),
                         "nonzero_coordinate_trace_count": nonzero_coordinate_traces,
                         "trace_sequence_unique": len(set(sequence_numbers)) == len(sequence_numbers),
                         "trace_sequence_strictly_increasing": all(b > a for a, b in zip(sequence_numbers, sequence_numbers[1:])),
                         "inline_ids_selected": selected_inline, "crossline_ids_selected": selected_crossline,
                         "sample_range_selected": [sample_start, sample_stop]})
    return SeismicVolume(cube, metadata, missing)


def validate_pair_files(baseline_path: str | Path, monitor_path: str | Path, *, endian: str = "big",
                        sample_axis_domain: str | None = None, sample_origin: float | None = None,
                        sample_origin_units: str | None = None, geometry_verified: bool = False,
                        geometry_verification_reference: str | None = None,
                        coordinate_reference: str | None = None,
                        expected_baseline_size_bytes: int | None = None,
                        expected_monitor_size_bytes: int | None = None,
                        expected_baseline_sha256: str | None = None,
                        expected_monitor_sha256: str | None = None,
                        expected_baseline_trace_count: int | None = None,
                        expected_monitor_trace_count: int | None = None) -> dict[str, object]:
    """Return machine-readable integrity/geometry checks without loading seismic samples.

    Equal trace count or trace order alone never establishes correspondence. A caller must
    explicitly assert verified survey geometry and provide the known sample-domain convention
    before the result is eligible for paired processing.
    """
    try:
        import segyio
    except ImportError as exc:
        raise RuntimeError("SEG-Y validation requires segyio") from exc

    def signature(path):
        with segyio.open(str(path), "r", ignore_geometry=True, endian=endian) as segy:
            trace_pairs, coordinates, trace_ns, trace_intervals, trace_delays = [], {}, set(), set(), set()
            tf = segyio.TraceField

            def header_value(header, names):
                for field_name in names:
                    field = getattr(tf, field_name, None)
                    if field is not None:
                        try:
                            return int(header[field])
                        except (KeyError, TypeError, ValueError):
                            continue
                return None

            x_names = ("CDP_X", "CDPX", "SourceX")
            y_names = ("CDP_Y", "CDPY", "SourceY")
            for i in range(int(segy.tracecount)):
                header = segy.header[i]
                key = (int(header[tf.INLINE_3D]), int(header[tf.CROSSLINE_3D]))
                trace_pairs.append(key)
                x, y = header_value(header, x_names), header_value(header, y_names)
                scalar = header_value(header, ("SourceGroupScalar",))
                units = header_value(header, ("CoordinateUnits",))
                if x is not None and y is not None and (x != 0 or y != 0):
                    factor = 1.0 if scalar in (None, 0, 1) else (float(scalar) if scalar > 0 else 1.0 / abs(scalar))
                    coordinates[key] = (x * factor, y * factor, units)
                nsv = header_value(header, ("TRACE_SAMPLE_COUNT", "TRACE_SAMPLE_COUNT"))
                dtv = header_value(header, ("TRACE_SAMPLE_INTERVAL", "TRACE_SAMPLE_INTERVAL"))
                delay = header_value(header, ("DelayRecordingTime",))
                if nsv is not None:
                    trace_ns.add(nsv)
                if dtv is not None:
                    trace_intervals.add(dtv)
                if delay is not None:
                    trace_delays.add(delay)
            interval = int(segy.bin[segyio.BinField.Interval])
            format_code = int(segy.bin[segyio.BinField.Format])
            return {"trace_count": int(segy.tracecount), "samples_per_trace": len(segy.samples),
                    "sample_interval_us": interval, "sample_format_code": format_code,
                    "keys": trace_pairs, "coordinates": coordinates,
                    "trace_ns": trace_ns, "trace_intervals": trace_intervals, "trace_delays": trace_delays}

    base_path, monitor_path = Path(baseline_path), Path(monitor_path)
    integrity_a = inspect_segy_integrity(base_path, endian=endian, expected_size_bytes=expected_baseline_size_bytes,
                                         expected_sha256=expected_baseline_sha256,
                                         expected_trace_count=expected_baseline_trace_count, require_sha256=True)
    integrity_b = inspect_segy_integrity(monitor_path, endian=endian, expected_size_bytes=expected_monitor_size_bytes,
                                         expected_sha256=expected_monitor_sha256,
                                         expected_trace_count=expected_monitor_trace_count, require_sha256=True)
    a, b = signature(base_path), signature(monitor_path)
    from ..data.validation import ValidationFinding, ValidationReport, compare_coordinate_headers, compare_trace_keys

    keys = compare_trace_keys(a["keys"], b["keys"], baseline_name=str(base_path), monitor_name=str(monitor_path))
    findings = list(integrity_a.findings) + list(integrity_b.findings) + list(keys.report.findings)
    pair_inputs = (str(base_path), str(monitor_path))

    trace_headers_consistent = (
        (not a["trace_ns"] or a["trace_ns"] == {a["samples_per_trace"]})
        and (not b["trace_ns"] or b["trace_ns"] == {b["samples_per_trace"]})
        and (not a["trace_intervals"] or a["trace_intervals"] == {a["sample_interval_us"]})
        and (not b["trace_intervals"] or b["trace_intervals"] == {b["sample_interval_us"]})
        and a["trace_delays"] == b["trace_delays"]
    )
    for check_id, predicate, explanation, metadata, action in (
        ("sample_count", a["samples_per_trace"] == b["samples_per_trace"], "Baseline and monitor samples-per-trace differ.",
         {"baseline": a["samples_per_trace"], "monitor": b["samples_per_trace"]}, "Confirm sample axes or apply documented resampling."),
        ("sample_interval", a["sample_interval_us"] == b["sample_interval_us"], "Baseline and monitor binary-header intervals differ.",
         {"baseline_us": a["sample_interval_us"], "monitor_us": b["sample_interval_us"]}, "Confirm sample intervals and document any resampling."),
        ("sample_format", a["sample_format_code"] == b["sample_format_code"], "Baseline and monitor sample encodings differ.",
         {"baseline_code": a["sample_format_code"], "monitor_code": b["sample_format_code"]}, "Decode each file using its declared encoding before any comparison."),
        ("trace_header_sample_geometry", trace_headers_consistent,
         "Trace-level sample counts, intervals, or recording delays differ.",
         {"baseline_ns": sorted(a["trace_ns"]), "monitor_ns": sorted(b["trace_ns"]),
          "baseline_intervals": sorted(a["trace_intervals"]), "monitor_intervals": sorted(b["trace_intervals"]),
          "baseline_delays": sorted(a["trace_delays"]), "monitor_delays": sorted(b["trace_delays"]),
          "baseline_binary_samples": a["samples_per_trace"], "monitor_binary_samples": b["samples_per_trace"],
          "baseline_binary_interval_us": a["sample_interval_us"], "monitor_binary_interval_us": b["sample_interval_us"]},
         "Resolve trace-level sampling/time-origin inconsistencies before loading paired data."),
    ):
        findings.append(ValidationFinding(check_id, "pass" if predicate else "fail",
            "Header compatibility check passed." if predicate else explanation, pair_inputs, metadata,
            "" if predicate else action, not predicate))

    if sample_axis_domain not in {"two_way_time", "one_way_time", "depth", "index"}:
        findings.append(ValidationFinding("sample_domain", "warning", "A documented sample domain was not supplied; SEG-Y interval alone does not establish time/depth semantics.",
            pair_inputs, {"sample_axis_domain": sample_axis_domain}, "Supply the verified sample-domain convention and origin from authoritative metadata.", True))
    else:
        findings.append(ValidationFinding("sample_domain", "pass", "A supported explicit sample-domain convention was supplied.",
            pair_inputs, {"sample_axis_domain": sample_axis_domain}))
    if sample_origin is None or not sample_origin_units:
        findings.append(ValidationFinding("sample_origin", "warning", "Sample origin or its unit is missing.", pair_inputs,
            {"sample_origin": sample_origin, "sample_origin_units": sample_origin_units},
            "Supply the authoritative sample origin and unit; do not infer absolute time/depth from index.", True))
    else:
        findings.append(ValidationFinding("sample_origin", "pass", "An explicit sample origin and unit were supplied.", pair_inputs,
            {"sample_origin": sample_origin, "sample_origin_units": sample_origin_units}))
    if not geometry_verified or not geometry_verification_reference:
        findings.append(ValidationFinding("geometry_verification", "warning", "Matching trace identifiers do not by themselves prove survey correspondence.",
            pair_inputs, {"geometry_verified_flag": geometry_verified,
                          "verification_reference": geometry_verification_reference},
            "Verify the relationship using authoritative survey navigation/DUG metadata and record a durable reference.", True))
    else:
        findings.append(ValidationFinding("geometry_verification", "pass", "Caller supplied explicit geometry verification with a provenance reference.", pair_inputs,
            {"verification_reference": geometry_verification_reference}))

    common = set(a["keys"]) & set(b["keys"])
    coordinate_finding = compare_coordinate_headers(a["coordinates"], b["coordinates"], common,
                                                     baseline_name=str(base_path), monitor_name=str(monitor_path))
    findings.append(coordinate_finding)
    coordinate_headers_present = bool(a["coordinates"]) and bool(b["coordinates"])
    coordinates_match = coordinate_finding.status == "pass"
    known_coordinate_reference = (isinstance(coordinate_reference, str)
                                  and coordinate_reference.strip().casefold() not in {"", "unknown", "unverified", "not supplied"})
    if not known_coordinate_reference:
        findings.append(ValidationFinding("coordinate_reference", "warning", "No authoritative CRS/coordinate reference was supplied.", pair_inputs, {},
            "Record CRS, coordinate units, axis order, and transform provenance before map-space operations.", False))
    else:
        findings.append(ValidationFinding("coordinate_reference", "pass", "An explicit coordinate reference was supplied.", pair_inputs,
            {"coordinate_reference": coordinate_reference}))

    report = ValidationReport(tuple(findings), {"baseline": a, "monitor": b,
        "trace_key_correspondence": {"status": keys.report.status, "matched_count": keys.report.metadata["matched_trace_count"],
                                     "order_differs": a["keys"] != b["keys"]}})
    # Do not expose large key lists or per-trace coordinate dictionaries in the public report.
    summarized = {**report.to_dict(), "metadata": {
        "baseline_trace_count": a["trace_count"], "monitor_trace_count": b["trace_count"],
        "baseline_samples_per_trace": a["samples_per_trace"], "monitor_samples_per_trace": b["samples_per_trace"],
        "baseline_sample_interval_us": a["sample_interval_us"], "monitor_sample_interval_us": b["sample_interval_us"],
        "matched_trace_key_count": keys.report.metadata["matched_trace_count"],
        "trace_order_differs": a["keys"] != b["keys"],
        "baseline_integrity": integrity_a.to_dict(), "monitor_integrity": integrity_b.to_dict(),
    }}
    return {
        "trace_count_matches": a["trace_count"] == b["trace_count"],
        "samples_per_trace_matches": a["samples_per_trace"] == b["samples_per_trace"],
        "sample_interval_matches": a["sample_interval_us"] == b["sample_interval_us"],
        "trace_order_matches": a["keys"] == b["keys"],
        "trace_key_set_matches": set(a["keys"]) == set(b["keys"]),
        "coordinate_headers_present": coordinate_headers_present,
        "coordinate_headers_match": coordinates_match,
        "spatial_geometry_compatible": bool(keys.report.can_proceed and geometry_verified
                                              and geometry_verification_reference and known_coordinate_reference
                                              and coordinate_headers_present and coordinates_match),
        "pair_processing_eligible": report.can_proceed,
        "validation_report": summarized,
    }


def load_challenge3_pair(root: str | Path, *, max_pair_memory_bytes: int = 2_000_000_000,
                         endian: str = "big", cache_directory: str | Path | None = None,
                         inline_ids: set[int] | None = None, crossline_ids: set[int] | None = None,
                         sample_range: tuple[int, int] | None = None,
                         sample_axis_domain: str | None = None, sample_origin: float | None = None,
                         sample_origin_units: str | None = None, sample_units: str = "unknown",
                         coordinate_reference: str | None = None, geometry_verified: bool = False,
                         geometry_verification_reference: str | None = None,
                         expected_baseline_size_bytes: int | None = None,
                         expected_monitor_size_bytes: int | None = None,
                         expected_baseline_sha256: str | None = None,
                         expected_monitor_sha256: str | None = None,
                         expected_baseline_trace_count: int | None = None,
                         expected_monitor_trace_count: int | None = None) -> tuple[SeismicVolume, SeismicVolume]:
    """Load paired files only after file-integrity and explicit metadata gates pass."""
    from ..data_audit.audit import BASELINE, MONITOR
    from ..data.validation import validate_pair

    root = Path(root)
    baseline_path, monitor_path = root / BASELINE, root / MONITOR
    for path in (baseline_path, monitor_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required Challenge 3 volume is missing: {path}")
    compatibility = validate_pair_files(
        baseline_path, monitor_path, endian=endian, sample_axis_domain=sample_axis_domain,
        sample_origin=sample_origin, sample_origin_units=sample_origin_units,
        geometry_verified=geometry_verified, geometry_verification_reference=geometry_verification_reference,
        coordinate_reference=coordinate_reference,
        expected_baseline_size_bytes=expected_baseline_size_bytes,
        expected_monitor_size_bytes=expected_monitor_size_bytes,
        expected_baseline_sha256=expected_baseline_sha256,
        expected_monitor_sha256=expected_monitor_sha256,
        expected_baseline_trace_count=expected_baseline_trace_count,
        expected_monitor_trace_count=expected_monitor_trace_count,
    )
    if not compatibility["pair_processing_eligible"]:
        findings = compatibility["validation_report"]["findings"]
        blockers = [item for item in findings if item["status"] == "fail" or item["blocking"]]
        reasons = "; ".join(f"{item['check_id']}: {item['explanation']}" for item in blockers)
        raise ValueError(f"Baseline/monitor SEG-Y pairing rejected: {reasons}")
    per_volume_limit = max_pair_memory_bytes // 2
    if per_volume_limit <= 0:
        raise ValueError("max_pair_memory_bytes must be positive")
    cache = Path(cache_directory) if cache_directory else None
    baseline_map = cache / "challenge3_baseline.npy" if cache else None
    monitor_map = cache / "challenge3_monitor.npy" if cache else None
    if baseline_map and (baseline_map.exists() or monitor_map.exists()):
        raise FileExistsError("Refusing to overwrite existing baseline/monitor memmap cache files")
    baseline = load_segy_volume(baseline_path, endian=endian, max_memory_bytes=per_volume_limit,
                                memmap_path=baseline_map, label="Baseline", inline_ids=inline_ids,
                                crossline_ids=crossline_ids, sample_range=sample_range,
                                sample_axis_domain=sample_axis_domain, sample_units=sample_units,
                                sample_origin=sample_origin, sample_origin_units=sample_origin_units,
                                coordinate_reference=coordinate_reference, geometry_verified=geometry_verified,
                                geometry_verification_reference=geometry_verification_reference,
                                expected_size_bytes=expected_baseline_size_bytes,
                                expected_sha256=expected_baseline_sha256)
    monitor = load_segy_volume(monitor_path, endian=endian, max_memory_bytes=per_volume_limit,
                               memmap_path=monitor_map, label="Monitor", inline_ids=inline_ids,
                               crossline_ids=crossline_ids, sample_range=sample_range,
                               sample_axis_domain=sample_axis_domain, sample_units=sample_units,
                               sample_origin=sample_origin, sample_origin_units=sample_origin_units,
                               coordinate_reference=coordinate_reference, geometry_verified=geometry_verified,
                               geometry_verification_reference=geometry_verification_reference,
                               expected_size_bytes=expected_monitor_size_bytes,
                               expected_sha256=expected_monitor_sha256)
    baseline = replace(baseline, metadata=replace(baseline.metadata,
        source_metadata={**baseline.metadata.source_metadata,
                         "coordinate_headers_match": compatibility["coordinate_headers_match"],
                         "spatial_geometry_compatible": compatibility["spatial_geometry_compatible"]}))
    monitor = replace(monitor, metadata=replace(monitor.metadata,
        source_metadata={**monitor.metadata.source_metadata,
                         "coordinate_headers_match": compatibility["coordinate_headers_match"],
                         "spatial_geometry_compatible": compatibility["spatial_geometry_compatible"]}))
    validate_pair(baseline, monitor)
    return baseline, monitor
