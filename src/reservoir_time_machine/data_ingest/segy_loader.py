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

from dataclasses import dataclass


@dataclass(frozen=True)
class PairedTraceIndex:
    """Compact correspondence between Baseline and Monitor SEG-Y traces."""

    baseline_keys: tuple[tuple[int, int], ...]
    monitor_index_for_baseline: tuple[int, ...]
    monitor_only_indices: tuple[int, ...]
    monitor_only_keys: tuple[tuple[int, int], ...]

    @property
    def paired_trace_count(self) -> int:
        return len(self.baseline_keys)

    @property
    def monitor_only_count(self) -> int:
        return len(self.monitor_only_indices)


def build_paired_trace_index(
    baseline_path: str | Path,
    monitor_path: str | Path,
    *,
    endian: str = "big",
) -> PairedTraceIndex:
    """
    Build trace correspondence using (inline, crossline).

    No seismic samples are loaded. Original SEG-Y files are opened read-only.
    """

    try:
        import segyio
    except ImportError as exc:
        raise RuntimeError("Trace indexing requires segyio") from exc

    def read_keys(path):
        keys = []

        with segyio.open(
            str(path), "r", ignore_geometry=True, endian=endian
        ) as segy:
            for i in range(int(segy.tracecount)):
                header = segy.header[i]

                il = int(header[segyio.TraceField.INLINE_3D])
                xl = int(header[segyio.TraceField.CROSSLINE_3D])

                if il == 0 or xl == 0:
                    raise ValueError(
                        f"Invalid zero inline/crossline key at trace {i}: "
                        f"({il}, {xl}) in {path}"
                    )

                keys.append((il, xl))

        if len(keys) != len(set(keys)):
            raise ValueError(
                f"Duplicate inline/crossline keys detected in {path}"
            )

        return tuple(keys)

    baseline_keys = read_keys(baseline_path)
    monitor_keys = read_keys(monitor_path)

    monitor_lookup = {
        key: index for index, key in enumerate(monitor_keys)
    }

    baseline_set = set(baseline_keys)

    baseline_only = baseline_set - set(monitor_keys)

    if baseline_only:
        examples = sorted(baseline_only)[:10]
        raise ValueError(
            f"Monitor is missing {len(baseline_only)} Baseline keys. "
            f"Examples: {examples}"
        )

    monitor_index_for_baseline = tuple(
        monitor_lookup[key] for key in baseline_keys
    )

    monitor_only_indices = tuple(
        index
        for index, key in enumerate(monitor_keys)
        if key not in baseline_set
    )

    monitor_only_keys = tuple(
        monitor_keys[index] for index in monitor_only_indices
    )

    return PairedTraceIndex(
        baseline_keys=baseline_keys,
        monitor_index_for_baseline=monitor_index_for_baseline,
        monitor_only_indices=monitor_only_indices,
        monitor_only_keys=monitor_only_keys,
    )


def write_paired_segy_pilot(
    baseline_path: str | Path,
    monitor_path: str | Path,
    output_dir: str | Path,
    *,
    inline_ids: set[int],
    crossline_ids: set[int],
    endian: str = "big",
    sample_range: tuple[int, int] | None = None,
) -> tuple[SeismicVolume, SeismicVolume, dict]:
    """
    Write a small, real-data Baseline/Monitor pilot.

    Both volumes use Baseline inline/crossline indexing.
    Original SEG-Y files are opened read-only.
    Existing output directories are rejected to prevent overwrites.
    """
    import json

    try:
        import segyio
    except ImportError as exc:
        raise RuntimeError("Paired SEG-Y loading requires segyio") from exc

    baseline_path = Path(baseline_path)
    monitor_path = Path(monitor_path)
    output_dir = Path(output_dir)

    if not baseline_path.is_file() or not monitor_path.is_file():
        raise FileNotFoundError("Baseline or Monitor SEG-Y file is missing.")

    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists; refusing to overwrite: {output_dir}"
        )

    if not inline_ids or not crossline_ids:
        raise ValueError("Pilot requires non-empty inline_ids and crossline_ids.")

    # Build the correspondence using the already-tested trace-key function.
    paired_index = build_paired_trace_index(
        baseline_path, monitor_path, endian=endian
    )

    baseline_key_to_index = {
        key: i for i, key in enumerate(paired_index.baseline_keys)
    }

    with segyio.open(
        str(baseline_path), "r", ignore_geometry=True, endian=endian
    ) as base, segyio.open(
        str(monitor_path), "r", ignore_geometry=True, endian=endian
    ) as mon:

        base_ns = len(base.samples)
        mon_ns = len(mon.samples)
        base_dt = int(base.bin[segyio.BinField.Interval])
        mon_dt = int(mon.bin[segyio.BinField.Interval])
        base_format = int(base.bin[segyio.BinField.Format])
        mon_format = int(mon.bin[segyio.BinField.Format])

        if base_ns != mon_ns:
            raise ValueError(
                f"Sample-count mismatch: Baseline={base_ns}, Monitor={mon_ns}"
            )
        if base_dt != mon_dt:
            raise ValueError(
                f"Sample-interval mismatch: Baseline={base_dt}, Monitor={mon_dt}"
            )
        if base_format != mon_format:
            raise ValueError(
                f"Sample-format mismatch: Baseline={base_format}, Monitor={mon_format}"
            )

        sample_start, sample_stop = sample_range or (0, base_ns)
        if not 0 <= sample_start < sample_stop <= base_ns:
            raise ValueError("Invalid sample_range.")

        n_samples = sample_stop - sample_start

        # Establish the selected Baseline grid from actual trace keys.
        selected_keys = [
            key for key in paired_index.baseline_keys
            if key[0] in inline_ids and key[1] in crossline_ids
        ]

        if not selected_keys:
            raise ValueError("No Baseline traces found in the requested pilot region.")

        selected_inlines = sorted({key[0] for key in selected_keys})
        selected_crosslines = sorted({key[1] for key in selected_keys})

        inline_index = {value: i for i, value in enumerate(selected_inlines)}
        crossline_index = {value: i for i, value in enumerate(selected_crosslines)}

        shape = (
            len(selected_inlines),
            len(selected_crosslines),
            n_samples,
        )

        selected_set = set(selected_keys)
        key_to_location = {
            key: (inline_index[key[0]], crossline_index[key[1]])
            for key in selected_keys
        }

        # Confirm that each selected Baseline key has a Monitor counterpart.
        selected_monitor_indices = {
            key: paired_index.monitor_index_for_baseline[
                baseline_key_to_index[key]
            ]
            for key in selected_keys
        }

        # Create output only after all preflight checks pass.
        output_dir.mkdir(parents=True, exist_ok=False)

        baseline_map = np.lib.format.open_memmap(
            output_dir / "baseline_pilot.npy",
            mode="w+",
            dtype=np.float32,
            shape=shape,
        )
        monitor_map = np.lib.format.open_memmap(
            output_dir / "monitor_pilot.npy",
            mode="w+",
            dtype=np.float32,
            shape=shape,
        )

        baseline_map[:] = np.nan
        monitor_map[:] = np.nan

        missing_mask = np.ones(shape[:2], dtype=bool)

        # Read Baseline sequentially and write selected traces.
        for key in selected_keys:
            trace_index = baseline_key_to_index[key]
            trace = np.asarray(
                base.trace[trace_index][sample_start:sample_stop],
                dtype=np.float32,
            )
            if trace.size != n_samples:
                raise ValueError(f"Unexpected Baseline sample count at {key}")

            loc = key_to_location[key]
            baseline_map[loc[0], loc[1], :] = trace
            missing_mask[loc] = False

        # Read Monitor sequentially and place traces by key, never file order.
        monitor_key_to_location = key_to_location

        for monitor_trace_index in range(int(mon.tracecount)):
            header = mon.header[monitor_trace_index]
            key = (
                int(header[segyio.TraceField.INLINE_3D]),
                int(header[segyio.TraceField.CROSSLINE_3D]),
            )

            if key not in monitor_key_to_location:
                continue

            trace = np.asarray(
                mon.trace[monitor_trace_index][sample_start:sample_stop],
                dtype=np.float32,
            )
            if trace.size != n_samples:
                raise ValueError(f"Unexpected Monitor sample count at {key}")

            loc = monitor_key_to_location[key]
            monitor_map[loc[0], loc[1], :] = trace

        # Preserve the complete trace-key list for this pilot.
        np.save(
            output_dir / "trace_keys.npy",
            np.asarray(selected_keys, dtype=np.int32),
        )
        np.save(output_dir / "missing_trace_mask.npy", missing_mask)

        # Verify that every selected key was written into both arrays.
        paired_written = int(np.count_nonzero(~missing_mask))
        monitor_finite = np.isfinite(monitor_map).all(axis=2)
        baseline_finite = np.isfinite(baseline_map).all(axis=2)

        if not np.array_equal(baseline_finite, monitor_finite):
            raise RuntimeError(
                "Pilot QC failed: Baseline and Monitor trace coverage differs."
            )

        baseline_map.flush()
        monitor_map.flush()

        metadata = {
            "baseline_source": baseline_path.name,
            "monitor_source": monitor_path.name,
            "pairing_key": ["inline", "crossline"],
            "array_axes": ["inline", "crossline", "sample"],
            "shape": list(shape),
            "inline_ids": selected_inlines,
            "crossline_ids": selected_crosslines,
            "paired_trace_count": paired_written,
            "sample_count": n_samples,
            "sample_interval_us": base_dt,
            "sample_domain": "unverified",
            "sample_origin": None,
            "monitor_only_trace_count_in_source": paired_index.monitor_only_count,
            "coordinate_method": "Baseline trace-key grid",
            "monitor_coordinate_headers_used": False,
            "physical_4d_registration": "not established by trace-key pairing",
        }

        with (output_dir / "metadata.json").open("w") as stream:
            json.dump(metadata, stream, indent=2)

    baseline_metadata = SurveyMetadata(
        survey_id="challenge3_baseline_pilot",
        label="Challenge 3 Baseline Pilot",
        dimensions=shape,
        inline_units="SEG-Y inline identifier",
        crossline_units="SEG-Y crossline identifier",
        sample_interval=base_dt,
        sample_interval_units="microseconds",
        sample_units="unknown",
        sample_axis_domain="unknown",
        synthetic=False,
        source_metadata={
            **metadata,
            "source_file_name": baseline_path.name,
            "geometry_status": "baseline_grid_mapping_verified",
        },
    )

    monitor_metadata = SurveyMetadata(
        survey_id="challenge3_monitor_pilot",
        label="Challenge 3 Monitor Pilot",
        dimensions=shape,
        inline_units="SEG-Y inline identifier",
        crossline_units="SEG-Y crossline identifier",
        sample_interval=mon_dt,
        sample_interval_units="microseconds",
        sample_units="unknown",
        sample_axis_domain="unknown",
        synthetic=False,
        source_metadata={
            **metadata,
            "source_file_name": monitor_path.name,
            "geometry_status": "4d_registration_unverified",
        },
    )

    baseline_volume = SeismicVolume(
        baseline_map, baseline_metadata, missing_mask
    )
    monitor_volume = SeismicVolume(
        monitor_map, monitor_metadata, missing_mask.copy()
    )

    return baseline_volume, monitor_volume, metadata


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

def load_challenge3_paired_volumes(
    root: str | Path,
    *,
    output_dir: str | Path | None = None,
    endian: str = "big",
) -> tuple[SeismicVolume, SeismicVolume]:
    """Load Challenge 3 on the Baseline grid, pairing strictly by (inline, crossline).

    Monitor-only traces are preserved separately. Original SEG-Y files are read-only.
    """
    import json
    import segyio

    from ..data_audit.audit import BASELINE, MONITOR

    root = Path(root)
    baseline_path = root / BASELINE
    monitor_path = root / MONITOR

    for path in (baseline_path, monitor_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    if output_dir is None:
        output_dir = root / ".reservoir_time_machine_cache" / "challenge3_paired"

    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    if output_dir.exists():
        # Reuse an existing completed cache without rebuilding or overwriting it.
        required_files = (
            "metadata.json",
            "baseline.npy",
            "monitor.npy",
            "missing_trace_mask.npy",
            "monitor_only_traces.npy",
            "monitor_only_keys.npy",
        )
        missing_files = [
            name for name in required_files
            if not (output_dir / name).is_file()
        ]
        if missing_files:
            raise RuntimeError(
                f"Incomplete paired cache at {output_dir}; "
                f"missing files: {missing_files}. Refusing to overwrite it."
            )

        try:
            with (output_dir / "metadata.json").open("r") as stream:
                metadata = json.load(stream)

            shape = tuple(int(v) for v in metadata["shape"])
            inline_ids = metadata["inline_ids"]
            crossline_ids = metadata["crossline_ids"]
            base_dt = int(metadata["sample_interval_us"])

            if len(shape) != 3 or any(v < 1 for v in shape):
                raise ValueError(f"Invalid shape: {shape}")
            if shape != (
                len(inline_ids),
                len(crossline_ids),
                int(metadata["sample_count"]),
            ):
                raise ValueError("Metadata dimensions disagree")
            if metadata["pairing_key"] != ["inline", "crossline"]:
                raise ValueError("Unexpected pairing key")
            if metadata["array_axes"] != ["inline", "crossline", "sample"]:
                raise ValueError("Unexpected array axes")
            if metadata["trace_key_pairing_verified"] is not True:
                raise ValueError("Trace-key pairing is not verified")
            if metadata["physical_4d_registration"] != "not established":
                raise ValueError("Unexpected physical registration status")
            if metadata["monitor_coordinate_headers_used"] is not False:
                raise ValueError("Unexpected Monitor coordinate handling")
            if metadata["baseline_source"] != baseline_path.name:
                raise ValueError("Baseline source filename mismatch")
            if metadata["monitor_source"] != monitor_path.name:
                raise ValueError("Monitor source filename mismatch")
            if base_dt != 2000:
                raise ValueError(f"Unexpected sample interval: {base_dt}")

            paired_count = int(metadata["paired_trace_count"])
            monitor_only_count = int(metadata["monitor_only_trace_count"])

            if paired_count != int(metadata["baseline_trace_count"]):
                raise ValueError("Baseline and paired trace counts disagree")
            if paired_count < 1 or monitor_only_count < 0:
                raise ValueError("Invalid trace counts")

            baseline_data = np.load(
                output_dir / "baseline.npy",
                mmap_mode="r",
                allow_pickle=False,
            )
            monitor_data = np.load(
                output_dir / "monitor.npy",
                mmap_mode="r",
                allow_pickle=False,
            )
            missing = np.load(
                output_dir / "missing_trace_mask.npy",
                mmap_mode="r",
                allow_pickle=False,
            )
            monitor_only_data = np.load(
                output_dir / "monitor_only_traces.npy",
                mmap_mode="r",
                allow_pickle=False,
            )
            monitor_only_keys = np.load(
                output_dir / "monitor_only_keys.npy",
                mmap_mode="r",
                allow_pickle=False,
            )

            if baseline_data.shape != shape or monitor_data.shape != shape:
                raise ValueError("Seismic volume shape mismatch")
            if baseline_data.dtype != np.float32 or monitor_data.dtype != np.float32:
                raise ValueError("Seismic volumes must be float32")
            if missing.shape != shape[:2] or missing.dtype != np.bool_:
                raise ValueError("Missing-trace mask shape or dtype mismatch")
            if monitor_only_data.shape != (monitor_only_count, shape[2]):
                raise ValueError("Monitor-only trace archive shape mismatch")
            if monitor_only_data.dtype != np.float32:
                raise ValueError("Monitor-only archive must be float32")
            if monitor_only_keys.shape != (monitor_only_count, 2):
                raise ValueError("Monitor-only key shape mismatch")
            if monitor_only_keys.dtype != np.int32:
                raise ValueError("Monitor-only keys must be int32")
            if int(np.count_nonzero(missing)) != int(
                metadata["missing_grid_cell_count"]
            ):
                raise ValueError("Missing-trace count disagrees with metadata")

            arrays = (
                baseline_data,
                monitor_data,
                missing,
                monitor_only_data,
                monitor_only_keys,
            )
            if any(array.flags.writeable for array in arrays):
                raise ValueError("A cached array is not read-only")

        except Exception as exc:
            raise RuntimeError(
                f"Paired cache validation failed at {output_dir}: {exc}. "
                "Cache was not modified."
            ) from exc

        baseline_metadata = SurveyMetadata(
            survey_id="challenge3_baseline",
            label="Challenge 3 Baseline",
            dimensions=shape,
            inline_units="SEG-Y inline identifier",
            crossline_units="SEG-Y crossline identifier",
            sample_interval=base_dt,
            sample_interval_units="microseconds",
            sample_units="unknown",
            sample_axis_domain="unknown",
            synthetic=False,
            source_metadata={
                **metadata,
                "source_file_name": baseline_path.name,
                "spatial_geometry_compatible": True,
                "paired_grid_available": True,
            },
        )

        monitor_metadata = SurveyMetadata(
            survey_id="challenge3_monitor",
            label="Challenge 3 Monitor",
            dimensions=shape,
            inline_units="SEG-Y inline identifier",
            crossline_units="SEG-Y crossline identifier",
            sample_interval=base_dt,
            sample_interval_units="microseconds",
            sample_units="unknown",
            sample_axis_domain="unknown",
            synthetic=False,
            source_metadata={
                **metadata,
                "source_file_name": monitor_path.name,
                "spatial_geometry_compatible": True,
                "paired_grid_available": True,
                "physical_4d_registration_verified": False,
            },
        )

        return (
            SeismicVolume(baseline_data, baseline_metadata, missing),
            SeismicVolume(monitor_data, monitor_metadata, missing),
        )

    # Build correspondence by trace key, not file order.
    pair_index = build_paired_trace_index(
        baseline_path, monitor_path, endian=endian
    )

    with segyio.open(
        str(baseline_path), "r", ignore_geometry=True, endian=endian
    ) as base, segyio.open(
        str(monitor_path), "r", ignore_geometry=True, endian=endian
    ) as mon:

        base_ns = len(base.samples)
        mon_ns = len(mon.samples)
        base_dt = int(base.bin[segyio.BinField.Interval])
        mon_dt = int(mon.bin[segyio.BinField.Interval])
        base_format = int(base.bin[segyio.BinField.Format])
        mon_format = int(mon.bin[segyio.BinField.Format])

        if base_ns != mon_ns:
            raise ValueError(f"Sample-count mismatch: {base_ns} != {mon_ns}")
        if base_dt != mon_dt:
            raise ValueError(f"Sample-interval mismatch: {base_dt} != {mon_dt}")
        if base_format != mon_format:
            raise ValueError(f"Sample-format mismatch: {base_format} != {mon_format}")
        if base_dt != 2000:
            raise ValueError(f"Unexpected Challenge 3 sample interval: {base_dt} us")

        # Establish the common Baseline grid.
        baseline_keys = pair_index.baseline_keys
        inline_ids = sorted({key[0] for key in baseline_keys})
        crossline_ids = sorted({key[1] for key in baseline_keys})

        il_index = {value: i for i, value in enumerate(inline_ids)}
        xl_index = {value: i for i, value in enumerate(crossline_ids)}

        shape = (len(inline_ids), len(crossline_ids), base_ns)
        grid_shape = shape[:2]

        output_dir.mkdir(parents=True, exist_ok=False)

        try:
            baseline_map = np.lib.format.open_memmap(
                output_dir / "baseline.npy",
                mode="w+", dtype=np.float32, shape=shape
            )
            monitor_map = np.lib.format.open_memmap(
                output_dir / "monitor.npy",
                mode="w+", dtype=np.float32, shape=shape
            )

            baseline_map[:] = np.nan
            monitor_map[:] = np.nan

            missing_mask = np.ones(grid_shape, dtype=bool)

            # Write Baseline traces onto the Baseline grid.
            for trace_index, key in enumerate(baseline_keys):
                il, xl = key
                loc = (il_index[il], xl_index[xl])
                trace = np.asarray(base.trace[trace_index], dtype=np.float32)

                if trace.size != base_ns:
                    raise ValueError(f"Unexpected Baseline samples at {key}")

                baseline_map[loc[0], loc[1], :] = trace
                missing_mask[loc] = False

            # Monitor correspondence: key -> Monitor file trace index.
            monitor_lookup = {
                key: index
                for key, index in zip(
                    pair_index.baseline_keys,
                    pair_index.monitor_index_for_baseline
                )
            }

            # Read paired Monitor traces by key.
            for key, monitor_index in monitor_lookup.items():
                il, xl = key
                loc = (il_index[il], xl_index[xl])
                trace = np.asarray(
                    mon.trace[monitor_index], dtype=np.float32
                )

                if trace.size != base_ns:
                    raise ValueError(f"Unexpected Monitor samples at {key}")

                monitor_map[loc[0], loc[1], :] = trace

            # Preserve Monitor-only traces separately.
            extra_count = pair_index.monitor_only_count
            extra_shape = (extra_count, base_ns)

            extra_data = np.lib.format.open_memmap(
                output_dir / "monitor_only_traces.npy",
                mode="w+", dtype=np.float32, shape=extra_shape
            )
            extra_keys = np.asarray(
                pair_index.monitor_only_keys, dtype=np.int32
            )
            np.save(output_dir / "monitor_only_keys.npy", extra_keys)

            for out_index, source_index in enumerate(
                pair_index.monitor_only_indices
            ):
                extra_data[out_index, :] = np.asarray(
                    mon.trace[source_index], dtype=np.float32
                )

            baseline_map.flush()
            monitor_map.flush()
            extra_data.flush()

            metadata = {
                "baseline_source": baseline_path.name,
                "monitor_source": monitor_path.name,
                "pairing_key": ["inline", "crossline"],
                "array_axes": ["inline", "crossline", "sample"],
                "shape": list(shape),
                "inline_ids": inline_ids,
                "crossline_ids": crossline_ids,
                "baseline_trace_count": len(pair_index.baseline_keys),
                "paired_trace_count": len(pair_index.baseline_keys),
                "monitor_only_trace_count": extra_count,
                "missing_grid_cell_count": int(missing_mask.sum()),
                "sample_count": base_ns,
                "sample_interval_us": base_dt,
                "sample_domain": "unknown",
                "sample_origin": None,
                "baseline_finite_fraction": float(
                    np.isfinite(baseline_map).mean()
                ),
                "monitor_finite_fraction": float(
                    np.isfinite(monitor_map).mean()
                ),
                "monitor_only_archive": "monitor_only_traces.npy",
                "monitor_only_keys": "monitor_only_keys.npy",
                "coordinate_method": "Baseline inline/crossline grid",
                "monitor_coordinate_headers_used": False,
                "trace_key_pairing_verified": True,
                "physical_4d_registration": "not established",
                "source_sha256": "not calculated",
            }

            np.save(output_dir / "missing_trace_mask.npy", missing_mask)

            with (output_dir / "metadata.json").open("w") as stream:
                json.dump(metadata, stream, indent=2)

        except Exception:
            # Leave the incomplete directory for inspection/recovery.
            raise

    baseline_metadata = SurveyMetadata(
        survey_id="challenge3_baseline",
        label="Challenge 3 Baseline",
        dimensions=shape,
        inline_units="SEG-Y inline identifier",
        crossline_units="SEG-Y crossline identifier",
        sample_interval=base_dt,
        sample_interval_units="microseconds",
        sample_units="unknown",
        sample_axis_domain="unknown",
        synthetic=False,
        source_metadata={
            **metadata,
            "source_file_name": baseline_path.name,
            "spatial_geometry_compatible": True,
            "paired_grid_available": True,
        },
    )

    monitor_metadata = SurveyMetadata(
        survey_id="challenge3_monitor",
        label="Challenge 3 Monitor",
        dimensions=shape,
        inline_units="SEG-Y inline identifier",
        crossline_units="SEG-Y crossline identifier",
        sample_interval=mon_dt,
        sample_interval_units="microseconds",
        sample_units="unknown",
        sample_axis_domain="unknown",
        synthetic=False,
        source_metadata={
            **metadata,
            "source_file_name": monitor_path.name,
            "spatial_geometry_compatible": True,
            "paired_grid_available": True,
            "physical_4d_registration_verified": False,
        },
    )

    missing = np.load(output_dir / "missing_trace_mask.npy", mmap_mode="r")

    return (
        SeismicVolume(
            np.load(output_dir / "baseline.npy", mmap_mode="r"),
            baseline_metadata,
            missing,
        ),
        SeismicVolume(
            np.load(output_dir / "monitor.npy", mmap_mode="r"),
            monitor_metadata,
            missing,
        ),
    )
