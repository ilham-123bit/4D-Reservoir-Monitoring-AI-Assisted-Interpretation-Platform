"""Private, versioned offline QC products for paired seismic memmaps."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .qc import quality_report

SCHEMA = "reservoir-time-machine-survey-qc-1"
ANALYSIS_VERSION = "chunked-quality-report-1"


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def paired_sample_support(baseline, monitor, missing_mask, *, inline_chunk=2):
    """Count finite paired samples in chunks, excluding every missing baseline-grid trace."""
    if baseline.shape != monitor.shape or len(baseline.shape) != 3:
        raise ValueError("Baseline and Monitor must share (inline, crossline, sample) geometry")
    if missing_mask.shape != baseline.shape[:2]:
        raise ValueError("Missing-trace mask must match the paired grid")
    if isinstance(inline_chunk, bool) or not isinstance(inline_chunk, int) or inline_chunk < 1 or inline_chunk > 8:
        raise ValueError("inline_chunk must be an integer from 1 to 8")
    paired_trace_count = int(missing_mask.size - np.asarray(missing_mask).sum(dtype=np.int64))
    total_support = complete_traces = partial_traces = zero_support_traces = 0
    baseline_only_samples = monitor_only_samples = 0
    sample_count = int(baseline.shape[2])
    for start in range(0, baseline.shape[0], inline_chunk):
        stop = min(start + inline_chunk, baseline.shape[0])
        b = np.asarray(baseline[start:stop])
        m = np.asarray(monitor[start:stop])
        paired = ~np.asarray(missing_mask[start:stop], dtype=bool)
        bf, mf = np.isfinite(b), np.isfinite(m)
        both = bf & mf & paired[..., None]
        per_trace = both.sum(axis=2, dtype=np.int64)
        selected = per_trace[paired]
        total_support += int(selected.sum(dtype=np.int64))
        complete_traces += int((selected == sample_count).sum())
        partial_traces += int(((selected > 0) & (selected < sample_count)).sum())
        zero_support_traces += int((selected == 0).sum())
        baseline_only_samples += int((bf & ~mf & paired[..., None]).sum(dtype=np.int64))
        monitor_only_samples += int((~bf & mf & paired[..., None]).sum(dtype=np.int64))
    potential = paired_trace_count * sample_count
    return {
        "paired_grid_trace_count": paired_trace_count,
        "potential_paired_samples": potential,
        "valid_paired_samples": total_support,
        "paired_sample_support_fraction": total_support / potential if potential else None,
        "complete_paired_traces": complete_traces,
        "partial_support_traces": partial_traces,
        "zero_support_paired_traces": zero_support_traces,
        "baseline_finite_monitor_nonfinite_samples": baseline_only_samples,
        "monitor_finite_baseline_nonfinite_samples": monitor_only_samples,
    }


def build_qc_artifact(baseline, monitor, missing_mask, monitor_only, *, source_identity,
                      source_metadata, inline_chunk=2):
    if (not isinstance(source_identity, str) or len(source_identity) != 64
            or any(ch not in "0123456789abcdef" for ch in source_identity)):
        raise ValueError("source_identity must be a SHA256 cache identity")
    if baseline.shape != monitor.shape or len(baseline.shape) != 3:
        raise ValueError("Baseline and Monitor must have matching 3D shapes")
    if missing_mask.shape != baseline.shape[:2]:
        raise ValueError("missing mask does not match the paired grid")
    if np.asarray(missing_mask).dtype.kind != "b":
        raise ValueError("missing mask must be Boolean")
    if np.asarray(monitor_only).ndim != 2 or monitor_only.shape[1] != baseline.shape[2]:
        raise ValueError("monitor_only must have shape (monitor-only traces, sample)")

    missing_count = int(np.asarray(missing_mask).sum(dtype=np.int64))
    paired_count = int(missing_mask.size - missing_count)
    monitor_only_count = int(monitor_only.shape[0])
    metadata_counts = {
        "baseline_trace_count": source_metadata.get("baseline_trace_count"),
        "paired_trace_count": source_metadata.get("paired_trace_count"),
        "monitor_only_trace_count": source_metadata.get("monitor_only_trace_count"),
    }
    if metadata_counts["paired_trace_count"] is not None and int(metadata_counts["paired_trace_count"]) != paired_count:
        raise ValueError("Paired trace metadata disagrees with missing-mask occupancy")
    if metadata_counts["baseline_trace_count"] is not None and int(metadata_counts["baseline_trace_count"]) != paired_count:
        raise ValueError("Baseline trace metadata disagrees with paired-grid occupancy")
    if metadata_counts["monitor_only_trace_count"] is not None and int(metadata_counts["monitor_only_trace_count"]) != monitor_only_count:
        raise ValueError("Monitor-only metadata disagrees with preserved trace array")

    valid_traces = ~np.asarray(missing_mask, dtype=bool)
    baseline_qc = quality_report(baseline, inline_chunk=inline_chunk, valid_trace_mask=valid_traces)
    monitor_qc = quality_report(monitor, inline_chunk=inline_chunk, valid_trace_mask=valid_traces)
    monitor_only_qc = quality_report(monitor_only.reshape(1, *monitor_only.shape), inline_chunk=1)
    paired_support = paired_sample_support(baseline, monitor, missing_mask, inline_chunk=inline_chunk)
    body = {
        "schema": SCHEMA,
        "analysis_version": ANALYSIS_VERSION,
        "source_identity": source_identity,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "shape_inline_crossline_sample": list(map(int, baseline.shape)),
        "sample_domain": "unknown; sample index is not converted to time or depth",
        "survey_statistics_scope": "full paired-cache arrays; computed offline in bounded inline chunks",
        "trace_counts": {
            "baseline_grid_cells": int(missing_mask.size),
            "paired_trace_count": paired_count,
            "missing_baseline_grid_count": missing_count,
            "monitor_only_trace_count": monitor_only_count,
            "monitor_survey_trace_count": paired_count + monitor_only_count,
            "source_metadata_counts": metadata_counts,
        },
        "paired_sample_support": paired_support,
        "baseline": _plain(baseline_qc),
        "monitor_on_paired_baseline_grid": _plain(monitor_qc),
        "monitor_only": _plain(monitor_only_qc),
        "comparability": {
            "paired_amplitude_rms_difference": (
                float(monitor_qc["amplitude_rms"] - baseline_qc["amplitude_rms"])
                if math.isfinite(float(monitor_qc["amplitude_rms"]))
                and math.isfinite(float(baseline_qc["amplitude_rms"])) else None
            ),
            "paired_amplitude_rms_ratio": (
                float(monitor_qc["amplitude_rms"] / baseline_qc["amplitude_rms"])
                if math.isfinite(float(monitor_qc["amplitude_rms"]))
                and math.isfinite(float(baseline_qc["amplitude_rms"]))
                and float(baseline_qc["amplitude_rms"]) > 0 else None
            ),
            "interpretation": "descriptive amplitude comparison only; no repeatability threshold or registration correction is established",
        },
    }
    envelope = {"body": body, "sha256": hashlib.sha256(_canonical(body)).hexdigest()}
    return _plain(envelope)


def load_qc_artifact(path, *, source_identity, shape):
    path = Path(path).expanduser()
    if path.stat().st_size > 1_000_000:
        raise ValueError("QC artifact exceeds the bounded manifest limit")
    envelope = json.loads(path.read_text(encoding="utf-8"))
    body = envelope["body"]
    if (envelope.get("sha256") != hashlib.sha256(_canonical(body)).hexdigest()
            or body.get("schema") != SCHEMA or body.get("analysis_version") != ANALYSIS_VERSION):
        raise ValueError("QC artifact checksum or version is invalid")
    if body.get("source_identity") != source_identity or body.get("shape_inline_crossline_sample") != list(shape):
        raise ValueError("QC artifact source identity or geometry is stale")
    return body


def write_qc_artifact(artifact, path, *, protected_cache):
    path = Path(path).expanduser().resolve()
    protected_cache = Path(protected_cache).resolve()
    if path == protected_cache or protected_cache in path.parents:
        raise ValueError("QC artifacts must remain outside the protected paired cache")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    parent_stat = path.parent.stat()
    if parent_stat.st_uid != os.getuid() or stat.S_IMODE(parent_stat.st_mode) & 0o077:
        raise PermissionError("QC artifact directory must be user-owned and private (mode 0700)")
    if path.exists():
        raise FileExistsError("Refusing to overwrite an existing QC artifact")
    payload = _canonical(artifact) + b"\n"
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path
