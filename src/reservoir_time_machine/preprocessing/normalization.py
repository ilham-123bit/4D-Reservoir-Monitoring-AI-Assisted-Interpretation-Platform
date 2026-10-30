"""Explicit amplitude equalization; source arrays are never mutated."""

import numpy as np
from pathlib import Path


def normalize_pair(baseline: np.ndarray, monitor: np.ndarray, method: str = "none") -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Return copies. Global RMS scales monitor to baseline; trace RMS is destructive to change."""
    before, after = np.array(baseline, dtype=float, copy=True), np.array(monitor, dtype=float, copy=True)
    if before.shape != after.shape or before.ndim != 3:
        raise ValueError("paired inputs must be matching 3D volumes")
    if method == "none":
        return before, after, {"method": method, "scale": 1.0, "warning": None}
    if method == "global_rms":
        bvalid, mvalid = np.isfinite(before), np.isfinite(after)
        brms = float(np.sqrt(np.mean(before[bvalid]**2))) if bvalid.any() else 0.0
        mrms = float(np.sqrt(np.mean(after[mvalid]**2))) if mvalid.any() else 0.0
        scale = brms / mrms if mrms > 0 else 1.0
        return before, after * scale, {"method": method, "scale": scale,
            "warning": "Global equalization can suppress genuine field-wide amplitude change."}
    if method == "trace_rms":
        bcount, mcount = np.isfinite(before).sum(axis=2), np.isfinite(after).sum(axis=2)
        brms = np.sqrt(np.divide(np.nansum(before**2, axis=2), bcount,
                                 out=np.zeros(before.shape[:2]), where=bcount > 0))
        mrms = np.sqrt(np.divide(np.nansum(after**2, axis=2), mcount,
                                 out=np.zeros(after.shape[:2]), where=mcount > 0))
        scale = np.divide(brms, mrms, out=np.ones_like(brms), where=mrms > 0)
        return before, after * scale[:, :, None], {"method": method, "scale": scale,
            "warning": "Trace equalization can erase localized amplitude changes; use for diagnostics only."}
    raise ValueError("method must be none, global_rms, or trace_rms")


def normalize_pair_to_memmap(baseline: np.ndarray, monitor: np.ndarray, output_path: str | Path, *,
                             method: str = "global_rms", inline_chunk: int = 8) -> tuple[np.ndarray, np.memmap, dict[str, object]]:
    """Write an equalized monitor copy out-of-core while leaving both inputs untouched.

    Returns the original baseline view and a newly created monitor NPY memmap. Global RMS
    uses a finite-sample first pass; trace RMS stores one scale per trace and applies it
    in inline chunks. Trace scaling is diagnostic only because it can erase real changes.
    """
    before, after = np.asanyarray(baseline), np.asanyarray(monitor)
    if before.ndim != 3 or before.shape != after.shape:
        raise ValueError("paired inputs must be matching 3D volumes")
    if method not in {"global_rms", "trace_rms"}:
        raise ValueError("out-of-core normalization supports global_rms or trace_rms")
    if inline_chunk < 1:
        raise ValueError("inline_chunk must be positive")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite normalized monitor output: {output}")
    scales: float | np.ndarray
    if method == "global_rms":
        sum_before = sum_monitor = 0.0
        count_before = count_monitor = 0
        for start in range(0, before.shape[0], inline_chunk):
            bblock = np.asarray(before[start:start+inline_chunk], dtype=np.float64)
            mblock = np.asarray(after[start:start+inline_chunk], dtype=np.float64)
            bvalid, mvalid = np.isfinite(bblock), np.isfinite(mblock)
            sum_before += float(np.square(bblock[bvalid]).sum())
            sum_monitor += float(np.square(mblock[mvalid]).sum())
            count_before += int(bvalid.sum())
            count_monitor += int(mvalid.sum())
        rms_before = float(np.sqrt(sum_before / count_before)) if count_before else 0.0
        rms_monitor = float(np.sqrt(sum_monitor / count_monitor)) if count_monitor else 0.0
        scale = rms_before / rms_monitor if rms_monitor > 0 else 1.0
        scales = scale
    else:
        trace_before = np.zeros(before.shape[:2], dtype=np.float64)
        trace_monitor = np.zeros(after.shape[:2], dtype=np.float64)
        trace_count_before = np.zeros(before.shape[:2], dtype=np.int64)
        trace_count_monitor = np.zeros(after.shape[:2], dtype=np.int64)
        for start in range(0, before.shape[0], inline_chunk):
            bblock = np.asarray(before[start:start+inline_chunk], dtype=np.float64)
            mblock = np.asarray(after[start:start+inline_chunk], dtype=np.float64)
            bv, mv = np.isfinite(bblock), np.isfinite(mblock)
            trace_before[start:start+inline_chunk] = np.where(bv, bblock*bblock, 0).sum(axis=2)
            trace_monitor[start:start+inline_chunk] = np.where(mv, mblock*mblock, 0).sum(axis=2)
            trace_count_before[start:start+inline_chunk] = bv.sum(axis=2)
            trace_count_monitor[start:start+inline_chunk] = mv.sum(axis=2)
        rms_before = np.sqrt(np.divide(trace_before, trace_count_before,
                                      out=np.zeros_like(trace_before), where=trace_count_before > 0))
        rms_monitor = np.sqrt(np.divide(trace_monitor, trace_count_monitor,
                                       out=np.zeros_like(trace_monitor), where=trace_count_monitor > 0))
        scales = np.divide(rms_before, rms_monitor, out=np.ones_like(rms_before), where=rms_monitor > 0)
    result = np.lib.format.open_memmap(output, mode="w+", dtype=np.result_type(after.dtype, np.float32), shape=after.shape)
    for start in range(0, after.shape[0], inline_chunk):
        stop = min(after.shape[0], start+inline_chunk)
        scale_slice = scales if np.isscalar(scales) else scales[start:stop, :, None]
        result[start:stop] = after[start:stop] * scale_slice
    result.flush()
    diagnostics = {"method": method, "scale": float(scales) if np.isscalar(scales) else "per_trace_array",
                   "warning": "Equalization can suppress genuine production-related amplitude changes.",
                   "source_arrays_modified": False, "output_path": str(output),
                   "rms_baseline": rms_before if method == "global_rms" else "per_trace",
                   "rms_monitor": rms_monitor if method == "global_rms" else "per_trace"}
    return before, result, diagnostics
