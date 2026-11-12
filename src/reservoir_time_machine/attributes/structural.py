"""Bounded, display-scale seismic waveform-continuity diagnostics.

The returned similarity is an attribute proxy, not a calibrated coherence,
fault probability, lithology classifier, or reservoir-connectivity result.
"""
from __future__ import annotations

import numpy as np


def local_waveform_similarity(
    volume: np.ndarray,
    inline: int,
    crossline: int,
    sample: int,
    *,
    radius: int = 24,
    window_samples: int = 9,
    min_window_support: int = 7,
    excluded_trace_mask: np.ndarray | None = None,
) -> dict:
    """Compute a local map of absolute trace-to-neighbour waveform correlation.

    Only the requested 3-D context cube is accessed. Adjacent inline/crossline
    trace pairs are compared over the clipped window around ``sample``. An
    edge contributes to both endpoint cells only when its paired sample count
    and variance are sufficient. The result retains its source-array bounds.

    ``excluded_trace_mask`` is a 2-D boolean mask where true means the trace
    must not contribute (for example, a missing paired Baseline-grid trace).
    """
    data = np.asarray(volume)
    if data.ndim != 3 or any(length < 1 for length in data.shape):
        raise ValueError("volume must be a nonempty inline/crossline/sample cube")
    coordinates = (inline, crossline, sample)
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
           for value in coordinates):
        raise ValueError("selected source coordinates must be integer indices")
    ni, nx, ns = data.shape
    if not (0 <= inline < ni and 0 <= crossline < nx and 0 <= sample < ns):
        raise IndexError("selected source coordinate is outside the volume")
    if (isinstance(radius, (bool, np.bool_)) or not isinstance(radius, (int, np.integer))
            or not 0 <= radius <= 256):
        raise ValueError("radius must be an integer from 0 through 256")
    if (isinstance(window_samples, (bool, np.bool_)) or not isinstance(window_samples, (int, np.integer))
            or window_samples < 3 or window_samples % 2 != 1):
        raise ValueError("window_samples must be an odd integer of at least 3")
    if (isinstance(min_window_support, (bool, np.bool_))
            or not isinstance(min_window_support, (int, np.integer))
            or not 3 <= min_window_support <= window_samples):
        raise ValueError("min_window_support must be between 3 and window_samples")
    excluded = None if excluded_trace_mask is None else np.asarray(excluded_trace_mask, dtype=bool)
    if excluded is not None and excluded.shape != data.shape[:2]:
        raise ValueError("excluded_trace_mask must match the inline/crossline grid")

    i0, i1 = max(0, inline - radius), min(ni, inline + radius + 1)
    x0, x1 = max(0, crossline - radius), min(nx, crossline + radius + 1)
    half = window_samples // 2
    s0, s1 = max(0, sample - half), min(ns, sample + half + 1)
    cube = np.asarray(data[i0:i1, x0:x1, s0:s1], dtype=np.float64)
    trace_valid = np.isfinite(cube).all(axis=2)
    if excluded is not None:
        trace_valid &= ~excluded[i0:i1, x0:x1]

    total = np.zeros(trace_valid.shape, dtype=np.float64)
    support = np.zeros(trace_valid.shape, dtype=np.uint8)
    for di, dx in ((1, 0), (0, 1)):
        ai = slice(0, cube.shape[0] - di) if di else slice(None)
        bi = slice(di, cube.shape[0]) if di else slice(None)
        ax = slice(0, cube.shape[1] - dx) if dx else slice(None)
        bx = slice(dx, cube.shape[1]) if dx else slice(None)
        left, right = cube[ai, ax, :], cube[bi, bx, :]
        valid_pair = trace_valid[ai, ax] & trace_valid[bi, bx]
        count = np.full(valid_pair.shape, cube.shape[2], dtype=np.float64)
        sum_left = left.sum(axis=2)
        sum_right = right.sum(axis=2)
        sum_left_sq = np.einsum("ijk,ijk->ij", left, left)
        sum_right_sq = np.einsum("ijk,ijk->ij", right, right)
        sum_cross = np.einsum("ijk,ijk->ij", left, right)
        covariance = sum_cross - sum_left * sum_right / count
        variance_left = np.maximum(sum_left_sq - sum_left * sum_left / count, 0.0)
        variance_right = np.maximum(sum_right_sq - sum_right * sum_right / count, 0.0)
        denominator = np.sqrt(variance_left * variance_right)
        eligible = valid_pair & (cube.shape[2] >= min_window_support) & (denominator > 1e-12)
        correlation = np.divide(covariance, denominator,
                                out=np.zeros_like(covariance), where=denominator > 1e-12)
        similarity = np.where(eligible, np.clip(np.abs(correlation), 0.0, 1.0), 0.0)
        total[ai, ax] += similarity
        total[bi, bx] += similarity
        support[ai, ax] += eligible.astype(np.uint8)
        support[bi, bx] += eligible.astype(np.uint8)

    result = np.divide(total, support, out=np.full(total.shape, np.nan), where=support > 0)
    return {
        "similarity": result.astype(np.float32),
        "neighbor_pair_count": support,
        "array_bounds": ((i0, i1), (x0, x1)),
        "sample_bounds": (s0, s1),
        "center_local_index": (inline - i0, crossline - x0),
        "meaning": "mean absolute Pearson correlation with valid face-adjacent traces",
        "interpretation_limit": "continuity screening proxy only; not calibrated coherence or fault/connectivity probability",
    }
