"""Signed, absolute, normalized and windowed RMS/NRMS attributes."""

from numbers import Real

import numpy as np
from scipy.ndimage import uniform_filter

from ..models import ChangeAttributes


def _validate_parameters(window_samples: int, epsilon: float) -> None:
    if (isinstance(window_samples, (bool, np.bool_))
            or not isinstance(window_samples, (int, np.integer))
            or not 1 <= window_samples <= np.iinfo(np.int32).max):
        raise ValueError("window_samples must be a positive integer within int32 support capacity")
    if (isinstance(epsilon, (bool, np.bool_)) or not isinstance(epsilon, Real)
            or not np.isfinite(epsilon) or epsilon <= 0):
        raise ValueError("epsilon must be finite and positive")


def calculate_attributes(baseline: np.ndarray, monitor: np.ndarray, *, window_samples: int = 9, epsilon: float = 1e-8) -> ChangeAttributes:
    before, after = np.asarray(baseline, float), np.asarray(monitor, float)
    if before.shape != after.shape or before.ndim != 3 or 0 in before.shape:
        raise ValueError("paired inputs must be matching 3D volumes")
    _validate_parameters(window_samples, epsilon)
    valid = np.isfinite(before) & np.isfinite(after)
    b, m = np.where(valid, before, 0.0), np.where(valid, after, 0.0)
    signed = np.where(valid, m - b, np.nan)
    absolute = np.abs(signed)
    normalized = np.where(valid, (m - b) / (0.5 * (np.abs(b) + np.abs(m)) + epsilon), np.nan)
    # Count actual paired samples, excluding zero padding outside the survey.
    count = np.rint(uniform_filter(valid.astype(float), size=(1, 1, window_samples),
                                  mode="constant") * window_samples).astype(np.int32)
    def mean_energy(values):
        total = uniform_filter(values * values, size=(1, 1, window_samples),
                               mode="constant") * window_samples
        return np.divide(np.maximum(total, 0.0), count,
                         out=np.full(before.shape, np.nan), where=count > 0)
    rms_b = np.sqrt(mean_energy(b))
    rms_m = np.sqrt(mean_energy(m))
    mean_d2 = mean_energy(m-b)
    rms_d = np.sqrt(np.maximum(mean_d2, 0.0))
    nrms = np.where((rms_b + rms_m) > epsilon, 200.0 * rms_d / np.maximum(rms_b + rms_m, epsilon), 0.0)
    nrms = np.where(count > 0, nrms, np.nan)
    return ChangeAttributes(signed, absolute, normalized, rms_b, rms_m, nrms, count)


def calculate_sample_attributes(baseline: np.ndarray, monitor: np.ndarray, sample_index: int, *,
                                window_samples: int = 9, epsilon: float = 1e-8) -> ChangeAttributes:
    """Calculate one sample-plane and its local NRMS window without materializing full attributes."""
    before, after = np.asarray(baseline), np.asarray(monitor)
    if before.shape != after.shape or before.ndim != 3 or 0 in before.shape:
        raise ValueError("paired inputs must be matching 3D volumes")
    if isinstance(sample_index, (bool, np.bool_)) or not isinstance(sample_index, (int, np.integer)):
        raise ValueError("sample_index must be an integer")
    if not 0 <= sample_index < before.shape[2]:
        raise IndexError("sample index outside volume")
    _validate_parameters(window_samples, epsilon)
    half = window_samples // 2
    lo, hi = max(0, sample_index-half), min(before.shape[2], sample_index+(window_samples+1)//2)
    bwin, mwin = np.asarray(before[:, :, lo:hi], float), np.asarray(after[:, :, lo:hi], float)
    valid = np.isfinite(bwin) & np.isfinite(mwin)
    count = valid.sum(axis=2, dtype=np.int32)
    safe_b, safe_m = np.where(valid, bwin, 0.0), np.where(valid, mwin, 0.0)
    b2 = np.divide(np.sum(safe_b**2, axis=2), count,
                    out=np.full(before.shape[:2], np.nan), where=count > 0)
    m2 = np.divide(np.sum(safe_m**2, axis=2), count,
                    out=np.full(before.shape[:2], np.nan), where=count > 0)
    d2 = np.divide(np.sum((safe_m-safe_b)**2, axis=2), count,
                    out=np.full(before.shape[:2], np.nan), where=count > 0)
    rms_b, rms_m = np.sqrt(b2), np.sqrt(m2)
    nrms = np.where(rms_b+rms_m > epsilon, 200*np.sqrt(np.maximum(d2, 0))/np.maximum(rms_b+rms_m, epsilon), 0.0)
    nrms = np.where(count > 0, nrms, np.nan)
    plane_b = np.asarray(before[:, :, sample_index], dtype=float)
    plane_m = np.asarray(after[:, :, sample_index], dtype=float)
    valid_plane = np.isfinite(plane_b) & np.isfinite(plane_m)
    signed = np.subtract(plane_m, plane_b, out=np.full(plane_b.shape, np.nan), where=valid_plane)
    signed = np.where(valid_plane, signed, np.nan)
    denom = 0.5*(np.abs(plane_b)+np.abs(plane_m))+epsilon
    normalized = np.where(valid_plane, signed/denom, np.nan)
    return ChangeAttributes(signed, np.abs(signed), normalized, rms_b, rms_m, nrms, count)
