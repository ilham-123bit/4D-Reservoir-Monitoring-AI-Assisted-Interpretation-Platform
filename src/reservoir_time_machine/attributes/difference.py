"""Signed, absolute, normalized and windowed RMS/NRMS attributes."""

import numpy as np
from scipy.ndimage import uniform_filter
import math

from ..models import ChangeAttributes


def calculate_attributes(baseline: np.ndarray, monitor: np.ndarray, *, window_samples: int = 9, epsilon: float = 1e-8) -> ChangeAttributes:
    before, after = np.asarray(baseline, float), np.asarray(monitor, float)
    if before.shape != after.shape or before.ndim != 3:
        raise ValueError("paired inputs must be matching 3D volumes")
    if window_samples < 1 or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("window_samples and epsilon must be positive")
    valid = np.isfinite(before) & np.isfinite(after)
    b, m = np.where(valid, before, 0.0), np.where(valid, after, 0.0)
    signed = np.where(valid, m - b, np.nan)
    absolute = np.abs(signed)
    normalized = np.where(valid, (m - b) / (0.5 * (np.abs(b) + np.abs(m)) + epsilon), np.nan)
    count = uniform_filter(valid.astype(float), size=(1, 1, window_samples), mode="constant")
    rms_b = np.sqrt(uniform_filter(b*b, size=(1, 1, window_samples), mode="constant") / np.maximum(count, epsilon))
    rms_m = np.sqrt(uniform_filter(m*m, size=(1, 1, window_samples), mode="constant") / np.maximum(count, epsilon))
    mean_d2 = uniform_filter((m-b)**2, size=(1, 1, window_samples), mode="constant") / np.maximum(count, epsilon)
    rms_d = np.sqrt(np.maximum(mean_d2, 0.0))
    nrms = np.where((rms_b + rms_m) > epsilon, 200.0 * rms_d / np.maximum(rms_b + rms_m, epsilon), 0.0)
    nrms = np.where(count > 0, nrms, np.nan)
    return ChangeAttributes(signed, absolute, normalized, rms_b, rms_m, nrms)


def calculate_sample_attributes(baseline: np.ndarray, monitor: np.ndarray, sample_index: int, *,
                                window_samples: int = 9, epsilon: float = 1e-8) -> ChangeAttributes:
    """Calculate one sample-plane and its local NRMS window without materializing full attributes."""
    before, after = np.asarray(baseline), np.asarray(monitor)
    if before.shape != after.shape or before.ndim != 3:
        raise ValueError("paired inputs must be matching 3D volumes")
    if not 0 <= sample_index < before.shape[2]:
        raise IndexError("sample index outside volume")
    if window_samples < 1 or epsilon <= 0:
        raise ValueError("window_samples and epsilon must be positive")
    half = window_samples // 2
    lo, hi = max(0, sample_index-half), min(before.shape[2], sample_index+(window_samples+1)//2)
    bwin, mwin = np.asarray(before[:, :, lo:hi], float), np.asarray(after[:, :, lo:hi], float)
    valid = np.isfinite(bwin) & np.isfinite(mwin)
    count = valid.sum(axis=2)
    b2 = np.divide(np.nansum(np.where(valid, bwin, np.nan)**2, axis=2), count,
                    out=np.full(before.shape[:2], np.nan), where=count > 0)
    m2 = np.divide(np.nansum(np.where(valid, mwin, np.nan)**2, axis=2), count,
                    out=np.full(before.shape[:2], np.nan), where=count > 0)
    d2 = np.divide(np.nansum(np.where(valid, bwin-mwin, np.nan)**2, axis=2), count,
                    out=np.full(before.shape[:2], np.nan), where=count > 0)
    rms_b, rms_m = np.sqrt(b2), np.sqrt(m2)
    nrms = np.where(rms_b+rms_m > epsilon, 200*np.sqrt(np.maximum(d2, 0))/np.maximum(rms_b+rms_m, epsilon), 0.0)
    nrms = np.where(count > 0, nrms, np.nan)
    signed = np.asarray(monitor[:, :, sample_index], float) - np.asarray(baseline[:, :, sample_index], float)
    valid_plane = np.isfinite(signed)
    signed = np.where(valid_plane, signed, np.nan)
    denom = 0.5*(np.abs(baseline[:, :, sample_index])+np.abs(monitor[:, :, sample_index]))+epsilon
    normalized = np.where(valid_plane, signed/denom, np.nan)
    return ChangeAttributes(signed, np.abs(signed), normalized, rms_b, rms_m, nrms)
