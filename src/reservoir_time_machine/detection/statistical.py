"""Conventional attribute and robust statistical screeners (not probabilities)."""

import numpy as np
import math
from scipy.ndimage import gaussian_filter

from ..attributes.difference import calculate_attributes
from ..models import AnomalyDetectionResult


def statistical_screen(score: np.ndarray, *, threshold_sigma: float = 3.5) -> AnomalyDetectionResult:
    """Global median/MAD threshold over a supplied 2D or 3D attribute map."""
    if not math.isfinite(threshold_sigma) or threshold_sigma <= 0:
        raise ValueError("threshold_sigma must be finite and positive")
    values = np.asarray(score, dtype=float)
    if values.ndim not in (2, 3):
        raise ValueError("score must be a 2D slice or 3D volume")
    valid = np.isfinite(values)
    center = float(np.median(values[valid])) if valid.any() else 0.0
    mad = float(np.median(np.abs(values[valid]-center))) if valid.any() else 0.0
    cutoff = center + threshold_sigma * 1.4826 * mad
    mask = valid & (values > cutoff) if mad > 0 else valid & (values > center)
    return AnomalyDetectionResult(values, mask, {"method": "global robust MAD", "threshold": cutoff,
        "threshold_sigma": threshold_sigma, "score_is_probability": False})


def nrms_detector(baseline: np.ndarray, monitor: np.ndarray, *, threshold: float = 35.0,
                  window_samples: int = 9) -> AnomalyDetectionResult:
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and non-negative")
    attrs = calculate_attributes(baseline, monitor, window_samples=window_samples)
    score = attrs.nrms
    return AnomalyDetectionResult(score, np.isfinite(score) & (score >= threshold),
                                  {"method": "windowed NRMS threshold", "threshold_percent": threshold,
                                   "score_is_probability": False})


def difference_detector(baseline: np.ndarray, monitor: np.ndarray, *, threshold: float = 0.15) -> AnomalyDetectionResult:
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and non-negative")
    attrs = calculate_attributes(baseline, monitor)
    score = attrs.absolute_difference
    return AnomalyDetectionResult(score, np.isfinite(score) & (score >= threshold),
                                  {"method": "absolute difference threshold", "threshold": threshold,
                                   "score_is_probability": False})


def statistical_detector(baseline: np.ndarray, monitor: np.ndarray, *, threshold_sigma: float = 3.5) -> AnomalyDetectionResult:
    if not math.isfinite(threshold_sigma) or threshold_sigma <= 0:
        raise ValueError("threshold_sigma must be finite and positive")
    attrs = calculate_attributes(baseline, monitor)
    result = statistical_screen(attrs.absolute_difference, threshold_sigma=threshold_sigma)
    return AnomalyDetectionResult(result.score, result.mask, result.diagnostics)


def spatial_detector(baseline: np.ndarray, monitor: np.ndarray, *, threshold_sigma: float = 3.5,
                     spatial_sigma: float = 1.0) -> AnomalyDetectionResult:
    """Screen a lightly smoothed absolute difference to favor spatial coherence."""
    if not math.isfinite(threshold_sigma) or threshold_sigma <= 0 or not math.isfinite(spatial_sigma) or spatial_sigma <= 0:
        raise ValueError("threshold_sigma and spatial_sigma must be finite and positive")
    raw = calculate_attributes(baseline, monitor).absolute_difference
    finite = np.isfinite(raw)
    spatial_weights = gaussian_filter(finite.astype(float), sigma=(spatial_sigma, spatial_sigma, 0.0))
    smooth = gaussian_filter(np.nan_to_num(raw), sigma=(spatial_sigma, spatial_sigma, 0.0))
    score = np.divide(smooth, spatial_weights, out=np.zeros_like(smooth), where=spatial_weights > 1e-8)
    if not finite.any():
        return AnomalyDetectionResult(score, np.zeros(raw.shape, dtype=bool),
                                      {"method": "spatially smoothed robust difference", "threshold": None,
                                       "score_is_probability": False})
    center = float(np.median(score[finite]))
    mad = float(np.median(np.abs(score[finite] - center)))
    cutoff = center + threshold_sigma * 1.4826 * mad
    mask = finite & (score > cutoff) if mad > 0 else finite & (score > center)
    return AnomalyDetectionResult(score, mask, {"method": "spatially smoothed robust difference",
        "threshold": cutoff, "spatial_sigma_voxels": spatial_sigma, "score_is_probability": False})
