"""Trace-wise bounded lag estimation and optional integer-sample correction."""

import numpy as np
from dataclasses import dataclass
from scipy.ndimage import map_coordinates

from ..models import RegistrationResult


@dataclass(frozen=True)
class FractionalRegistrationResult:
    """Trace-wise fractional lag estimates; positive values mean monitor is later."""
    shifts_samples: np.ndarray
    correlation: np.ndarray
    accepted: np.ndarray
    registered_monitor: np.ndarray
    failure_reason: dict[str, str]


def estimate_fractional_temporal_registration(baseline: np.ndarray, monitor: np.ndarray, *,
                                               max_lag: int = 4, apply: bool = False,
                                               min_correlation: float = 0.2,
                                               minimum_valid_fraction: float = 0.8) -> FractionalRegistrationResult:
    """Estimate bounded fractional trace lags by local parabolic interpolation.

    Correlation is computed for integer lags over finite overlapping samples. A
    three-point parabola refines an interior maximum by at most half a sample.
    Application uses linear interpolation and finite-support weighting; unsupported
    edge samples are NaN. The operation is a derived correction and never changes inputs.
    """
    before, after = np.asarray(baseline, dtype=np.float64), np.asarray(monitor, dtype=np.float64)
    if before.ndim != 3 or before.shape != after.shape:
        raise ValueError("paired inputs must be matching 3D volumes")
    ni, nc, ns = before.shape
    if not isinstance(max_lag, int) or max_lag < 1 or max_lag >= ns:
        raise ValueError("max_lag must be an integer in [1, sample_count)")
    if not np.isfinite(min_correlation) or not 0 <= min_correlation <= 1:
        raise ValueError("min_correlation must be finite and in [0,1]")
    if not np.isfinite(minimum_valid_fraction) or not 0 < minimum_valid_fraction <= 1:
        raise ValueError("minimum_valid_fraction must be in (0,1]")
    shifts = np.full((ni, nc), np.nan, dtype=np.float32)
    quality = np.full((ni, nc), np.nan, dtype=np.float32)
    accepted = np.zeros((ni, nc), dtype=bool)
    reasons: dict[str, str] = {}
    registered = after.copy() if apply else after
    minimum_overlap = max(8, int(np.ceil(ns * minimum_valid_fraction)))
    lags = np.arange(-max_lag, max_lag + 1)
    for i in range(ni):
        for j in range(nc):
            a, b = before[i, j], after[i, j]
            correlations = np.full(lags.shape, np.nan, dtype=np.float64)
            for index, lag in enumerate(lags):
                if lag >= 0:
                    av, bv = a[:ns-lag], b[lag:]
                else:
                    av, bv = a[-lag:], b[:ns+lag]
                valid = np.isfinite(av) & np.isfinite(bv)
                if int(valid.sum()) < minimum_overlap:
                    continue
                av, bv = av[valid], bv[valid]
                av, bv = av-av.mean(), bv-bv.mean()
                denom = np.linalg.norm(av) * np.linalg.norm(bv)
                if denom > np.finfo(float).eps:
                    correlations[index] = float(np.dot(av, bv) / denom)
            if not np.isfinite(correlations).any():
                reasons[f"{i},{j}"] = "no lag has enough finite, non-zero-energy overlap"
                continue
            peak = int(np.nanargmax(correlations))
            peak_corr = float(correlations[peak])
            if peak_corr < min_correlation:
                reasons[f"{i},{j}"] = "best correlation below configured minimum"
                continue
            fractional_offset = 0.0
            if 0 < peak < len(lags)-1 and np.isfinite(correlations[peak-1:peak+2]).all():
                left, center, right = correlations[peak-1:peak+2]
                curvature = left - 2.0*center + right
                if curvature < -1e-12:
                    fractional_offset = float(np.clip(0.5*(left-right)/curvature, -0.5, 0.5))
            lag_estimate = float(lags[peak] + fractional_offset)
            shifts[i, j], quality[i, j], accepted[i, j] = lag_estimate, peak_corr, True
            if apply:
                positions = np.arange(ns, dtype=np.float64) + lag_estimate
                finite = np.isfinite(b)
                numerator = map_coordinates(np.where(finite, b, 0.0), [positions], order=1,
                                            mode="constant", cval=0.0, prefilter=False)
                support = map_coordinates(finite.astype(float), [positions], order=1,
                                          mode="constant", cval=0.0, prefilter=False)
                corrected = np.divide(numerator, support, out=np.full(ns, np.nan),
                                      where=support >= minimum_valid_fraction)
                registered[i, j] = corrected
    return FractionalRegistrationResult(shifts, quality, accepted, registered, reasons)


def estimate_temporal_registration(baseline: np.ndarray, monitor: np.ndarray, *, max_lag: int = 4,
                                   apply: bool = False, min_correlation: float = 0.2) -> RegistrationResult:
    before, after = np.asarray(baseline, float), np.asarray(monitor, float)
    if before.shape != after.shape or before.ndim != 3:
        raise ValueError("paired inputs must be matching 3D volumes")
    ni, nc, ns = before.shape
    if not isinstance(max_lag, int) or max_lag < 0 or max_lag >= ns:
        raise ValueError("max_lag must be a non-negative integer smaller than sample count")
    if not np.isfinite(min_correlation) or not 0.0 <= min_correlation <= 1.0:
        raise ValueError("min_correlation must be finite and in [0,1]")
    shifts = np.zeros((ni, nc), dtype=np.int16)
    quality = np.full((ni, nc), np.nan, dtype=np.float32)
    accepted = np.zeros((ni, nc), dtype=bool)
    failure_reason: dict[str, str] = {}
    # Avoid a second full-volume allocation for diagnostics-only registration.
    registered = after.copy() if apply else after
    for i in range(ni):
        for j in range(nc):
            a, b = before[i, j], after[i, j]
            valid = np.isfinite(a) & np.isfinite(b)
            if valid.sum() < max(8, ns // 4):
                failure_reason[f"{i},{j}"] = "insufficient finite overlapping samples"
                continue
            best_corr, best_lag = -np.inf, 0
            for lag in range(-max_lag, max_lag + 1):
                if lag >= 0:
                    av, bv = a[:ns-lag or ns], b[lag:]
                else:
                    av, bv = a[-lag:], b[:ns+lag]
                pair_valid = np.isfinite(av) & np.isfinite(bv)
                if int(pair_valid.sum()) < max(8, ns // 4):
                    continue
                av, bv = av[pair_valid], bv[pair_valid]
                av, bv = av - av.mean(), bv - bv.mean()
                denom = np.linalg.norm(av) * np.linalg.norm(bv)
                if denom <= np.finfo(float).eps:
                    continue
                corr = float(np.dot(av, bv) / denom)
                if corr > best_corr + 1e-12 or (abs(corr-best_corr) <= 1e-12 and abs(lag) < abs(best_lag)):
                    best_corr, best_lag = corr, lag
            if not np.isfinite(best_corr):
                failure_reason[f"{i},{j}"] = "no lag had sufficient finite, non-zero-energy overlap"
                continue
            quality[i, j] = best_corr
            if best_corr < min_correlation:
                failure_reason[f"{i},{j}"] = f"best correlation {best_corr:.6g} is below minimum {min_correlation:.6g}"
                continue
            shifts[i, j], accepted[i, j] = best_lag, True
            if apply and best_lag:
                # Positive lag means monitor events occur later. Shift with NaN edge padding;
                # never wrap samples or extrapolate values beyond measured support.
                shifted = np.full(ns, np.nan, dtype=float)
                if best_lag > 0:
                    shifted[:ns-best_lag] = after[i, j, best_lag:]
                else:
                    shifted[-best_lag:] = after[i, j, :ns+best_lag]
                registered[i, j] = shifted
    return RegistrationResult(shifts, quality, registered, accepted, failure_reason)
