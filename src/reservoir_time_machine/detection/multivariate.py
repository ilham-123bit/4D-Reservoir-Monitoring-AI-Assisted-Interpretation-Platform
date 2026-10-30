"""Robust local thresholds and interpretable multi-attribute change fusion."""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np
from scipy.ndimage import generic_filter, uniform_filter

from ..models import AnomalyDetectionResult
from ..processing.chunks import iter_chunks
from pathlib import Path


def _finite_median(window: np.ndarray) -> float:
    values = window[np.isfinite(window)]
    return float(np.median(values)) if values.size else float("nan")


def local_robust_screen(score: np.ndarray, *, footprint: tuple[int, int, int] = (5, 5, 9),
                        threshold_sigma: float = 3.5, min_valid_fraction: float = 0.5) -> AnomalyDetectionResult:
    """Return a local robust z-score and threshold mask using reflected volume edges.

    Median/MAD are computed in a 3D neighborhood. NaNs are excluded from each
    neighborhood; neighborhoods below ``min_valid_fraction`` are not detected.
    Local scores are anomaly strengths, not calibrated probabilities. The generic
    median filter is computationally costly for large footprints and should be
    chunked/out-of-core before HPC-scale use.
    """
    values = np.asarray(score, dtype=np.float32)
    if values.ndim != 3 or any(size < 1 or size % 2 == 0 for size in footprint):
        raise ValueError("score must be 3D and footprint dimensions must be positive odd integers")
    if not math.isfinite(threshold_sigma) or threshold_sigma <= 0:
        raise ValueError("threshold_sigma must be finite and positive")
    if not math.isfinite(min_valid_fraction) or not 0 < min_valid_fraction <= 1:
        raise ValueError("min_valid_fraction must be in (0,1]")
    valid = np.isfinite(values)
    median = generic_filter(values, _finite_median, size=footprint, mode="reflect")
    deviations = np.where(valid, np.abs(values - median), np.nan)
    mad = generic_filter(deviations, _finite_median, size=footprint, mode="reflect")
    valid_fraction = uniform_filter(valid.astype(np.float32), size=footprint, mode="reflect")
    scale = 1.4826 * mad
    local_z = np.divide(values - median, scale, out=np.full(values.shape, np.nan, dtype=np.float32),
                        where=valid & np.isfinite(scale) & (scale > np.finfo(np.float32).eps))
    # A locally constant neighborhood is not assigned infinite confidence. If the
    # center differs from its median, report an unbounded score cap and flag degeneracy.
    degenerate = valid & (valid_fraction >= min_valid_fraction) & (scale <= np.finfo(np.float32).eps)
    local_z[degenerate & (values > median)] = np.finfo(np.float32).max ** 0.25
    mask = valid & (valid_fraction >= min_valid_fraction) & (local_z >= threshold_sigma)
    uncertainty = np.where(valid, 1.0 - valid_fraction, np.nan).astype(np.float32)
    return AnomalyDetectionResult(local_z, mask, {
        "method": "local median/MAD robust screen",
        "footprint_voxels": list(footprint), "threshold_sigma": threshold_sigma,
        "min_valid_fraction": min_valid_fraction,
        "edge_mode": "reflect", "uncertainty_definition": "1 - finite-sample fraction in local footprint",
        "uncertainty": uncertainty, "degenerate_scale_voxel_count": int(degenerate.sum()),
        "score_is_probability": False,
    })


def local_robust_screen_chunked(score: np.ndarray, output_directory: str | Path, *,
                               chunk_shape: tuple[int, int, int] = (16, 16, 64),
                               footprint: tuple[int, int, int] = (5, 5, 9),
                               threshold_sigma: float = 3.5,
                               min_valid_fraction: float = 0.5) -> AnomalyDetectionResult:
    """Out-of-core local median/MAD with a two-radius halo for median-of-MAD exactness.

    For a footprint radius r, the local median at neighboring samples required by
    the MAD calculation also needs radius r. Therefore the input halo is 2r. Output
    score, mask and support-uncertainty maps are stored as separate NPY memmaps.
    """
    values = np.asanyarray(score)
    if values.ndim != 3 or any(size < 1 or size % 2 == 0 for size in footprint):
        raise ValueError("score must be 3D and footprint dimensions must be positive odd integers")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    paths = {name: output / f"{name}.npy" for name in ("score", "mask", "uncertainty")}
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite detector outputs: {', '.join(existing)}")
    scores = np.lib.format.open_memmap(paths["score"], mode="w+", dtype=np.float32, shape=values.shape)
    masks = np.lib.format.open_memmap(paths["mask"], mode="w+", dtype=np.bool_, shape=values.shape)
    uncertainty = np.lib.format.open_memmap(paths["uncertainty"], mode="w+", dtype=np.float32, shape=values.shape)
    radius = tuple(size // 2 for size in footprint)
    halo = tuple(2 * size for size in radius)
    for chunk in iter_chunks(values.shape, chunk_shape, halo=halo):
        result = local_robust_screen(values[chunk.expanded], footprint=footprint,
                                     threshold_sigma=threshold_sigma, min_valid_fraction=min_valid_fraction)
        core = chunk.core
        local_core = chunk.core_in_expanded
        scores[core] = result.score[local_core]
        masks[core] = result.mask[local_core]
        uncertainty[core] = result.diagnostics["uncertainty"][local_core]
    for array in (scores, masks, uncertainty):
        array.flush()
    return AnomalyDetectionResult(scores, masks, {
        "method": "chunked local median/MAD robust screen",
        "footprint_voxels": list(footprint), "threshold_sigma": threshold_sigma,
        "min_valid_fraction": min_valid_fraction, "edge_mode": "reflect",
        "chunk_shape": list(chunk_shape), "halo_voxels": list(halo),
        "uncertainty_definition": "1 - finite-sample fraction in local footprint",
        "uncertainty": uncertainty, "score_is_probability": False,
        "exactness_reference": "local_robust_screen with same SciPy edge mode and finite handling",
    })


def fuse_attributes(attributes: Mapping[str, np.ndarray], *, weights: Mapping[str, float] | None = None,
                    threshold_sigma: float = 3.5, min_attributes: int = 2) -> AnomalyDetectionResult:
    """Normalize each attribute robustly, fuse absolute robust z-scores, and screen.

    Each attribute is centered by its global finite median and scaled by 1.4826*MAD.
    Fusion is a weighted mean of absolute standardized scores over available inputs.
    The output is an uncalibrated strength; uncertainty is the across-attribute
    coefficient of variation plus a missing-attribute fraction.
    """
    if len(attributes) < 1 or min_attributes < 1 or min_attributes > len(attributes):
        raise ValueError("provide attributes and a feasible min_attributes")
    if not math.isfinite(threshold_sigma) or threshold_sigma <= 0:
        raise ValueError("threshold_sigma must be finite and positive")
    arrays = {name: np.asarray(values, dtype=np.float32) for name, values in attributes.items()}
    shapes = {array.shape for array in arrays.values()}
    if len(shapes) != 1 or len(next(iter(shapes))) != 3:
        raise ValueError("all attributes must have one identical 3D shape")
    configured_weights = dict(weights or {})
    if set(configured_weights) - set(arrays):
        raise ValueError("weights contain unknown attribute names")
    if any(not math.isfinite(float(configured_weights.get(name, 1.0))) or configured_weights.get(name, 1.0) <= 0
           for name in arrays):
        raise ValueError("attribute weights must be finite and positive")
    normalized: list[np.ndarray] = []
    valid_masks: list[np.ndarray] = []
    scales: dict[str, dict[str, float | int | None]] = {}
    for name, array in arrays.items():
        valid = np.isfinite(array)
        center = float(np.median(array[valid])) if valid.any() else 0.0
        mad = float(np.median(np.abs(array[valid] - center))) if valid.any() else 0.0
        scale = 1.4826 * mad
        score = np.divide(np.abs(array - center), scale,
                          out=np.full(array.shape, np.nan, dtype=np.float32),
                          where=valid & (scale > np.finfo(np.float32).eps))
        if scale <= np.finfo(np.float32).eps:
            score[valid] = np.where(np.abs(array[valid] - center) > 0, np.finfo(np.float32).max ** 0.25, 0.0)
        normalized.append(score)
        valid_masks.append(valid)
        scales[name] = {"median": center, "mad": mad, "valid_voxels": int(valid.sum())}
    stack = np.stack(normalized)
    mask_stack = np.stack(valid_masks)
    weight_array = np.asarray([configured_weights.get(name, 1.0) for name in arrays], dtype=np.float32)[:, None, None, None]
    effective = mask_stack * weight_array
    denominator = effective.sum(axis=0)
    score = np.divide(np.nansum(stack * effective, axis=0), denominator,
                      out=np.full(next(iter(shapes)), np.nan, dtype=np.float32), where=denominator > 0)
    support = mask_stack.sum(axis=0)
    valid_score = np.isfinite(score)
    center = float(np.median(score[valid_score])) if valid_score.any() else 0.0
    mad = float(np.median(np.abs(score[valid_score] - center))) if valid_score.any() else 0.0
    cutoff = center + threshold_sigma * 1.4826 * mad
    detected = valid_score & (support >= min_attributes) & (score >= cutoff if mad > 0 else score > center)
    # Attribute disagreement is an evidence-instability diagnostic, not probability uncertainty.
    with np.errstate(invalid="ignore", divide="ignore"):
        disagreement = np.nanstd(stack, axis=0)
    uncertainty = np.where(support > 0, disagreement / (np.abs(score) + 1e-6), np.nan).astype(np.float32)
    return AnomalyDetectionResult(score, detected, {
        "method": "global robust multi-attribute fusion",
        "attribute_names": list(arrays), "attribute_robust_scales": scales,
        "weights": {name: float(configured_weights.get(name, 1.0)) for name in arrays},
        "minimum_attribute_support": min_attributes,
        "threshold": cutoff, "threshold_sigma": threshold_sigma,
        "uncertainty_definition": "relative standard deviation of normalized attribute strengths",
        "uncertainty": uncertainty, "attribute_support": support,
        "score_is_probability": False,
    })
