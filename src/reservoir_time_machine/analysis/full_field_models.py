"""Offline, resumable classical-ML scoring over validated paired attributes.

All model scores remain detector-specific and uncalibrated. Spatial reference and
threshold populations are supplied explicitly; this module never calls a score a
probability or interprets it as fluid movement.
"""

from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from contextlib import nullcontext
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import time

import numpy as np

from ..ai.detectors import (GaussianMixtureDetector, IsolationForestDetector,
                            LinearAutoencoderDetector, MahalanobisDetector)
from ..detection.contracts import DetectionRequest
from ..processing.chunks import iter_chunks
from ..processing.full_field import (_append, _atomic_json, _canonical, _hash_bytes,
                                     _hash_file, _read_journal)
from .features import FEATURE_NAMES, robust_scaling

MODEL_VERSION = "classical-field-screen-1"
MIN_REFERENCE_ROWS = 64
MIN_CALIBRATION_ROWS = 32
MAX_POPULATION_ROWS = 50_000


@dataclass(frozen=True)
class FeaturePopulation:
    features: np.ndarray
    population_id: str
    bounds: tuple[int, int, int, int]
    source_identity: str
    eligible_voxels_scanned: int
    feature_sha256: str
    window_samples: int
    sample_bounds: tuple[int, int]


def _nonoverlap(a, b) -> bool:
    ai0, ai1, ax0, ax1 = a
    bi0, bi1, bx0, bx1 = b
    return ai1 <= bi0 or bi1 <= ai0 or ax1 <= bx0 or bx1 <= ax0


def _validate_bounds(bounds, shape):
    if len(bounds) != 4:
        raise ValueError("spatial bounds must be (inline_start, inline_stop, crossline_start, crossline_stop)")
    i0, i1, x0, x1 = map(int, bounds)
    if not (0 <= i0 < i1 <= shape[0] and 0 <= x0 < x1 <= shape[1]):
        raise ValueError("population bounds lie outside the survey grid")
    return (i0, i1, x0, x1)


def validate_spatial_calibration_design(reference_bounds, calibration_bounds, *,
                                        shape, missing_trace_mask,
                                        buffer_grid_cells: int,
                                        minimum_occupied_fraction: float,
                                        review: dict) -> dict:
    """Fail-closed validation for unlabeled, spatially separated populations.

    The grid buffer limits local spatial leakage; it does not establish that
    either region is geologically unchanged. Region choice must be documented
    as score-blind and based on survey geometry/validity, not low NRMS.
    """
    shape = tuple(shape)
    reference = _validate_bounds(reference_bounds, shape)
    calibration = _validate_bounds(calibration_bounds, shape)
    if (isinstance(buffer_grid_cells, bool) or not isinstance(buffer_grid_cells, int)
            or buffer_grid_cells < 1):
        raise ValueError("calibration design requires a positive grid-cell separation buffer")
    if (not np.isfinite(minimum_occupied_fraction)
            or not 0.5 <= float(minimum_occupied_fraction) <= 1):
        raise ValueError("minimum occupied fraction must be in [0.5, 1]")
    if np.shape(missing_trace_mask) != shape[:2]:
        raise ValueError("calibration occupancy mask differs from the survey grid")
    if not isinstance(review, dict):
        raise ValueError("a calibration design record is required")
    exploratory = review.get("execution_mode") == "unlabeled_exploratory_screening"
    required = {"selection_basis", "ground_truth_status", "physical_registration_status",
                "occupancy_proposal"}
    if exploratory:
        required.add("limitations")
        if review.get("record_status") != "not_a_geological_approval":
            raise ValueError("exploratory design must not claim scientific approval")
        limitations = set(review.get("limitations", []))
        if not {"physical_4d_registration_unverified", "sample_origin_unverified",
                "geological_comparability_uncertain", "acquisition_processing_comparability_unverified"}.issubset(limitations):
            raise ValueError("exploratory design must retain registration, sample, geology, and acquisition limitations")
    else:
        required.update({"review_id", "reviewed_by"})
    string_fields = required - {"occupancy_proposal", "limitations"}
    if (required - set(review) or any(not isinstance(review.get(k), str) or not review[k].strip()
                                      for k in string_fields)
            or not isinstance(review.get("occupancy_proposal"), dict)):
        raise ValueError("calibration design record is incomplete")
    if review["ground_truth_status"] != "unlabeled_screening_only":
        raise ValueError("unlabeled seismic populations cannot be declared geological ground truth")
    if review["physical_registration_status"] != "unverified":
        raise ValueError("this analysis requires physical registration to remain explicitly unverified")
    allowed_basis = "grid_partition_selected_without_inspecting_anomaly_scores"
    if review["selection_basis"] != allowed_basis:
        raise ValueError("regions must be selected without inspecting anomaly scores")
    proposal = review["occupancy_proposal"]
    if (not isinstance(proposal, dict)
            or proposal.get("schema") != "rtm-calibration-region-proposal-1"
            or proposal.get("status") != "requires_human_review"
            or proposal.get("reference_bounds") != list(reference)
            or proposal.get("calibration_bounds") != list(calibration)
            or proposal.get("selection_basis", "").find("occupancy only") < 0):
        raise ValueError("review must include the matching score-blind occupancy proposal")

    # Minimum Euclidean distance between two axis-aligned, half-open grid boxes.
    gaps = []
    for a0, a1, b0, b1 in ((reference[0], reference[1], calibration[0], calibration[1]),
                           (reference[2], reference[3], calibration[2], calibration[3])):
        gaps.append(max(0, b0 - a1, a0 - b1))
    separation = float(np.hypot(*gaps))
    if separation < buffer_grid_cells:
        raise ValueError(f"reference/calibration regions need >= {buffer_grid_cells} grid-cell buffer; got {separation:g}")

    mask = np.asarray(missing_trace_mask, dtype=bool)
    occupancy = {}
    for name, bounds in (("reference", reference), ("calibration", calibration)):
        i0, i1, x0, x1 = bounds
        occupied = int((~mask[i0:i1, x0:x1]).sum(dtype=np.int64))
        total = (i1 - i0) * (x1 - x0)
        fraction = occupied / total
        if fraction < minimum_occupied_fraction:
            raise ValueError(f"{name} region occupied-trace fraction {fraction:.4f} is below the reviewed minimum")
        occupancy[name] = {"occupied_traces": occupied, "grid_cells": total,
                           "occupied_fraction": fraction, "bounds": list(bounds)}
    return {"schema": "rtm-spatial-calibration-design-1", "review": dict(review),
            "minimum_buffer_grid_cells": buffer_grid_cells,
            "measured_region_separation_grid_cells": separation,
            "minimum_occupied_fraction": float(minimum_occupied_fraction),
            "regions": occupancy,
            "interpretation": "unlabeled spatial reference and threshold populations; not unchanged-ground-truth labels"}


def propose_spatial_regions_from_occupancy(missing_trace_mask, *, region_shape=(128, 128),
                                           stride=(64, 64), buffer_grid_cells=32,
                                           minimum_occupied_fraction=0.95) -> dict:
    """Propose separated rectangles using occupancy only, never seismic scores.

    This is a review aid, not an automatic calibration approval. It deliberately
    does not inspect NRMS, amplitudes, attributes, or model results.
    """
    mask = np.asarray(missing_trace_mask, dtype=bool)
    if mask.ndim != 2 or len(region_shape) != 2 or len(stride) != 2:
        raise ValueError("occupancy proposal requires a 2D mask and 2D window/stride")
    hi, hx = map(int, region_shape); si, sx = map(int, stride)
    if min(hi, hx, si, sx, buffer_grid_cells) < 1 or not 0.5 <= minimum_occupied_fraction <= 1:
        raise ValueError("invalid occupancy proposal window, stride, buffer, or threshold")
    if hi > mask.shape[0] or hx > mask.shape[1]:
        raise ValueError("calibration proposal windows exceed the grid")
    integral = np.pad((~mask).astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    candidates = []
    for i0 in range(0, mask.shape[0] - hi + 1, si):
        for x0 in range(0, mask.shape[1] - hx + 1, sx):
            i1, x1 = i0 + hi, x0 + hx
            occupied = int(integral[i1, x1] - integral[i0, x1] - integral[i1, x0] + integral[i0, x0])
            fraction = occupied / (hi * hx)
            if fraction >= minimum_occupied_fraction:
                candidates.append(((i0, i1, x0, x1), fraction, occupied))
    best = None
    for left_index, left in enumerate(candidates):
        for right in candidates[left_index + 1:]:
            gaps = (max(0, right[0][0] - left[0][1], left[0][0] - right[0][1]),
                    max(0, right[0][2] - left[0][3], left[0][2] - right[0][3]))
            distance = float(np.hypot(*gaps))
            if distance < buffer_grid_cells:
                continue
            score = (min(left[1], right[1]), left[1] + right[1], distance,
                     tuple(-v for v in left[0]), tuple(-v for v in right[0]))
            if best is None or score > best[0]:
                best = (score, left, right)
    if best is None:
        raise ValueError("no pair of occupancy-eligible regions satisfies the spatial buffer")
    _, first, second = best
    return {"schema": "rtm-calibration-region-proposal-1", "status": "requires_human_review",
            "selection_basis": "Baseline-grid occupancy only; no amplitude, NRMS, attribute, detector, or finding scores inspected",
            "minimum_occupied_fraction": float(minimum_occupied_fraction),
            "minimum_buffer_grid_cells": int(buffer_grid_cells),
            "measured_separation_grid_cells": best[0][2],
            "reference_bounds": list(first[0]), "calibration_bounds": list(second[0]),
            "reference_occupied_fraction": first[1], "calibration_occupied_fraction": second[1],
            "limitation": "These unlabeled spatial populations are not verified unchanged geology; sample domain and physical registration remain unresolved."}


def _feature_block(attributes, missing, region, min_support_fraction, window_samples, *,
                   sample_origin=0, global_sample_count=None, timing_metrics=None):
    read_started = time.perf_counter() if timing_metrics is not None else None
    # Copies in profiling mode make page-fault/data access time observable. Normal
    # production scoring retains the existing bounded memmap-view behavior.
    copy = timing_metrics is not None
    def read(name, dtype):
        return np.array(attributes[name][region], dtype=dtype, copy=True) if copy else np.asarray(attributes[name][region], dtype=dtype)
    signed = read("signed_difference", np.float32)
    absolute = read("absolute_difference", np.float32)
    normalized = read("normalized_difference", np.float32)
    rms_b = read("rms_baseline", np.float32)
    rms_m = read("rms_monitor", np.float32)
    nrms = read("nrms", np.float32)
    support = read("paired_sample_count", None)
    spatial_missing = np.array(missing[region[:2]], dtype=bool, copy=copy)
    if timing_metrics is not None:
        timing_metrics["input_read_seconds"] = timing_metrics.get("input_read_seconds", 0.0) + time.perf_counter() - read_started
    prep_started = time.perf_counter() if timing_metrics is not None else None
    energy = rms_b + rms_m
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        cube = np.stack((normalized, nrms / 200.0,
                         np.log((rms_m + 1e-8) / (rms_b + 1e-8)),
                         absolute / (energy + 1e-8)), axis=-1)
    valid = (np.isfinite(cube).all(axis=-1) & np.isfinite(signed) & np.isfinite(energy)
             & (energy > 1e-8) & (support > 0) & ~spatial_missing[..., None])
    total_samples = int(global_sample_count if global_sample_count is not None
                        else np.shape(attributes["signed_difference"])[2])
    sample_slice = region[2]
    centers = np.arange((sample_slice.start or 0) + sample_origin,
                        (sample_slice.stop or np.shape(attributes["signed_difference"])[2]) + sample_origin)
    left, right = window_samples // 2, (window_samples - 1) // 2
    available = np.minimum(total_samples, centers + right + 1) - np.maximum(0, centers - left)
    valid &= support >= np.ceil(min_support_fraction * available)[None, None, :]
    cube[~valid] = np.nan
    if timing_metrics is not None:
        timing_metrics["preprocessing_seconds"] = timing_metrics.get("preprocessing_seconds", 0.0) + time.perf_counter() - prep_started
        timing_metrics["total_positions"] = timing_metrics.get("total_positions", 0) + int(valid.size)
        timing_metrics["valid_positions"] = timing_metrics.get("valid_positions", 0) + int(valid.sum())
    return cube, valid


def sample_spatial_population(attributes, missing_trace_mask, *, bounds, population_id: str,
                              source_identity: str, max_rows: int = 10_000,
                              chunk_shape=(16, 32, 128), min_support_fraction: float = 0.8,
                              window_samples: int = 9, seed: int = 2026,
                              sample_bounds: tuple[int, int] | None = None,
                              timing_metrics: dict | None = None) -> FeaturePopulation:
    """Reservoir-sample feature rows from an explicitly declared spatial region."""
    required = {"signed_difference", "absolute_difference", "normalized_difference",
                "rms_baseline", "rms_monitor", "nrms", "paired_sample_count"}
    if not required.issubset(attributes):
        raise ValueError("validated full-field attributes are incomplete")
    shape = tuple(np.shape(attributes["signed_difference"]))
    if len(shape) != 3 or any(np.shape(attributes[name]) != shape for name in required):
        raise ValueError("all full-field attributes must share one 3D shape")
    if np.shape(missing_trace_mask) != shape[:2]:
        raise ValueError("missing-trace mask shape differs from attribute grid")
    bounds = _validate_bounds(bounds, shape)
    sample_bounds = (0, shape[2]) if sample_bounds is None else tuple(map(int, sample_bounds))
    if len(sample_bounds) != 2 or not (0 <= sample_bounds[0] < sample_bounds[1] <= shape[2]):
        raise ValueError("sample bounds must be a nonempty half-open range within the source axis")
    if not population_id.strip() or len(source_identity) != 64:
        raise ValueError("population ID and source SHA-256 identity are required")
    if (not 1 <= max_rows <= MAX_POPULATION_ROWS or not 0 < min_support_fraction <= 1
            or isinstance(window_samples, bool) or int(window_samples) < 1):
        raise ValueError("population sampling budget or support threshold is invalid")
    window_samples = int(window_samples)
    i0, i1, x0, x1 = bounds
    view_shape = (i1-i0, x1-x0, sample_bounds[1] - sample_bounds[0])
    rng = np.random.default_rng(seed)
    reservoir = np.empty((max_rows, len(FEATURE_NAMES)), dtype=np.float32)
    eligible = 0
    for chunk in iter_chunks(view_shape, chunk_shape):
        local = chunk.core
        global_region = (slice(i0 + local[0].start, i0 + local[0].stop),
                         slice(x0 + local[1].start, x0 + local[1].stop),
                         slice(sample_bounds[0] + local[2].start, sample_bounds[0] + local[2].stop))
        cube, valid = _feature_block(attributes, missing_trace_mask, global_region,
                                     min_support_fraction, window_samples,
                                     timing_metrics=timing_metrics)
        rows = cube[valid]
        for row in rows:
            if eligible < max_rows:
                reservoir[eligible] = row
            else:
                replacement = int(rng.integers(0, eligible + 1))
                if replacement < max_rows:
                    reservoir[replacement] = row
            eligible += 1
    selected = reservoir[:min(eligible, max_rows)].copy()
    if len(selected) < MIN_REFERENCE_ROWS:
        raise ValueError(f"population {population_id!r} has only {len(selected)} eligible rows")
    digest = hashlib.sha256(np.ascontiguousarray(selected).tobytes()).hexdigest()
    return FeaturePopulation(selected, population_id, bounds, source_identity, eligible,
                             digest, window_samples, sample_bounds)


def sample_attribute_reference(array, *, bounds, max_rows=10_000, chunk_shape=(64, 128), seed=2026):
    """Bounded deterministic sample of a scalar attribute over spatial bounds."""
    values = np.asanyarray(array)
    if values.ndim != 3:
        raise ValueError("attribute reference must be a 3D map")
    i0, i1, x0, x1 = _validate_bounds(bounds, values.shape)
    if not 1 <= max_rows <= MAX_POPULATION_ROWS:
        raise ValueError("attribute sampling budget is invalid")
    rng = np.random.default_rng(seed)
    reservoir = np.empty(max_rows, dtype=np.float64)
    count = 0
    for i in range(i0, i1, int(chunk_shape[0])):
        for x in range(x0, x1, int(chunk_shape[1])):
            block = np.asarray(values[i:min(i+chunk_shape[0], i1),
                                      x:min(x+chunk_shape[1], x1), :], dtype=np.float64)
            finite = block[np.isfinite(block)]
            for value in finite:
                if count < max_rows:
                    reservoir[count] = value
                else:
                    index = int(rng.integers(0, count + 1))
                    if index < max_rows:
                        reservoir[index] = value
                count += 1
    result = reservoir[:min(count, max_rows)].copy()
    if result.size < MIN_CALIBRATION_ROWS:
        raise ValueError("statistical reference contains too few finite values")
    return result


def _score_one_chunk(attributes, missing_trace_mask, region, *, models, thresholds,
                     source_identity, fit_record_sha256, min_support_fraction,
                     window_samples, sample_origin, global_sample_count, existing_event=None,
                     output_arrays=None):
    """Read and score one disjoint chunk; workers never write shared outputs."""
    started = time.perf_counter()
    raw = {name: np.asarray(value[region]) for name, value in attributes.items()}
    missing = np.asarray(missing_trace_mask[region[:2]], dtype=bool)
    input_hash = _hash_bytes(*(raw[name] for name in sorted(raw)), missing)
    read_seconds = time.perf_counter() - started
    if existing_event is not None:
        if existing_event["input_sha256"] != input_hash:
            raise ValueError(f"attribute input changed since model checkpoint: {','.join(str(s.start) for s in region)}")
        for filename, digest in existing_event["output_sha256"].items():
            if _hash_bytes(np.asarray(output_arrays[filename][region])) != digest:
                raise ValueError(f"model output block checksum mismatch: {filename} / {','.join(str(s.start) for s in region)}")
        return {"input_sha256": input_hash, "skip": True,
                "input_read_seconds": read_seconds, "preprocessing_seconds": 0.0,
                "model_scoring_seconds": {}}

    # Use the already-read block for feature construction. This avoids a second
    # pass over seven Lustre-backed attribute maps for each chunk.
    local_region = tuple(slice(0, region[i].stop - region[i].start) for i in range(3))
    prep_started = time.perf_counter()
    cube, valid = _feature_block(raw, missing, local_region,
        min_support_fraction, int(window_samples), sample_origin=sample_origin + region[2].start,
        global_sample_count=global_sample_count)
    preprocessing_seconds = time.perf_counter() - prep_started
    key = ",".join(str(s.start) for s in region)
    request = DetectionRequest({feature: cube[..., k] for k, feature in enumerate(FEATURE_NAMES)},
        input_provenance={"source_identity": source_identity, "fit_record_sha256": fit_record_sha256},
        processing_configuration={"version": MODEL_VERSION, "chunk_key": key})
    scores, checksums = {}, {}
    model_times = {}
    for name in sorted(models):
        score_started = time.perf_counter()
        result = models[name].detect(request)
        model_times[name] = time.perf_counter() - score_started
        score = np.asarray(result.score, dtype=np.float32)
        score[~valid] = np.nan
        flag = (valid & np.isfinite(score) & (score >= thresholds[name])).astype(np.uint8)
        scores[f"{name}-score.npy"] = score
        scores[f"{name}-flag.npy"] = flag
        checksums[f"{name}-score.npy"] = _hash_bytes(score)
        checksums[f"{name}-flag.npy"] = _hash_bytes(flag)
    return {"input_sha256": input_hash, "output_sha256": checksums, "scores": scores,
            "valid_voxels": int(valid.sum()),
            "flagged": {name: int(scores[f"{name}-flag.npy"].sum()) for name in models},
            "input_read_seconds": read_seconds,
            "preprocessing_seconds": preprocessing_seconds,
            "model_scoring_seconds": model_times, "skip": False}


def fit_classical_models(reference: FeaturePopulation, calibration: FeaturePopulation, *,
                         quantile: float = 0.99, random_state: int = 2026,
                         max_rows: int = MAX_POPULATION_ROWS,
                         calibration_design: dict | None = None,
                         timing_metrics: dict | None = None) -> dict:
    """Fit four models on reference rows and set exploratory cutoffs on disjoint rows."""
    if reference.population_id == calibration.population_id or reference.source_identity != calibration.source_identity:
        raise ValueError("reference and calibration need distinct IDs tied to the same attribute source")
    if not _nonoverlap(reference.bounds, calibration.bounds):
        raise ValueError("reference and calibration spatial bounds must not overlap")
    if reference.window_samples != calibration.window_samples:
        raise ValueError("reference and calibration feature windows differ")
    if reference.sample_bounds != calibration.sample_bounds:
        raise ValueError("reference and calibration sample windows differ")
    if (reference.features.ndim != 2 or calibration.features.ndim != 2
            or reference.features.shape[1] != len(FEATURE_NAMES)
            or calibration.features.shape[1] != len(FEATURE_NAMES)
            or not np.isfinite(reference.features).all() or not np.isfinite(calibration.features).all()):
        raise ValueError("population feature arrays must be finite matrices with the declared feature order")
    if len(reference.features) < MIN_REFERENCE_ROWS or len(calibration.features) < MIN_CALIBRATION_ROWS:
        raise ValueError("reference/calibration populations are too small")
    if len(reference.features) > max_rows or len(calibration.features) > max_rows:
        raise ValueError("population row budget exceeded")
    if not 0.90 <= quantile < 1:
        raise ValueError("exploratory threshold quantile must be in [0.90,1)")
    center, scale, varying = robust_scaling(reference.features)
    if int(varying.sum()) < 2:
        raise ValueError("reference population has insufficient feature variability")
    fit = (reference.features - center) / scale
    calibration_x = (calibration.features - center) / scale
    factories = {
        "mahalanobis": lambda: MahalanobisDetector(threshold_squared_distance=None),
        "pca": lambda: LinearAutoencoderDetector(latent_dimensions=2, threshold_quantile=quantile),
        "isolation_forest": lambda: IsolationForestDetector(contamination=0.01, random_state=random_state, n_estimators=200),
        "gmm": lambda: GaussianMixtureDetector(components=2, threshold_quantile=quantile, random_state=random_state),
    }
    parameters = {"mahalanobis": {"regularization": 1e-3, "robust_winsor_z": 4.5},
                  "pca": {"latent_dimensions": 2, "threshold_quantile": quantile},
                  "isolation_forest": {"contamination": 0.01, "random_state": random_state,
                                       "n_estimators": 200},
                  "gmm": {"components": 2, "random_state": random_state,
                          "threshold_quantile": quantile}}
    models, statuses = {}, {}
    for name, factory in factories.items():
        model_started = time.perf_counter() if timing_metrics is not None else None
        try:
            model = factory().fit(fit, feature_names=FEATURE_NAMES)
            fit_seconds = time.perf_counter() - model_started if timing_metrics is not None else None
            if name == "gmm" and not model.estimator.converged_:
                raise ValueError("GMM did not converge on declared reference population")
            request = DetectionRequest({feature: calibration_x[:, k, None, None]
                                        for k, feature in enumerate(FEATURE_NAMES)},
                                       input_provenance={"population_id": calibration.population_id},
                                       processing_configuration={"model_scope": "field_screening"})
            threshold_started = time.perf_counter() if timing_metrics is not None else None
            output = model.detect(request)
            if timing_metrics is not None:
                timing_metrics.setdefault("model_fit_seconds", {})[name] = fit_seconds
                timing_metrics.setdefault("threshold_scoring_seconds", {})[name] = time.perf_counter() - threshold_started
            scores = np.asarray(output.score).reshape(-1)
            if len(scores) != len(calibration_x) or not np.isfinite(scores).all():
                raise ValueError("non-finite or misaligned calibration scores")
            threshold = float(np.quantile(scores, quantile))
            models[name] = model
            statuses[name] = {"status": "FITTED_FOR_FIELD_SCREENING", "model_id": model.model_id,
                               "score_meaning": output.score_meaning, "threshold": threshold,
                               "threshold_quantile": quantile, "threshold_population_id": calibration.population_id,
                               "score_is_probability": False, "calibration_status": "heldout_spatial_quantile_uncalibrated"}
        except (ImportError, RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            if timing_metrics is not None:
                timing_metrics.setdefault("model_fit_seconds", {}).setdefault(
                    name, time.perf_counter() - model_started)
            statuses[name] = {"status": "FAILED", "reason": f"{type(exc).__name__}: {exc}"}
    if not models:
        raise RuntimeError("no classical detector fitted successfully")
    fit_record = {"schema": "rtm-classical-model-fit-1", "version": MODEL_VERSION,
                  "source_identity": reference.source_identity, "feature_names": list(FEATURE_NAMES),
                  "window_samples": reference.window_samples,
                  "sample_bounds": list(reference.sample_bounds),
                  "feature_scaling": {"center": center.tolist(), "scale": scale.tolist(),
                                      "method": "reference-only median/MAD with standard-deviation fallback"},
                  "reference": {"id": reference.population_id, "bounds": list(reference.bounds),
                                "rows": len(reference.features), "eligible_scanned": reference.eligible_voxels_scanned,
                                "features_sha256": reference.feature_sha256},
                  "calibration": {"id": calibration.population_id, "bounds": list(calibration.bounds),
                                  "rows": len(calibration.features), "eligible_scanned": calibration.eligible_voxels_scanned,
                                  "features_sha256": calibration.feature_sha256},
                  "spatial_design": calibration_design,
                  "model_parameters": parameters, "random_state": random_state,
                  "models": statuses, "thresholds": {name: record["threshold"] for name, record in statuses.items()
                                                       if "threshold" in record},
                  "limitations": ["Reference and cutoff regions are unlabeled spatial samples, not verified no-change ground truth.",
                                  "Quantile cutoffs are detector-specific screening thresholds, not false-positive rates.",
                                  "Detector agreement is not independent confirmation; all share related seismic attributes.",
                                  "Scores do not identify fluid, pressure, saturation, or geological cause."]}
    return {"models": models, "center": center, "scale": scale, "fit_record": fit_record,
            "thresholds": fit_record["thresholds"]}


def score_full_field_models(attributes, missing_trace_mask, output_root, *, bundle: dict,
                            source_identity: str, min_support_fraction: float = 0.8,
                            window_samples: int = 9, chunk_shape=(16, 32, 128), protected_cache=None,
                            max_chunks: int | None = None, sample_origin: int = 0,
                            global_sample_count: int | None = None,
                            timing_metrics: dict | None = None, workers: int = 1) -> dict:
    """Score bounded chunks concurrently, committing outputs/journals in order."""
    models, thresholds = bundle["models"], bundle["thresholds"]
    shape = tuple(np.shape(attributes["signed_difference"]))
    if not models or set(models) - set(thresholds):
        raise ValueError("fitted detector models and thresholds are inconsistent")
    for name, value in attributes.items():
        if np.shape(value) != shape:
            raise ValueError(f"attribute {name} shape differs from signed_difference")
    if np.shape(missing_trace_mask) != shape[:2]:
        raise ValueError("missing-trace mask shape differs from attributes")
    if len(source_identity) != 64:
        raise ValueError("source_identity must be SHA-256")
    if not 0 < min_support_fraction <= 1:
        raise ValueError("min_support_fraction must be in (0,1]")
    if int(window_samples) != int(bundle["fit_record"].get("window_samples", -1)):
        raise ValueError("inference window differs from the feature-population window")
    if isinstance(workers, bool) or int(workers) < 1:
        raise ValueError("workers must be a positive integer")
    workers = int(workers)
    chunk_shape = tuple(map(int, chunk_shape))
    global_sample_count = int(global_sample_count if global_sample_count is not None else shape[2])
    if (isinstance(sample_origin, bool) or int(sample_origin) < 0
            or int(sample_origin) + shape[2] > global_sample_count):
        raise ValueError("scoring sample crop lies outside the declared global sample axis")
    sample_origin = int(sample_origin)
    selected = tuple(sorted(models))
    config = {"schema": "rtm-full-field-model-scores-1", "version": MODEL_VERSION,
              "source_identity": source_identity, "fit_record": bundle["fit_record"],
              "min_support_fraction": min_support_fraction, "window_samples": int(window_samples),
              "chunk_shape": list(chunk_shape),
              "shape": list(shape), "models": list(selected)}
    config_sha = hashlib.sha256(_canonical(config)).hexdigest()
    root = Path(output_root).expanduser().resolve()
    if protected_cache is not None:
        cache = Path(protected_cache).expanduser().resolve()
        if root == cache or cache in root.parents or root in cache.parents:
            raise ValueError("model output root must not overlap source cache")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if root.stat().st_uid != os.getuid() or stat.S_IMODE(root.stat().st_mode) & 0o077:
        raise PermissionError("model output root must be private and user-owned")
    run_dir = root / f"models-{config_sha[:20]}"
    run_dir.mkdir(mode=0o700, exist_ok=True)
    if run_dir.stat().st_uid != os.getuid() or stat.S_IMODE(run_dir.stat().st_mode) & 0o077:
        raise PermissionError("model run directory must be private and user-owned")
    manifest_path, journal_path, done_path = run_dir / "manifest.json", run_dir / "chunks.jsonl", run_dir / "complete.json"
    out_names = [f"{name}-{kind}.npy" for name in selected for kind in ("score", "flag")]
    required_output_bytes = int(np.prod(shape, dtype=np.int64)) * 5 * len(selected)
    existing_output_bytes = sum((run_dir / name).stat().st_size for name in out_names if (run_dir / name).is_file())
    fs = os.statvfs(run_dir)
    if fs.f_bavail * fs.f_frsize < required_output_bytes - existing_output_bytes:
        raise OSError("insufficient storage for full-field model score and flag maps")
    with (run_dir / ".lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        events = _read_journal(journal_path)
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("configuration_sha256") != config_sha:
                raise ValueError("model output manifest belongs to another fit/source/configuration")
        else:
            if events or done_path.exists():
                raise ValueError("model output journal exists without its manifest")
            if any((run_dir / name).exists() for name in out_names):
                raise FileExistsError("untracked model output files exist; refusing to overwrite")
            manifest = {"configuration_sha256": config_sha, "configuration": config,
                        "execution": {"worker_count_peak": 0,
                                      "worker_policy": "bounded chunk threads; serialized output and journal commits"},
                        "status": "running"}
            _atomic_json(manifest_path, manifest)
            _append(journal_path, {"event": "start", "configuration_sha256": config_sha})
            events = [{"event": "start", "configuration_sha256": config_sha}]
        if not events or events[0].get("event") != "start" or events[0].get("configuration_sha256") != config_sha:
            raise ValueError("model chunk journal has no matching start event")
        outputs = {}
        for filename in out_names:
            path = run_dir / filename
            if path.exists():
                outputs[filename] = np.load(path, mmap_mode="r+", allow_pickle=False)
            else:
                if any(e.get("event") == "chunk_complete" for e in events):
                    raise FileNotFoundError("model output array is missing despite completed journal chunks")
                is_flag = filename.endswith("-flag.npy")
                outputs[filename] = np.lib.format.open_memmap(path, mode="w+",
                    dtype=np.uint8 if is_flag else np.float32, shape=shape)
                outputs[filename].flush()
        completed = {e["chunk_key"]: e for e in events if e.get("event") == "chunk_complete"}
        total = sum(1 for _ in iter_chunks(shape, chunk_shape))
        processed = 0
        manifest.setdefault("execution", {})
        manifest["execution"].update(worker_count_peak=max(
            int(manifest["execution"].get("worker_count_peak", 0)), workers),
            worker_policy="bounded chunk threads; serialized output and journal commits")
        _atomic_json(manifest_path, manifest)
        try:
            if workers > 1:
                from threadpoolctl import threadpool_limits
                native_limit = threadpool_limits(limits=1)
            else:
                native_limit = nullcontext()
        except ImportError as exc:
            raise RuntimeError("threadpoolctl is required for parallel chunk scoring") from exc
        chunks = iter(iter_chunks(shape, chunk_shape))
        pending = deque()
        exhausted = False
        submitted_new = 0
        fit_record_sha = hashlib.sha256(_canonical(bundle["fit_record"])).hexdigest()
        with native_limit:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="rtm-score") as executor:
                while pending or not exhausted:
                    while len(pending) < workers and not exhausted:
                        try:
                            chunk = next(chunks)
                        except StopIteration:
                            exhausted = True
                            break
                        region = chunk.core
                        key = ",".join(str(s.start) for s in region)
                        old = completed.get(key)
                        if old is None and max_chunks is not None and submitted_new >= max_chunks:
                            exhausted = True
                            break
                        future = executor.submit(_score_one_chunk, attributes,
                            missing_trace_mask, region, models=models, thresholds=thresholds,
                            source_identity=source_identity, fit_record_sha256=fit_record_sha,
                            min_support_fraction=min_support_fraction, window_samples=window_samples,
                            sample_origin=sample_origin, global_sample_count=global_sample_count,
                            existing_event=old, output_arrays=outputs)
                        pending.append((region, key, old, future))
                        if old is None:
                            submitted_new += 1
                    if not pending:
                        break
                    region, key, old, future = pending.popleft()
                    result = future.result()
                    if timing_metrics is not None:
                        timing_metrics["input_read_seconds"] = timing_metrics.get("input_read_seconds", 0.0) + result["input_read_seconds"]
                        timing_metrics["preprocessing_seconds"] = timing_metrics.get("preprocessing_seconds", 0.0) + result["preprocessing_seconds"]
                        model_times = timing_metrics.setdefault("model_scoring_seconds", {})
                        for name, seconds in result["model_scoring_seconds"].items():
                            model_times[name] = model_times.get(name, 0.0) + seconds
                    if old is not None:
                        continue
                    write_started = time.perf_counter() if timing_metrics is not None else None
                    for filename, array in result["scores"].items():
                        outputs[filename][region] = array
                    for value in outputs.values():
                        value.flush()
                    event = {"event": "chunk_complete", "chunk_key": key,
                        "input_sha256": result["input_sha256"],
                        "output_sha256": result["output_sha256"],
                        "voxels": int(np.prod([s.stop-s.start for s in region])),
                        "valid_voxels": result["valid_voxels"],
                        "flagged": result["flagged"]}
                    _append(journal_path, event)
                    if timing_metrics is not None:
                        timing_metrics["scoring_io_seconds"] = timing_metrics.get("scoring_io_seconds", 0.0) + time.perf_counter() - write_started
                    completed[key] = event
                    processed += 1
        if len(completed) != total:
            manifest.update(status="incomplete", completed_chunks=len(completed), total_chunks=total)
            _atomic_json(manifest_path, manifest)
            return {**manifest, "run_directory": str(run_dir), "cache_hit": False}
        if done_path.exists():
            complete = json.loads(done_path.read_text())
            if complete.get("configuration_sha256") != config_sha:
                raise ValueError("completion artifact has incompatible configuration")
            for name, item in complete["outputs"].items():
                path = run_dir / item["file"]
                if not path.is_file() or _hash_file(path) != item["sha256"]:
                    raise ValueError(f"completed model output checksum mismatch: {name}")
            return {**complete, "run_directory": str(run_dir), "cache_hit": True}
        output_records = {}
        for filename, array in outputs.items():
            path = run_dir / filename
            array.flush()
            output_records[filename] = {"file": filename, "size_bytes": path.stat().st_size,
                                        "sha256": _hash_file(path), "dtype": str(array.dtype), "shape": list(shape)}
        body = {"schema": "rtm-full-field-model-scores-1", "version": MODEL_VERSION,
                "configuration_sha256": config_sha, "status": "complete", "total_chunks": total,
                "completed_chunks": len(completed), "valid_voxels": sum(e["valid_voxels"] for e in completed.values()),
                "flagged_voxels": {name: sum(e["flagged"][name] for e in completed.values()) for name in selected},
                "outputs": output_records, "fit_record": bundle["fit_record"],
                "execution": manifest.get("execution", {}),
                "limitations": bundle["fit_record"]["limitations"],
                "ranking_scope": "detector-specific screening; not a probability or field-wide geological priority"}
        _atomic_json(done_path, {**body, "configuration_sha256": config_sha})
        manifest.update(status="complete", completed_chunks=total, total_chunks=total)
        _atomic_json(manifest_path, manifest)
        return {**body, "run_directory": str(run_dir), "cache_hit": False}
