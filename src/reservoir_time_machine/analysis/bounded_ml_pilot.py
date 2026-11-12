"""Bounded benchmarking of the production Stage 2 classical ML functions."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import resource
import tempfile
import time

import numpy as np

from .full_field_models import (fit_classical_models, sample_spatial_population,
                                sample_attribute_reference, score_full_field_models)
from .edge_qc import summarize_bounded_coverage_edges

MAX_PILOT_SAMPLES = 50_000
DEFAULT_MAX_SCORE_VOXELS = 1_000_000
DEFAULT_MAX_OUTPUT_BYTES = 128 * 1024 * 1024


def _validate_box4(bounds, name, shape):
    if len(bounds) != 4:
        raise ValueError(f"{name} must be four half-open array bounds: i0 i1 x0 x1")
    i0, i1, x0, x1 = map(int, bounds)
    if not (0 <= i0 < i1 <= shape[0] and 0 <= x0 < x1 <= shape[1]):
        raise ValueError(f"{name} lies outside the inline/crossline array geometry")
    return (i0, i1, x0, x1)


def _contains(outer, inner):
    return outer[0] <= inner[0] < inner[1] <= outer[1] and outer[2] <= inner[2] < inner[3] <= outer[3]


def _box_distance(a, b):
    di = max(0, b[0] - a[1], a[0] - b[1])
    dx = max(0, b[2] - a[3], a[2] - b[3])
    return float(np.hypot(di, dx))


def pilot_execution_blockers(*, config_valid, in_slurm_allocation):
    """Check static pilot safety settings and require Slurm execution.

    Full-field screening continues to use its independent strict review gate.
    """
    blockers = [] if config_valid else ["bounded-pilot scientific configuration has not passed validation"]
    if not in_slurm_allocation:
        blockers.append("real-data pilot must run inside an approved Slurm allocation")
    return blockers


def validate_crop_plan(*, shape, reference_bounds, calibration_bounds,
                       approved_reference_bounds, approved_calibration_bounds,
                       sample_bounds, scoring_bounds, max_samples,
                       max_score_voxels=DEFAULT_MAX_SCORE_VOXELS,
                       max_output_bytes=DEFAULT_MAX_OUTPUT_BYTES,
                       minimum_separation=32, window_samples=9,
                       chunk_shape=(16, 16, 64)):
    """Validate bounded, separated half-open array crops before any model work."""
    shape = tuple(map(int, shape))
    if len(shape) != 3 or any(n < 1 for n in shape):
        raise ValueError("source geometry must be positive (inline,crossline,sample)")
    ref = _validate_box4(reference_bounds, "reference_bounds", shape)
    cal = _validate_box4(calibration_bounds, "calibration_bounds", shape)
    approved_ref = _validate_box4(approved_reference_bounds, "approved_reference_bounds", shape)
    approved_cal = _validate_box4(approved_calibration_bounds, "approved_calibration_bounds", shape)
    if not _contains(approved_ref, ref) or not _contains(approved_cal, cal):
        raise ValueError("reference/calibration crops must be subsets of the approved regions")
    if _box_distance(ref, cal) < minimum_separation:
        raise ValueError("reference and calibration crops violate the spatial separation buffer")
    if len(sample_bounds) != 2:
        raise ValueError("sample_bounds must be half-open [start,stop]")
    s0, s1 = map(int, sample_bounds)
    if not (0 <= s0 < s1 <= shape[2]):
        raise ValueError("training sample bounds lie outside the source sample axis")
    halo = int(window_samples) // 2
    if s1 - s0 <= 2 * halo or s0 < halo or s1 > shape[2] - halo:
        raise ValueError("training sample interval must contain an interior window and its valid context")
    if isinstance(max_samples, bool) or not 64 <= int(max_samples) <= MAX_PILOT_SAMPLES:
        raise ValueError(f"max_samples must be between 64 and {MAX_PILOT_SAMPLES}")
    if len(scoring_bounds) != 6:
        raise ValueError("scoring_bounds must be half-open [i0,i1,x0,x1,s0,s1]")
    i0, i1, x0, x1, q0, q1 = map(int, scoring_bounds)
    score_box = _validate_box4((i0, i1, x0, x1), "scoring_bounds", shape)
    if not (0 <= q0 < q1 <= shape[2]) or q0 < halo or q1 > shape[2] - halo:
        raise ValueError("scoring crop must be an in-range interior sample window")
    if (i1-i0)*(x1-x0)*(q1-q0) > int(max_score_voxels):
        raise ValueError("scoring crop exceeds max_score_voxels")
    if min(_box_distance(score_box, ref), _box_distance(score_box, cal)) < minimum_separation:
        raise ValueError("scoring crop must be spatially separated from both populations")
    score_voxels = (i1-i0)*(x1-x0)*(q1-q0)
    expected_output_bytes = score_voxels * 5 * 4  # float32 score + uint8 flag per detector
    if expected_output_bytes > int(max_output_bytes):
        raise ValueError("pilot score maps exceed max_output_bytes")
    # Conservative workspace estimate, excluding the OS page cache.
    chunk_shape = tuple(map(int, chunk_shape))
    if len(chunk_shape) != 3 or min(chunk_shape) < 1:
        raise ValueError("chunk_shape must contain three positive dimensions")
    chunk_voxels = min(score_voxels, int(np.prod(chunk_shape)))
    estimated_memory_mib = int(np.ceil((128 * 1024**2 + 256 * chunk_voxels +
                                       2 * int(max_samples) * 4 * 4) / 1024**2))
    return {"reference_bounds": list(ref), "calibration_bounds": list(cal),
            "sample_bounds": [s0, s1], "scoring_bounds": [i0, i1, x0, x1, q0, q1],
            "scoring_voxels": score_voxels, "max_samples_per_population": int(max_samples),
            "expected_score_output_bytes": expected_output_bytes,
            "chunk_shape": list(chunk_shape),
            "estimated_working_memory_mib": estimated_memory_mib,
            "minimum_population_separation_cells": float(_box_distance(ref, cal)),
            "score_scope": "bounded benchmark crop only; not full-field findings"}


def _private_fresh_directory(path: Path):
    path = Path(path).expanduser().resolve()
    if path.exists():
        raise FileExistsError("pilot output directory already exists; choose a unique new path")
    if not path.parent.is_dir():
        raise FileNotFoundError("pilot output parent must already exist")
    parent = path.parent.stat()
    if parent.st_uid != os.getuid() or parent.st_mode & 0o077:
        raise PermissionError("pilot output parent must be user-owned and private")
    path.mkdir(mode=0o700)
    return path


def run_bounded_ml_pilot(*, attributes, missing_trace_mask, source_identity,
                         plan, output_dir, max_samples, random_seed=2026,
                         chunk_shape=(16, 16, 64), min_support_fraction=0.8,
                         window_samples=9, threshold_quantile=0.99, mad_multiplier=3.5,
                         protected_paths=(), chunk_shapes=None,
                         configuration_snapshot=None, experiment_id=None,
                         inline_ids=None, crossline_ids=None, monitor_only_keys=()):
    """Fit once and score bounded arrays using one or several chunk shapes."""
    chunk_shapes = [tuple(map(int, x)) for x in (chunk_shapes or [chunk_shape])]
    if not chunk_shapes or any(len(x) != 3 or min(x) < 1 for x in chunk_shapes):
        raise ValueError("chunk_shapes must contain positive 3D chunk dimensions")
    if not np.isfinite(mad_multiplier) or mad_multiplier <= 0:
        raise ValueError("mad_multiplier must be finite and positive")
    experiment_id = experiment_id or Path(output_dir).name
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", str(experiment_id)):
        raise ValueError("experiment_id must be a short, stable identifier")
    out = _private_fresh_directory(Path(output_dir))
    for protected in protected_paths:
        p = Path(protected).expanduser().resolve()
        if out == p or out in p.parents or p in out.parents:
            out.rmdir()
            raise ValueError("pilot output directory overlaps protected input data")
    snapshot = {"schema": "rtm-bounded-ml-pilot-config-1",
                "source_identity": source_identity, "plan": plan,
                "random_seed": int(random_seed), "max_samples": int(max_samples),
                "threshold_quantile": float(threshold_quantile),
                "mad_multiplier": float(mad_multiplier),
                "min_support_fraction": float(min_support_fraction),
                "window_samples": int(window_samples),
                "chunk_shapes": [list(x) for x in chunk_shapes],
                "experiment_id": str(experiment_id),
                "operator_configuration": configuration_snapshot or {}}
    (out / "pilot_config_snapshot.json").write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    os.chmod(out / "pilot_config_snapshot.json", 0o600)
    started = time.perf_counter()
    timings = {"input_read_seconds": 0.0, "preprocessing_seconds": 0.0}
    ref = sample_spatial_population(
        attributes, missing_trace_mask, bounds=plan["reference_bounds"],
        population_id="bounded-pilot-reference", source_identity=source_identity,
        max_rows=max_samples, chunk_shape=(16, 16, 64),
        min_support_fraction=min_support_fraction, window_samples=window_samples,
        seed=random_seed, sample_bounds=tuple(plan["sample_bounds"]), timing_metrics=timings)
    cal = sample_spatial_population(
        attributes, missing_trace_mask, bounds=plan["calibration_bounds"],
        population_id="bounded-pilot-calibration", source_identity=source_identity,
        max_rows=max_samples, chunk_shape=(16, 16, 64),
        min_support_fraction=min_support_fraction, window_samples=window_samples,
        seed=random_seed, sample_bounds=tuple(plan["sample_bounds"]), timing_metrics=timings)
    population_preprocessing_seconds = time.perf_counter() - started
    fit_metrics = {"model_fit_seconds": {}, "threshold_scoring_seconds": {}}
    fit_started = time.perf_counter()
    bundle = fit_classical_models(ref, cal, quantile=threshold_quantile,
                                  random_state=random_seed, max_rows=max_samples,
                                  timing_metrics=fit_metrics)
    fitting_seconds = time.perf_counter() - fit_started
    for key in fit_metrics:
        timings[key] = fit_metrics[key]

    # Describe, but never fit or calibrate on, the independent scoring region.
    score_population_timing = {"input_read_seconds": 0.0, "preprocessing_seconds": 0.0}
    score_population = sample_spatial_population(
        attributes, missing_trace_mask, bounds=plan["scoring_bounds"][:4],
        population_id=f"{experiment_id}-scoring-diagnostic", source_identity=source_identity,
        max_rows=max_samples, chunk_shape=(16, 16, 64),
        min_support_fraction=min_support_fraction, window_samples=window_samples,
        seed=random_seed, sample_bounds=tuple(plan["scoring_bounds"][4:]),
        timing_metrics=score_population_timing)
    feature_distribution = _feature_distribution_comparison(ref, cal, score_population, bundle)

    i0, i1, x0, x1, s0, s1 = plan["scoring_bounds"]
    score_attrs = {name: value[i0:i1, x0:x1, s0:s1] for name, value in attributes.items()}
    score_missing = missing_trace_mask[i0:i1, x0:x1]
    score_runs, score_records = [], []
    for tested_shape in chunk_shapes:
        label = "chunk-" + "x".join(map(str, tested_shape))
        score_metrics = {"input_read_seconds": 0.0, "preprocessing_seconds": 0.0,
                         "model_scoring_seconds": {}, "scoring_io_seconds": 0.0}
        score_started = time.perf_counter()
        score_result = score_full_field_models(
            score_attrs, score_missing, out / label / "model-scores", bundle=bundle,
            source_identity=source_identity, min_support_fraction=min_support_fraction,
            window_samples=window_samples, chunk_shape=tested_shape,
            sample_origin=s0, global_sample_count=np.shape(attributes["signed_difference"])[2],
            timing_metrics=score_metrics)
        score_seconds = time.perf_counter() - score_started
        if score_result.get("status") != "complete":
            raise RuntimeError(f"bounded pilot scoring did not complete for {tested_shape}")
        run_dir = Path(score_result["run_directory"])
        run_files = [p for p in run_dir.rglob("*") if p.is_file()]
        score_records.append({
            "chunk_shape": list(tested_shape), "status": score_result["status"],
            "scoring_wall_seconds": score_seconds,
            "scoring_positions_per_second": int(plan["scoring_voxels"]) / score_seconds if score_seconds else None,
            "input_read_seconds": score_metrics["input_read_seconds"],
            "feature_preprocessing_seconds": score_metrics["preprocessing_seconds"],
            "per_model_scoring_seconds": score_metrics["model_scoring_seconds"],
            "scoring_io_and_journal_seconds": score_metrics["scoring_io_seconds"],
            "valid_positions": int(score_result.get("valid_voxels", 0)),
            "unsupported_positions": int(plan["scoring_voxels"])-int(score_result.get("valid_voxels", 0)),
            "threshold_exceedance_positions": score_result.get("flagged_voxels", {}),
            "score_distributions": _score_distributions(run_dir, score_result.get("outputs", {}),
                tuple(np.shape(score_attrs["signed_difference"]))),
            "output_bytes": sum(p.stat().st_size for p in run_files),
            "run_directory": str(run_dir.relative_to(out)),
        })
        score_runs.append((score_result, run_dir, tested_shape))
    consistency = _compare_chunk_scores(score_runs, score_attrs, comparison_chunk=(16, 16, 64))
    if inline_ids is not None and crossline_ids is not None:
        flags_by_chunk = {}
        for result, run_dir, tested_shape in score_runs:
            maps = {}
            for filename, item in result.get("outputs", {}).items():
                if filename.endswith("-flag.npy"):
                    name = filename[:-len("-flag.npy")]
                    maps[name] = np.load(Path(result["run_directory"]) / item["file"],
                                         mmap_mode="r", allow_pickle=False)
            flags_by_chunk["x".join(map(str, tested_shape))] = maps
        edge_coverage = summarize_bounded_coverage_edges(
            support=attributes["paired_sample_count"], missing_trace_mask=missing_trace_mask,
            scoring_bounds=plan["scoring_bounds"], inline_ids=inline_ids,
            crossline_ids=crossline_ids, monitor_only_keys=monitor_only_keys,
            detector_flags_by_chunk=flags_by_chunk, window_samples=window_samples,
            min_support_fraction=min_support_fraction)
    else:
        edge_coverage = {"status": "unavailable",
                         "reason": "verified SEG-Y axes were not supplied to bounded pilot"}
    first_score_file = next((name for name in score_runs[0][0].get("outputs", {})
                             if name.endswith("-score.npy")), None)
    mad_screening = _bounded_mad_screen(
        attributes=attributes, score_attrs=score_attrs,
        valid_score_path=score_runs[0][1] / first_score_file if first_score_file else None,
        plan=plan, max_samples=max_samples, random_seed=random_seed,
        mad_multiplier=mad_multiplier)
    completed = time.perf_counter()
    files = [p for p in out.rglob("*") if p.is_file()]
    output_bytes = sum(p.stat().st_size for p in files)
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_mib = usage / 1024 if os.name == "posix" else None
    valid_positions = int(score_runs[0][0].get("valid_voxels", 0))
    total_positions = int(plan["scoring_voxels"])
    si0, si1, sx0, sx1, ss0, ss1 = plan["scoring_bounds"]
    unoccupied_samples = int(np.asarray(missing_trace_mask[si0:si1, sx0:sx1], dtype=bool).sum()) * (ss1-ss0)
    support_counts = {}
    from ..processing.chunks import iter_chunks
    for chunk in iter_chunks(tuple(np.shape(score_attrs["paired_sample_count"])), (16, 16, 64)):
        values, counts = np.unique(np.asarray(score_attrs["paired_sample_count"][chunk.core]),
                                   return_counts=True)
        for value, count in zip(values.tolist(), counts.tolist()):
            label = str(int(value)) if np.isfinite(value) else "nonfinite"
            support_counts[label] = support_counts.get(label, 0) + int(count)
    report = {
        "schema": "rtm-bounded-ml-pilot-1", "status": "complete",
        "output_class": "disposable_stage2_benchmark",
        "source_identity": source_identity,
        "experiment_id": str(experiment_id),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "array_shape": list(np.shape(attributes["signed_difference"])),
        "plan": plan, "random_seed": int(random_seed),
        "configuration_snapshot": configuration_snapshot or {},
        "reference_population_rows": int(len(ref.features)),
        "reference_population_eligible_scanned": int(ref.eligible_voxels_scanned),
        "calibration_population_rows": int(len(cal.features)),
        "calibration_population_eligible_scanned": int(cal.eligible_voxels_scanned),
        "input_read_seconds": timings["input_read_seconds"],
        "feature_preprocessing_seconds": timings["preprocessing_seconds"],
        "population_sampling_wall_seconds": population_preprocessing_seconds,
        "model_fitting_wall_seconds": fitting_seconds,
        "scoring_population_diagnostic_read_seconds": score_population_timing["input_read_seconds"],
        "scoring_population_diagnostic_preprocessing_seconds": score_population_timing["preprocessing_seconds"],
        "feature_distribution_comparison": feature_distribution,
        "per_model_fit_seconds": timings["model_fit_seconds"],
        "per_model_threshold_scoring_seconds": timings["threshold_scoring_seconds"],
        "chunk_benchmarks": score_records,
        "chunk_score_consistency": consistency,
        "edge_and_coverage_qc": edge_coverage,
        "mad_screening": mad_screening,
        "bounded_scoring_wall_seconds": sum(x["scoring_wall_seconds"] for x in score_records),
        "per_model_scoring_seconds": score_records[0]["per_model_scoring_seconds"],
        "scoring_io_and_journal_seconds": sum(x["scoring_io_and_journal_seconds"] for x in score_records),
        "score_total_positions": total_positions,
        "score_valid_positions": valid_positions,
        "score_unsupported_positions": total_positions - valid_positions,
        "paired_sample_support_distribution": support_counts,
        "unoccupied_grid_sample_positions": unoccupied_samples,
        "unsupported_non_unoccupied_positions": max(0, total_positions-valid_positions-unoccupied_samples),
        "scoring_positions_per_second": total_positions / score_records[0]["scoring_wall_seconds"] if score_records[0]["scoring_wall_seconds"] else None,
        "peak_process_rss_mib": peak_rss_mib,
        "output_bytes": output_bytes,
        "model_status": bundle["fit_record"]["models"],
        "errors": {name: status.get("reason") for name, status in bundle["fit_record"]["models"].items()
                   if status.get("status") == "FAILED"},
        "thresholds_are_probabilities": False,
        "scientific_scope": "performance benchmark only; no geological validation or full-field ranking",
        "warnings": ["Pilot timings are not full-field runtime guarantees.",
                     "Physical 4D registration and sample domain remain unverified."],
        "elapsed_seconds": completed - started,
        "model_score_run_directory": score_records[0]["run_directory"],
        "resumable": False,
    }
    target = out / "pilot_metrics.json"
    def write_report():
        fd, temporary = tempfile.mkstemp(prefix=".pilot-metrics-", dir=out)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)
    write_report()
    for _ in range(3):
        actual_output_bytes = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
        if report["output_bytes"] == actual_output_bytes:
            break
        report["output_bytes"] = actual_output_bytes
        write_report()
    return report


def _score_distributions(run_dir: Path, output_records: dict, shape):
    """Exact score summaries, read incrementally from the saved memmaps."""
    from ..processing.chunks import iter_chunks
    summaries = {}
    for filename in sorted(output_records):
        if not filename.endswith("-score.npy"):
            continue
        name = filename[:-len("-score.npy")]
        values = np.load(run_dir / filename, mmap_mode="r", allow_pickle=False)
        pieces = []
        for chunk in iter_chunks(shape, (16, 16, 64)):
            block = np.asarray(values[chunk.core])
            finite = block[np.isfinite(block)]
            if finite.size:
                pieces.append(finite)
        if not pieces:
            summaries[name] = {"finite_score_count": 0}
            continue
        sample = np.concatenate(pieces)
        summaries[name] = {"finite_score_count": int(sample.size),
            "min": float(sample.min()), "p05": float(np.percentile(sample, 5)),
            "median": float(np.median(sample)), "p95": float(np.percentile(sample, 95)),
            "max": float(sample.max()), "score_is_probability": False}
    return summaries


def _feature_distribution_comparison(reference, calibration, scoring, bundle):
    """Robust descriptive shift; scoring rows do not affect fitting/cutoffs."""
    names = bundle["fit_record"]["feature_names"]
    center = np.asarray(bundle["center"], dtype=float)
    scale = np.asarray(bundle["scale"], dtype=float)
    records = {}
    for index, name in enumerate(names):
        values = {}
        for label, population in (("reference", reference), ("calibration", calibration),
                                  ("scoring", scoring)):
            column = np.asarray(population.features[:, index], dtype=float)
            values[label] = {"median": float(np.median(column)),
                             "p05": float(np.percentile(column, 5)),
                             "p95": float(np.percentile(column, 95)),
                             "rows": int(column.size)}
        values["scoring_minus_reference_in_reference_robust_scales"] = float(
            (values["scoring"]["median"] - center[index]) / scale[index])
        values["calibration_minus_reference_in_reference_robust_scales"] = float(
            (values["calibration"]["median"] - center[index]) / scale[index])
        records[name] = values
    return {"sample_intervals_match": reference.sample_bounds == calibration.sample_bounds == scoring.sample_bounds,
            "reference_sample_bounds": list(reference.sample_bounds),
            "calibration_sample_bounds": list(calibration.sample_bounds),
            "scoring_sample_bounds": list(scoring.sample_bounds),
            "scoring_population_id": scoring.population_id,
            "scoring_population_rows": int(len(scoring.features)),
            "scoring_population_eligible_scanned": int(scoring.eligible_voxels_scanned),
            "scoring_population_features_sha256": scoring.feature_sha256,
            "features": records,
            "interpretation": "Descriptive distribution comparison only; the scoring population did not fit models or set thresholds."}


def _bounded_mad_screen(*, attributes, score_attrs, valid_score_path, plan,
                        max_samples, random_seed, mad_multiplier):
    """Apply the existing calibration-region median/MAD cutoff to a bounded crop."""
    sample_start, sample_stop = plan["sample_bounds"]
    bounded_absolute = attributes["absolute_difference"][:, :, sample_start:sample_stop]
    reference = sample_attribute_reference(
        bounded_absolute, bounds=plan["calibration_bounds"], max_rows=max_samples,
        chunk_shape=(16, 16), seed=random_seed)
    center = float(np.median(reference))
    mad = float(np.median(np.abs(reference - center)))
    if mad <= np.finfo(float).eps:
        return {"status": "unavailable_zero_reference_spread",
                "reference_population_id": "bounded-pilot-calibration-absolute-difference",
                "reference_rows": int(reference.size), "median": center, "mad": mad,
                "multiplier": float(mad_multiplier), "cutoff": None,
                "flagged_positions": None, "valid_positions": None,
                "interpretation": "No MAD cutoff is reported because the calibration reference has zero spread."}
    cutoff = center + float(mad_multiplier) * 1.4826 * mad
    if valid_score_path is None:
        return {"status": "unavailable_no_valid_detector_map", "cutoff": float(cutoff),
                "reference_rows": int(reference.size), "median": center, "mad": mad,
                "multiplier": float(mad_multiplier), "flagged_positions": None,
                "valid_positions": None}
    valid_map = np.load(valid_score_path, mmap_mode="r", allow_pickle=False)
    absolute = score_attrs["absolute_difference"]
    valid = np.isfinite(valid_map)
    values = np.asarray(absolute)
    flagged = valid & np.isfinite(values) & (values > cutoff)
    return {"status": "screened", "reference_population_id": "bounded-pilot-calibration-absolute-difference",
            "reference_bounds": plan["calibration_bounds"],
            "reference_sample_bounds": [sample_start, sample_stop],
            "reference_rows": int(reference.size), "median": center, "mad": mad,
            "multiplier": float(mad_multiplier), "cutoff": float(cutoff),
            "valid_positions": int(valid.sum()), "flagged_positions": int(flagged.sum()),
            "flagged_fraction": float(flagged.sum() / max(1, valid.sum())),
            "score_is_probability": False,
            "interpretation": "Exploratory reference-region MAD screen; cutoff is not a geological probability."}


def _compare_chunk_scores(score_runs, score_attrs, *, comparison_chunk):
    """Compare score/flag maps incrementally without materializing full crops."""
    if len(score_runs) < 2:
        return {"status": "single_chunk_shape; comparison not applicable"}
    from ..processing.chunks import iter_chunks
    shape = tuple(np.shape(score_attrs["signed_difference"]))
    reference = score_runs[0][1]
    results = []
    for result, run_dir, chunk_shape in score_runs[1:]:
        models = sorted({p.name.rsplit("-", 1)[0] for p in reference.glob("*-score.npy")})
        model_results = {}
        for name in models:
            ref_score = np.load(reference / f"{name}-score.npy", mmap_mode="r", allow_pickle=False)
            ref_flag = np.load(reference / f"{name}-flag.npy", mmap_mode="r", allow_pickle=False)
            cur_score = np.load(run_dir / f"{name}-score.npy", mmap_mode="r", allow_pickle=False)
            cur_flag = np.load(run_dir / f"{name}-flag.npy", mmap_mode="r", allow_pickle=False)
            finite_mismatches = flag_mismatches = over_tolerance = compared = 0
            maximum_absolute_error = 0.0
            for chunk in iter_chunks(shape, comparison_chunk):
                region = chunk.core
                a = np.asarray(ref_score[region])
                b = np.asarray(cur_score[region])
                finite_a, finite_b = np.isfinite(a), np.isfinite(b)
                finite_mismatches += int(np.count_nonzero(finite_a != finite_b))
                mask = finite_a & finite_b
                if mask.any():
                    delta = np.abs(a[mask] - b[mask])
                    allowed = 1e-6 + 1e-6 * np.abs(a[mask])
                    maximum_absolute_error = max(maximum_absolute_error, float(delta.max()))
                    over_tolerance += int(np.count_nonzero(delta > allowed))
                    compared += int(delta.size)
                flag_mismatches += int(np.count_nonzero(ref_flag[region] != cur_flag[region]))
            model_results[name] = {"finite_mask_mismatches": finite_mismatches,
                "flag_mismatches": flag_mismatches, "score_values_compared": compared,
                "score_values_over_1e-6_relative_tolerance": over_tolerance,
                "maximum_absolute_score_difference": maximum_absolute_error,
                "within_tolerance": finite_mismatches == 0 and flag_mismatches == 0 and over_tolerance == 0}
        results.append({"chunk_shape": list(chunk_shape), "models": model_results,
                        "all_models_within_tolerance": all(x["within_tolerance"] for x in model_results.values())})
    return {"reference_chunk_shape": list(score_runs[0][2]),
            "comparisons": results,
            "all_configurations_within_tolerance": all(x["all_models_within_tolerance"] for x in results)}
