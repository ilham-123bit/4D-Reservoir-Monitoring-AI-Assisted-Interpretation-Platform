#!/usr/bin/env python3
"""Run a bounded Stage 2 benchmark with the production classical ML functions.

This command never submits Slurm jobs. Real Angsi execution is allowed only
inside an approved Slurm allocation; this pilot does not approve full-field interpretation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


def _pilot_config_validator():
    path = Path(__file__).with_name("validate_bounded_ml_pilot_config.py")
    spec = importlib.util.spec_from_file_location("rtm_pilot_config_gate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _under(root, value):
    candidate = Path(value).expanduser()
    result = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if result != root and root not in result.parents:
        raise ValueError("attribute run directory must remain under the private output root")
    return result


def _load_attributes(attr_dir, np):
    from reservoir_time_machine.processing.full_field import _canonical
    envelope = json.loads((attr_dir / "complete.json").read_text(encoding="utf-8"))
    body = envelope.get("body", {})
    digest = hashlib.sha256(_canonical(body)).hexdigest()
    if envelope.get("sha256") != digest or body.get("status") != "complete":
        raise ValueError("attribute completion manifest failed validation")
    configuration = body.get("configuration", {})
    shape = tuple(configuration.get("shape", ()))
    if len(shape) != 3 or configuration.get("source_domain_shape") != list(shape):
        raise ValueError("attribute manifest is not for a complete 3D source domain")
    expected = {"signed_difference", "absolute_difference", "normalized_difference",
                "rms_baseline", "rms_monitor", "nrms", "paired_sample_count"}
    if set(body.get("outputs", {})) != expected:
        raise ValueError("completed attribute manifest does not contain exactly seven required products")
    arrays = {}
    for name, item in body["outputs"].items():
        path = attr_dir / item["file"]
        if not path.is_file() or path.stat().st_size != int(item["size_bytes"]):
            raise ValueError(f"attribute file missing or size differs from manifest: {name}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if tuple(array.shape) != shape or list(array.shape) != item.get("shape"):
            raise ValueError(f"attribute geometry differs from manifest: {name}")
        arrays[name] = array
    source_identity = body.get("configuration_sha256")
    if not isinstance(source_identity, str) or len(source_identity) != 64:
        # The completion envelope hashes its canonical body; use that stable identity
        # when older Stage 1 manifests omit the convenience field.
        source_identity = envelope["sha256"]
    return arrays, body, source_identity


def _parse_chunk_shape(value):
    try:
        shape = tuple(int(part) for part in value.lower().split("x"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("chunk shapes use INLINExCROSSLINExSAMPLE, e.g. 16x16x64") from exc
    if len(shape) != 3 or min(shape) < 1:
        raise argparse.ArgumentTypeError("chunk shapes require three positive dimensions")
    return shape


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--paired-cache", required=True, type=Path)
    parser.add_argument("--challenge3-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path, help="new, unique private directory; must not exist")
    parser.add_argument("--experiment-id", help="stable label for this separate bounded scientific trial")
    parser.add_argument("--reference-bounds", nargs=4, type=int, required=True, metavar=("I0", "I1", "X0", "X1"))
    parser.add_argument("--calibration-bounds", nargs=4, type=int, required=True, metavar=("I0", "I1", "X0", "X1"))
    parser.add_argument("--sample-bounds", nargs=2, type=int, required=True, metavar=("S0", "S1"), help="half-open shared reference/calibration sample range")
    parser.add_argument("--scoring-bounds", nargs=6, type=int, required=True, metavar=("I0", "I1", "X0", "X1", "S0", "S1"))
    parser.add_argument("--max-samples", type=int, required=True, help="maximum reservoir-sampled feature rows per population")
    parser.add_argument("--max-score-voxels", type=int, default=1_000_000)
    parser.add_argument("--max-output-bytes", type=int, default=128 * 1024 * 1024)
    parser.add_argument("--max-memory-mib", type=int, default=2048,
                        help="maximum estimated working set; the Slurm allocation enforces the hard limit")
    parser.add_argument("--chunk-shape", nargs=3, type=int, default=(16, 16, 64),
                        metavar=("INLINE", "CROSSLINE", "SAMPLE"))
    parser.add_argument("--compare-chunks", nargs="+", type=_parse_chunk_shape,
                        help="score one common crop with these chunk shapes after one shared fit")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dry-run", action="store_true", help="validate geometry, manifests and budgets; no model fitting")
    args = parser.parse_args(argv)
    if not 1 <= args.threads <= 128:
        parser.error("--threads must be in [1,128]")
    if any(value < 1 for value in args.chunk_shape):
        parser.error("--chunk-shape values must be positive")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(args.threads)

    source_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(source_root / "src"))
    try:
        import numpy as np
        from reservoir_time_machine.analysis.bounded_ml_pilot import (
            pilot_execution_blockers, run_bounded_ml_pilot, validate_crop_plan)
    except ImportError as exc:
        parser.error(f"pilot dependencies unavailable: {exc}")
    tested_chunk_shapes = args.compare_chunks or [tuple(args.chunk_shape)]
    largest_chunk_shape = max(tested_chunk_shapes, key=lambda value: int(np.prod(value)))

    config_path = args.config.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    cache = args.paired_cache.expanduser().resolve()
    data_root = args.challenge3_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not cache.is_dir() or not data_root.is_dir() or not output_root.is_dir():
        parser.error("paired cache, authorized source root and output root must exist")
    if output_dir.exists() or not output_dir.parent.is_dir() or output_dir == output_root:
        parser.error("output directory must be a new path with an existing parent")
    try:
        output_dir.relative_to(output_root)
    except ValueError:
        parser.error("pilot output directory must be under RTM_HPC_OUTPUT_ROOT")
    attrs_dir = _under(output_root, config.get("attribute_run_directory", ""))
    attributes, attribute_body, source_identity = _load_attributes(attrs_dir, np)
    shape = tuple(attributes["signed_difference"].shape)
    metadata_path = cache / "metadata.json"
    mask_path = cache / "missing_trace_mask.npy"
    if not metadata_path.is_file() or not mask_path.is_file():
        parser.error("paired cache metadata and occupancy mask are required")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if tuple(metadata.get("shape", ())) != shape:
        parser.error("paired-cache and verified-attribute geometry differ")
    inline_ids, crossline_ids = metadata.get("inline_ids"), metadata.get("crossline_ids")
    if inline_ids is None or crossline_ids is None or len(inline_ids) != shape[0] or len(crossline_ids) != shape[1]:
        parser.error("paired cache lacks valid original SEG-Y axes")
    monitor_only_path = cache / "monitor_only_keys.npy"
    if monitor_only_path.is_file():
        monitor_only_keys = np.load(monitor_only_path, mmap_mode="r", allow_pickle=False)
        if monitor_only_keys.ndim != 2 or monitor_only_keys.shape[1] != 2:
            parser.error("monitor-only coverage key archive has invalid geometry")
    else:
        monitor_only_keys = np.empty((0, 2), dtype=np.int32)
    config_validator = _pilot_config_validator()
    config_valid = True
    try:
        config_validator.validate_pilot_config(config, output_root=output_root)
    except (TypeError, ValueError, OSError) as exc:
        config_valid = False
        config_error = str(exc)
    for name, actual, expected in (
            ("reference", list(args.reference_bounds), config.get("reference_bounds")),
            ("calibration", list(args.calibration_bounds), config.get("calibration_bounds")),
            ("sample", list(args.sample_bounds), config.get("sample_bounds")),
            ("scoring", list(args.scoring_bounds), config.get("scoring_bounds"))):
        if actual != expected:
            parser.error(f"CLI {name} bounds disagree with the bounded-pilot configuration")
    configured_chunks = [tuple(map(int, shape)) for shape in config.get("compare_chunks", [])]
    if configured_chunks and tested_chunk_shapes != configured_chunks:
        parser.error("CLI chunk comparison differs from the bounded-pilot configuration")
    if args.max_samples > int(config.get("max_samples", 0)):
        parser.error("CLI max-samples exceeds the bounded-pilot configuration")
    if args.max_score_voxels > int(config.get("max_score_voxels", 0)):
        parser.error("CLI max-score-voxels exceeds the bounded-pilot configuration")
    if args.max_output_bytes > int(config.get("max_output_bytes", 0)):
        parser.error("CLI output budget exceeds the bounded-pilot configuration limit")
    if args.max_memory_mib > int(config.get("max_memory_mib", 0)):
        parser.error("CLI max-memory-mib exceeds the bounded-pilot configuration")
    if args.threads > int(config.get("threads", 0)) or args.seed != int(config.get("random_seed", -1)):
        parser.error("CLI threads or random seed differ from the bounded-pilot configuration")
    min_buffer = int(config.get("minimum_separation_grid_cells", 32))
    plan = validate_crop_plan(
        shape=shape, reference_bounds=args.reference_bounds,
        calibration_bounds=args.calibration_bounds,
        approved_reference_bounds=config.get("reference_bounds", []),
        approved_calibration_bounds=config.get("calibration_bounds", []),
        sample_bounds=args.sample_bounds, scoring_bounds=args.scoring_bounds,
        max_samples=args.max_samples, max_score_voxels=args.max_score_voxels,
        max_output_bytes=args.max_output_bytes, minimum_separation=min_buffer,
        window_samples=int(attribute_body.get("configuration", {}).get("window_samples", 9)),
        chunk_shape=largest_chunk_shape)
    # Four detector score/flag maps cost 20 bytes per scored voxel. Estimate
    # each shape's journal separately because smaller chunks create more rows.
    score_dims = (args.scoring_bounds[1] - args.scoring_bounds[0],
                  args.scoring_bounds[3] - args.scoring_bounds[2],
                  args.scoring_bounds[5] - args.scoring_bounds[4])
    journal_overhead_estimate = sum(
        1_048_576 + 8192 * int(np.prod([int(np.ceil(n / c)) for n, c in zip(score_dims, shape)]))
        for shape in tested_chunk_shapes)
    total_output_estimate = (plan["expected_score_output_bytes"] * len(tested_chunk_shapes)
                             + journal_overhead_estimate)
    if total_output_estimate > args.max_output_bytes:
        parser.error("all chunk comparisons plus journals exceed --max-output-bytes")
    if total_output_estimate > args.max_output_bytes:
        parser.error("all chunk comparisons plus journals exceed --max-output-bytes")
    if plan["estimated_working_memory_mib"] > args.max_memory_mib:
        parser.error("bounded workspace estimate exceeds --max-memory-mib")
    if output_root == cache or output_root in cache.parents or cache in output_root.parents:
        parser.error("output root overlaps the paired cache")
    if output_root == data_root or output_root in data_root.parents or data_root in output_root.parents:
        parser.error("output root overlaps original Challenge 3 data")
    if (output_dir == attrs_dir or output_dir in attrs_dir.parents or attrs_dir in output_dir.parents):
        parser.error("pilot output directory overlaps completed attributes")

    allocation_ok = bool(os.environ.get("SLURM_JOB_ID"))
    auth_blockers = pilot_execution_blockers(config_valid=config_valid,
        in_slurm_allocation=allocation_ok)
    if not config_valid:
        auth_blockers.insert(0, config_error)
    if args.dry_run:
        print(json.dumps({"status": "dry_run_validated", "plan": plan,
                          "source_identity": source_identity,
                          "reference_segy_ids": [[inline_ids[args.reference_bounds[0]], inline_ids[args.reference_bounds[1]-1]],
                                                 [crossline_ids[args.reference_bounds[2]], crossline_ids[args.reference_bounds[3]-1]]],
                          "calibration_segy_ids": [[inline_ids[args.calibration_bounds[0]], inline_ids[args.calibration_bounds[1]-1]],
                                                   [crossline_ids[args.calibration_bounds[2]], crossline_ids[args.calibration_bounds[3]-1]]],
                          "scoring_segy_ids": [[inline_ids[args.scoring_bounds[0]], inline_ids[args.scoring_bounds[1]-1]],
                                               [crossline_ids[args.scoring_bounds[2]], crossline_ids[args.scoring_bounds[3]-1]]],
                          "real_angsi_execution_allowed": not auth_blockers,
                          "blockers": auth_blockers,
                          "models_fit": False}, sort_keys=True, indent=2))
        return 0
    if auth_blockers:
        parser.error("real Angsi pilot blocked: " + "; ".join(auth_blockers))
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if not slurm_cpus:
        parser.error("SLURM_CPUS_PER_TASK is required to enforce the thread allocation")
    if not 1 <= args.threads <= int(slurm_cpus):
        parser.error("--threads exceeds the Slurm CPU allocation")
    if os.environ.get("SLURM_MEM_PER_NODE"):
        allocated_memory_mib = int(os.environ["SLURM_MEM_PER_NODE"])
    elif os.environ.get("SLURM_MEM_PER_CPU"):
        allocated_memory_mib = int(os.environ["SLURM_MEM_PER_CPU"]) * int(slurm_cpus)
    else:
        parser.error("Slurm memory allocation is unavailable; refusing a real-data pilot")
    if plan["estimated_working_memory_mib"] > min(args.max_memory_mib, allocated_memory_mib):
        parser.error("estimated bounded workspace exceeds the configured or allocated memory")
    missing = np.load(mask_path, mmap_mode="r", allow_pickle=False)
    if tuple(missing.shape) != shape[:2]:
        parser.error("paired occupancy mask geometry differs from attributes")
    plan["segy_reference_ids"] = [[int(inline_ids[args.reference_bounds[0]]), int(inline_ids[args.reference_bounds[1]-1])],
                                   [int(crossline_ids[args.reference_bounds[2]]), int(crossline_ids[args.reference_bounds[3]-1])]]
    plan["segy_calibration_ids"] = [[int(inline_ids[args.calibration_bounds[0]]), int(inline_ids[args.calibration_bounds[1]-1])],
                                        [int(crossline_ids[args.calibration_bounds[2]]), int(crossline_ids[args.calibration_bounds[3]-1])]]
    plan["segy_scoring_ids"] = [[int(inline_ids[args.scoring_bounds[0]]), int(inline_ids[args.scoring_bounds[1]-1])],
                                     [int(crossline_ids[args.scoring_bounds[2]]), int(crossline_ids[args.scoring_bounds[3]-1])]]
    plan["sample_indices_only"] = True
    report = run_bounded_ml_pilot(
        attributes=attributes, missing_trace_mask=missing, source_identity=source_identity,
        plan=plan, output_dir=output_dir, max_samples=args.max_samples,
        random_seed=args.seed, chunk_shape=tuple(args.chunk_shape),
        chunk_shapes=tested_chunk_shapes,
        min_support_fraction=float(config.get("min_support_fraction", 0.8)),
        window_samples=int(attribute_body["configuration"].get("window_samples", 9)),
        threshold_quantile=float(config.get("threshold_quantile", 0.99)),
        mad_multiplier=float(config.get("mad_multiplier", 3.5)),
        protected_paths=(cache, data_root, attrs_dir),
        configuration_snapshot={"calibration_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "reference_bounds": config.get("reference_bounds"),
            "calibration_bounds": config.get("calibration_bounds"),
            "pilot_configuration": {key: config.get(key) for key in (
                "experiment_id", "reference_bounds", "calibration_bounds", "sample_bounds",
                "scoring_bounds", "scientific_limitations", "full_field_screening_status")},
            "threads": args.threads,
            "estimated_total_output_bytes": total_output_estimate},
        experiment_id=args.experiment_id, inline_ids=inline_ids,
        crossline_ids=crossline_ids, monitor_only_keys=monitor_only_keys)
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
