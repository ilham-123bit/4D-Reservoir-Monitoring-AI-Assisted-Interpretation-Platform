#!/usr/bin/env python3
"""Fit/scoring/grouping stage for completed full-field attribute products."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from reservoir_time_machine.analysis.full_field_findings import build_full_field_findings  # noqa: E402
from reservoir_time_machine.analysis.full_field_models import (fit_classical_models,
    sample_attribute_reference, sample_spatial_population, score_full_field_models,
    validate_spatial_calibration_design)  # noqa: E402
from reservoir_time_machine.processing.full_field import _canonical, _hash_file  # noqa: E402
from reservoir_time_machine.data_ingest.cache_identity import paired_cache_provenance  # noqa: E402


def _under(root, value):
    candidate = Path(value).expanduser()
    path = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if path != root and root not in path.parents:
        raise ValueError("analysis product path must remain under RTM_HPC_OUTPUT_ROOT")
    return path


def _publish_findings_package(staging: Path, destination: Path, source_identity: str) -> dict:
    """Validate a complete findings package, then expose it with one directory rename."""
    staging, destination = Path(staging), Path(destination)
    if staging.parent != destination.parent or destination.exists():
        raise FileExistsError("findings destination exists or staging is not a sibling; refusing publication")
    package = json.loads((staging / "findings.json").read_text(encoding="utf-8"))
    if (package.get("status") != "complete_for_supplied_full_volume"
            or package.get("source_identity") != source_identity):
        raise ValueError("staged finding package is incomplete or belongs to another source")
    for record_key in ("additional_candidate_catalogue", "quarantined_qc_catalogue"):
        record = package.get(record_key)
        if not isinstance(record, dict) or Path(record.get("file", "")).name != record.get("file"):
            raise ValueError(f"staged {record_key} has an invalid artifact path")
        from reservoir_time_machine.analysis.full_field_findings import _verify_streamed_envelope
        expected_schema = ("rtm-full-field-candidate-catalogue-1" if record_key == "additional_candidate_catalogue"
                           else "rtm-full-field-qc-catalogue-1")
        if not _verify_streamed_envelope(staging / record["file"], record.get("sha256", ""),
                                         source_identity, expected_schema):
            raise ValueError(f"staged {record_key} failed identity/checksum verification")
    os.rename(staging, destination)
    return package


def _reuse_completed_model_run(model_root: Path, *, source_identity: str, shape: tuple,
                              reference_bounds: tuple, calibration_bounds: tuple,
                              fit_sample_bounds: tuple, config: dict, window_samples: int,
                              min_support_fraction: float, calibration_design: dict) -> dict | None:
    """Return an integrity-checked completed score run matching this fit design."""
    expected_models = {"gmm", "isolation_forest", "mahalanobis", "pca"}
    matches = []
    for run_dir in sorted(Path(model_root).glob("models-*")):
        done_path, manifest_path = run_dir / "complete.json", run_dir / "manifest.json"
        if not done_path.is_file() or not manifest_path.is_file():
            continue
        complete = json.loads(done_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_config = manifest.get("configuration", {})
        fit = complete.get("fit_record", {})
        ref, cal = fit.get("reference", {}), fit.get("calibration", {})
        if (complete.get("status") != "complete"
                or complete.get("completed_chunks") != complete.get("total_chunks")
                or complete.get("schema") != "rtm-full-field-model-scores-1"
                or run_config.get("source_identity") != source_identity
                or run_config.get("shape") != list(shape)
                or run_config.get("models") != sorted(expected_models)
                or run_config.get("min_support_fraction") != min_support_fraction
                or run_config.get("window_samples") != window_samples
                or run_config.get("chunk_shape") != list(config.get("chunk_shape", (16, 32, 128)))
                or run_config.get("fit_record") != fit
                or complete.get("configuration_sha256") != hashlib.sha256(_canonical(run_config)).hexdigest()
                or fit.get("source_identity") != source_identity
                or fit.get("sample_bounds") != list(fit_sample_bounds)
                or fit.get("window_samples") != window_samples
                or fit.get("random_state") != int(config.get("random_state", 2026))
                or ref.get("bounds") != list(reference_bounds)
                or cal.get("bounds") != list(calibration_bounds)
                or ref.get("id") != str(config.get("reference_population_id", "spatial-reference"))
                or cal.get("id") != str(config.get("calibration_population_id", "spatial-calibration"))
                or fit.get("spatial_design", {}).get("review") != calibration_design.get("review")
                or any(item.get("threshold_quantile") != float(config.get("threshold_quantile", 0.99))
                       for item in fit.get("models", {}).values())):
            continue
        output_records = complete.get("outputs", {})
        required_files = {f"{name}-{kind}.npy" for name in expected_models for kind in ("score", "flag")}
        if set(output_records) != required_files:
            continue
        for filename, item in output_records.items():
            path = run_dir / item.get("file", "")
            if (path.name != filename or not path.is_file()
                    or item.get("shape") != list(shape)
                    or path.stat().st_size != item.get("size_bytes")
                    or _hash_file(path) != item.get("sha256")):
                raise ValueError(f"completed model output failed reuse integrity check: {filename}")
        matches.append({**complete, "run_directory": str(run_dir), "cache_hit": True})
    if len(matches) > 1:
        raise ValueError("multiple completed classical score runs match this configuration; refusing ambiguous reuse")
    return matches[0] if matches else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="private JSON with completed attribute run and spatial populations")
    args = parser.parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("field-wide model fit/scoring requires an approved Slurm allocation")
    output_root_text = os.environ.get("RTM_HPC_OUTPUT_ROOT")
    cache_text = os.environ.get("RTM_PAIRED_CACHE_DIR")
    if not output_root_text or not cache_text:
        parser.error("RTM_HPC_OUTPUT_ROOT and RTM_PAIRED_CACHE_DIR are required")
    output_root, cache = Path(output_root_text).expanduser().resolve(), Path(cache_text).expanduser().resolve()
    config = json.loads(args.config.expanduser().read_text(encoding="utf-8"))
    attr_dir = _under(output_root, config["attribute_run_directory"])
    analysis_root = _under(output_root, config["analysis_output_directory"])
    analysis_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if analysis_root.stat().st_uid != os.getuid() or analysis_root.stat().st_mode & 0o077:
        parser.error("analysis output must be private and user-owned")
    complete = json.loads((attr_dir / "complete.json").read_text(encoding="utf-8"))
    body = complete["body"]
    if complete.get("sha256") != hashlib.sha256(_canonical(body)).hexdigest() or body.get("status") != "complete":
        parser.error("attribute completion manifest is invalid or incomplete")
    source_configuration = body.get("configuration", {})
    shape_from_manifest = tuple(source_configuration.get("shape", ()))
    cache_identity, cache_details = paired_cache_provenance(cache)
    source_details = {"schema": "rtm-full-field-input-identity-2",
                      "paired_cache_sha256": cache_identity, "paired_cache": cache_details,
                      "source_bounds": [0, shape_from_manifest[0], 0, shape_from_manifest[1],
                                        0, shape_from_manifest[2]] if len(shape_from_manifest) == 3 else [],
                      "source_domain_shape": list(shape_from_manifest)}
    expected_source_identity = hashlib.sha256(json.dumps(
        source_details, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if (source_configuration.get("source_bounds") != source_details["source_bounds"]
            or source_configuration.get("source_domain_shape") != list(shape_from_manifest)
            or source_configuration.get("source_identity_details") != source_details
            or source_configuration.get("source_identity") != expected_source_identity):
        parser.error("Attribute provenance does not authenticate the paired cache and complete source domain")
    if os.environ.get("RTM_OPERATOR_SUBMISSION_APPROVED") != "YES":
        parser.error("explicit operator authorization must be exported to the Slurm worker")
    status = config.get("status")
    if status == "APPROVED_FOR_SCREENING":
        calibration_design = config.get("calibration_review")
        if not isinstance(calibration_design, dict):
            parser.error("approved screening config requires calibration_review")
    elif status == "UNLABELED_EXPLORATORY_SCREENING":
        if config.get("execution_scope") != "operator_authorized_unlabeled_full_field_screening":
            parser.error("exploratory config has an unsupported execution scope")
        calibration_design = config.get("calibration_design")
        if not isinstance(calibration_design, dict):
            parser.error("exploratory config requires a score-blind calibration_design record")
    else:
        parser.error("config must be approved screening or explicit unlabeled exploratory screening")
    reference_bounds = tuple(config["reference_bounds"])
    calibration_bounds = tuple(config["calibration_bounds"])
    missing = np.load(cache / "missing_trace_mask.npy", mmap_mode="r", allow_pickle=False)
    try:
        calibration_design = validate_spatial_calibration_design(
            reference_bounds, calibration_bounds,
            shape=tuple(body["configuration"]["shape"]),
            missing_trace_mask=missing,
            buffer_grid_cells=int(calibration_design["minimum_buffer_grid_cells"]),
            minimum_occupied_fraction=float(calibration_design["minimum_occupied_fraction"]),
            review=calibration_design)
    except (KeyError, TypeError, ValueError) as exc:
        parser.error(f"spatial calibration design rejected: {exc}")
    for name, item in body["outputs"].items():
        path = attr_dir / item["file"]
        if not path.is_file() or _hash_file(path) != item["sha256"]:
            parser.error(f"attribute product checksum failed: {name}")
    attributes = {name: np.load(attr_dir / item["file"], mmap_mode="r", allow_pickle=False)
                  for name, item in body["outputs"].items()}
    # Keep survey amplitudes memory-mapped. Candidate edge diagnostics read
    # bounded component boxes only; they do not materialize either full cube.
    survey_amplitudes = {
        "baseline": np.load(cache / "baseline.npy", mmap_mode="r", allow_pickle=False),
        "monitor": np.load(cache / "monitor.npy", mmap_mode="r", allow_pickle=False),
    }
    analysis_shape = tuple(np.shape(attributes["absolute_difference"]))
    if analysis_shape != tuple(map(int, config.get("expected_shape", analysis_shape))):
        parser.error("configured full-field geometry differs from the verified Stage 1 attribute geometry")
    if int(config.get("window_samples", body["configuration"]["window_samples"])) != int(body["configuration"]["window_samples"]):
        parser.error("configured paired-sample window differs from the completed Stage 1 attribute contract")
    if (tuple(survey_amplitudes["baseline"].shape) != analysis_shape
            or tuple(survey_amplitudes["monitor"].shape) != analysis_shape
            or tuple(missing.shape) != analysis_shape[:2]):
        parser.error("findings/edge-QC stage requires attributes covering the complete paired-cache geometry")
    metadata = json.loads((cache / "metadata.json").read_text(encoding="utf-8"))
    inline_ids = metadata.get("inline_ids")
    crossline_ids = metadata.get("crossline_ids")
    if inline_ids is None or crossline_ids is None:
        parser.error("paired cache lacks explicit original SEG-Y header axes")
    monitor_only_path = cache / "monitor_only_keys.npy"
    monitor_only_keys = (np.load(monitor_only_path, mmap_mode="r", allow_pickle=False)
                         if monitor_only_path.is_file() else np.empty((0, 2), dtype=np.int32))
    if monitor_only_keys.ndim != 2 or monitor_only_keys.shape[1] != 2:
        parser.error("monitor-only coverage key archive has invalid geometry")
    source_identity = body["configuration_sha256"]
    common = {"source_identity": source_identity, "max_rows": int(config.get("max_population_rows", 10000)),
              "chunk_shape": tuple(config.get("chunk_shape", (16, 32, 128))),
              "min_support_fraction": float(config.get("min_support_fraction", 0.8)),
              "window_samples": int(body["configuration"]["window_samples"])}
    fit_sample_bounds = tuple(map(int, config.get("fit_sample_bounds", (0, analysis_shape[2]))))
    scoring_workers = int(config.get("scoring_workers", 1))
    allocated_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    if scoring_workers < 1 or scoring_workers > allocated_cpus:
        parser.error(f"scoring_workers={scoring_workers} exceeds the Slurm CPU allocation ({allocated_cpus})")
    model_result = _reuse_completed_model_run(analysis_root / "classical-models",
        source_identity=source_identity, shape=analysis_shape,
        reference_bounds=reference_bounds, calibration_bounds=calibration_bounds,
        fit_sample_bounds=fit_sample_bounds, config=config,
        window_samples=common["window_samples"], min_support_fraction=common["min_support_fraction"],
        calibration_design=calibration_design)
    if model_result is not None:
        fit_record = model_result["fit_record"]
        bundle = {"fit_record": fit_record, "models": {name: None for name in fit_record["models"]}}
        print(json.dumps({"model_run": model_result["run_directory"],
                          "model_stage": "reused_completed_score_maps_without_refit_or_rescoring"}))
    else:
        reference = sample_spatial_population(attributes, missing, bounds=reference_bounds,
            population_id=str(config.get("reference_population_id", "spatial-reference")),
            sample_bounds=fit_sample_bounds, **common)
        calibration = sample_spatial_population(attributes, missing, bounds=calibration_bounds,
            population_id=str(config.get("calibration_population_id", "spatial-calibration")),
            sample_bounds=fit_sample_bounds, **common)
        bundle = fit_classical_models(reference, calibration, quantile=float(config.get("threshold_quantile", 0.99)),
                                      random_state=int(config.get("random_state", 2026)),
                                      calibration_design=calibration_design)
        model_result = score_full_field_models(attributes, missing, analysis_root / "classical-models",
            bundle=bundle, source_identity=source_identity,
            min_support_fraction=common["min_support_fraction"], window_samples=common["window_samples"],
            chunk_shape=common["chunk_shape"], protected_cache=cache, workers=scoring_workers)
    if model_result["status"] != "complete":
        parser.error("model scoring is incomplete; findings were not generated")
    model_dir = Path(model_result["run_directory"])
    model_outputs = {name: {"score": np.load(model_dir / f"{name}-score.npy", mmap_mode="r", allow_pickle=False),
                            "flag": np.load(model_dir / f"{name}-flag.npy", mmap_mode="r", allow_pickle=False)}
                     for name in bundle["models"]}
    reference_abs = sample_attribute_reference(attributes["absolute_difference"], bounds=calibration_bounds,
        max_rows=common["max_rows"], chunk_shape=common["chunk_shape"][:2],
        seed=int(config.get("random_state", 2026)))
    findings_final = analysis_root / "full-field-findings"
    if findings_final.exists():
        parser.error("a published findings package already exists; refusing to overwrite it")
    findings_stage = analysis_root / f".full-field-findings-staging-{os.environ['SLURM_JOB_ID']}-{time.time_ns()}"
    findings = build_full_field_findings(attributes, model_outputs, missing, reference_abs,
        findings_stage, source_identity=source_identity,
        inline_ids=inline_ids, crossline_ids=crossline_ids,
        survey_amplitudes=survey_amplitudes, monitor_only_keys=monitor_only_keys,
        min_support_fraction=common["min_support_fraction"],
        window_samples=common["window_samples"],
        min_region_voxels=int(config.get("min_region_voxels", 16)),
        min_ml_votes=int(config.get("min_ml_votes", 2)),
        mad_multiplier=float(config.get("mad_multiplier", 3.5)),
        chunk_shape=tuple(config.get("component_chunk_shape", (32, 32, 64))),
        max_findings=min(10, int(config.get("max_findings", 10))))
    _publish_findings_package(findings_stage, findings_final, source_identity)
    findings["run_directory"] = str(findings_final)
    findings["component_labels"] = str(findings_final / "components" / "component_labels.npy")
    print(json.dumps({"analysis_root": str(analysis_root), "attribute_run": str(attr_dir), "model_run": model_result["run_directory"],
                      "finding_run": findings["run_directory"], "coverage_fraction": findings["coverage_fraction"],
                      "finding_count": findings["finding_count"],
                      "screening_status": status,
                      "status": "preliminary_full-volume_screening"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
