#!/usr/bin/env python3
"""Validate fixed scientific/resource settings for the bounded Well-2 pilot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REFERENCE = [64, 192, 1088, 1216]
CALIBRATION = [869, 997, 64, 192]
SCORING = [600, 728, 131, 259, 850, 901]
LIMITATIONS = {
    "unlabeled_exploratory_screening_only", "physical_4d_registration_unverified",
    "well2_seismic_tie_unverified", "acquisition_processing_comparability_unverified",
    "geological_comparability_uncertain", "sample_domain_unverified",
}
ATTRIBUTE_DIRECTORY = "full-field-attributes/attributes-0a361aa415beca729b53"


def resolve_attribute_directory(output_root: Path, config: dict) -> Path:
    relative = Path(str(config.get("attribute_run_directory", "")))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError("Stage 1 attribute directory must be a safe relative path")
    if relative.as_posix() != ATTRIBUTE_DIRECTORY:
        raise ValueError("pilot must use the verified Stage 1 attribute run")
    root = output_root.expanduser().resolve()
    resolved = (root / relative).resolve()
    if root not in resolved.parents:
        raise ValueError("Stage 1 attribute directory must remain under the private output root")
    return resolved


def validate_pilot_config(config: dict, *, output_root: Path | None = None) -> dict:
    if config.get("schema") != "rtm-bounded-ml-pilot-config-1":
        raise ValueError("unsupported bounded-pilot configuration schema")
    if config.get("status") != "BOUNDED_EXPLORATORY_CONFIGURATION":
        raise ValueError("pilot config must identify an exploratory configuration, not an approval")
    if config.get("full_field_screening_status") != "REVIEW_REQUIRED_BEFORE_EXECUTION":
        raise ValueError("bounded pilot cannot approve full-field screening")
    if config.get("experiment_id") != "well2-coal-control-v1":
        raise ValueError("pilot experiment ID mismatch")
    if config.get("attribute_run_directory") != ATTRIBUTE_DIRECTORY:
        raise ValueError("pilot Stage 1 attribute directory mismatch")
    if config.get("reference_bounds") != REFERENCE or config.get("calibration_bounds") != CALIBRATION:
        raise ValueError("Design A reference/calibration bounds mismatch")
    if config.get("calibration_separation_grid_cells") != 1123.0071237530062:
        raise ValueError("Design A separation metadata mismatch")
    if config.get("sample_bounds") != [850, 901] or config.get("scoring_bounds") != SCORING:
        raise ValueError("Well-2 fitting/scoring geometry mismatch")
    expected = {
        "minimum_separation_grid_cells": 32,
        "min_support_fraction": 0.8,
        "threshold_quantile": 0.99,
        "mad_multiplier": 3.5,
        "max_samples": 10000,
        "max_score_voxels": 1000000,
        "max_output_bytes": 134217728,
        "max_memory_mib": 2048,
        "compare_chunks": [[16, 16, 64], [32, 32, 128], [64, 64, 128]],
        "threads": 4,
        "random_seed": 2026,
        "max_priority_discoveries": 10,
        "candidate_catalogue": True,
        "qc_catalogue": True,
        "ground_truth_status": "unlabeled_screening_only",
        "physical_registration_status": "unverified",
        "well2_seismic_tie_status": "unverified",
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"bounded-pilot scientific/resource setting mismatch: {key}")
    if config.get("mandatory_edge_qc") is not True:
        raise ValueError("mandatory edge QC cannot be disabled")
    if config.get("monitor_only_paired_use") is not False:
        raise ValueError("Monitor-only traces must remain excluded from paired calculations")
    if not LIMITATIONS.issubset(set(config.get("scientific_limitations", []))):
        raise ValueError("required exploratory scientific limitations are missing")
    if output_root is not None:
        output_root = output_root.expanduser().resolve()
        if (not output_root.is_dir() or output_root.stat().st_uid != os.getuid()
                or output_root.stat().st_mode & 0o077):
            raise ValueError("pilot output root must be existing, private and user-owned")
        attribute_dir = resolve_attribute_directory(output_root, config)
        if not attribute_dir.is_dir() or not (attribute_dir / "complete.json").is_file():
            raise ValueError(f"verified Stage 1 attribute manifest is missing: {attribute_dir / 'complete.json'}")
    return {"status": "pilot_configuration_valid", "experiment_id": config["experiment_id"],
            "full_field_screening_status": "REVIEW_REQUIRED_BEFORE_EXECUTION"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    try:
        result = validate_pilot_config(json.loads(args.config.read_text(encoding="utf-8")),
                                       output_root=args.output_root)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
