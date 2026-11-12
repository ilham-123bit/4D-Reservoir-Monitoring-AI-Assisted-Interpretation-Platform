#!/usr/bin/env python3
"""Validate the distinct, bounded Stage 2 pilot authorization record."""

from __future__ import annotations

import json
import os
from pathlib import Path


SCOPES = {"design_a_reference_calibration_fit", "well2_coal_control_scoring"}
LIMITATIONS = {
    "unlabeled_exploratory_screening_only",
    "physical_4d_registration_unverified",
    "well2_seismic_tie_unverified",
    "acquisition_processing_comparability_unverified",
    "geological_comparability_uncertain",
    "sample_domain_unverified",
}
REFERENCE = [64, 192, 1088, 1216]
CALIBRATION = [869, 997, 64, 192]
SCORING = [600, 728, 131, 259, 850, 901]


def _real_text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"authorization.{field} must contain actual operator-provided evidence")
    if any(token in value.upper() for token in ("REPLACE", "PLACEHOLDER", "INSERT", "TODO", "ACTUAL_")):
        raise ValueError(f"authorization.{field} contains a placeholder")
    return value.strip()


def validate_pilot_config(config: dict, *, output_root: Path | None = None) -> dict:
    if config.get("schema") != "rtm-bounded-ml-pilot-config-1":
        raise ValueError("unsupported bounded-pilot config schema")
    if config.get("status") != "AUTHORIZED_FOR_BOUNDED_PILOT":
        raise ValueError("bounded pilot has no recorded authorized decision")
    if config.get("full_field_screening_status") != "REVIEW_REQUIRED_BEFORE_EXECUTION":
        raise ValueError("pilot authorization must not approve full-field screening")
    if config.get("experiment_id") != "well2-coal-control-v1":
        raise ValueError("pilot experiment ID does not match the approved Well-2 trial")
    if config.get("reference_bounds") != REFERENCE or config.get("calibration_bounds") != CALIBRATION:
        raise ValueError("pilot config does not preserve the approved Design A regions")
    if config.get("sample_bounds") != [850, 901] or config.get("scoring_bounds") != SCORING:
        raise ValueError("pilot config does not match the Well-2 fitting/scoring geometry")
    if config.get("calibration_separation_grid_cells") != 1123.0071237530062:
        raise ValueError("Design A calibration separation metadata is inconsistent")
    if (config.get("minimum_separation_grid_cells") != 32
            or config.get("sample_bounds") != [850, 901]
            or config.get("scoring_bounds") != SCORING):
        raise ValueError("pilot geometry and sample context differ from the reviewed Well-2 experiment")
    if (config.get("max_samples") != 10000 or config.get("max_score_voxels") != 1000000
            or config.get("max_output_bytes") != 134217728 or config.get("max_memory_mib") != 2048
            or config.get("compare_chunks") != [[16, 16, 64], [32, 32, 128], [64, 64, 128]]
            or config.get("threads") != 4 or config.get("random_seed") != 2026):
        raise ValueError("pilot resource, chunk, sampling, or seed settings differ from the reviewed bounded configuration")
    if (config.get("min_support_fraction") != 0.8 or config.get("threshold_quantile") != 0.99
            or config.get("mad_multiplier") != 3.5):
        raise ValueError("support and exploratory screening thresholds differ from the reviewed configuration")
    authorization = config.get("authorization")
    if not isinstance(authorization, dict):
        raise ValueError("bounded-pilot authorization record is required")
    for field in ("approved_by", "approver_authority", "decision_evidence", "review_notes",
                  "storage_evidence"):
        _real_text(authorization.get(field), field)
    if authorization.get("decision") != "approved_with_conditions":
        raise ValueError("authorization decision must be approved_with_conditions")
    if not SCOPES.issubset(set(authorization.get("approved_scope", []))):
        raise ValueError("authorization scope must include Design A fitting and independent Well-2 scoring")
    if not LIMITATIONS.issubset(set(authorization.get("accepted_limitations", []))):
        raise ValueError("pilot authorization must explicitly accept all recorded scientific limitations")
    try:
        authorized_bytes = int(authorization["authorized_output_limit_bytes"])
        max_output_bytes = int(config["max_output_bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("authorized output limit and pilot output budget must be integer bytes") from exc
    if max_output_bytes != 134217728 or authorized_bytes < max_output_bytes:
        raise ValueError("authorization must cover the configured 128 MiB bounded-pilot output ceiling")
    if config.get("mandatory_edge_qc") is not True or config.get("monitor_only_paired_use") is not False:
        raise ValueError("mandatory edge QC and Monitor-only exclusion must remain enabled")
    if config.get("max_priority_discoveries") != 10 or config.get("candidate_catalogue") is not True or config.get("qc_catalogue") is not True:
        raise ValueError("pilot must preserve variable 0-10 ranking and separate candidate/QC catalogues")
    if output_root is not None:
        output_root = output_root.expanduser().resolve()
        if not output_root.is_dir() or output_root.stat().st_uid != os.getuid() or output_root.stat().st_mode & 0o077:
            raise ValueError("bounded pilot output root must be existing, private and user-owned")
    return {"status": "pilot_authorization_valid", "approved_by": authorization["approved_by"],
            "scope": sorted(SCOPES), "full_field_screening_status": "REVIEW_REQUIRED_BEFORE_EXECUTION"}


def main(argv=None):
    import argparse
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
