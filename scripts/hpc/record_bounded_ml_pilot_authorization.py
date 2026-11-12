#!/usr/bin/env python3
"""Create a separate, private authorization record for the bounded Well-2 pilot.

This does not edit the strict full-field review config and never submits Slurm.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone

try:
    from validate_bounded_ml_pilot_authorization import validate_pilot_config
except ModuleNotFoundError:  # Package import in unit tests.
    from scripts.hpc.validate_bounded_ml_pilot_authorization import validate_pilot_config


REFERENCE = [64, 192, 1088, 1216]
CALIBRATION = [869, 997, 64, 192]
SCORING = [600, 728, 131, 259, 850, 901]
LIMITATIONS = [
    "unlabeled_exploratory_screening_only", "physical_4d_registration_unverified",
    "well2_seismic_tie_unverified", "acquisition_processing_comparability_unverified",
    "geological_comparability_uncertain", "sample_domain_unverified",
]
SCOPE = ["design_a_reference_calibration_fit", "well2_coal_control_scoring"]


def ask(prompt, input_fn=input):
    value = input_fn(prompt).strip()
    if not value or any(token in value.upper() for token in ("REPLACE", "PLACEHOLDER", "INSERT", "TODO", "ACTUAL_")):
        raise ValueError(f"{prompt} requires actual information; placeholders are refused")
    return value


def build_config(base, answers):
    decision = answers["decision"].strip()
    if decision != "APPROVED WITH CONDITIONS":
        raise ValueError("no approved-with-conditions decision was recorded; pilot remains unauthorized")
    if answers["scope_confirmation"].strip() != "YES":
        raise ValueError("authorizing authority did not confirm both Design A fitting and Well-2 scoring")
    if answers["limitations_confirmation"].strip() != "YES":
        raise ValueError("scientific limitations were not accepted")
    try:
        capacity = int(answers["authorized_output_limit_bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("authorized output limit must be entered as integer bytes") from exc
    if capacity < 134217728:
        raise ValueError("authorized output limit must cover the 134217728-byte pilot cap")
    config = {
        "schema": "rtm-bounded-ml-pilot-config-1",
        "status": "AUTHORIZED_FOR_BOUNDED_PILOT",
        "full_field_screening_status": "REVIEW_REQUIRED_BEFORE_EXECUTION",
        "experiment_id": "well2-coal-control-v1",
        "reference_bounds": REFERENCE,
        "calibration_bounds": CALIBRATION,
        "calibration_separation_grid_cells": 1123.0071237530062,
        "sample_bounds": [850, 901],
        "scoring_bounds": SCORING,
        "minimum_separation_grid_cells": 32,
        "min_support_fraction": float(base.get("min_support_fraction", 0.8)),
        "threshold_quantile": float(base.get("threshold_quantile", 0.99)),
        "mad_multiplier": float(base.get("mad_multiplier", 3.5)),
        "max_samples": 10000,
        "max_score_voxels": 1000000,
        "max_output_bytes": 134217728,
        "max_memory_mib": 2048,
        "compare_chunks": [[16, 16, 64], [32, 32, 128], [64, 64, 128]],
        "threads": 4,
        "random_seed": 2026,
        "mandatory_edge_qc": True,
        "monitor_only_paired_use": False,
        "max_priority_discoveries": 10,
        "candidate_catalogue": True,
        "qc_catalogue": True,
        "ground_truth_status": "unlabeled_screening_only",
        "physical_registration_status": "unverified",
        "well2_seismic_tie_status": "unverified",
        "scientific_limitations": LIMITATIONS,
        "authorization": {
            "approved_by": answers["approved_by"].strip(),
            "approver_authority": answers["approver_authority"].strip(),
            "decision": "approved_with_conditions",
            "approved_scope": SCOPE,
            "decision_evidence": answers["decision_evidence"].strip(),
            "review_notes": answers["review_notes"].strip(),
            "accepted_limitations": LIMITATIONS,
            "storage_evidence": answers["storage_evidence"].strip(),
            "authorized_output_limit_bytes": capacity,
            "recorded_by": answers["recorded_by"].strip(),
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    }
    validate_pilot_config(config)
    return config


def create_authorization(output: Path, full_field_config: Path, output_root: Path, input_fn=input):
    output = output.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite an existing pilot authorization: {output}")
    base = json.loads(full_field_config.read_text(encoding="utf-8"))
    print("This authorization is only for the bounded unlabeled Well-2 trial.")
    print("It does not approve full-field screening. The reviewer must accept:")
    for limitation in LIMITATIONS:
        print(f"  - {limitation}")
    answers = {
        "approved_by": ask("Actual name/team of the authority approving this bounded pilot", input_fn),
        "approver_authority": ask("Basis of that authority to approve this bounded experiment", input_fn),
        "decision": ask("Type exactly APPROVED WITH CONDITIONS only if that is the actual decision", input_fn),
        "scope_confirmation": ask("Confirm Design A fitting/calibration AND independent Well-2 scoring; type YES", input_fn),
        "review_notes": ask("Conditions and notes recorded by the approving authority", input_fn),
        "limitations_confirmation": ask("Confirm acceptance of all displayed scientific limitations; type YES", input_fn),
        "decision_evidence": ask("Evidence available for the decision (for example, actual meeting/email record; no ticket required)", input_fn),
        "storage_evidence": ask("Actual HPC permission/evidence for this private, recyclable output location", input_fn),
        "authorized_output_limit_bytes": ask("Authorized pilot output limit in bytes (must be at least 134217728)", input_fn),
        "recorded_by": ask("Actual operator entering this record", input_fn),
    }
    config = build_config(base, answers)
    validate_pilot_config(config, output_root=output_root)
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    temporary = Path(name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(config, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Hard-link creation is atomic and refuses to replace a concurrent file.
        os.link(temporary, output)
        directory_fd = os.open(output.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Private bounded-pilot authorization created: {output}")
    print("The full-field review JSON was not changed. No Slurm job was submitted.")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag, env_name in (("--output", "RTM_BOUNDED_PILOT_CONFIG"),
                           ("--full-field-config", "RTM_FIELD_ML_CONFIG"),
                           ("--output-root", "RTM_HPC_OUTPUT_ROOT")):
        value = os.environ.get(env_name)
        parser.add_argument(flag, type=Path, default=value, required=not bool(value))
    args = parser.parse_args(argv)
    try:
        create_authorization(args.output, args.full_field_config, args.output_root)
        return 0
    except (EOFError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: bounded pilot authorization was not created: {exc}", file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
