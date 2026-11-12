#!/usr/bin/env python3
"""Interactively record genuine Stage 2 approvals, validate, and replace safely.

Run in a normal HPC login terminal. This tool never submits a Slurm job.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

from validate_full_field_ml_review import validate_config


REQUIRED_LIMITATIONS = [
    "neither_region_verified_unchanged",
    "geological_comparability_uncertain",
    "acquisition_processing_comparability_unverified",
    "physical_4d_registration_unverified",
    "sample_domain_unverified",
    "well_to_seismic_tie_unverified",
]
APPROVED_SCOPE = ["design_a_reference_calibration_fit", "well2_coal_control_scoring"]


def _ask(label: str) -> str:
    value = input(f"{label}: ").strip()
    if not value or any(token in value.upper() for token in ("REPLACE", "PLACEHOLDER", "INSERT", "TODO", "ACTUAL_")):
        raise ValueError(f"{label} must be a genuine, non-placeholder value")
    return value


def build_candidate(current: dict, answers: dict) -> dict:
    """Apply only the reviewed Design A bounds and genuine review fields."""
    result = json.loads(json.dumps(current))
    review = result.get("calibration_review")
    if not isinstance(review, dict):
        raise ValueError("calibration_review object is missing")
    if answers.get("decision_confirmation") != "APPROVED WITH CONDITIONS":
        raise ValueError("operator did not confirm the approved-with-conditions decision")
    if answers.get("scope_confirmation") != "YES":
        raise ValueError("review scope must explicitly include Design A and independent Well-2 scoring")
    if answers.get("limitations_confirmation") != "YES":
        raise ValueError("all stated scientific limitations must be accepted")
    try:
        capacity = int(answers["authorized_capacity_bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("authorized capacity must be entered as an integer number of bytes") from exc
    if capacity < 134217728:
        raise ValueError("authorized capacity must be at least 134217728 bytes for the bounded pilot")
    for name in ("reviewed_by", "review_id", "review_notes", "storage_authorization_reference",
                 "storage_authorization_evidence"):
        value = str(answers.get(name, "")).strip()
        if not value or any(token in value.upper() for token in ("REPLACE", "PLACEHOLDER", "INSERT", "TODO", "ACTUAL_")):
            raise ValueError(f"{name} must contain the genuine approved record")
        review[name] = value
    review.update({
        "decision": "approved_with_conditions",
        "approved_scope": APPROVED_SCOPE,
        "decision_scope": "unlabeled_exploratory_4d_anomaly_screening_only",
        "authorized_capacity_bytes": capacity,
        "accepted_limitations": REQUIRED_LIMITATIONS,
        "ground_truth_status": "unlabeled_screening_only",
        "physical_registration_status": "unverified",
        "selection_basis": "grid_partition_selected_without_inspecting_anomaly_scores",
    })
    result["reference_bounds"] = [64, 192, 1088, 1216]
    result["calibration_bounds"] = [869, 997, 64, 192]
    proposal = review.get("occupancy_proposal")
    if not isinstance(proposal, dict):
        raise ValueError("embedded occupancy_proposal is missing")
    proposal["reference_bounds"] = [64, 192, 1088, 1216]
    proposal["calibration_bounds"] = [869, 997, 64, 192]
    proposal["measured_separation_grid_cells"] = 1123.0071237530062
    result["status"] = "APPROVED_FOR_SCREENING"
    return result


def _write_private_candidate(path: Path, payload: dict, mode: int) -> Path:
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.candidate.", suffix=".json", dir=path.parent)
    candidate = Path(name)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        return candidate
    except Exception:
        candidate.unlink(missing_ok=True)
        raise


def _run_validator(code_root: Path, candidate: Path, output_root: Path, cache: Path, data: Path) -> None:
    validate_config(json.loads(candidate.read_text(encoding="utf-8")), output_root, cache, data)
    subprocess.run([sys.executable, str(code_root / "scripts/hpc/validate_full_field_ml_review.py"),
                    "--config", str(candidate), "--output-root", str(output_root),
                    "--paired-cache", str(cache), "--protected-data-root", str(data)], check=True)


def _dry_run_launcher(code_root: Path, candidate: Path) -> None:
    env = os.environ.copy()
    env["RTM_FIELD_ML_CONFIG"] = str(candidate)
    subprocess.run(["bash", str(code_root / "scripts/hpc/submit_bounded_ml_pilot.sh"), "--dry-run"],
                   check=True, env=env)


def record_and_validate(config_path: Path, code_root: Path, output_root: Path,
                        cache: Path, data: Path, input_fn=input) -> Path:
    config_path = config_path.expanduser().absolute()
    if config_path.is_symlink() or not config_path.is_file():
        raise ValueError("configuration must be a regular, non-symlink file")
    config_path = config_path.resolve(strict=True)
    for path in (config_path.parent, output_root, cache, data):
        if not path.exists():
            raise ValueError(f"required path does not exist: {path}")
    if not os.access(config_path.parent, os.W_OK):
        raise PermissionError("configuration directory is not writable by this operator")

    print("Enter values from the actual scientific review and storage authorization records.")
    print("No approval is inferred from this prompt; answer the confirmations only if the records support them.")
    answers = {
        "reviewed_by": input_fn("Actual scientific reviewer identity").strip(),
        "review_id": input_fn("Genuine reviewer-assigned review ID").strip(),
        "decision_confirmation": input_fn("Type exactly APPROVED WITH CONDITIONS if the reviewer approved").strip(),
        "review_notes": input_fn("Review notes: Design A fitting/calibration plus independent Well-2 scoring").strip(),
        "scope_confirmation": input_fn("Confirm that scope explicitly covers both Design A and Well-2; type YES").strip(),
        "limitations_confirmation": input_fn("Confirm acceptance of registration, sample/well tie, acquisition and geology limits; type YES").strip(),
        "storage_authorization_reference": input_fn("Actual HPC storage authorization reference").strip(),
        "storage_authorization_evidence": input_fn("Storage approval evidence and approved destination/capacity scope").strip(),
        "authorized_capacity_bytes": input_fn("Authorized private capacity in bytes").strip(),
    }
    current = json.loads(config_path.read_text(encoding="utf-8"))
    candidate_payload = build_candidate(current, answers)
    candidate = _write_private_candidate(config_path, candidate_payload,
                                         stat.S_IMODE(config_path.stat().st_mode))
    backup = None
    try:
        _run_validator(code_root, candidate, output_root, cache, data)
        _dry_run_launcher(code_root, candidate)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = config_path.with_name(f"{config_path.name}.backup.{timestamp}")
        # Exclusive create: never replace a prior backup.
        with backup.open("xb") as stream:
            stream.write(config_path.read_bytes())
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(backup, stat.S_IMODE(config_path.stat().st_mode))
        os.replace(candidate, config_path)
        dir_fd = os.open(config_path.parent, os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
        try:
            _dry_run_launcher(code_root, config_path)
        except Exception:
            rollback = _write_private_candidate(config_path,
                json.loads(backup.read_text(encoding="utf-8")), stat.S_IMODE(config_path.stat().st_mode))
            os.replace(rollback, config_path)
            raise
        print(f"Approval recorded and validated. Backup: {backup}")
        print("No Slurm job was submitted. Use the separate --submit launcher only after operator authorization.")
        return backup
    finally:
        candidate.unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag, env_name in (("--config", "RTM_FIELD_ML_CONFIG"),
                           ("--output-root", "RTM_HPC_OUTPUT_ROOT"),
                           ("--paired-cache", "RTM_PAIRED_CACHE_DIR"),
                           ("--protected-data-root", "RTM_CHALLENGE3_ROOT")):
        value = os.environ.get(env_name)
        parser.add_argument(flag, type=Path, default=value, required=not bool(value))
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args(argv)
    try:
        record_and_validate(args.config, args.source_root.resolve(), args.output_root,
                            args.paired_cache, args.protected_data_root)
        return 0
    except (EOFError, OSError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: approval was not safely recorded: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
