#!/usr/bin/env python3
"""Fail-closed, lightweight pre-submission check for a human-reviewed ML config."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _bounds(value, name):
    if (not isinstance(value, list) or len(value) != 4
            or any(isinstance(x, bool) or not isinstance(x, int) for x in value)):
        raise ValueError(f"{name} must be four integer half-open grid bounds")
    i0, i1, x0, x1 = value
    if min(i0, x0) < 0 or i1 <= i0 or x1 <= x0:
        raise ValueError(f"{name} is not a valid nonempty half-open region")
    return value


PREFERRED_REFERENCE = [64, 192, 1088, 1216]
PREFERRED_CALIBRATION = [869, 997, 64, 192]


def validate_config(config: dict, output_root: Path, paired_cache: Path,
                    protected_data_root: Path) -> dict:
    """Validate scientific mode, calibration geometry, capacity and protected paths."""
    if config.get("schema") != "rtm-full-field-ml-config-1":
        raise ValueError("unsupported full-field ML config schema")
    mode = config.get("status")
    if mode == "APPROVED_FOR_SCREENING":
        review = config.get("calibration_review")
        if not isinstance(review, dict):
            raise ValueError("calibration_review record is required")
        for field in ("reviewed_by", "review_id"):
            value = review.get(field)
            if not isinstance(value, str) or not value.strip() or value.strip().upper().startswith("REPLACE_"):
                raise ValueError(f"calibration_review.{field} must contain the actual human review value")
        for field in ("storage_authorization_reference", "storage_authorization_evidence", "review_notes"):
            value = review.get(field)
            if not isinstance(value, str) or not value.strip() or any(
                    token in value.upper() for token in ("REPLACE", "PLACEHOLDER", "INSERT", "TODO", "ACTUAL_")):
                raise ValueError(f"calibration_review.{field} must contain the actual authorization or review record")
        if review.get("decision") != "approved_with_conditions":
            raise ValueError("calibration_review.decision must record the genuine approved-with-conditions decision")
        required_scope = {"design_a_reference_calibration_fit", "well2_coal_control_scoring"}
        if not isinstance(review.get("approved_scope"), list) or not required_scope.issubset(set(review["approved_scope"])):
            raise ValueError("calibration_review.approved_scope must include Design A and Well-2")
        try:
            capacity = int(review["authorized_capacity_bytes"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("calibration_review.authorized_capacity_bytes is required") from exc
        if capacity < 134217728:
            raise ValueError("approved private capacity must cover the bounded pilot's 128 MiB output ceiling")
        if review.get("decision_scope") != "unlabeled_exploratory_4d_anomaly_screening_only":
            raise ValueError("decision_scope must remain unlabeled exploratory 4D anomaly screening only")
        limitations = set(review.get("accepted_limitations", []))
        required_limitations = {"neither_region_verified_unchanged", "geological_comparability_uncertain",
            "acquisition_processing_comparability_unverified", "physical_4d_registration_unverified", "sample_domain_unverified"}
        if not required_limitations.issubset(limitations):
            raise ValueError("review must record acceptance of all known scientific limitations")
        if (review.get("ground_truth_status") != "unlabeled_screening_only"
                or review.get("physical_registration_status") != "unverified"
                or review.get("selection_basis") != "grid_partition_selected_without_inspecting_anomaly_scores"):
            raise ValueError("review must preserve unlabeled, unregistered, score-blind limitations")
        design = review
    elif mode == "UNLABELED_EXPLORATORY_SCREENING":
        if config.get("execution_scope") != "operator_authorized_unlabeled_full_field_screening":
            raise ValueError("exploratory execution scope is missing or unsupported")
        design = config.get("calibration_design")
        if not isinstance(design, dict):
            raise ValueError("score-blind calibration_design is required for exploratory screening")
        if design.get("record_status") != "not_a_geological_approval":
            raise ValueError("exploratory design must not imply independent scientific approval")
        required_limits = {"physical_4d_registration_unverified", "sample_origin_unverified",
            "geological_comparability_uncertain", "acquisition_processing_comparability_unverified",
            "well_to_seismic_tie_unverified", "thresholds_are_not_probabilities"}
        if not required_limits.issubset(set(design.get("limitations", []))):
            raise ValueError("exploratory config omits a required scientific limitation")
        if design.get("ground_truth_status") != "unlabeled_screening_only":
            raise ValueError("exploratory config must remain unlabeled")
    else:
        raise ValueError("config status must be APPROVED_FOR_SCREENING or UNLABELED_EXPLORATORY_SCREENING")

    reference = _bounds(config.get("reference_bounds"), "reference_bounds")
    calibration = _bounds(config.get("calibration_bounds"), "calibration_bounds")
    if reference != PREFERRED_REFERENCE or calibration != PREFERRED_CALIBRATION:
        raise ValueError("reviewed configuration must use the team's preferred reference and calibration bounds")
    proposal = design.get("occupancy_proposal")
    if not isinstance(proposal, dict):
        raise ValueError("occupancy_proposal is required")
    if proposal.get("schema") != "rtm-calibration-region-proposal-1":
        raise ValueError("unsupported occupancy proposal schema")
    if proposal.get("status") != "requires_human_review":
        raise ValueError("embedded occupancy proposal must retain its review provenance status")
    if proposal.get("reference_bounds") != reference or proposal.get("calibration_bounds") != calibration:
        raise ValueError("top-level bounds disagree with embedded occupancy proposal bounds")
    if "occupancy only" not in str(proposal.get("selection_basis", "")).lower():
        raise ValueError("embedded proposal must document score-blind occupancy-only selection")
    try:
        minimum_occupancy = float(design["minimum_occupied_fraction"])
        minimum_buffer = int(design["minimum_buffer_grid_cells"])
        ref_occ = float(proposal["reference_occupied_fraction"])
        cal_occ = float(proposal["calibration_occupied_fraction"])
        recorded_separation = float(proposal["measured_separation_grid_cells"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("review occupancy and separation measurements are incomplete") from exc
    if not 0.5 <= minimum_occupancy <= 1 or minimum_buffer < 1:
        raise ValueError("invalid minimum occupancy or spatial buffer")
    if (proposal.get("minimum_buffer_grid_cells") != minimum_buffer
            or abs(float(proposal.get("minimum_occupied_fraction", -1)) - minimum_occupancy) > 1e-12):
        raise ValueError("review thresholds disagree with embedded occupancy proposal")
    if ref_occ < minimum_occupancy or cal_occ < minimum_occupancy:
        raise ValueError("reviewed regions do not meet the minimum occupancy")
    gaps = (max(0, calibration[0] - reference[1], reference[0] - calibration[1]),
            max(0, calibration[2] - reference[3], reference[2] - calibration[3]))
    separation = (gaps[0] ** 2 + gaps[1] ** 2) ** 0.5
    if separation + 1e-9 < minimum_buffer:
        raise ValueError("reference/calibration regions do not meet the spatial buffer")
    if abs(recorded_separation - separation) > 1e-6:
        raise ValueError("recorded proposal separation disagrees with the region bounds")
    if mode == "UNLABELED_EXPLORATORY_SCREENING":
        if design.get("execution_mode") != "unlabeled_exploratory_screening":
            raise ValueError("calibration design execution mode is invalid")
        if (design.get("selection_basis") != "grid_partition_selected_without_inspecting_anomaly_scores"
                or design.get("physical_registration_status") != "unverified"):
            raise ValueError("exploratory design must remain score-blind and physically unregistered")
    if mode == "UNLABELED_EXPLORATORY_SCREENING":
        fit_bounds = config.get("fit_sample_bounds")
        if (not isinstance(fit_bounds, list) or len(fit_bounds) != 2
                or any(isinstance(v, bool) or not isinstance(v, int) for v in fit_bounds)
                or not 0 <= fit_bounds[0] < fit_bounds[1] <= int(config.get("expected_shape", [0, 0, 0])[2])):
            raise ValueError("fit_sample_bounds must be valid half-open source sample indices")

    output_root = output_root.expanduser().resolve()
    paired_cache = paired_cache.expanduser().resolve()
    protected_data_root = protected_data_root.expanduser().resolve()
    if not output_root.is_dir():
        raise ValueError("private ML output root must already exist")
    output_stat = output_root.stat()
    if output_stat.st_uid != os.getuid() or output_stat.st_mode & 0o077:
        raise ValueError("ML output root must be user-owned and inaccessible to group/others")
    for label, protected in (("paired cache", paired_cache), ("original data", protected_data_root)):
        if output_root == protected or output_root in protected.parents or protected in output_root.parents:
            raise ValueError(f"ML output root must not overlap {label}")
    attr_rel = Path(config.get("attribute_run_directory", ""))
    analysis_rel = Path(config.get("analysis_output_directory", ""))
    for name, rel in (("attribute_run_directory", attr_rel), ("analysis_output_directory", analysis_rel)):
        if not str(rel) or rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"{name} must be a safe relative path under the private output root")
    attr_path = (output_root / attr_rel).resolve()
    analysis_path = (output_root / analysis_rel).resolve()
    if output_root not in attr_path.parents or output_root not in analysis_path.parents:
        raise ValueError("configured products must remain under the private output root")
    if attr_path == analysis_path or attr_path in analysis_path.parents or analysis_path in attr_path.parents:
        raise ValueError("ML analysis output must be separate from completed attribute products")
    # Budget the four float32 scores, four uint8 detector flags, three uint8
    # candidate masks, int64 global component labels, journals and QC overhead.
    shape = config.get("expected_shape", [1029, 1248, 1301])
    if len(shape) != 3 or any(int(n) <= 0 for n in shape):
        raise ValueError("expected_shape must be a positive 3D source geometry")
    estimated = int(shape[0]) * int(shape[1]) * int(shape[2]) * (4 * 5 + 3 + 8)
    minimum_free = int(config.get("minimum_free_output_bytes", 65_000_000_000))
    if minimum_free < estimated or minimum_free < 1:
        raise ValueError("configured free-space budget is below the estimated score/component output size")
    usage = os.statvfs(output_root)
    free_bytes = usage.f_bavail * usage.f_frsize
    if free_bytes < minimum_free:
        raise ValueError(f"private output filesystem has {free_bytes} free bytes; need at least {minimum_free}")
    if analysis_path.exists() and (analysis_path.is_symlink() or not analysis_path.is_dir()):
        raise ValueError("analysis output path must be a directory, not a symlink or file")
    result = {"status": "exploratory_config_valid" if mode == "UNLABELED_EXPLORATORY_SCREENING" else "review_config_valid",
              "reference_bounds": reference, "calibration_bounds": calibration,
              "separation_grid_cells": separation, "estimated_analysis_output_bytes": estimated,
              "minimum_free_bytes": minimum_free, "available_filesystem_bytes": free_bytes}
    if mode == "APPROVED_FOR_SCREENING":
        result.update(reviewed_by=review["reviewed_by"], review_id=review["review_id"])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--paired-cache", required=True, type=Path)
    parser.add_argument("--protected-data-root", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        result = validate_config(config, args.output_root, args.paired_cache,
                                 args.protected_data_root)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
