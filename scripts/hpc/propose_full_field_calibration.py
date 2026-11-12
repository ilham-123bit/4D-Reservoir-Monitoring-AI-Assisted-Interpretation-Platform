#!/usr/bin/env python3
"""Create a score-blind spatial calibration proposal from occupancy QC only."""
import argparse
import json
import os
import tempfile
import uuid
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from reservoir_time_machine.analysis.full_field_models import propose_spatial_regions_from_occupancy
from reservoir_time_machine.processing.full_field import _canonical


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--attributes", type=Path, required=True, help="completed full-volume attribute run")
    p.add_argument("--output", type=Path, required=True, help="new private JSON configuration path")
    p.add_argument("--minimum-occupied-fraction", type=float, default=0.95)
    p.add_argument("--buffer-grid-cells", type=int, default=32)
    p.add_argument("--region-inline", type=int, default=128)
    p.add_argument("--region-crossline", type=int, default=128)
    p.add_argument("--stride-inline", type=int, default=64)
    p.add_argument("--stride-crossline", type=int, default=64)
    a = p.parse_args(argv)
    output_root_text = os.environ.get("RTM_HPC_OUTPUT_ROOT")
    if not output_root_text:
        p.error("RTM_HPC_OUTPUT_ROOT must identify approved private storage")
    output_root = Path(output_root_text).expanduser().resolve()
    output = a.output.expanduser().resolve()
    if not output_root.is_dir() or (output != output_root and output_root not in output.parents):
        p.error("Output must be a new path under approved RTM_HPC_OUTPUT_ROOT")
    if (not output.parent.is_dir() or output.exists()
            or output.parent.stat().st_mode & 0o077):
        p.error("Output must not exist; parent must be private (mode 0700)")
    if not (a.cache / "metadata.json").is_file() or not (a.cache / "missing_trace_mask.npy").is_file():
        p.error("Paired cache metadata and occupancy mask are required")
    envelope = json.loads((a.attributes / "complete.json").read_text(encoding="utf-8"))
    body = envelope.get("body", {})
    domain = body.get("configuration", {}).get("source_domain_shape", [0, 0, 0])
    full_bounds = [0, domain[0], 0, domain[1], 0, domain[2]] if len(domain) == 3 else []
    if (envelope.get("sha256") != __import__("hashlib").sha256(_canonical(body)).hexdigest()
            or body.get("status") != "complete"
            or body.get("configuration", {}).get("source_bounds") != full_bounds):
        p.error("Attribute completion artifact is invalid or not full-volume")
    metadata = json.loads((a.cache / "metadata.json").read_text(encoding="utf-8"))
    shape = tuple(metadata["shape"])
    if tuple(body.get("configuration", {}).get("shape", ())) != shape:
        p.error("Completed attributes and paired cache geometry differ")
    try:
        a.attributes.expanduser().resolve().relative_to(output_root)
    except ValueError:
        p.error("Completed attribute run must be under RTM_HPC_OUTPUT_ROOT")
    missing = np.load(a.cache / "missing_trace_mask.npy", mmap_mode="r", allow_pickle=False)
    proposal = propose_spatial_regions_from_occupancy(
        missing, region_shape=(a.region_inline, a.region_crossline),
        stride=(a.stride_inline, a.stride_crossline),
        buffer_grid_cells=a.buffer_grid_cells,
        minimum_occupied_fraction=a.minimum_occupied_fraction)
    result = {
        "schema": "rtm-full-field-ml-config-1",
        "status": "REVIEW_REQUIRED_BEFORE_EXECUTION",
        "attribute_run_directory": str(a.attributes.expanduser().resolve().relative_to(output_root)),
        "analysis_output_directory": f"full-field-ml-runs/{uuid.uuid4().hex}",
        "reference_bounds": proposal["reference_bounds"],
        "calibration_bounds": proposal["calibration_bounds"],
        "reference_population_id": "score-blind-occupancy-reference",
        "calibration_population_id": "score-blind-occupancy-calibration",
        "calibration_review": {
            "review_id": "REPLACE_WITH_REVIEW_ID",
            "reviewed_by": "REPLACE_WITH_REVIEWER",
            "selection_basis": "grid_partition_selected_without_inspecting_anomaly_scores",
            "ground_truth_status": "unlabeled_screening_only",
            "physical_registration_status": "unverified",
            "minimum_buffer_grid_cells": a.buffer_grid_cells,
            "minimum_occupied_fraction": a.minimum_occupied_fraction,
            "occupancy_proposal": proposal,
            "required_review": "Confirm geometry and survey coverage without inspecting NRMS, amplitude, detector scores, or findings. Do not describe either population as unchanged geology."
        },
        "threshold_quantile": 0.99,
        "max_population_rows": 10000,
        "random_state": 2026,
        "min_region_voxels": 16,
        "min_ml_votes": 2,
        "mad_multiplier": 3.5,
        "max_findings": 10,
        "limitations": ["Screening thresholds are not probabilities.",
                        "Physical 4D registration and sample origin are unverified.",
                        "Calibration populations are unlabeled spatial samples."]
    }
    fd, temporary = tempfile.mkstemp(prefix=".calibration-proposal-", dir=output.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(_canonical(result) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        Path(temporary).unlink(missing_ok=True)
    print(json.dumps({"status": result["status"], "config": str(output),
                      "reference_bounds": result["reference_bounds"],
                      "calibration_bounds": result["calibration_bounds"],
                      "reference_occupied_fraction": proposal["reference_occupied_fraction"],
                      "calibration_occupied_fraction": proposal["calibration_occupied_fraction"],
                      "separation_grid_cells": proposal["measured_separation_grid_cells"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
