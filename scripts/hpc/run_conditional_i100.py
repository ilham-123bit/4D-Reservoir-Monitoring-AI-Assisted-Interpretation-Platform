#!/usr/bin/env python3
"""Publish bounded real-data I-100 scenario measurements without full-field reruns.

Operator-run on the authorized HPC with --root, --cache-dir, --output.  Private
seismic and LAS inputs are only read; all sampling uses existing paired memmaps.
This is a conditional interpretation experiment, never a drilling-target run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from reservoir_time_machine.analysis.conditional_reservoir import (
    conditional_i100_locations, evaluate_conditional_window,
)
from reservoir_time_machine.data_ingest.cache_identity import paired_cache_provenance
from reservoir_time_machine.data_ingest.segy_loader import load_challenge3_paired_volumes
from reservoir_time_machine.data_ingest.spatial_geometry import (
    conditionally_map_well_coordinates, sample_segy_header_geometry,
)
from reservoir_time_machine.data_ingest.well_evidence import (
    build_well_evidence_matrix, well_evidence_summary,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Organizer Challenge 3 dataset root")
    parser.add_argument("--cache-dir", type=Path, required=True, help="Existing completed paired cache")
    parser.add_argument("--output", type=Path, required=True, help="Private JSON results location")
    args = parser.parse_args(argv)
    root, cache, output = args.root.resolve(), args.cache_dir.resolve(), args.output.expanduser().resolve()
    required = ("baseline.npy", "monitor.npy", "metadata.json", "missing_trace_mask.npy",
                "monitor_only_traces.npy", "monitor_only_keys.npy")
    if not all((cache / name).is_file() for name in required):
        parser.error("Existing completed paired cache required; this script will not build one")
    if output.is_file():
        parser.error("Output already exists; choose a new versioned output path")
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if output.parent.stat().st_mode & 0o077:
        parser.error("Output directory must not be accessible to group or other users")
    cache_hash, cache_provenance = paired_cache_provenance(cache)
    baseline_volume, monitor_volume = load_challenge3_paired_volumes(root, output_dir=cache)
    b, m = baseline_volume.data, monitor_volume.data
    source = monitor_volume.metadata.source_metadata
    inline_ids = source["inline_ids"]
    crossline_ids = source["crossline_ids"]
    paired_missing = np.asarray(baseline_volume.missing_trace_mask) | np.asarray(
        monitor_volume.missing_trace_mask)
    wells = build_well_evidence_matrix(well_evidence_summary(root))
    baseline_path = root / "volumes" / "Challenge3_Baseline_FilteredScaled_FullStack.sgy"
    geometry = sample_segy_header_geometry(baseline_path, sample_count=1024)
    projected = conditionally_map_well_coordinates(
        wells, geometry, inline_ids=inline_ids, crossline_ids=crossline_ids)
    results = []
    for origin_ms in (-20.0, 0.0, 20.0):
        cases = conditional_i100_locations(
            wells, projected, inline_ids, crossline_ids, b.shape[2],
            sample_interval_ms=2.0, origin_ms=origin_ms)
        for case in cases:
            results.append({
                "candidate": case,
                "measurement": evaluate_conditional_window(b, m, paired_missing, case),
            })
    body = {
        "schema": "rtm-conditional-i100-evidence-1",
        "status": "bounded_conditional_scientific_test_completed" if results else "no_eligible_conditional_markers",
        "data_identity": cache_hash,
        "data_identity_method": cache_provenance["identity_scope"],
        "source_shape": list(b.shape),
        "source_axes": {"inline_ids": [int(inline_ids[0]), int(inline_ids[-1])],
                        "crossline_ids": [int(crossline_ids[0]), int(crossline_ids[-1])]},
        "coordinate_fit_summary": geometry.get("affine_geometry"),
        "conditional_well_projections": projected,
        "assumptions": {
            "common_crs": "NOT VERIFIED; numeric workbook XY projected through Baseline CDP affine fit",
            "seismic_domain": "zero-origin TWT is a scenario, NOT verified",
            "time_origin_cases_ms": [-20, 0, 20],
            "registration": "trace-key paired; physical registration not verified",
            "hydrocarbon_probability": "not computed",
        },
        "measurements": results,
    }
    raw = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    envelope = {"body": body, "sha256": hashlib.sha256(raw).hexdigest()}
    serialized = json.dumps(envelope, sort_keys=True, indent=2, allow_nan=False) + "\n"
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                         prefix=".conditional-i100-", suffix=".tmp",
                                         dir=output.parent, delete=False) as stream:
            tmp = Path(stream.name)
            os.fchmod(stream.fileno(), 0o600)
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, output)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
    print(json.dumps({
        "status": body["status"],
        "candidate_scenarios": len(results),
        "wells": sorted({x["candidate"]["well_id"] for x in results}),
        "result": str(output),
        "sha256": envelope["sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
