#!/usr/bin/env python3
"""Build a private survey-wide QC artifact in a scheduled HPC allocation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reservoir_time_machine.data_ingest.cache_identity import paired_cache_identity
from reservoir_time_machine.data_ingest.segy_loader import load_challenge3_paired_volumes
from reservoir_time_machine.preprocessing.qc_artifact import (
    build_qc_artifact,
    load_qc_artifact,
    write_qc_artifact,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-full-scan", action="store_true",
                        help="required because this reads all survey samples in chunks")
    parser.add_argument("--inline-chunk", type=int, default=2, choices=range(1, 9))
    parser.add_argument("--output-dir", type=Path,
                        default=Path.home() / ".reservoir_time_machine/analysis/qc")
    args = parser.parse_args()
    if not args.confirm_full_scan:
        parser.error("pass --confirm-full-scan only from a scheduled HPC compute allocation")
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("survey-wide QC is restricted to a scheduled SLURM job; no login-node scan started")
    root_value = os.environ.get("RTM_CHALLENGE3_ROOT")
    if not root_value:
        parser.error("RTM_CHALLENGE3_ROOT must identify the authorized read-only dataset directory")
    root = Path(root_value).expanduser().resolve()
    cache = Path(os.environ.get("RTM_CACHE_DIR", "")).resolve()
    if not cache.is_dir() or not (cache / "metadata.json").is_file():
        parser.error("RTM_CACHE_DIR must point to the existing paired cache; this tool never rebuilds it")
    identity = paired_cache_identity(root, cache)
    target = args.output_dir.expanduser().resolve() / f"survey-qc-{identity}.json"
    if target == root or root in target.parents or target == ROOT or ROOT in target.parents:
        parser.error("private QC artifacts must be stored outside the input dataset and project tree")
    if target.exists():
        metadata = __import__("json").loads((cache / "metadata.json").read_text())
        existing = load_qc_artifact(target, source_identity=identity, shape=metadata["shape"])
        print(f"Validated existing QC artifact: {target} · {existing['schema']}")
        return 0

    baseline_volume, monitor_volume = load_challenge3_paired_volumes(root, output_dir=cache)
    if baseline_volume.shape != monitor_volume.shape:
        raise RuntimeError("Paired volumes have inconsistent geometry; no QC artifact written")
    print(f"Starting chunked QC in SLURM job {os.environ['SLURM_JOB_ID']}; shape={baseline_volume.shape}; "
          f"inline_chunk={args.inline_chunk}; source identity={identity}", flush=True)
    artifact = build_qc_artifact(
        baseline_volume.data,
        monitor_volume.data,
        baseline_volume.missing_trace_mask,
        __import__("numpy").load(cache / "monitor_only_traces.npy", mmap_mode="r", allow_pickle=False),
        source_identity=identity,
        source_metadata=monitor_volume.metadata.source_metadata,
        inline_chunk=args.inline_chunk,
    )
    if paired_cache_identity(root, cache) != identity:
        raise RuntimeError("Paired source/cache identity changed during QC; artifact was not written")
    output = write_qc_artifact(artifact, target, protected_cache=cache)
    # Read back through the same validator used by the dashboard.
    validated = load_qc_artifact(output, source_identity=identity, shape=baseline_volume.shape)
    print(f"QC artifact validated: {output}; schema={validated['schema']}; "
          f"paired traces={validated['trace_counts']['paired_trace_count']:,}; "
          f"monitor-only traces={validated['trace_counts']['monitor_only_trace_count']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
