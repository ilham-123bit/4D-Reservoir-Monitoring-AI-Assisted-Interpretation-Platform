#!/usr/bin/env python3
"""Precompute finding-directed display figures on an offline HPC allocation.

This reads existing paired memmaps and a validated registry. It does not run
seismic analysis or fit models. Files are published atomically outside source
and cache directories.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from reservoir_time_machine.analysis.registry import load_registry
from reservoir_time_machine.data_ingest.cache_identity import paired_cache_identity
from reservoir_time_machine.data_ingest.segy_loader import load_challenge3_paired_volumes
from reservoir_time_machine.visualization.finding_animation import finding_animation
from reservoir_time_machine.visualization.presentation_cache import (
    get_or_build_presentation,
    presentation_identity,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=os.environ.get("RTM_CHALLENGE3_ROOT"))
    parser.add_argument("--cache-dir", type=Path, default=os.environ.get("RTM_CACHE_DIR"))
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.dataset_root is None or args.cache_dir is None:
        parser.error("--dataset-root/RTM_CHALLENGE3_ROOT and --cache-dir/RTM_CACHE_DIR are required")

    dataset_root = args.dataset_root.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    registry_path = args.registry.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output == cache_dir or cache_dir in output.parents or output in cache_dir.parents:
        raise ValueError("Presentation output must be separate from the paired seismic cache")
    if output == dataset_root or dataset_root in output.parents:
        raise ValueError("Presentation output must be outside the original dataset")

    t0 = time.perf_counter()
    source_identity = paired_cache_identity(dataset_root, cache_dir)
    baseline, monitor = load_challenge3_paired_volumes(dataset_root, output_dir=cache_dir)
    envelope = __import__("json").loads(registry_path.read_text(encoding="utf-8"))
    body = envelope["body"]
    if body["dataset_id"] != source_identity:
        raise ValueError("Registry dataset identity does not match the current paired cache")
    registry = load_registry(registry_path, source_identity, tuple(baseline.data.shape))
    missing = baseline.missing_trace_mask | monitor.missing_trace_mask
    output.mkdir(parents=True, exist_ok=True, mode=0o700)

    created = 0
    for entry in registry["entries"]:
        finding, result = entry["finding"], entry["result"]
        identity = presentation_identity(registry["id"], result["run_id"], finding["id"],
                                         source_identity)
        _, cache_hit = get_or_build_presentation(
            output, identity,
            lambda r=result, f=finding: finding_animation(
                baseline.data, monitor.data, missing, r, f),
        )
        created += not cache_hit
        print(f"finding={finding['id']} cache={'hit' if cache_hit else 'built'}")
    print(f"registry={registry['id']} findings={len(registry['entries'])} built={created} "
          f"elapsed_seconds={time.perf_counter()-t0:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
