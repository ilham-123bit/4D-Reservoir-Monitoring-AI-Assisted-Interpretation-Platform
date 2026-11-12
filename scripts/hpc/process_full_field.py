#!/usr/bin/env python3
"""Run bounded paired-volume attributes from an existing NPY cache on an allocation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from reservoir_time_machine.processing.full_field import run_full_field_attributes  # noqa: E402
from reservoir_time_machine.data_ingest.cache_identity import paired_cache_provenance  # noqa: E402


def _input_identity(cache: Path, bounds, source_shape):
    cache_sha, cache_details = paired_cache_provenance(cache)
    details = {"schema": "rtm-full-field-input-identity-2", "paired_cache_sha256": cache_sha,
               "paired_cache": cache_details, "source_bounds": list(bounds),
               "source_domain_shape": list(source_shape)}
    identity = hashlib.sha256(json.dumps(details, sort_keys=True, separators=(",", ":"),
                               allow_nan=False).encode()).hexdigest()
    return identity, details


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True, help="existing paired NPY cache directory")
    parser.add_argument("--output", type=Path, required=True, help="separate private derived-output root")
    parser.add_argument("--chunk", type=int, nargs=3, default=(16, 32, 128), metavar=("INLINE", "CROSSLINE", "SAMPLE"))
    parser.add_argument("--window", type=int, default=9)
    parser.add_argument("--min-support", type=float, default=0.8)
    parser.add_argument("--min-free-gib", type=float, default=5.0)
    parser.add_argument("--pilot-bounds", type=int, nargs=6, metavar=("I0", "I1", "X0", "X1", "S0", "S1"),
                        help="optional bounded pilot ROI in absolute half-open array indices; each axis <=64")
    args = parser.parse_args(argv)
    cache, output = args.cache.expanduser().resolve(), args.output.expanduser().resolve()
    if not cache.is_dir() or not (cache / "metadata.json").is_file():
        parser.error("cache must contain metadata.json and paired NPY arrays")
    if output == cache or cache in output.parents or output in cache.parents:
        parser.error("derived output must not overlap the source cache")
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("run this full-domain job inside an approved Slurm allocation")
    try:
        arrays = {name: np.load(cache / f"{name}.npy", mmap_mode="r", allow_pickle=False)
                  for name in ("baseline", "monitor", "missing_trace_mask")}
    except (OSError, ValueError) as exc:
        parser.error(f"could not open paired cache safely: {type(exc).__name__}")
    metadata_path = cache / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_shape = tuple(metadata.get("shape", arrays["baseline"].shape))
    if arrays["baseline"].shape != arrays["monitor"].shape or tuple(arrays["baseline"].shape) != expected_shape:
        parser.error("paired volume shapes do not match each other and metadata")
    if arrays["missing_trace_mask"].shape != arrays["baseline"].shape[:2]:
        parser.error("missing-trace mask does not match the paired grid")
    full_shape = tuple(arrays["baseline"].shape)
    bounds = tuple(args.pilot_bounds) if args.pilot_bounds else (
        0, full_shape[0], 0, full_shape[1], 0, full_shape[2])
    limits = ((bounds[0], bounds[1], full_shape[0]),
              (bounds[2], bounds[3], full_shape[1]),
              (bounds[4], bounds[5], full_shape[2]))
    if any(low < 0 or high <= low or high > axis_size for low, high, axis_size in limits):
        parser.error("pilot bounds must be valid half-open ranges within the full array shape")
    if args.pilot_bounds and any(high - low > 64 for low, high, _ in limits):
        parser.error("bounded pilot ROI is limited to 64 cells per axis")
    spatial = (slice(bounds[0], bounds[1]), slice(bounds[2], bounds[3]))
    volume = spatial + (slice(bounds[4], bounds[5]),)
    selected_arrays = {
        "baseline": arrays["baseline"][volume],
        "monitor": arrays["monitor"][volume],
        "missing_trace_mask": arrays["missing_trace_mask"][spatial],
    }
    source_identity, source_identity_details = _input_identity(cache, bounds, full_shape)
    result = run_full_field_attributes(selected_arrays["baseline"], selected_arrays["monitor"],
        selected_arrays["missing_trace_mask"],
        output, source_identity=source_identity, chunk_shape=args.chunk, window_samples=args.window,
        min_support_fraction=args.min_support, protected_cache=cache,
        source_bounds=bounds, source_domain_shape=full_shape,
        source_identity_details=source_identity_details,
        min_free_bytes=int(args.min_free_gib * (1024 ** 3)))
    print(json.dumps({key: result[key] for key in
        ("status", "run_directory", "completed_chunks", "total_chunks", "source_voxels_scanned",
         "finite_paired_sample_positions", "output_bytes", "cache_hit") if key in result}, indent=2))
    print(f"source_bounds={list(bounds)} full_shape={list(full_shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
