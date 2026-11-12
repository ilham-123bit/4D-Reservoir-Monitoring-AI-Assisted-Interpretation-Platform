#!/usr/bin/env python3
"""Verified checkpoint inference over the paired cache using context-overlap cores."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import argparse

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from reservoir_time_machine.hpc.patch_inference import (load_verified_trained_model,
    run_overlap_volume_inference)  # noqa: E402


def cache_identity(cache):
    metadata = cache / "metadata.json"
    body = {"metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
            "files": {name: {"size": (cache / name).stat().st_size,
                             "mtime_ns": (cache / name).stat().st_mtime_ns,
                             "inode": (cache / name).stat().st_ino}
                      for name in ("baseline.npy", "monitor.npy", "missing_trace_mask.npy")}}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="private inference JSON")
    args = parser.parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("neural inference requires an approved Slurm GPU allocation")
    import torch
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this allocation; CPU fallback is disabled")
    cache_text = os.environ.get("RTM_PAIRED_CACHE_DIR")
    output_text = os.environ.get("RTM_HPC_OUTPUT_ROOT")
    if not cache_text or not output_text:
        parser.error("RTM_PAIRED_CACHE_DIR and RTM_HPC_OUTPUT_ROOT are required")
    cache, output = Path(cache_text).expanduser().resolve(), Path(output_text).expanduser().resolve()
    config = json.loads(args.config.expanduser().read_text(encoding="utf-8"))
    checkpoint = (output / config["checkpoint"]).resolve()
    completion = (output / config["training_complete"]).resolve()
    verified = load_verified_trained_model(checkpoint, completion, output, device="cuda:0")
    baseline = np.load(cache / "baseline.npy", mmap_mode="r", allow_pickle=False)
    monitor = np.load(cache / "monitor.npy", mmap_mode="r", allow_pickle=False)
    missing = np.load(cache / "missing_trace_mask.npy", mmap_mode="r", allow_pickle=False)
    result = run_overlap_volume_inference(verified, baseline, monitor, missing,
        output / "neural-volume-inference", source_identity=cache_identity(cache),
        patch_shape=tuple(config.get("patch_shape", (32, 32, 64))),
        context_overlap=tuple(config.get("context_overlap", (8, 8, 16))),
        protected_cache=cache)
    print(json.dumps({key: result.get(key) for key in
        ("status", "run_directory", "total_cores", "completed_cores", "valid_output_voxels", "excluded_cores")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
