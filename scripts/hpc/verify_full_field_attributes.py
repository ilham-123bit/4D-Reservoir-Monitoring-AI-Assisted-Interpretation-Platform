#!/usr/bin/env python3
"""Read-only integrity verification of a completed full-field attribute run.

Run only inside an approved compute allocation. It streams the paired cache,
completion outputs, and chunk journals; it does not alter the run or source data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT / "src"))
from reservoir_time_machine.data_ingest.cache_identity import (paired_cache_provenance,
    validate_cache_index_metadata)
from reservoir_time_machine.processing.chunks import iter_chunks
from reservoir_time_machine.processing.full_field import FIELDS, _canonical


def _array_digest(*arrays) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode())
        digest.update(repr(value.shape).encode())
        digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def _file_digest(path: Path, block_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def verify(cache: Path, run_dir: Path) -> dict:
    started = time.perf_counter()
    cache, run_dir = cache.resolve(), run_dir.resolve()
    completion = json.loads((run_dir / "complete.json").read_text(encoding="utf-8"))
    body = completion.get("body")
    if not isinstance(body, dict) or hashlib.sha256(_canonical(body)).hexdigest() != completion.get("sha256"):
        raise ValueError("Completion envelope checksum failed")
    config = body.get("configuration", {})
    shape = tuple(config.get("shape", ()))
    if (body.get("schema") != "rtm-full-field-attributes-1" or body.get("status") != "complete"
            or len(shape) != 3 or config.get("source_bounds") != [0, shape[0], 0, shape[1], 0, shape[2]]
            or config.get("source_domain_shape") != list(shape)
            or config.get("chunk_shape") != [64, 64, 256]
            or config.get("window_samples") != 9 or config.get("min_support_fraction") != 0.8):
        raise ValueError("Completion does not describe the expected full-domain scientific run")
    expected_source, source_details = paired_cache_provenance(cache)
    expected_details = {"schema": "rtm-full-field-input-identity-2",
                        "paired_cache_sha256": expected_source, "paired_cache": source_details,
                        "source_bounds": [0, shape[0], 0, shape[1], 0, shape[2]],
                        "source_domain_shape": list(shape)}
    source_identity = hashlib.sha256(json.dumps(expected_details, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if (config.get("source_identity") != source_identity
            or config.get("source_identity_details") != expected_details):
        raise ValueError("Run source identity does not match this paired cache")

    metadata = json.loads((cache / "metadata.json").read_text(encoding="utf-8"))
    baseline = np.load(cache / "baseline.npy", mmap_mode="r", allow_pickle=False)
    monitor = np.load(cache / "monitor.npy", mmap_mode="r", allow_pickle=False)
    missing = np.load(cache / "missing_trace_mask.npy", mmap_mode="r", allow_pickle=False)
    if tuple(baseline.shape) != shape or tuple(monitor.shape) != shape or tuple(missing.shape) != shape[:2]:
        raise ValueError("Paired-cache geometry differs from completed output")
    if (tuple(metadata.get("shape", ())) != shape
            or len(metadata.get("inline_ids", [])) != shape[0]
            or len(metadata.get("crossline_ids", [])) != shape[1]):
        raise ValueError("Cache metadata/header axes do not cover the full source grid")
    monitor_only_path = cache / "monitor_only_keys.npy"
    monitor_only = (np.load(monitor_only_path, mmap_mode="r", allow_pickle=False)
                    if monitor_only_path.is_file() else np.empty((0, 2), dtype=np.int32))
    validate_cache_index_metadata(metadata, missing, monitor_only)

    processing_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if (processing_manifest.get("status") != "complete"
            or processing_manifest.get("configuration_sha256") != body.get("configuration_sha256")
            or processing_manifest.get("completion_sha256") != completion.get("sha256")
            or processing_manifest.get("completed_chunks") != body.get("completed_chunks")
            or processing_manifest.get("total_chunks") != body.get("total_chunks")):
        raise ValueError("Processing manifest and completion artifact disagree")

    output_records = body.get("outputs", {})
    if set(output_records) != set(FIELDS):
        raise ValueError("Completion does not contain all seven required attributes")
    outputs = {}
    for name in FIELDS:
        record = output_records[name]
        filename = record.get("file")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError(f"Unsafe output filename for {name}")
        path = run_dir / filename
        expected_dtype = np.dtype("uint8" if name == "paired_sample_count" else "float32")
        if not path.is_file() or path.stat().st_size != record.get("size_bytes"):
            raise ValueError(f"Output size/existence check failed: {name}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if tuple(array.shape) != shape or array.dtype != expected_dtype or record.get("shape") != list(shape):
            raise ValueError(f"Output dtype/geometry check failed: {name}")
        if _file_digest(path) != record.get("sha256"):
            raise ValueError(f"Complete-file SHA-256 check failed: {name}")
        outputs[name] = array
    if sum(item["size_bytes"] for item in output_records.values()) != body.get("output_bytes"):
        raise ValueError("Output byte total disagrees with completion manifest")

    journal_path = run_dir / "chunks.jsonl"
    if _file_digest(journal_path) != body.get("journal_sha256"):
        raise ValueError("Chunk journal SHA-256 check failed")
    events = [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines() if line]
    expected_chunks = list(iter_chunks(shape, config["chunk_shape"], halo=(0, 0, config["window_samples"] // 2)))
    if not events or events[0] != {"event": "start", "configuration_sha256": body["configuration_sha256"]}:
        raise ValueError("Journal start event does not match run configuration")
    chunks = [event for event in events if event.get("event") == "chunk_complete"]
    if len(chunks) != len(expected_chunks) or len({event.get("chunk_key") for event in chunks}) != len(chunks):
        raise ValueError("Journal is incomplete or contains duplicate chunk records")
    by_key = {event["chunk_key"]: event for event in chunks}
    total_scanned = total_finite = 0
    support = outputs["paired_sample_count"]
    float_outputs = [outputs[name] for name in FIELDS if name != "paired_sample_count"]
    for chunk in expected_chunks:
        core, expanded = chunk.core, chunk.expanded
        key = ",".join(str(part.start) for part in core)
        event = by_key.get(key)
        if event is None:
            raise ValueError(f"Journal omitted expected chunk {key}")
        source_hash = _array_digest(np.asarray(baseline[expanded]), np.asarray(monitor[expanded]),
                                    np.asarray(missing[expanded[:2]], dtype=bool))
        if event.get("input_sha256") != source_hash:
            raise ValueError(f"Source chunk hash mismatch: {key}")
        actual_outputs = {name: np.asarray(outputs[name][core]) for name in FIELDS}
        if event.get("output_sha256") != {name: _array_digest(value) for name, value in actual_outputs.items()}:
            raise ValueError(f"Journaled output chunk hash mismatch: {key}")
        if event.get("scanned_voxels") != int(np.prod([s.stop-s.start for s in core])):
            raise ValueError(f"Journal core voxel count mismatch: {key}")
        b = np.asarray(baseline[core]); m = np.asarray(monitor[core])
        mask = np.asarray(missing[core[:2]], dtype=bool)
        finite = np.isfinite(b) & np.isfinite(m) & ~mask[..., None]
        count = int(finite.sum(dtype=np.int64))
        if event.get("finite_paired_sample_positions") != count:
            raise ValueError(f"Journal paired-support count mismatch: {key}")
        local_support = actual_outputs["paired_sample_count"]
        if local_support.size and int(local_support.max()) > config["window_samples"]:
            raise ValueError(f"Paired sample support exceeds configured window: {key}")
        centers = np.arange(core[2].start, core[2].stop)
        left, right = config["window_samples"] // 2, (config["window_samples"] - 1) // 2
        available = (np.minimum(shape[2], centers + right + 1) - np.maximum(0, centers - left))
        required = np.ceil(config["min_support_fraction"] * available).astype(np.uint8)
        unsupported = local_support < required[None, None, :]
        for name in ("rms_baseline", "rms_monitor", "nrms"):
            if np.isfinite(actual_outputs[name][unsupported]).any():
                raise ValueError(f"Unsupported samples contain finite {name} values: {key}")
        if mask.any():
            invalid_support = local_support[mask]
            if np.any(invalid_support != 0):
                raise ValueError(f"Unoccupied Baseline-grid cells have nonzero support: {key}")
            for values in (actual_outputs[n] for n in FIELDS if n != "paired_sample_count"):
                if not np.isnan(values[mask]).all():
                    raise ValueError(f"Unoccupied grid cells are not NaN-masked: {key}")
        total_scanned += event["scanned_voxels"]
        total_finite += count
    voxel_count = int(np.prod(shape, dtype=np.int64))
    if (total_scanned != voxel_count or body.get("source_voxels_scanned") != voxel_count
            or body.get("completed_chunks") != len(expected_chunks)
            or body.get("total_chunks") != len(expected_chunks)
            or body.get("finite_paired_sample_positions") != total_finite
            or body.get("masked_baseline_grid_cells") != int(np.asarray(missing).sum(dtype=np.int64))):
        raise ValueError("Full-domain coverage/support totals disagree")
    return {"status": "verified", "run_directory": str(run_dir), "source_identity": source_identity,
            "shape": list(shape), "chunk_shape": config["chunk_shape"],
            "chunks_verified": len(expected_chunks), "output_files_verified": len(FIELDS),
            "output_bytes": body["output_bytes"], "finite_paired_sample_positions": total_finite,
            "unoccupied_grid_cells": int(np.asarray(missing).sum(dtype=np.int64)),
            "journal_input_and_output_chunk_hashes": len(chunks),
            "elapsed_seconds": time.perf_counter()-started,
            "claim_limit": "Integrity and processing coverage verified; physical registration and geological interpretation are not validated."}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", required=True, type=Path)
    p.add_argument("--run-dir", required=True, type=Path)
    a = p.parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID"):
        p.error("run this read-only verification inside an approved Slurm allocation")
    result = verify(a.cache, a.run_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
