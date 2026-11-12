"""Resumable, memory-bounded full-domain paired seismic attribute processing.

This creates scientific products only; it does not establish physical registration
or a geological cause. Use on a scheduled HPC allocation after checking output space.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import stat
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np

from ..attributes.difference import calculate_attributes, _validate_parameters
from .chunks import iter_chunks

SCHEMA = "rtm-full-field-attributes-1"
VERSION = "paired-attributes-out-of-core-1"
FIELDS = ("signed_difference", "absolute_difference", "normalized_difference",
          "rms_baseline", "rms_monitor", "nrms", "paired_sample_count")
_FLOAT_FIELDS = set(FIELDS) - {"paired_sample_count"}


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _hash_bytes(*arrays) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode())
        digest.update(repr(value.shape).encode())
        digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def _hash_file(path: Path, block_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def estimate_storage_bytes(shape, *, fields: Iterable[str] = FIELDS) -> int:
    shape = tuple(int(value) for value in shape)
    selected = tuple(fields)
    if len(shape) != 3 or any(value < 1 for value in shape):
        raise ValueError("shape must be a positive (inline, crossline, sample) triple")
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(FIELDS):
        raise ValueError("fields must be unique names from the supported attribute set")
    voxels = math.prod(shape)
    return sum(voxels * (1 if name == "paired_sample_count" else 4) for name in selected)


def _read_journal(path: Path) -> list[dict]:
    if not path.exists():
        return []
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    events, valid_end = [], 0
    for n, line in enumerate(lines):
        if n == len(lines) - 1 and not line.endswith(b"\n"):
            with path.open("r+b") as stream:
                stream.truncate(valid_end)
                stream.flush()
                os.fsync(stream.fileno())
            break
        events.append(json.loads(line))
        valid_end += len(line)
    return events


def _append(path: Path, event: dict) -> None:
    payload = _canonical(event) + b"\n"
    with path.open("ab") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_json(path: Path, value: dict) -> None:
    payload = _canonical(value) + b"\n"
    with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
                                     delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def run_full_field_attributes(baseline, monitor, missing_trace_mask, output_root, *,
                              source_identity: str, chunk_shape=(32, 32, 128),
                              window_samples: int = 9, min_support_fraction: float = 0.8,
                              epsilon: float = 1e-8, fields=FIELDS,
                              source_bounds=None, source_domain_shape=None,
                              source_identity_details: dict | None = None,
                              protected_cache=None, min_free_bytes: int = 0,
                              max_chunks: int | None = None) -> dict:
    """Create full 3D paired attributes in resumable NPY memmaps.

    Chunks partition inline/crossline/sample cores without overlap. A sample-axis
    halo of ``window_samples//2`` supplies exact centered-window context; only core
    outputs are committed. Missing grid traces are set to NaN in temporary blocks
    and retain zero support, never measured zero amplitude. ``max_chunks`` exists
    for interruption/recovery tests and controlled pilots; omit it for full runs.
    """
    before, after = np.asanyarray(baseline), np.asanyarray(monitor)
    missing = np.asanyarray(missing_trace_mask)
    if before.ndim != 3 or before.shape != after.shape or missing.shape != before.shape[:2]:
        raise ValueError("paired 3D arrays and a matching 2D missing-trace mask are required")
    if missing.dtype.kind != "b":
        raise ValueError("missing_trace_mask must be Boolean")
    if (not isinstance(source_identity, str) or len(source_identity) != 64
            or any(c not in "0123456789abcdef" for c in source_identity)):
        raise ValueError("source_identity must be a SHA-256 identity")
    _validate_parameters(window_samples, epsilon)
    if not 0 < min_support_fraction <= 1 or not np.isfinite(min_support_fraction):
        raise ValueError("min_support_fraction must be finite and in (0,1]")
    selected = tuple(fields)
    if "paired_sample_count" in selected and window_samples > np.iinfo(np.uint8).max:
        raise ValueError("paired_sample_count output supports at most 255 samples; use a wider output dtype")
    required_bytes = estimate_storage_bytes(before.shape, fields=selected)
    if isinstance(min_free_bytes, bool) or not isinstance(min_free_bytes, int) or min_free_bytes < 0:
        raise ValueError("min_free_bytes must be a non-negative integer")
    if max_chunks is not None and (isinstance(max_chunks, bool) or not isinstance(max_chunks, int) or max_chunks < 1):
        raise ValueError("max_chunks must be a positive integer when specified")
    chunks = tuple(int(value) for value in chunk_shape)
    if len(chunks) != 3 or any(value < 1 for value in chunks):
        raise ValueError("chunk_shape must contain three positive sizes")
    if source_bounds is None:
        source_bounds = (0, before.shape[0], 0, before.shape[1], 0, before.shape[2])
    if source_domain_shape is None:
        source_domain_shape = before.shape
    source_domain_shape = tuple(source_domain_shape)
    source_bounds = tuple(source_bounds)
    if (len(source_domain_shape) != 3 or any(type(value) is not int or value < 1 for value in source_domain_shape)
            or len(source_bounds) != 6 or any(type(value) is not int for value in source_bounds)
            or any(low < 0 or high <= low for low, high in
                   ((source_bounds[0], source_bounds[1]), (source_bounds[2], source_bounds[3]),
                    (source_bounds[4], source_bounds[5])))
            or tuple(high - low for low, high in
                     ((source_bounds[0], source_bounds[1]), (source_bounds[2], source_bounds[3]),
                      (source_bounds[4], source_bounds[5]))) != before.shape
            or any(source_bounds[index + 1] > source_domain_shape[axis]
                   for axis, index in enumerate((0, 2, 4)))):
        raise ValueError("source_bounds must be three valid half-open absolute array ranges matching input shape")

    root = Path(output_root).expanduser().resolve()
    if protected_cache is not None:
        cache = Path(protected_cache).expanduser().resolve()
        if root == cache or cache in root.parents or root in cache.parents:
            raise ValueError("analysis output root must be separate from the paired cache")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if root.stat().st_uid != os.getuid() or stat.S_IMODE(root.stat().st_mode) & 0o077:
        raise PermissionError("output root must be user-owned and private (mode 0700)")
    config = {"schema": SCHEMA, "version": VERSION, "source_identity": source_identity,
              "source_identity_details": source_identity_details,
              "shape": list(before.shape), "source_bounds": list(source_bounds),
              "source_domain_shape": list(source_domain_shape),
              "dtype_inputs": [str(before.dtype), str(after.dtype)],
              "missing_mask_dtype": str(missing.dtype), "chunk_shape": list(chunks),
              "sample_halo": window_samples // 2, "window_samples": window_samples,
              "min_support_fraction": float(min_support_fraction), "epsilon": float(epsilon),
              "fields": list(selected), "output_dtypes": {n: ("uint8" if n == "paired_sample_count" else "float32") for n in selected}}
    config_digest = hashlib.sha256(_canonical(config)).hexdigest()
    run_dir = root / f"attributes-{config_digest[:20]}"
    run_dir.mkdir(mode=0o700, exist_ok=True)
    if run_dir.stat().st_uid != os.getuid() or stat.S_IMODE(run_dir.stat().st_mode) & 0o077:
        raise PermissionError("run directory must be user-owned and private (mode 0700)")
    expected_existing_bytes = sum((run_dir / f"{name}.npy").stat().st_size
                                  for name in selected if (run_dir / f"{name}.npy").is_file())
    bytes_to_allocate = required_bytes - expected_existing_bytes
    if shutil_disk_free(run_dir) < bytes_to_allocate + min_free_bytes:
        raise OSError(f"insufficient output space: need at least {bytes_to_allocate + min_free_bytes} additional bytes")
    manifest_path = run_dir / "manifest.json"
    journal_path = run_dir / "chunks.jsonl"
    completion_path = run_dir / "complete.json"
    lock_path = run_dir / ".lock"

    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        events = _read_journal(journal_path)
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("configuration_sha256") != config_digest or manifest.get("configuration") != config:
                raise ValueError("existing output manifest belongs to a different input/configuration")
        else:
            if events or completion_path.exists():
                raise ValueError("journal or completion artifact exists without its manifest")
            manifest = {"schema": SCHEMA, "configuration": config,
                        "configuration_sha256": config_digest, "status": "running"}
            _atomic_json(manifest_path, manifest)
            _append(journal_path, {"event": "start", "configuration_sha256": config_digest})
            events = _read_journal(journal_path)
        if not events or events[0].get("event") != "start" or events[0].get("configuration_sha256") != config_digest:
            raise ValueError("chunk journal has no matching start record")

        if completion_path.exists():
            complete = json.loads(completion_path.read_text())
            body = complete["body"]
            if complete.get("sha256") != hashlib.sha256(_canonical(body)).hexdigest():
                raise ValueError("completed artifact manifest checksum mismatch")
            if body.get("configuration_sha256") != config_digest:
                raise ValueError("completed artifact configuration mismatch")
            committed_chunks = {event["chunk_key"]: event for event in events
                                if event.get("event") == "chunk_complete"}
            expected_chunk_count = sum(1 for _ in iter_chunks(before.shape, chunks, halo=(0, 0, window_samples // 2)))
            if len(committed_chunks) != expected_chunk_count:
                raise ValueError("completed journal has an unexpected number of source chunks")
            for chunk in iter_chunks(before.shape, chunks, halo=(0, 0, window_samples // 2)):
                key = ",".join(str(item.start) for item in chunk.core)
                source_digest = _hash_bytes(np.asarray(before[chunk.expanded]),
                                            np.asarray(after[chunk.expanded]),
                                            np.asarray(missing[chunk.expanded[:2]], dtype=bool))
                if committed_chunks[key].get("input_sha256") != source_digest:
                    raise ValueError(f"completed artifact source data changed: chunk {key}")
            for name, item in body["outputs"].items():
                path = run_dir / item["file"]
                if not path.is_file() or path.stat().st_size != item["size_bytes"]:
                    raise ValueError(f"completed output is missing or has wrong size: {name}")
                if _hash_file(path) != item["sha256"]:
                    raise ValueError(f"completed output checksum mismatch: {name}")
            return {**body, "run_directory": str(run_dir), "cache_hit": True}

        expected_fields = {name: run_dir / f"{name}.npy" for name in selected}
        if any(path.exists() for path in expected_fields.values()):
            if not all(path.exists() for path in expected_fields.values()):
                raise ValueError("partial output field set is incomplete; refusing unsafe resume")
            outputs = {name: np.load(path, mmap_mode="r+", allow_pickle=False) for name, path in expected_fields.items()}
        else:
            if any(event.get("event") == "chunk_complete" for event in events):
                raise FileNotFoundError("output arrays are missing although completed chunks were journaled")
            outputs = {name: np.lib.format.open_memmap(path, mode="w+",
                dtype=np.uint8 if name == "paired_sample_count" else np.float32,
                shape=before.shape) for name, path in expected_fields.items()}
            for out in outputs.values(): out.flush()

        completed = {event["chunk_key"]: event for event in events[1:] if event.get("event") == "chunk_complete"}
        total_chunks = sum(1 for _ in iter_chunks(before.shape, chunks, halo=(0, 0, window_samples // 2)))
        processed_now = 0
        stats = {"source_voxels_scanned": 0, "finite_paired_sample_positions": 0,
                 "masked_baseline_grid_cells": int(missing.sum(dtype=np.int64)),
                 "chunks_completed": len(completed)}
        for chunk in iter_chunks(before.shape, chunks, halo=(0, 0, window_samples // 2)):
            key = ",".join(str(item.start) for item in chunk.core)
            core = chunk.core
            current_b = np.asarray(before[chunk.expanded])
            current_m = np.asarray(after[chunk.expanded])
            spatial_mask = np.asarray(missing[chunk.expanded[:2]], dtype=bool)
            input_hash = _hash_bytes(current_b, current_m, spatial_mask)
            previous = completed.get(key)
            if previous is not None:
                if previous.get("input_sha256") != input_hash:
                    raise ValueError(f"source data changed since checkpoint for chunk {key}")
                for name, digest in previous["output_sha256"].items():
                    if _hash_bytes(np.asarray(outputs[name][core])) != digest:
                        raise ValueError(f"completed chunk output was modified: {key} / {name}")
                continue
            if max_chunks is not None and processed_now >= max_chunks:
                break

            bblock = np.array(current_b, copy=True)
            mblock = np.array(current_m, copy=True)
            bblock[spatial_mask] = np.nan
            mblock[spatial_mask] = np.nan
            attrs = calculate_attributes(bblock, mblock, window_samples=window_samples, epsilon=epsilon)
            local = chunk.core_in_expanded
            # Compare paired support with the samples that physically exist inside
            # the survey at each center. Do not penalize a valid survey boundary
            # for samples outside the acquired volume.
            left = window_samples // 2
            right = (window_samples - 1) // 2
            sample_centers = source_bounds[4] + np.arange(core[2].start, core[2].stop)
            available = (np.minimum(source_domain_shape[2], sample_centers + right + 1)
                         - np.maximum(0, sample_centers - left))
            required = np.ceil(min_support_fraction * available).astype(np.int32)
            local_count = attrs.paired_sample_count[local]
            under_supported = local_count < required[None, None, :]
            for name in ("rms_baseline", "rms_monitor", "nrms"):
                value = getattr(attrs, name)[local]
                value[under_supported] = np.nan
            result_hashes = {}
            for name in selected:
                values = np.asarray(getattr(attrs, name)[local], dtype=outputs[name].dtype)
                outputs[name][core] = values
                result_hashes[name] = _hash_bytes(values)
            for out in outputs.values(): out.flush()

            core_b = np.asarray(before[core]); core_m = np.asarray(after[core])
            core_mask = np.asarray(missing[core[:2]], dtype=bool)
            pair_valid = np.isfinite(core_b) & np.isfinite(core_m) & ~core_mask[..., None]
            event = {"event": "chunk_complete", "chunk_key": key,
                     "input_sha256": input_hash, "output_sha256": result_hashes,
                     "core_shape": [part.stop - part.start for part in core],
                     "scanned_voxels": int(core_b.size),
                     "finite_paired_sample_positions": int(pair_valid.sum(dtype=np.int64))}
            _append(journal_path, event)
            completed[key] = event
            stats["source_voxels_scanned"] += int(core_b.size)
            stats["finite_paired_sample_positions"] += int(pair_valid.sum(dtype=np.int64))
            stats["chunks_completed"] += 1
            processed_now += 1

        if len(completed) != total_chunks:
            manifest.update(status="incomplete", completed_chunks=len(completed), total_chunks=total_chunks)
            _atomic_json(manifest_path, manifest)
            return {**manifest, "run_directory": str(run_dir), "cache_hit": False}

        output_manifest = {}
        for name, path in expected_fields.items():
            outputs[name].flush()
            output_manifest[name] = {"file": path.name, "size_bytes": path.stat().st_size,
                                     "sha256": _hash_file(path), "dtype": str(outputs[name].dtype),
                                     "shape": list(outputs[name].shape)}
        # Recompute exact run summaries from committed journal entries, including resumed chunks.
        final_events = _read_journal(journal_path)
        chunk_events = [event for event in final_events if event.get("event") == "chunk_complete"]
        body = {"schema": SCHEMA, "version": VERSION, "configuration": config,
                "configuration_sha256": config_digest, "status": "complete",
                "total_chunks": total_chunks, "completed_chunks": len(chunk_events),
                "source_voxels_scanned": sum(event["scanned_voxels"] for event in chunk_events),
                "finite_paired_sample_positions": sum(event["finite_paired_sample_positions"] for event in chunk_events),
                "masked_baseline_grid_cells": int(missing.sum(dtype=np.int64)),
                "output_bytes": sum(item["size_bytes"] for item in output_manifest.values()),
                "outputs": output_manifest,
                "journal_sha256": _hash_file(journal_path),
                "interpretation": "array-aligned preliminary attributes; physical 4D registration and geological cause are not established"}
        _atomic_json(completion_path, {"body": body,
            "sha256": hashlib.sha256(_canonical(body)).hexdigest()})
        manifest.update(status="complete", completed_chunks=total_chunks, total_chunks=total_chunks,
                        completion_sha256=hashlib.sha256(_canonical(body)).hexdigest())
        _atomic_json(manifest_path, manifest)
        return {**body, "run_directory": str(run_dir), "cache_hit": False}


def shutil_disk_free(path: Path) -> int:
    """Return free bytes without importing broad filesystem utilities at module import."""
    return os.statvfs(path).f_bavail * os.statvfs(path).f_frsize
