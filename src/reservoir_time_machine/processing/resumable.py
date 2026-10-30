"""Restartable chunk processing with checksummed append-only checkpoint journals."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .chunks import Chunk, iter_chunks
from .provenance import configuration_fingerprint, sha256_file


def _array_hash(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(repr(contiguous.shape).encode("ascii"))
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _chunk_key(chunk: Chunk) -> str:
    return ",".join(str(part.start) for part in chunk.core)


def _append_event(path: Path, event: dict[str, Any]) -> None:
    """Append one fsynced JSONL event; a torn final line is ignored on recovery."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(event, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    events: list[dict[str, Any]] = []
    valid_end = 0
    for index, line in enumerate(lines):
        if index == len(lines)-1 and not line.endswith((b"\n", b"\r")):
            # A record is committed only with its newline terminator. Remove a torn
            # final append before a later event is appended to the journal.
            with path.open("r+b") as stream:
                stream.truncate(valid_end)
                stream.flush()
                os.fsync(stream.fileno())
            break
        try:
            events.append(json.loads(line))
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise ValueError(f"checkpoint journal contains a corrupt non-final record at line {index+1}")
        valid_end += len(line)
    return events


def _create_completion_sentinel(path: Path) -> None:
    """Exclusive file creation atomically publishes that a validated completion event exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="ascii") as stream:
            stream.write("complete\n")
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        # A concurrent/repeated call must validate the journal/output before reusing it.
        return


def run_resumable_chunks(source: np.ndarray, processor: Callable[[np.ndarray], np.ndarray],
                         output_path: str | Path, *, chunk_shape: Sequence[int],
                         processor_name: str, parameters: dict[str, Any] | None = None,
                         halo: int | Sequence[int] = 0,
                         output_dtype: np.dtype | str | None = None) -> np.memmap:
    """Process deterministic chunks and resume only checksum-valid completed tiles.

    The JSONL journal is append-only: each chunk event is written and fsynced only after
    the corresponding NPY memmap region is flushed. An incomplete final journal line is
    discarded during recovery and its chunk is recomputed. A final JSON completion event
    includes a whole-output SHA-256; an exclusive `.complete` sentinel publishes readiness.
    Concurrent writers to the same output are unsupported.
    """
    array = np.asanyarray(source)
    if array.ndim < 1 or not processor_name.strip():
        raise ValueError("source array and processor_name are required")
    # Validate shape/chunk/halo before creating artifacts.
    next(iter_chunks(array.shape, chunk_shape, halo=halo))
    output = Path(output_path)
    journal_path = output.with_name(output.name + ".journal.jsonl")
    complete_path = output.with_name(output.name + ".complete")
    output.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.dtype(output_dtype) if output_dtype is not None else array.dtype
    config = {"processor_name": processor_name, "parameters": parameters or {},
              "chunk_shape": [int(value) for value in chunk_shape],
              "halo": [int(halo)] * array.ndim if isinstance(halo, int) else [int(value) for value in halo],
              "source_shape": list(array.shape), "source_dtype": str(array.dtype), "output_dtype": str(dtype)}
    config_hash = configuration_fingerprint(config)
    events = _read_events(journal_path)
    if events:
        start_event = events[0]
        if start_event.get("event") != "start" or start_event.get("configuration_sha256") != config_hash:
            raise ValueError("checkpoint journal belongs to another source shape or processing configuration")
    elif output.exists() or complete_path.exists():
        raise FileExistsError("output/marker exists without a valid journal; refusing to overwrite")
    else:
        _append_event(journal_path, {"event": "start", "configuration_sha256": config_hash,
                                     "configuration": config, "schema_version": "1.0"})
        events = _read_events(journal_path)

    completed: dict[str, dict[str, Any]] = {}
    completion_event = None
    for event in events[1:]:
        if event.get("event") == "chunk_complete":
            completed[event["chunk_key"]] = event
        elif event.get("event") == "complete":
            completion_event = event

    def verify_completed_inputs() -> None:
        expected_chunks = 0
        for chunk in iter_chunks(array.shape, chunk_shape, halo=halo):
            expected_chunks += 1
            key = _chunk_key(chunk)
            record = completed.get(key)
            if record is None or record.get("input_sha256") != _array_hash(np.asarray(array[chunk.expanded])):
                raise ValueError(f"completed output input checksum mismatch or missing chunk record: {key}")
        if completion_event and completion_event.get("completed_chunk_count") != expected_chunks:
            raise ValueError("completion record has an unexpected chunk count")

    if complete_path.exists():
        if completion_event is None or not output.is_file():
            raise ValueError("completion sentinel exists without a valid completion event and output")
        if sha256_file(output) != completion_event.get("output_sha256"):
            raise ValueError("completed output checksum mismatch")
        if completion_event.get("configuration_sha256") != config_hash:
            raise ValueError("completion record belongs to another configuration")
        verify_completed_inputs()
        return np.load(output, mmap_mode="r+")
    if completion_event is not None:
        if not output.is_file() or sha256_file(output) != completion_event.get("output_sha256"):
            raise ValueError("journal says complete but output checksum is missing or invalid")
        verify_completed_inputs()
        _create_completion_sentinel(complete_path)
        return np.load(output, mmap_mode="r+")
    if not output.exists():
        if completed:
            raise FileNotFoundError("resume output is missing although completed chunks were journaled")
        np.lib.format.open_memmap(output, mode="w+", dtype=dtype, shape=array.shape).flush()
    result = np.load(output, mmap_mode="r+")
    if result.shape != array.shape or result.dtype != dtype:
        raise ValueError("existing resume output shape/dtype differs from the journal configuration")
    for chunk in iter_chunks(array.shape, chunk_shape, halo=halo):
        key = _chunk_key(chunk)
        input_block = np.asarray(array[chunk.expanded])
        input_hash = _array_hash(input_block)
        recorded = completed.get(key)
        if recorded is not None:
            if recorded.get("input_sha256") != input_hash:
                raise ValueError(f"source data changed since checkpoint for chunk {key}")
            if recorded.get("output_sha256") != _array_hash(np.asarray(result[chunk.core])):
                raise ValueError(f"checkpoint output checksum mismatch for chunk {key}")
            continue
        processed = np.asarray(processor(input_block))
        if processed.shape != input_block.shape:
            raise ValueError(f"processor changed chunk shape from {input_block.shape} to {processed.shape}")
        core_output = np.asarray(processed[chunk.core_in_expanded], dtype=dtype)
        result[chunk.core] = core_output
        result.flush()
        event = {"event": "chunk_complete", "chunk_key": key, "input_sha256": input_hash,
                 "output_sha256": _array_hash(core_output), "core_shape": list(core_output.shape)}
        _append_event(journal_path, event)
        completed[key] = event
    result.flush()
    output_hash = sha256_file(output)
    _append_event(journal_path, {"event": "complete", "configuration_sha256": config_hash,
                                 "output_sha256": output_hash, "completed_chunk_count": len(completed)})
    _create_completion_sentinel(complete_path)
    return result
