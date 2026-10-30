"""Deterministic analysis-run manifests for derived analytical products."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_file(path: str | Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Hash an input incrementally without loading the file into memory."""
    if chunk_bytes < 1:
        raise ValueError("chunk_bytes must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def configuration_fingerprint(configuration: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(configuration).encode("utf-8")).hexdigest()


def write_analysis_manifest(path: str | Path, *, operation: str, inputs: Iterable[dict[str, Any]],
                           parameters: dict[str, Any], outputs: Iterable[dict[str, Any]],
                           software: dict[str, str] | None = None, created_utc: str | None = None) -> dict[str, Any]:
    """Write a reproducible JSON manifest exclusively; never replace an existing record."""
    if not operation.strip():
        raise ValueError("operation is required")
    input_records, output_records = list(inputs), list(outputs)
    config_hash = configuration_fingerprint(parameters)
    manifest = {
        "schema_version": "1.0",
        "operation": operation,
        "created_utc": created_utc or datetime.now(timezone.utc).isoformat(),
        "inputs": input_records,
        "parameters": parameters,
        "configuration_sha256": config_hash,
        "outputs": output_records,
        "software": software or {},
        "result_type": "derived_analytical_product",
        "interpretation_status": "measurement_or_detection_only; not a geological conclusion",
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))
        stream.write("\n")
    return manifest
