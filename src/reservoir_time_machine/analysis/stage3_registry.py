"""Validated adapter for the separately checksummed Stage 3 Angsi assessment.

The adapter preserves Stage 2 findings as historical seismic screening records
and exposes Stage 3 QC follow-up actions separately. It never labels those actions
as reservoir targets or calibrated recovery probabilities.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA = "rtm-stage3-evidence-assessment-1"
REGISTRY_SCHEMA = "rtm-stage3-interpretation-registry-1"
MAX_ASSESSMENT_BYTES = 2_000_000
MAX_MANIFEST_BYTES = 16_000


def detector_threshold_flag_count(item: dict) -> int:
    """Return the total detector flag count across supported Stage 3 schemas.

    Stage 3 assessments store per-detector counts as a mapping, while older
    presentation records may contain a scalar total. Keep this conversion at
    the registry boundary so callers never try to cast a mapping to ``int``.
    """
    value = item.get("detector_threshold_flagged_voxels",
                     item.get("detector_flagged_voxels", 0))
    if isinstance(value, dict):
        counts = value.values()
    elif type(value) is int:
        counts = (value,)
    else:
        raise ValueError("Detector threshold flags must be an integer or per-detector mapping")
    if any(type(count) is not int or count < 0 for count in counts):
        raise ValueError("Detector threshold flag counts must be non-negative integers")
    return sum(counts)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_manifest(directory: Path) -> dict[str, str]:
    manifest = directory / "stage3_manifest.sha256"
    if not manifest.is_file() or manifest.stat().st_size > MAX_MANIFEST_BYTES:
        raise ValueError("Stage 3 checksum manifest is missing or oversized")
    files: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        parts = line.split(None, 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ValueError("Malformed Stage 3 checksum manifest")
        digest, name = parts[0], parts[1].strip().lstrip("*")
        if (any(c not in "0123456789abcdef" for c in digest)
                or Path(name).name != name or name in files):
            raise ValueError("Unsafe or duplicate Stage 3 manifest entry")
        files[name] = digest
    if "stage3_assessment.json" not in files:
        raise ValueError("Stage 3 assessment is not covered by its checksum manifest")
    return files


def load_stage3_registry(path, *, expected_shape, inline_ids, crossline_ids) -> dict:
    """Load and validate a Stage 3 assessment and its small companion artifacts.

    Only the package's manifest-listed outputs are hashed (the assessment, CSVs,
    and PDF); no seismic volumes or full-field model maps are read or hashed.
    """
    assessment_path = Path(path).expanduser().resolve(strict=True)
    directory = assessment_path.parent
    if assessment_path.name != "stage3_assessment.json":
        raise ValueError("Expected stage3_assessment.json")
    if assessment_path.stat().st_size > MAX_ASSESSMENT_BYTES:
        raise ValueError("Stage 3 assessment exceeds the bounded manifest size")
    manifest = _load_manifest(directory)
    for name, expected_digest in manifest.items():
        artifact = directory / name
        if not artifact.is_file() or _sha256(artifact) != expected_digest:
            raise ValueError(f"Stage 3 artifact checksum mismatch: {name}")
    raw = assessment_path.read_bytes()
    assessment = json.loads(raw)
    if assessment.get("schema") != SCHEMA or assessment.get("status") != "separate_scientific_assessment_not_published":
        raise ValueError("Unsupported or incomplete Stage 3 assessment")
    provenance = assessment.get("provenance", {})
    shape = tuple(int(v) for v in expected_shape)
    if (len(shape) != 3 or provenance.get("source_shape") != list(shape)
            or len(inline_ids) != shape[0] or len(crossline_ids) != shape[1]
            or provenance.get("source_inline_ids") != [int(inline_ids[0]), int(inline_ids[-1])]
            or provenance.get("source_crossline_ids") != [int(crossline_ids[0]), int(crossline_ids[-1])]):
        raise ValueError("Stage 3 source geometry or SEG-Y axes do not match the loaded survey")

    review = assessment.get("stage2_top_ten_review", {})
    records = review.get("finding_records_detailed", [])
    if len(records) != 10 or not review.get("all_ten_mad_only") or not review.get("all_ten_four_detector_threshold_flags_zero"):
        raise ValueError("Stage 3 historical discovery inventory is incomplete or incompatible")
    axes = (np.asarray(inline_ids, dtype=np.int64), np.asarray(crossline_ids, dtype=np.int64))
    indexed: dict[str, dict] = {}
    for item in records:
        ident = item.get("id")
        loc = item.get("array_location")
        extent = item.get("extent")
        if (not isinstance(ident, str) or not ident.startswith("field-") or ident in indexed
                or not isinstance(loc, list) or len(loc) != 3
                or any(type(v) is not int or not 0 <= v < shape[i] for i, v in enumerate(loc))
                or not isinstance(extent, list) or len(extent) != 3):
            raise ValueError("Invalid Stage 3 finding ID, location, or extent")
        if (int(item.get("inline_id", -1)) != int(axes[0][loc[0]])
                or int(item.get("crossline_id", -1)) != int(axes[1][loc[1]])
                or int(item.get("sample_index", -1)) != loc[2]):
            raise ValueError(f"Finding {ident} does not map to original SEG-Y axes")
        for axis, bounds in enumerate(extent):
            if (not isinstance(bounds, list) or len(bounds) != 2
                    or any(type(v) is not int for v in bounds)
                    or not 0 <= bounds[0] <= bounds[1] < shape[axis]):
                raise ValueError(f"Finding {ident} has invalid component extent")
        if not isinstance(item.get("detector_flagged_voxels"), dict) or not isinstance(item.get("edge_qc"), dict):
            raise ValueError(f"Finding {ident} lacks detector or edge-QC provenance")
        indexed[ident] = item

    candidate = assessment.get("candidate_interpretation", {})
    if candidate.get("qualified_recovery_opportunity_count") != 0:
        raise ValueError("This Stage 3 adapter expects the reviewed zero-recovery-opportunity assessment")
    qc_ids = candidate.get("follow_up_qc_queue", [])
    if len(qc_ids) > 10 or any(ident not in indexed for ident in qc_ids):
        raise ValueError("Stage 3 QC follow-up queue references invalid findings")
    corrected = {item["id"]: float(item["corrected_index"])
                 for item in review.get("corrected_priority_order_without_geological_reweighting", [])}
    if set(corrected) != set(indexed) or not np.isfinite(list(corrected.values())).all():
        raise ValueError("Stage 3 corrected screening indices are incomplete or invalid")

    findings = [indexed[item["id"]] for item in sorted(records, key=lambda record: record["rank"])]
    actions = [{"rank": rank, "finding": indexed[ident], "basis": candidate.get("queue_basis", "")}
               for rank, ident in enumerate(qc_ids, 1)]
    return {
        "schema": REGISTRY_SCHEMA,
        "id": hashlib.sha256(raw).hexdigest(),
        "path": str(assessment_path),
        "assessment": assessment,
        "historical_findings": findings,
        "stage3_indices": corrected,
        "qc_follow_up_actions": actions,
        "recovery_opportunities": [],
        "checksums": manifest,
    }
