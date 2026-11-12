"""Validated, versioned publication adapter for full-field findings.

This adapter is deliberately separate from the bounded ``registry-3.1`` format.
It preserves the full-field candidate record and refuses incomplete or incompatible
products. Publication is atomic: ``latest.json`` changes only after validation.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from ..processing.full_field import _canonical, _hash_file

SCHEMA = "rtm-full-field-registry-1"


def _read_complete(path: Path, schema: str) -> dict:
    envelope = json.loads(path.read_text(encoding="utf-8"))
    body = envelope.get("body")
    if (not isinstance(body, dict) or body.get("schema") != schema
            or hashlib.sha256(_canonical(body)).hexdigest() != envelope.get("sha256")):
        raise ValueError(f"Invalid or incomplete {schema} completion artifact")
    return body


def _verify_outputs(directory: Path, records: dict, shape: tuple[int, ...]) -> None:
    for name, item in records.items():
        filename = item.get("file")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError(f"Unsafe artifact filename: {name}")
        path = directory / filename
        if (not path.is_file() or path.stat().st_size != item["size_bytes"]
                or _hash_file(path) != item["sha256"]):
            raise ValueError(f"Output checksum or size mismatch: {name}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if tuple(array.shape) != shape or list(array.shape) != item["shape"]:
            raise ValueError(f"Output geometry mismatch: {name}")


def _validate_findings(manifest: dict, shape: tuple[int, ...], inline_ids, crossline_ids) -> None:
    if (manifest.get("schema") != "rtm-full-field-findings-1"
            or manifest.get("status") != "complete_for_supplied_full_volume"
            or tuple(manifest.get("array_shape", ())) != shape
            or manifest.get("processed_voxels") != int(np.prod(shape, dtype=np.int64))
            or not 0 <= manifest.get("coverage_fraction", -1) <= 1
            or not 0 <= manifest.get("finding_count", -1) <= 10
            or manifest.get("finding_count") != len(manifest.get("findings", []))):
        raise ValueError("Findings are incomplete, bounded, or incompatible with full-volume publication")
    if len(inline_ids) != shape[0] or len(crossline_ids) != shape[1]:
        raise ValueError("SEG-Y header axes do not match finding grid")
    ids = set()
    for finding in manifest["findings"]:
        ident = finding.get("id")
        loc = finding.get("array_location")
        segy = finding.get("segy_location", {})
        if (not isinstance(ident, str) or not ident.startswith("field-") or ident in ids
                or not isinstance(loc, list) or len(loc) != 3
                or any(type(v) is not int or v < 0 or v >= shape[i] for i, v in enumerate(loc))):
            raise ValueError("Invalid or duplicate full-field finding identity/location")
        if (segy.get("inline_id") != int(inline_ids[loc[0]])
                or segy.get("crossline_id") != int(crossline_ids[loc[1]])
                or segy.get("sample_index") != loc[2]):
            raise ValueError("Finding array location does not map to supplied SEG-Y axes")
        if (not isinstance(finding.get("edge_metrics"), dict)
                or not np.isfinite(finding.get("edge_artifact_risk", np.nan))
                or not 0 <= finding["edge_artifact_risk"] <= 1
                or finding.get("member_mask_sha256") is None):
            raise ValueError("Finding lacks valid edge QC or exact component provenance")
        ids.add(ident)


def _interpret(finding: dict) -> dict:
    loc = finding["segy_location"]
    metrics = finding["measurements"]
    evidence = {
        "location": {"inline_id": loc["inline_id"], "crossline_id": loc["crossline_id"],
                     "sample_index": loc["sample_index"]},
        "mean_signed_difference": metrics.get("mean_signed_difference"),
        "mean_absolute_difference": metrics.get("mean_absolute_difference"),
        "mean_nrms_percent": metrics.get("mean_nrms_percent"),
        "paired_support_minimum": metrics.get("minimum_paired_sample_support"),
        "component_voxels": finding["component_voxels"],
        "edge_artifact_risk": finding["edge_artifact_risk"],
        "edge_metrics": finding["edge_metrics"],
        "detectors": finding.get("detectors", {}),
    }
    return {
        "source": "deterministic_local_full_field_interpretation",
        "validation": "evidence-linked; not geological validation",
        "result": "A full-volume statistical/ML candidate was identified in the paired seismic attributes.",
        "discussion": ("Possible reservoir response is a hypothesis. Physical 4D registration, acquisition/processing repeatability, "
                       "and independent geological evidence remain unresolved."),
        "recovery_relevance": "Investigation candidate only; no recovery opportunity is established.",
        "limitations": ["Trace-key pairing is not verified physical registration.",
                        "Sample is an index; time/depth origin is unresolved.",
                        "Detector scores and ranking are not calibrated probabilities.",
                        "No well-to-seismic association is asserted."],
        "actions": [
            {"discipline": "Geophysics", "action": "Check registration, repeatability, edge risk, and processing history.",
             "evidence_ids": [finding["id"] + ":edge", finding["id"] + ":seismic"]},
            {"discipline": "Petrophysics", "action": "Review source logs and markers after a verified well tie.",
             "evidence_ids": [finding["id"] + ":well_context"]},
            {"discipline": "Reservoir engineering", "action": "Gather pressure and production history before recovery screening.",
             "evidence_ids": [finding["id"] + ":well_context", finding["id"] + ":registration"]},
        ],
        "evidence": evidence,
    }


def publish_full_field_registry(*, findings_path, attributes_directory, model_directory,
                                labels_path, output_root, dataset_identity, shape,
                                inline_ids, crossline_ids, well_context):
    """Validate completed HPC products and atomically publish an immutable registry."""
    findings_path = Path(findings_path).resolve()
    attributes_directory = Path(attributes_directory).resolve()
    model_directory = Path(model_directory).resolve()
    labels_path = Path(labels_path).resolve()
    output_root = Path(output_root).resolve()
    shape = tuple(int(x) for x in shape)
    if len(shape) != 3 or len(dataset_identity) != 64:
        raise ValueError("Invalid dataset identity or source geometry")
    attr = _read_complete(attributes_directory / "complete.json", "rtm-full-field-attributes-1")
    model = json.loads((model_directory / "complete.json").read_text(encoding="utf-8"))
    model_manifest = json.loads((model_directory / "manifest.json").read_text(encoding="utf-8"))
    if (attr.get("status") != "complete" or tuple(attr["configuration"]["shape"]) != shape
            or attr["configuration"]["source_domain_shape"] != list(shape)
            or attr["configuration"]["source_bounds"] != [0, shape[0], 0, shape[1], 0, shape[2]]):
        raise ValueError("Attribute artifact does not cover the complete source volume")
    required_attributes = {"signed_difference", "absolute_difference", "normalized_difference",
                          "rms_baseline", "rms_monitor", "nrms", "paired_sample_count"}
    if set(attr.get("outputs", {})) != required_attributes:
        raise ValueError("The complete seven-field attribute product is required")
    if (model.get("status") != "complete" or model.get("schema") != "rtm-full-field-model-scores-1"
            or model.get("configuration_sha256") != model_manifest.get("configuration_sha256")
            or model_manifest.get("configuration", {}).get("source_identity")
               != attr.get("configuration_sha256")):
        raise ValueError("Model scores are incomplete or do not derive from these attributes")
    expected_model_outputs = {f"{name}-{kind}.npy" for name in model_manifest.get("configuration", {}).get("models", [])
                              for kind in ("score", "flag")}
    if not expected_model_outputs or expected_model_outputs != set(model.get("outputs", {})):
        raise ValueError("Model output inventory does not match its completed model manifest")
    _verify_outputs(attributes_directory, attr["outputs"], shape)
    _verify_outputs(model_directory, model["outputs"], shape)
    findings = json.loads(findings_path.read_text(encoding="utf-8"))
    if findings.get("source_identity") != attr.get("configuration_sha256"):
        raise ValueError("Finding source identity does not match completed attributes")
    _validate_findings(findings, shape, inline_ids, crossline_ids)
    if labels_path.stat().st_size <= 0:
        raise ValueError("Component labels are unavailable")
    label_map = np.load(labels_path, mmap_mode="r", allow_pickle=False)
    if tuple(label_map.shape) != shape:
        raise ValueError("Component-label geometry mismatch")
    for finding in findings["findings"]:
        ext = finding.get("extent_in_array_indices")
        if (not isinstance(ext, list) or len(ext) != 3
                or any(not isinstance(axis, list) or len(axis) != 2 for axis in ext)):
            raise ValueError("Finding component extent is invalid")
        sl = tuple(slice(int(axis[0]), int(axis[1]) + 1) for axis in ext)
        member = np.asarray(label_map[sl] == int(finding["component_label"]), dtype=bool)
        if (int(member.sum()) != finding["component_voxels"]
                or hashlib.sha256(np.ascontiguousarray(member).tobytes()).hexdigest()
                   != finding["member_mask_sha256"]):
            raise ValueError("Finding membership differs from the completed 3D component labels")
        location = finding["array_location"]
        if not bool(label_map[tuple(location)] == int(finding["component_label"])):
            raise ValueError("Representative location is not a member of the finding")
    label_digest = _hash_file(labels_path)
    source_hashes = {"attributes_complete_sha256": _hash_file(attributes_directory / "complete.json"),
                     "models_complete_sha256": _hash_file(model_directory / "complete.json"),
                     "findings_sha256": _hash_file(findings_path),
                     "component_labels_sha256": label_digest}
    records = []
    for rank, finding in enumerate(findings["findings"], 1):
        records.append({"id": finding["id"], "rank": rank, "finding": finding,
                        "interpretation": _interpret(finding),
                        "well_evidence": {"association_available": False,
                                          "source_context": well_context,
                                          "limitation": "Well data are contextual only; no spatial/depth tie is verified."}})
    body = {"schema": SCHEMA, "dataset_identity": dataset_identity, "source_identity": findings["source_identity"],
            "array_shape": list(shape), "inline_ids": [int(x) for x in inline_ids],
            "crossline_ids": [int(x) for x in crossline_ids],
            "coverage": {"status": "complete_full_volume", "fraction": 1.0,
                         "valid_voxels": findings["valid_voxels"], "processed_voxels": findings["processed_voxels"],
                         "ranking_scope": findings["ranking_scope"]},
            "provenance": {"attribute_run": str(attributes_directory),
                           "model_run": str(model_directory), "findings_path": str(findings_path),
                           "component_labels_path": str(labels_path), "sha256": source_hashes},
            "model_names": findings["model_names"], "limitations": findings["limitations"],
            "findings": records}
    registry_id = hashlib.sha256(_canonical(body)).hexdigest()
    envelope = {"registry_id": registry_id, "body": body, "sha256": registry_id}
    output_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if output_root.stat().st_uid != os.getuid() or output_root.stat().st_mode & 0o077:
        raise PermissionError("Registry output must be private (mode 0700)")
    target = output_root / f"registry-{registry_id}.json"
    payload = _canonical(envelope) + b"\n"
    if not target.exists():
        fd, temporary = tempfile.mkstemp(prefix=".registry-", dir=output_root)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload); stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, target)
            directory_fd = os.open(output_root, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            Path(temporary).unlink(missing_ok=True)
    pointer_body = {"schema": "rtm-full-field-registry-pointer-1", "registry_id": registry_id,
                    "filename": target.name, "sha256": hashlib.sha256(payload).hexdigest()}
    pointer_payload = _canonical(pointer_body) + b"\n"
    fd, temporary = tempfile.mkstemp(prefix=".latest-", dir=output_root)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(pointer_payload); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, output_root / "latest.json")
        directory_fd = os.open(output_root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return {"registry_id": registry_id, "registry_path": str(target), "pointer_path": str(output_root / "latest.json"),
            "finding_count": len(records), "source_hashes": source_hashes}


def load_full_field_registry(path, *, dataset_identity, shape, inline_ids, crossline_ids):
    """Load an immutable registry or latest pointer without scanning seismic volumes."""
    path = Path(path).resolve()
    if path.stat().st_size > 4_000_000:
        raise ValueError("Registry exceeds bounded manifest size")
    raw = path.read_bytes()
    item = json.loads(raw)
    if item.get("schema") == "rtm-full-field-registry-pointer-1":
        filename = item.get("filename", "")
        if Path(filename).name != filename or not filename.startswith("registry-"):
            raise ValueError("Unsafe registry pointer")
        target = path.parent / filename
        payload = target.read_bytes()
        if hashlib.sha256(payload).hexdigest() != item.get("sha256"):
            raise ValueError("Published registry pointer checksum mismatch")
        item = json.loads(payload)
    body = item.get("body")
    digest = hashlib.sha256(_canonical(body)).hexdigest() if isinstance(body, dict) else ""
    if (item.get("sha256") != digest or item.get("registry_id") != digest
            or body.get("schema") != SCHEMA or body.get("dataset_identity") != dataset_identity
            or body.get("array_shape") != list(shape)
            or body.get("inline_ids") != [int(x) for x in inline_ids]
            or body.get("crossline_ids") != [int(x) for x in crossline_ids]
            or body.get("coverage", {}).get("status") != "complete_full_volume"
            or body.get("coverage", {}).get("fraction") != 1.0):
        raise ValueError("Full-field registry identity, geometry, or completion status is invalid")
    # Reuse the same coordinate/edge/schema validation as publication.
    manifest = {"schema": "rtm-full-field-findings-1", "status": "complete_for_supplied_full_volume",
                "array_shape": body["array_shape"],
                "coverage_fraction": body["coverage"].get("valid_voxels", 0)/max(1,body["coverage"].get("processed_voxels", 0)),
                "processed_voxels": body["coverage"].get("processed_voxels"),
                "finding_count": len(body["findings"]), "findings": [r["finding"] for r in body["findings"]]}
    _validate_findings(manifest, tuple(shape), inline_ids, crossline_ids)
    return {"id": digest, "body": body, "entries": body["findings"], "path": str(path)}
