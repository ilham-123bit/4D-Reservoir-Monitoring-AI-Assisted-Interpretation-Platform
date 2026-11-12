"""Preliminary 3D component extraction and transparent investigation ranking."""

from __future__ import annotations

import hashlib
import gzip
import json
import os
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt

from ..detection.clustering import connected_zones_chunked
from .edge_qc import summarize_candidate_edges
from ..processing.chunks import iter_chunks
from ..processing.full_field import _atomic_json, _canonical


def _write_streamed_envelope(path: Path, body: dict, list_key: str, records) -> str:
    """Write a canonical JSON envelope while streaming its potentially large list."""
    if list_key not in body:
        raise ValueError(f"streamed catalogue body has no {list_key!r} field")
    digest = hashlib.sha256()
    with Path(path).open("xb") as stream:
        stream.write(b'{"body":')

        def emit(value: bytes) -> None:
            stream.write(value)
            digest.update(value)

        emit(b"{")
        first = True
        for key in sorted(body):
            if not first:
                emit(b",")
            first = False
            emit(json.dumps(key, ensure_ascii=False, separators=(",", ":")).encode())
            emit(b":")
            if key != list_key:
                emit(_canonical(body[key]))
                continue
            emit(b"[")
            first_record = True
            for record in records:
                if not first_record:
                    emit(b",")
                first_record = False
                emit(_canonical(record))
            emit(b"]")
        emit(b"}")
        checksum = digest.hexdigest()
        stream.write(b',"sha256":"' + checksum.encode("ascii") + b'"}')
        stream.flush()
        os.fsync(stream.fileno())
    return checksum


def _verify_streamed_envelope(path: Path, expected_sha256: str, source_identity: str,
                              schema: str) -> bool:
    """Verify a canonical streamed catalogue without loading it into memory."""
    prefix = b'{"body":'
    suffix = b',"sha256":"' + expected_sha256.encode("ascii") + b'"}'
    identity_marker = b'"source_identity":"' + source_identity.encode("ascii") + b'"'
    schema_marker = b'"schema":"' + schema.encode("ascii") + b'"'
    status_marker = b'"status":"complete_for_supplied_full_volume"'
    try:
        size = Path(path).stat().st_size
        if size < len(prefix) + len(suffix):
            return False
        digest = hashlib.sha256()
        tail = b""
        markers = {"identity": False, "schema": False, "status": False}
        with Path(path).open("rb") as stream:
            if stream.read(len(prefix)) != prefix:
                return False
            stream.seek(-len(suffix), os.SEEK_END)
            if stream.read() != suffix:
                return False
            body_start = len(prefix)
            body_end = size - len(suffix)
            stream.seek(body_start)
            remaining = body_end - body_start
            while remaining:
                block = stream.read(min(8 * 1024 * 1024, remaining))
                if not block:
                    return False
                remaining -= len(block)
                digest.update(block)
                searchable = tail + block
                markers["identity"] |= identity_marker in searchable
                markers["schema"] |= schema_marker in searchable
                markers["status"] |= status_marker in searchable
                tail = searchable[-max(len(identity_marker), len(schema_marker), len(status_marker)):]
        return (digest.hexdigest() == expected_sha256 and all(markers.values()))
    except (OSError, ValueError):
        return False


def build_full_field_findings(attributes, model_outputs: dict, missing_trace_mask,
                              statistical_reference_absolute_difference, output_directory, *,
                              source_identity: str, inline_ids, crossline_ids,
                              min_region_voxels: int = 16, min_ml_votes: int = 2,
                              mad_multiplier: float = 3.5, chunk_shape=(32, 32, 64),
                              max_findings: int = 10, survey_amplitudes=None,
                              monitor_only_keys=None,
                              window_samples: int = 9, min_support_fraction: float = 0.8) -> dict:
    """Group 3D candidates after full ML completion; outputs are screening targets.

    The statistical cutoff is estimated from a caller-supplied spatial reference
    population. Candidates are MAD exceedances OR explicit multi-model agreement.
    Priority combines amplitude strength, component extent, model agreement and
    support; it is an investigation index, never geological probability.
    """
    shape = tuple(np.shape(attributes["absolute_difference"]))
    if len(shape) != 3 or np.shape(attributes["signed_difference"]) != shape or np.shape(attributes["nrms"]) != shape:
        raise ValueError("full 3D signed difference, absolute difference, and NRMS maps are required")
    if np.shape(attributes["paired_sample_count"]) != shape or np.shape(missing_trace_mask) != shape[:2]:
        raise ValueError("support/missing-trace grids do not match the attribute volume")
    if len(inline_ids) != shape[0] or len(crossline_ids) != shape[1]:
        raise ValueError("verified SEG-Y header axes must match the source array grid")
    if not model_outputs or min_ml_votes < 1 or min_ml_votes > len(model_outputs):
        raise ValueError("at least one eligible model output and valid vote requirement are needed")
    if not 1 <= max_findings <= 10 or len(source_identity) != 64:
        raise ValueError("source identity must be SHA-256 and max_findings must be in 1..10")
    if (isinstance(window_samples, bool) or not isinstance(window_samples, int) or window_samples < 1
            or not np.isfinite(min_support_fraction) or not 0 < min_support_fraction <= 1):
        raise ValueError("window/support configuration is invalid")
    if survey_amplitudes is not None and any(np.shape(survey_amplitudes.get(name)) != shape
                                             for name in ("baseline", "monitor")):
        raise ValueError("survey Baseline/Monitor arrays must match attribute geometry")
    if survey_amplitudes is not None and any(
            name not in attributes or np.shape(attributes[name]) != shape
            for name in ("rms_baseline", "rms_monitor")):
        raise ValueError("edge QC requires validated Baseline and Monitor RMS attribute volumes")
    if any(np.shape(item.get("score")) != shape or np.shape(item.get("flag")) != shape for item in model_outputs.values()):
        raise ValueError("each detector score and flag map must match the full 3D grid")
    reference = np.asarray(statistical_reference_absolute_difference, dtype=np.float64)
    reference = reference[np.isfinite(reference)]
    if reference.size < 32 or mad_multiplier <= 0 or not np.isfinite(mad_multiplier):
        raise ValueError("MAD reference needs >=32 finite independent spatial samples and a positive multiplier")
    center = float(np.median(reference))
    mad = float(np.median(np.abs(reference - center)))
    if mad <= np.finfo(float).eps:
        raise ValueError("MAD reference has zero spread; no defensible MAD cutoff is available")
    cutoff = center + mad_multiplier * 1.4826 * mad
    root = Path(output_directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if root.stat().st_uid != os.getuid() or root.stat().st_mode & 0o077:
        raise PermissionError("finding artifacts must be user-owned and private")
    stat_mask_path, vote_path, candidate_path = (root / "statistical_mad_mask.npy",
        root / "model_vote_count.npy", root / "candidate_union.npy")
    catalogue_path, qc_catalogue_path = root / "candidate_catalogue.json", root / "qc_catalogue.json"
    if any(path.exists() for path in (stat_mask_path, vote_path, candidate_path,
                                      root / "findings.json", catalogue_path, qc_catalogue_path)):
        raise FileExistsError("candidate artifacts already exist; refusing to overwrite")
    stat_mask = np.lib.format.open_memmap(stat_mask_path, mode="w+", dtype=np.uint8, shape=shape)
    votes = np.lib.format.open_memmap(vote_path, mode="w+", dtype=np.uint8, shape=shape)
    candidates = np.lib.format.open_memmap(candidate_path, mode="w+", dtype=np.uint8, shape=shape)
    total_valid = total_stat = total_union = 0
    missing_grid = np.asarray(missing_trace_mask, dtype=bool)
    near_missing_grid = binary_dilation(missing_grid, iterations=1)
    distance_to_unoccupied = (distance_transform_edt(~missing_grid)
                              if missing_grid.any() else None)
    monitor_keys = np.asarray(monitor_only_keys if monitor_only_keys is not None else [], dtype=np.int64)
    if monitor_keys.size == 0:
        monitor_keys = np.empty((0, 2), dtype=np.int64)
    if monitor_keys.ndim != 2 or monitor_keys.shape[1] != 2:
        raise ValueError("Monitor-only coverage keys must be inline/crossline pairs")
    if len(np.unique(monitor_keys, axis=0)) != len(monitor_keys):
        raise ValueError("Monitor-only coverage keys must be unique")
    inline_map = {int(value): index for index, value in enumerate(inline_ids)}
    crossline_map = {int(value): index for index, value in enumerate(crossline_ids)}
    monitor_only_grid = np.zeros(shape[:2], dtype=bool)
    unmapped_monitor_only = 0
    for inline, crossline in monitor_keys:
        ii, xx = inline_map.get(int(inline)), crossline_map.get(int(crossline))
        if ii is None or xx is None:
            unmapped_monitor_only += 1
        else:
            if not missing_grid[ii, xx]:
                raise ValueError("Monitor-only key maps to an occupied Baseline-grid position")
            monitor_only_grid[ii, xx] = True
    near_monitor_only_grid = (binary_dilation(monitor_only_grid, iterations=1)
                              if monitor_only_grid.any() else monitor_only_grid)
    distance_to_monitor_only = (distance_transform_edt(~monitor_only_grid)
                                if monitor_only_grid.any() else None)
    sample_centers = np.arange(shape[2])
    sample_left = window_samples // 2
    sample_right = (window_samples - 1) // 2
    available_samples = (np.minimum(shape[2], sample_centers + sample_right + 1)
                         - np.maximum(0, sample_centers - sample_left))
    required_support = np.ceil(min_support_fraction * available_samples).astype(np.uint16)
    pre_group_edge_counts = {"candidate_voxels_near_unoccupied_grid": 0,
        "candidate_voxels_near_monitor_only_coverage": 0,
        "candidate_voxels_on_rectangular_spatial_boundary": 0,
        "invalid_or_unsupported_voxels_excluded": 0}
    for chunk in iter_chunks(shape, chunk_shape):
        region = chunk.core
        absdiff = np.asarray(attributes["absolute_difference"][region], dtype=np.float32)
        signed_block = np.asarray(attributes["signed_difference"][region], dtype=np.float32)
        nrms_block = np.asarray(attributes["nrms"][region], dtype=np.float32)
        support = np.asarray(attributes["paired_sample_count"][region])
        missing = np.asarray(missing_trace_mask[region[:2]], dtype=bool)
        sample_required = required_support[region[2]]
        valid = (np.isfinite(absdiff) & np.isfinite(signed_block) & np.isfinite(nrms_block)
                 & (support >= sample_required[None, None, :]) & ~missing[..., None])
        stat = valid & (absdiff > cutoff)
        vote = np.zeros(absdiff.shape, dtype=np.uint8)
        for output in model_outputs.values():
            score = np.asarray(output["score"][region])
            flag = np.asarray(output["flag"][region], dtype=bool)
            vote += (valid & np.isfinite(score) & flag).astype(np.uint8)
        union = stat | (valid & (vote >= min_ml_votes))
        # Coverage risk is measured before grouping. Boundary-adjacent candidates
        # remain in the mask for review; component-level edge metrics determine
        # later quarantine/priority eligibility rather than silently deleting them.
        near_missing = np.asarray(near_missing_grid[region[:2]], dtype=bool)
        near_monitor = np.asarray(near_monitor_only_grid[region[:2]], dtype=bool)
        pre_group_edge_counts["candidate_voxels_near_unoccupied_grid"] += int(
            np.count_nonzero(union & near_missing[..., None]))
        pre_group_edge_counts["candidate_voxels_near_monitor_only_coverage"] += int(
            np.count_nonzero(union & near_monitor[..., None]))
        at_spatial_edge = np.zeros(union.shape[:2], dtype=bool)
        if region[0].start == 0:
            at_spatial_edge[0, :] = True
        if region[0].stop == shape[0]:
            at_spatial_edge[-1, :] = True
        if region[1].start == 0:
            at_spatial_edge[:, 0] = True
        if region[1].stop == shape[1]:
            at_spatial_edge[:, -1] = True
        pre_group_edge_counts["candidate_voxels_on_rectangular_spatial_boundary"] += int(
            np.count_nonzero(union & at_spatial_edge[..., None]))
        pre_group_edge_counts["invalid_or_unsupported_voxels_excluded"] += int(
            np.count_nonzero(~valid))
        stat_mask[region] = stat
        votes[region] = vote
        candidates[region] = union
        total_valid += int(valid.sum())
        total_stat += int(stat.sum())
        total_union += int(union.sum())
    for array in (stat_mask, votes, candidates):
        array.flush()
    # Keep the exact voxel candidate mask above. Group only globally connected
    # components meeting the existing minimum extent; the clustering routine
    # retains seam-touching fragments until global merging and summarizes smaller
    # components instead of allocating one Python record per isolated MAD voxel.
    grouping_summary = {}
    labels, zones = connected_zones_chunked(candidates, root / "components",
        chunk_shape=tuple(chunk_shape), score=attributes["absolute_difference"],
        min_size=min_region_voxels, connectivity=1, component_summary=grouping_summary,
        component_capacity=total_union)
    component_count = int(grouping_summary["total_components"])
    candidate_spool_path = root / ".candidate-records.jsonl.gz"
    qc_spool_path = root / ".qc-records.jsonl.gz"
    if candidate_spool_path.exists() or qc_spool_path.exists():
        raise FileExistsError("temporary candidate catalogue spool already exists; refusing to overwrite")
    candidate_spool = gzip.open(candidate_spool_path, "xb")
    qc_spool = gzip.open(qc_spool_path, "xb")
    candidate_count = 0
    quarantined_count = 0
    priority_eligible_count = 0
    priority_eligible = []
    total_voxels = int(np.prod(shape, dtype=np.int64))
    model_names = tuple(sorted(model_outputs))
    for zone in zones:
        box = tuple(slice(low, high + 1) for low, high in zone.bounding_box)
        local_labels = np.asarray(labels[box])
        member = local_labels == zone.label
        if not member.any():
            continue
        absdiff = np.asarray(attributes["absolute_difference"][box], dtype=np.float32)
        signed = np.asarray(attributes["signed_difference"][box], dtype=np.float32)
        nrms = np.asarray(attributes["nrms"][box], dtype=np.float32)
        support = np.asarray(attributes["paired_sample_count"][box])
        local_vote = np.asarray(votes[box])
        candidate_strength = np.where(member, absdiff, -np.inf)
        rep_local = np.unravel_index(int(np.argmax(candidate_strength)), candidate_strength.shape)
        rep = tuple(int(part.start + index) for part, index in zip(box, rep_local))
        i0, i1 = zone.bounding_box[0]
        x0, x1 = zone.bounding_box[1]
        s0, s1 = zone.bounding_box[2]
        member_mask_hash = hashlib.sha256(np.ascontiguousarray(member).tobytes()).hexdigest()
        finding_id = "field-" + hashlib.sha256((source_identity + member_mask_hash +
            json.dumps([zone.bounding_box, zone.voxel_count], separators=(",", ":"))).encode()).hexdigest()[:20]
        if zone.voxel_count < min_region_voxels:
            member_coords = np.nonzero(member)
            global_i = member_coords[0] + i0
            global_x = member_coords[1] + x0
            global_s = member_coords[2] + s0
            survey_edge_fraction = float(np.mean(
                (global_i == 0) | (global_i == shape[0]-1) |
                (global_x == 0) | (global_x == shape[1]-1) |
                (global_s == 0) | (global_s == shape[2]-1)))
            missing_near = np.asarray(near_missing_grid[box[:2]], dtype=bool)[..., None]
            monitor_near = np.asarray(near_monitor_only_grid[box[:2]], dtype=bool)[..., None]
            missing_fraction = float(np.mean(np.broadcast_to(missing_near, member.shape)[member]))
            monitor_fraction = float(np.mean(np.broadcast_to(monitor_near, member.shape)[member]))
            reasons = [f"component has {zone.voxel_count} voxels; configured minimum is {min_region_voxels}"]
            record = {"id": finding_id, "component_label": zone.label,
                "source_identity": source_identity,
                "array_location": list(rep), "segy_location": {
                    "inline_id": int(inline_ids[rep[0]]), "crossline_id": int(crossline_ids[rep[1]]),
                    "sample_index": rep[2]},
                "extent_in_array_indices": [[i0, i1], [x0, x1], [s0, s1]],
                "component_voxels": zone.voxel_count, "member_mask_sha256": member_mask_hash,
                "measurements": {"mean_absolute_difference": float(absdiff[member].mean()),
                    "mean_signed_difference": float(signed[member].mean()),
                    "mean_nrms_percent": float(nrms[member].mean()),
                    "minimum_paired_sample_support": int(support[member].min())},
                "detector_vote_fraction": float(local_vote[member].mean() / len(model_names)),
                "survey_boundary_member_fraction": survey_edge_fraction,
                "unoccupied_grid_neighborhood_fraction": missing_fraction,
                "monitor_only_neighborhood_fraction": monitor_fraction,
                "priority_eligible": False, "priority_exclusion_reasons": reasons,
                "catalogue_status": "retained_candidate_below_minimum_extent"}
            candidate_records.append(record)
            if survey_edge_fraction == 1.0 or missing_fraction == 1.0 or monitor_fraction == 1.0:
                artifact_reasons = []
                if survey_edge_fraction == 1.0:
                    artifact_reasons.append("all component voxels lie on the rectangular survey boundary")
                if missing_fraction == 1.0:
                    artifact_reasons.append("all component voxels are adjacent to unoccupied Baseline-grid cells")
                if monitor_fraction == 1.0:
                    artifact_reasons.append("all component voxels are adjacent to Monitor-only coverage")
                quarantined_records.append({"finding_id": finding_id,
                    "reason_codes": artifact_reasons, "component_voxels": zone.voxel_count,
                    "disposition": "quarantined_from_priority_pending_QC_review"})
                record["catalogue_status"] = "quarantined_data_quality"
            continue
        method = {}
        for name, output in model_outputs.items():
            flags = np.asarray(output["flag"][box], dtype=bool)
            scores = np.asarray(output["score"][box], dtype=np.float32)
            supported = member & np.isfinite(scores)
            method[name] = {"flagged_voxels": int((member & flags & supported).sum()),
                            "valid_component_voxels": int(supported.sum()),
                            "flagged_fraction": float((flags[supported]).mean()) if supported.any() else None,
                            "score_mean": float(scores[supported].mean()) if supported.any() else None,
                            "score_is_probability": False}
        member_votes = local_vote[member]
        valid_support = support[member]
        member_coords = np.nonzero(member)
        near_missing = np.asarray(near_missing_grid[box[:2]], dtype=bool)[..., None]
        unoccupied_neighborhood_fraction = float(np.mean(np.broadcast_to(near_missing, member.shape)[member]))
        near_monitor = np.asarray(near_monitor_only_grid[box[:2]], dtype=bool)[..., None]
        monitor_only_neighborhood_fraction = float(
            np.mean(np.broadcast_to(near_monitor, member.shape)[member])) if member.any() else 0.0
        if distance_to_unoccupied is not None:
            member_spatial_distance = distance_to_unoccupied[
                member_coords[0] + i0, member_coords[1] + x0]
            min_unoccupied_distance = float(member_spatial_distance.min())
            mean_unoccupied_distance = float(member_spatial_distance.mean())
        else:
            min_unoccupied_distance = None
            mean_unoccupied_distance = None
        if distance_to_monitor_only is not None:
            member_monitor_distance = distance_to_monitor_only[
                member_coords[0] + i0, member_coords[1] + x0]
            min_monitor_only_distance = float(member_monitor_distance.min())
            mean_monitor_only_distance = float(member_monitor_distance.mean())
        else:
            min_monitor_only_distance = None
            mean_monitor_only_distance = None
        valid_nrms = nrms[member & np.isfinite(nrms)]
        valid_signed = signed[member & np.isfinite(signed)]
        mean_abs = float(absdiff[member].mean())
        strength = min(1.0, mean_abs / max(cutoff, np.finfo(float).eps))
        extent = min(1.0, np.log1p(zone.voxel_count) / np.log1p(total_voxels))
        agreement = float(member_votes.mean() / len(model_names))
        member_sample_index = np.nonzero(member)[2] + s0
        member_required = required_support[member_sample_index]
        support_fraction = float(np.mean(valid_support / np.maximum(member_required, 1))) if valid_support.size else 0.0
        edge_distance = min(i0, shape[0]-1-i1, x0, shape[1]-1-x1, s0, shape[2]-1-s1)
        boundary_member = np.zeros(zone.voxel_count, dtype=bool)
        for axis, axis_size in enumerate(shape):
            pos = member_coords[axis] + zone.bounding_box[axis][0]
            boundary_member |= (pos <= 0) | (pos >= axis_size-1)
        survey_boundary_fraction = float(boundary_member.mean()) if boundary_member.size else 1.0
        if survey_amplitudes is not None:
            edge_metrics = summarize_candidate_edges(
                baseline=survey_amplitudes["baseline"], monitor=survey_amplitudes["monitor"],
                signed_difference=attributes["signed_difference"],
                absolute_difference=attributes["absolute_difference"], nrms=attributes["nrms"],
                support=attributes["paired_sample_count"],
                rms_baseline=attributes["rms_baseline"], rms_monitor=attributes["rms_monitor"],
                unoccupied_mask=missing_trace_mask, member_mask=member, bounds=box,
                min_support_by_sample=required_support)
            gradient_coverage = (edge_metrics["gradient_coverage_fraction"]
                                 if edge_metrics["status"] == "computed" else 0.0)
        else:
            edge_metrics = {"status": "unavailable", "reason": "survey amplitude arrays not supplied"}
            gradient_coverage = 0.0
        boundary_factor = 1.0 - 0.5 * survey_boundary_fraction
        unoccupied_factor = 1.0 - 0.5 * unoccupied_neighborhood_fraction
        monitor_only_factor = 1.0 - 0.5 * monitor_only_neighborhood_fraction
        gradient_factor = 0.5 + 0.5 * gradient_coverage if edge_metrics["status"] == "computed" else 0.75
        priority = boundary_factor * unoccupied_factor * monitor_only_factor * gradient_factor * (
            0.35*strength + 0.25*extent + 0.25*agreement + 0.15*support_fraction)
        record = {"id": finding_id, "component_label": zone.label, "source_identity": source_identity,
            "array_location": list(rep), "segy_location": {"inline_id": int(inline_ids[rep[0]]),
                "crossline_id": int(crossline_ids[rep[1]]), "sample_index": rep[2],
                "xy": "N/A — verified coordinate transform unavailable"},
            "extent_in_array_indices": [[i0, i1], [x0, x1], [s0, s1]],
            "component_voxels": zone.voxel_count, "member_mask_sha256": member_mask_hash,
            "measurements": {"mean_absolute_difference": mean_abs,
                "mean_signed_difference": float(valid_signed.mean()) if valid_signed.size else None,
                "mean_nrms_percent": float(valid_nrms.mean()) if valid_nrms.size else None,
                "minimum_paired_sample_support": int(valid_support.min()) if valid_support.size else 0,
                "maximum_absolute_difference": float(zone.max_strength)},
            "detectors": method, "mean_detector_vote_fraction": agreement,
            "edge_distance_grid_cells": int(edge_distance), "survey_boundary_member_fraction": survey_boundary_fraction,
            "survey_edge_factor": boundary_factor,
            "unoccupied_grid_neighborhood_fraction": unoccupied_neighborhood_fraction,
            "monitor_only_neighborhood_fraction": monitor_only_neighborhood_fraction,
            "distance_to_monitor_only_coverage_keys": {
                "minimum_grid_cells": min_monitor_only_distance,
                "mean_grid_cells": mean_monitor_only_distance,
                "status": "coverage only; excluded from paired calculations" if monitor_only_grid.any()
                         else "no monitor-only keys provided or mapped",
            },
            "distance_to_nearest_unoccupied_grid_cell": {
                "minimum_grid_cells": min_unoccupied_distance,
                "mean_grid_cells": mean_unoccupied_distance,
                "status": "measured" if min_unoccupied_distance is not None else "no unoccupied cells in grid",
            },
            "unoccupied_grid_factor": unoccupied_factor,
            "monitor_only_coverage_factor": monitor_only_factor,
            "edge_gradient_coverage_fraction": gradient_coverage,
            "edge_gradient_factor": gradient_factor, "edge_metrics": edge_metrics,
            "edge_artifact_risk": float(max(survey_boundary_fraction,
                unoccupied_neighborhood_fraction, monitor_only_neighborhood_fraction,
                1.0-gradient_coverage)),
            "investigation_priority_index": float(priority),
            "priority_components": {"change_strength": strength, "component_extent": extent,
                                    "detector_vote_fraction": agreement, "support_fraction": support_fraction},
            "status": "preliminary 3D change candidate; physical registration and geological cause unverified",
            "well_association": "N/A — CRS, datum, and seismic well tie are not verified",
            "recovery_status": "investigation opportunity only; no recovery outcome established"}
        exclusion_reasons = []
        if survey_boundary_fraction == 1.0:
            exclusion_reasons.append("all component voxels lie on the rectangular survey boundary")
        if unoccupied_neighborhood_fraction == 1.0:
            exclusion_reasons.append("all component voxels are adjacent to unoccupied Baseline-grid cells")
        if monitor_only_neighborhood_fraction == 1.0:
            exclusion_reasons.append("all component voxels are adjacent to Monitor-only coverage")
        if gradient_coverage == 0.0:
            exclusion_reasons.append("no component voxels have complete validity-aware 3D change-gradient support")
        record["priority_eligible"] = not exclusion_reasons
        record["priority_exclusion_reasons"] = exclusion_reasons
        record["catalogue_status"] = ("quarantined_data_quality" if exclusion_reasons
                                       else "eligible_for_priority_ranking")
        candidate_spool.write(_canonical(record) + b"\n")
        candidate_count += 1
        if exclusion_reasons:
            qc_record = {"finding_id": finding_id,
                "reason_codes": exclusion_reasons,
                "edge_artifact_risk": record["edge_artifact_risk"],
                "component_voxels": zone.voxel_count,
                "disposition": "quarantined_from_priority_pending_QC_review"}
            qc_spool.write(_canonical(qc_record) + b"\n")
            quarantined_count += 1
        else:
            priority_eligible_count += 1
            priority_eligible.append(record)
            priority_eligible.sort(key=lambda item: (-item["investigation_priority_index"], item["id"]))
            if len(priority_eligible) > max_findings:
                priority_eligible.pop()
    small_component_count = (int(grouping_summary["interior_subminimum_components"])
                             + int(grouping_summary["postmerge_subminimum_components"]))
    small_component_voxels = (int(grouping_summary["interior_subminimum_voxels"])
                              + int(grouping_summary["postmerge_subminimum_voxels"]))
    if small_component_count:
        summary_id = "subminimum-" + hashlib.sha256(
            source_identity.encode() + _canonical(grouping_summary)).hexdigest()[:20]
        summary_record = {"id": summary_id, "record_type": "component_summary",
            "source_identity": source_identity, "component_count": small_component_count,
            "component_voxels": small_component_voxels,
            "component_size_histogram": dict(sorted(grouping_summary["subminimum_size_histogram"].items(),
                                                      key=lambda item: int(item[0]))),
            "candidate_mask_file": candidate_path.name,
            "priority_eligible": False,
            "priority_exclusion_reasons": [
                f"components below configured minimum extent of {min_region_voxels} voxels; "
                "exact candidate voxels remain in candidate_union.npy"],
            "catalogue_status": "subminimum_components_summarized_not_ranked"}
        candidate_spool.write(_canonical(summary_record) + b"\n")
        candidate_count += 1
    findings = priority_eligible
    for rank, record in enumerate(findings, start=1):
        record["priority_rank"] = rank
        record["catalogue_status"] = "priority_discovery"

    candidate_spool.close()
    qc_spool.close()
    selected_ids = {record["id"] for record in findings}

    def candidate_records():
        # Keep the small discovery shortlist first; stream every other eligible,
        # quarantined, and summarized candidate from the compressed spool.
        yield from findings
        with gzip.open(candidate_spool_path, "rb") as stream:
            for line in stream:
                record = json.loads(line)
                if record["id"] in selected_ids:
                    continue
                if record.get("priority_eligible"):
                    record["catalogue_status"] = "additional_valid_candidate_not_in_top_display_limit"
                yield record

    def quarantined_records():
        with gzip.open(qc_spool_path, "rb") as stream:
            for line in stream:
                yield json.loads(line)
    catalogue_body = {"schema": "rtm-full-field-candidate-catalogue-1",
        "status": "complete_for_supplied_full_volume", "source_identity": source_identity,
        "candidate_count": candidate_count,
        "spatial_component_count": component_count,
        "subminimum_component_summary": {
            "component_count": small_component_count,
            "candidate_voxels": small_component_voxels,
            "size_histogram": dict(grouping_summary["subminimum_size_histogram"]),
            "candidate_mask_file": candidate_path.name,
            "exact_voxel_membership_preserved": True},
        "candidate_voxel_mask": {"file": candidate_path.name, "shape": list(shape),
            "dtype": "uint8", "candidate_voxels": total_union},
        "candidates": None}
    qc_body = {"schema": "rtm-full-field-qc-catalogue-1",
        "status": "complete_for_supplied_full_volume", "source_identity": source_identity,
        "quarantined_count": quarantined_count, "items": None}
    catalogue_sha256 = _write_streamed_envelope(catalogue_path, catalogue_body,
                                                "candidates", candidate_records())
    qc_sha256 = _write_streamed_envelope(qc_catalogue_path, qc_body,
                                         "items", quarantined_records())
    candidate_spool_path.unlink()
    qc_spool_path.unlink()
    manifest = {"schema": "rtm-full-field-findings-1", "source_identity": source_identity,
        "status": "complete_for_supplied_full_volume", "array_shape": list(shape),
        "processed_voxels": total_voxels, "valid_voxels": total_valid,
        "coverage_fraction": total_valid / total_voxels, "statistical_cutoff": cutoff,
        "statistical_reference_count": int(reference.size), "statistical_method": "reference-population median + multiplier x 1.4826 MAD",
        "statistical_voxels": total_stat, "candidate_voxels": total_union,
        "component_count_before_top_limit": component_count, "candidate_count": candidate_count,
        "priority_eligible_count": priority_eligible_count,
        "quarantined_qc_count": quarantined_count,
        "connected_component_statistics": grouping_summary,
        "pre_group_edge_quality_counts": pre_group_edge_counts,
        "finding_count": len(findings), "priority_discovery_count": len(findings),
        "additional_candidate_catalogue": {"file": catalogue_path.name,
            "sha256": catalogue_sha256, "candidate_count": candidate_count},
        "quarantined_qc_catalogue": {"file": qc_catalogue_path.name,
            "sha256": qc_sha256, "quarantined_count": quarantined_count},
        "monitor_only_coverage": {"provided_key_count": int(len(monitor_keys)),
            "mapped_baseline_axis_positions": int(monitor_only_grid.sum()),
            "unmapped_key_count": int(unmapped_monitor_only),
            "used_in_paired_calculations": False},
        "ranking_scope": "full-volume screening across supplied attribute products; local detector cutoffs; priority is not probability",
        "processing_boundary_policy": "voxelwise detector scores are journaled in disjoint chunks; 3D components merge across chunk boundaries; edge gradients use a one-cell source-grid halo; processing chunks are not survey edges",
        "model_names": list(model_names), "findings": findings,
        "limitations": ["Trace-key paired attributes; physical 4D registration is not verified.",
                         "Sample is an array index; sample domain and origin are unresolved.",
                         "Unsupervised scores and quantile thresholds are not calibrated probabilities.",
                         "Well association and fluid, pressure, saturation, or recovery conclusions are unavailable without independent evidence."]}
    _atomic_json(root / "findings.json", manifest)
    return {**manifest, "run_directory": str(root),
            "component_labels": str(root / "components" / "component_labels.npy")}
