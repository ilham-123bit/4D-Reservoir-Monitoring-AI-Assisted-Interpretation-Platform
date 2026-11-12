"""Validity-aware, array-index seismic edge diagnostics for candidate QC."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

from ..processing.chunks import iter_chunks

AXIS_NAMES = ("inline_index", "crossline_index", "sample_index")


def summarize_bounded_coverage_edges(*, support, missing_trace_mask, scoring_bounds,
                                     inline_ids, crossline_ids, monitor_only_keys=(),
                                     detector_flags_by_chunk, window_samples=9,
                                     min_support_fraction=0.8):
    """Report survey/footprint/crop/chunk edge context for a bounded ML score.

    Monitor-only keys are used only to map a coverage footprint. They are never
    paired with Baseline or included in attribute or detector calculations.
    Scoring-crop edges are explicitly not treated as survey edges.
    """
    support_shape = tuple(np.shape(support))
    missing = np.asarray(missing_trace_mask, dtype=bool)
    if len(support_shape) != 3 or missing.shape != support_shape[:2]:
        raise ValueError("support and survey occupancy geometry must match")
    if len(scoring_bounds) != 6:
        raise ValueError("scoring_bounds must contain six half-open array bounds")
    i0, i1, x0, x1, s0, s1 = map(int, scoring_bounds)
    ni, nx, ns = support_shape
    if not (0 <= i0 < i1 <= ni and 0 <= x0 < x1 <= nx and 0 <= s0 < s1 <= ns):
        raise ValueError("scoring bounds lie outside the source array")
    inline_ids, crossline_ids = np.asarray(inline_ids), np.asarray(crossline_ids)
    if inline_ids.shape != (ni,) or crossline_ids.shape != (nx,):
        raise ValueError("SEG-Y header axes do not match the source grid")
    if len(np.unique(inline_ids)) != ni or len(np.unique(crossline_ids)) != nx:
        raise ValueError("SEG-Y header axes must be unique for coverage mapping")
    crop = (slice(i0, i1), slice(x0, x1), slice(s0, s1))
    support_crop = np.asarray(support[crop])
    missing_crop = missing[i0:i1, x0:x1]
    centers = np.arange(s0, s1)
    left, right = window_samples // 2, (window_samples - 1) // 2
    available = np.minimum(ns, centers + right + 1) - np.maximum(0, centers - left)
    required = np.ceil(float(min_support_fraction) * available).astype(np.uint16)
    valid = ((support_crop >= required[None, None, :])
             & ~missing_crop[..., None] & np.isfinite(support_crop))

    monitor_keys = np.asarray(monitor_only_keys, dtype=np.int64)
    if monitor_keys.size == 0:
        monitor_keys = np.empty((0, 2), dtype=np.int64)
    if monitor_keys.ndim != 2 or monitor_keys.shape[1] != 2:
        raise ValueError("monitor-only key table must have inline/crossline pairs")
    if len(np.unique(monitor_keys, axis=0)) != len(monitor_keys):
        raise ValueError("monitor-only coverage keys must be unique")
    i_map = {int(value): index for index, value in enumerate(inline_ids)}
    x_map = {int(value): index for index, value in enumerate(crossline_ids)}
    monitor_grid = np.zeros((ni, nx), dtype=bool)
    unmapped_monitor_keys = 0
    for inline, crossline in monitor_keys:
        ii, xx = i_map.get(int(inline)), x_map.get(int(crossline))
        if ii is None or xx is None:
            unmapped_monitor_keys += 1
        else:
            if not missing[ii, xx]:
                raise ValueError("Monitor-only key maps to an occupied Baseline-grid position")
            monitor_grid[ii, xx] = True
    missing_distance = distance_transform_edt(~missing) if missing.any() else None
    monitor_distance = distance_transform_edt(~monitor_grid) if monitor_grid.any() else None
    yy, xx = np.ogrid[i0:i1, x0:x1]
    survey_distance = np.minimum.reduce((np.broadcast_to(yy, (i1-i0, x1-x0)),
        np.broadcast_to(ni - 1 - yy, (i1-i0, x1-x0)),
        np.broadcast_to(xx, (i1-i0, x1-x0)),
        np.broadcast_to(nx - 1 - xx, (i1-i0, x1-x0))))
    crop_i = np.minimum(np.arange(i1-i0), np.arange(i1-i0)[::-1])
    crop_x = np.minimum(np.arange(x1-x0), np.arange(x1-x0)[::-1])
    crop_s = np.minimum(np.arange(s1-s0), np.arange(s1-s0)[::-1])
    crop_edge = ((crop_i[:, None, None] == 0) | (crop_x[None, :, None] == 0)
                 | (crop_s[None, None, :] == 0))
    survey_edge = np.broadcast_to((survey_distance <= 1)[..., None], valid.shape)
    missing_edge = (np.broadcast_to((missing_distance[i0:i1, x0:x1] <= 1)[..., None], valid.shape)
                    if missing_distance is not None else np.zeros(valid.shape, dtype=bool))
    monitor_edge = (np.broadcast_to((monitor_distance[i0:i1, x0:x1] <= 1)[..., None], valid.shape)
                    if monitor_distance is not None else np.zeros(valid.shape, dtype=bool))

    def fractions(mask, selected):
        count = int(selected.sum())
        return {"positions": count,
                "fraction": float(np.count_nonzero(selected & mask) / count) if count else None}

    by_detector = {}
    chunk_reports = {}
    for chunk_label, flag_maps in detector_flags_by_chunk.items():
        seam = np.zeros(valid.shape, dtype=bool)
        dims = tuple(int(v) for v in chunk_label.split("x"))
        for axis, (size, step) in enumerate(zip(valid.shape, dims)):
            if step < size:
                positions = np.arange(size)
                on_seam = (positions > 0) & ((positions % step == 0) | (positions % step == step - 1))
                reshape = [1, 1, 1]; reshape[axis] = size
                seam |= np.broadcast_to(on_seam.reshape(reshape), valid.shape)
        chunk_reports[chunk_label] = {}
        for name, raw_flags in flag_maps.items():
            flags = np.asarray(raw_flags, dtype=bool)
            if flags.shape != valid.shape:
                raise ValueError("bounded detector flag geometry differs from scoring crop")
            flagged = flags & valid
            info = {"valid_flagged_positions": int(flagged.sum()),
                    "flagged_near_rectangular_survey_edge": fractions(survey_edge, flagged),
                    "flagged_adjacent_to_unoccupied_grid": fractions(missing_edge, flagged),
                    "flagged_adjacent_to_monitor_only_footprint": fractions(monitor_edge, flagged),
                    "flagged_on_scoring_crop_boundary": fractions(crop_edge, flagged),
                    "high_confidence_eligible": False}
            risk = any(info[k]["fraction"] not in (None, 0.0) for k in (
                "flagged_near_rectangular_survey_edge", "flagged_adjacent_to_unoccupied_grid",
                "flagged_adjacent_to_monitor_only_footprint", "flagged_on_scoring_crop_boundary"))
            info["edge_proximity_requires_manual_review"] = bool(risk)
            by_detector.setdefault(name, {})[chunk_label] = info
            inside, outside = flagged & seam, flagged & ~seam
            seam_valid, interior_valid = valid & seam, valid & ~seam
            chunk_reports[chunk_label][name] = {
                "internal_chunk_seam_positions": int(seam_valid.sum()),
                "seam_flag_fraction": float(inside.sum() / seam_valid.sum()) if seam_valid.any() else None,
                "nonseam_flag_fraction": float(outside.sum() / interior_valid.sum()) if interior_valid.any() else None,
                "seams_are_processing_boundaries_not_survey_edges": True,
            }
    crop_missing_count = int(missing_crop.sum())
    monitor_in_crop = int(monitor_grid[i0:i1, x0:x1].sum())
    return {
        "status": "computed", "coordinate_system": "array indices mapped by explicit SEG-Y axes",
        "source_shape": list(support_shape), "scoring_bounds_half_open_array_indices": list(scoring_bounds),
        "scoring_crop_edges_are_survey_edges": False,
        "crop_to_survey_edge_clearance_cells": [i0, ni-i1, x0, nx-x1],
        "valid_paired_scoring_positions": int(valid.sum()),
        "unsupported_or_unoccupied_scoring_positions": int(valid.size-valid.sum()),
        "unoccupied_grid_traces_in_crop": crop_missing_count,
        "monitor_only_keys_total": int(len(monitor_keys)),
        "monitor_only_keys_mapped_to_baseline_axes_for_coverage_only": int(monitor_grid.sum()),
        "monitor_only_keys_unmapped_to_baseline_axes": unmapped_monitor_keys,
        "monitor_only_footprint_positions_in_scoring_crop": monitor_in_crop,
        "monitor_only_used_in_paired_calculation": False,
        "near_edge_definition": "within one array-grid cell; descriptive flag only",
        "detectors": by_detector, "processing_chunk_seams": chunk_reports,
        "interpretation_limit": "Coverage/processing QC only; not a geological edge or fluid-change classification.",
    }


def validity_aware_gradient(values, valid_mask, *, global_origin=(0, 0, 0), global_shape=None):
    """Return centered finite-difference gradients without crossing invalid cells.

    Interior gradients require valid samples on both sides and at the center.
    One-sided gradients are permitted only at actual global volume boundaries.
    All derivatives are per array-index step; no physical spacing is implied.
    """
    data = np.asarray(values, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    if data.ndim != 3 or data.shape != valid.shape or any(size < 1 for size in data.shape):
        raise ValueError("values and validity mask must be matching nonempty 3D arrays")
    valid = valid & np.isfinite(data)
    origin = tuple(int(x) for x in global_origin)
    full = tuple(int(x) for x in (global_shape or data.shape))
    if len(origin) != 3 or len(full) != 3 or any(o < 0 or o+n > f for o, n, f in zip(origin, data.shape, full)):
        raise ValueError("global origin/shape do not contain the local block")
    gradients = {}
    for axis, axis_name in enumerate(AXIS_NAMES):
        gradient = np.full(data.shape, np.nan, dtype=np.float32)
        if data.shape[axis] >= 3:
            center = [slice(None)] * 3
            previous = [slice(None)] * 3
            following = [slice(None)] * 3
            center[axis] = slice(1, -1)
            previous[axis] = slice(0, -2)
            following[axis] = slice(2, None)
            center, previous, following = tuple(center), tuple(previous), tuple(following)
            ok = valid[center] & valid[previous] & valid[following]
            vals = (data[following].astype(np.float64) - data[previous].astype(np.float64)) * 0.5
            view = gradient[center]
            view[ok] = vals[ok].astype(np.float32)
        # A local block edge is not a physical edge. Use one-sided differences
        # only when this block coincides with the corresponding survey boundary.
        if origin[axis] == 0 and data.shape[axis] >= 2:
            c = [slice(None)] * 3; n = [slice(None)] * 3
            c[axis], n[axis] = 0, 1
            c, n = tuple(c), tuple(n)
            ok = valid[c] & valid[n]
            view = gradient[c]
            vals = data[n].astype(np.float64) - data[c].astype(np.float64)
            view[ok] = vals[ok].astype(np.float32)
        if origin[axis] + data.shape[axis] == full[axis] and data.shape[axis] >= 2:
            c = [slice(None)] * 3; p = [slice(None)] * 3
            c[axis], p[axis] = -1, -2
            c, p = tuple(c), tuple(p)
            ok = valid[c] & valid[p]
            view = gradient[c]
            vals = data[c].astype(np.float64) - data[p].astype(np.float64)
            view[ok] = vals[ok].astype(np.float32)
        gradients[axis_name] = gradient
    return gradients


def gradient_magnitude(gradients):
    """Combine directional gradients only where all three are supported."""
    names = AXIS_NAMES
    arrays = [np.asarray(gradients[name], dtype=np.float32) for name in names]
    if len({a.shape for a in arrays}) != 1:
        raise ValueError("directional gradients must share a shape")
    valid = np.logical_and.reduce([np.isfinite(a) for a in arrays])
    magnitude = np.full(arrays[0].shape, np.nan, dtype=np.float32)
    total = sum(a.astype(np.float64) ** 2 for a in arrays)
    magnitude[valid] = np.sqrt(total[valid]).astype(np.float32)
    return magnitude


def gradient_magnitude_chunked(values, valid_mask, *, chunk_shape=(32, 32, 64)):
    """Reference/test helper that assembles chunked magnitude with one-cell halos."""
    data = np.asanyarray(values)
    valid = np.asanyarray(valid_mask)
    if data.shape != valid.shape or data.ndim != 3:
        raise ValueError("values and mask must be matching 3D arrays")
    output = np.full(data.shape, np.nan, dtype=np.float32)
    for chunk in iter_chunks(data.shape, chunk_shape, halo=1):
        block = np.asarray(data[chunk.expanded], dtype=np.float32)
        block_valid = np.asarray(valid[chunk.expanded], dtype=bool)
        gradients = validity_aware_gradient(block, block_valid,
            global_origin=tuple(part.start for part in chunk.expanded), global_shape=data.shape)
        magnitude = gradient_magnitude(gradients)
        output[chunk.core] = magnitude[chunk.core_in_expanded]
    return output


def summarize_candidate_edges(*, baseline, monitor, signed_difference, absolute_difference, nrms,
                              support, rms_baseline, rms_monitor,
                              unoccupied_mask, member_mask, bounds,
                              min_support_by_sample, denominator_epsilon=1e-8,
                              max_context_voxels=1_000_000):
    """Summarize survey and change gradients only around one candidate component.

    `bounds` are absolute half-open array slices for the component. The function
    reads at most a one-voxel halo and refuses oversized bounding boxes rather
    than allocating an unbounded temporary. It never labels an edge as a fault
    or fluid front.
    """
    shape = tuple(np.shape(signed_difference))
    if len(shape) != 3 or any(np.shape(a) != shape for a in
            (baseline, monitor, absolute_difference, nrms, support, rms_baseline,
             rms_monitor)) or np.shape(unoccupied_mask) != shape[:2]:
        raise ValueError("edge-QC inputs have inconsistent source geometry")
    if (len(bounds) != 3 or len(min_support_by_sample) != shape[2]
            or not np.isfinite(denominator_epsilon) or denominator_epsilon <= 0):
        raise ValueError("candidate bounds or sample support thresholds are invalid")
    core = tuple(slice(int(s.start), int(s.stop)) for s in bounds)
    if any(s.start < 0 or s.stop > n or s.start >= s.stop for s, n in zip(core, shape)):
        raise ValueError("candidate bounds are outside the source volume")
    expanded = tuple(slice(max(0, s.start-1), min(n, s.stop+1)) for s, n in zip(core, shape))
    context_shape = tuple(s.stop-s.start for s in expanded)
    voxel_count = int(np.prod(context_shape, dtype=np.int64))
    if voxel_count > max_context_voxels:
        return {"status": "not_computed_size_cap", "context_voxels": voxel_count,
                "max_context_voxels": int(max_context_voxels), "gradient_units": "per array-index step"}
    offsets = tuple(c.start-e.start for c, e in zip(core, expanded))
    local_core = tuple(slice(off, off+(c.stop-c.start)) for off, c in zip(offsets, core))
    member = np.asarray(member_mask, dtype=bool)
    expected_member_shape = tuple(s.stop-s.start for s in core)
    if member.shape != expected_member_shape:
        raise ValueError("component membership does not match candidate bounds")
    member_context = np.zeros(context_shape, dtype=bool)
    member_context[local_core] = member
    b = np.asarray(baseline[expanded], dtype=np.float32)
    m = np.asarray(monitor[expanded], dtype=np.float32)
    d = np.asarray(signed_difference[expanded], dtype=np.float32)
    ad = np.asarray(absolute_difference[expanded], dtype=np.float32)
    nrm = np.asarray(nrms[expanded], dtype=np.float32)
    rb = np.asarray(rms_baseline[expanded], dtype=np.float32)
    rm = np.asarray(rms_monitor[expanded], dtype=np.float32)
    sup = np.asarray(support[expanded])
    unocc = np.asarray(unoccupied_mask[expanded[:2]], dtype=bool)
    min_support = np.asarray(min_support_by_sample[expanded[2]], dtype=np.uint8)
    paired_valid = (np.isfinite(b) & np.isfinite(m) & np.isfinite(d)
                    & (sup >= min_support[None, None, :]) & ~unocc[..., None])
    # NRMS can be invalid when its normalization denominator is unstable. Keep
    # that limitation local to the NRMS gradient instead of suppressing valid
    # Baseline/Monitor amplitude gradients.
    fields = {"baseline": (b, paired_valid), "monitor": (m, paired_valid),
              "signed_difference": (d, paired_valid),
              "absolute_difference": (ad, paired_valid & np.isfinite(ad)),
              "nrms": (nrm, paired_valid & np.isfinite(nrm))}
    summary = {}
    change_axis_coverage = []
    for name, (values, field_valid) in fields.items():
        gradients = validity_aware_gradient(values, field_valid,
            global_origin=tuple(s.start for s in expanded), global_shape=shape)
        per_axis = {}
        for axis_name, grad in gradients.items():
            selected = np.abs(grad[member_context & np.isfinite(grad)])
            per_axis[axis_name] = {"p50": float(np.percentile(selected, 50)) if selected.size else None,
                                   "p95": float(np.percentile(selected, 95)) if selected.size else None,
                                   "supported_component_voxels": int(selected.size)}
            if name == "signed_difference":
                change_axis_coverage.append(np.isfinite(grad[member_context]))
        summary[name] = per_axis
    member_voxels = int(member.sum())
    nrms_denominator = rb[local_core] + rm[local_core]
    selected_denominator = nrms_denominator[member & np.isfinite(nrms_denominator)]
    denominator_qc = {
        "minimum_rms_sum": float(selected_denominator.min()) if selected_denominator.size else None,
        "p05_rms_sum": float(np.percentile(selected_denominator, 5)) if selected_denominator.size else None,
        "median_rms_sum": float(np.median(selected_denominator)) if selected_denominator.size else None,
        "epsilon": float(denominator_epsilon),
        "finite_member_voxels": int(selected_denominator.size),
        "at_or_below_epsilon_voxels": int((selected_denominator <= denominator_epsilon).sum()),
        "at_or_below_epsilon_fraction": (float((selected_denominator <= denominator_epsilon).mean())
                                         if selected_denominator.size else None),
        "meaning": "NRMS normalization support diagnostic; not an amplitude-independent confidence measure",
    }
    gradient_coverage = (float(np.logical_and.reduce(change_axis_coverage).sum()) / member_voxels
                         if member_voxels and change_axis_coverage else 0.0)
    coords = np.nonzero(member)
    global_coords = [coords[axis] + core[axis].start for axis in range(3)]
    boundary_member = np.zeros(member_voxels, dtype=bool)
    for axis, n in enumerate(shape):
        pos = global_coords[axis]
        boundary_member |= ((pos <= 0) | (pos >= n-1))
    boundary_fraction = float(boundary_member.mean()) if member_voxels else 1.0
    unoccupied_fraction = float(np.broadcast_to(unocc[local_core[:2]][..., None], member.shape)[member].mean()) if member_voxels else 1.0
    base_p95 = [summary["baseline"][axis]["p95"] for axis in AXIS_NAMES]
    monitor_p95 = [summary["monitor"][axis]["p95"] for axis in AXIS_NAMES]
    change_p95 = [summary["signed_difference"][axis]["p95"] for axis in AXIS_NAMES]
    structural = [max(a or 0.0, b or 0.0) for a, b in zip(base_p95, monitor_p95)]
    ratios = [c / max(s, 1e-12) for c, s in zip(change_p95, structural) if c is not None]
    return {"status": "computed", "gradient_units": "amplitude/NRMS per array-index step",
            "edge_method": "validity-aware centered finite difference; one-sided only at survey boundary",
            "gradient_coverage_fraction": gradient_coverage,
            "survey_boundary_member_fraction": boundary_fraction,
            "unoccupied_grid_member_fraction": unoccupied_fraction,
            "nrms_normalization_denominator": denominator_qc,
            "directional_gradient_percentiles": summary,
            "change_to_survey_edge_gradient_ratio_by_axis": ratios,
            "interpretation": "descriptive discontinuity QC only; not a geological fault or fluid classification"}
