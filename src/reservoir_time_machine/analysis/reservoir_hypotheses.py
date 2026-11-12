"""Numerical discovery of spatially coherent 4D seismic response regions.

These are measured differences at matching source trace keys, not fluid labels,
sweep fractions or drill-ready hydrocarbon targets. Maps are evaluated lazily
on bounded patches around actual source-linked finding locations.
"""
from __future__ import annotations

from math import log1p
import numpy as np
from scipy.ndimage import label

STRUCTURE = np.ones((3, 3), dtype=np.uint8)


def _box_record(region, region_type, values, support, i0, x0, sample, inline_ids, crossline_ids, median, threshold):
    coords = np.argwhere(region)
    if not len(coords):
        raise ValueError("Empty region")
    center = np.mean(coords, axis=0)
    ci, cx = min((tuple(c) for c in coords), key=lambda c: (float(c[0]-center[0])**2 + float(c[1]-center[1])**2, c[0], c[1]))
    ai, ax = int(ci+i0), int(cx+x0)
    i_min, x_min = coords.min(axis=0).tolist()
    i_max, x_max = coords.max(axis=0).tolist()
    observed = float(np.median(values[region]))
    return {
        "type": region_type,
        "center_array": [ai, ax, int(sample)],
        "center_segy": {"inline_id": int(inline_ids[ai]), "crossline_id": int(crossline_ids[ax]), "sample_index": int(sample)},
        "bbox_array": [[int(i_min+i0), int(i_max+i0)], [int(x_min+x0), int(x_max+x0)], [int(sample), int(sample)]],
        "trace_count": int(len(coords)),
        "median_nrms_percent": round(observed, 3),
        "contrast_to_local_median_pp": round(observed-median, 3),
        "valid_support_median": round(float(np.median(support[region])), 4),
        "threshold_nrms_percent": round(float(threshold), 3),
        "evidence_level": "paired_seismic_response_only",
        "reservoir_sweep_class": "not_established",
        "hydrocarbon_status": "not_established",
    }


def screen_response_regions(
    baseline, monitor, missing_trace_mask, source_location, inline_ids, crossline_ids,
    *, radius=22, half_window=10, min_region_traces=9, min_support=0.8,
):
    """Identify connected stronger/quieter *seismic* regions in one bounded tile.

    Operates only on finite, exactly paired samples; local amplitude detectability
    and a robust MAD contrast are required. Results are not a reservoir mask.
    """
    shape = tuple(np.shape(baseline))
    if len(shape) != 3 or np.shape(monitor) != shape or np.shape(missing_trace_mask) != shape[:2]:
        raise ValueError("Incompatible paired seismic and mask")
    if len(inline_ids) != shape[0] or len(crossline_ids) != shape[1]:
        raise ValueError("Source SEG-Y axes do not match seismic grid")
    if len(source_location) != 3:
        raise ValueError("Expected inline/crossline/sample array indices")
    indices = tuple(int(v) for v in source_location)
    if any(float(source_location[d]) != indices[d] or not 0 <= indices[d] < shape[d] for d in range(3)):
        raise ValueError("Source location outside volume")
    if not (8 <= radius <= 80 and 2 <= half_window <= 64 and
            3 <= min_region_traces <= (2*radius+1)**2 and 0.5 <= min_support <= 1):
        raise ValueError("Invalid screening parameters")
    i, x, s = indices
    i0, i1 = max(0, i-radius), min(shape[0], i+radius+1)
    x0, x1 = max(0, x-radius), min(shape[1], x+radius+1)
    s0, s1 = max(0, s-half_window), min(shape[2], s+half_window+1)
    b = np.asarray(baseline[i0:i1, x0:x1, s0:s1], dtype=np.float64)
    m = np.asarray(monitor[i0:i1, x0:x1, s0:s1], dtype=np.float64)
    valid = (np.isfinite(b) & np.isfinite(m) &
             ~np.asarray(missing_trace_mask[i0:i1, x0:x1], dtype=bool)[..., None])
    sample_count = valid.sum(axis=2)
    support = sample_count / b.shape[-1]
    denominator = np.maximum(sample_count, 1)
    bv = np.where(valid, b, 0.0)
    mv = np.where(valid, m, 0.0)
    rb = np.sqrt((bv*bv).sum(axis=2) / denominator)
    rm = np.sqrt((mv*mv).sum(axis=2) / denominator)
    amplitude = rb+rm
    eligible = (support >= min_support) & (sample_count >= min(9, b.shape[2]))
    reference = amplitude[eligible]
    detectable = float(np.median(reference)) * 0.2 if reference.size else float("inf")
    eligible &= amplitude > max(detectable, 1e-10)
    response = np.full(amplitude.shape, np.nan)
    difference = np.sqrt(np.square(mv-bv).sum(axis=2) / denominator)
    np.divide(200*difference, amplitude, out=response, where=eligible)
    response[~eligible] = np.nan
    sample_number = int(np.count_nonzero(eligible))
    result = {
        "source_location": list(indices),
        "bounds": [i0, i1, x0, x1, s0, s1],
        "evaluated_traces": sample_number,
        "eligible_fraction": round(sample_number/max(response.size, 1), 5),
        "response_map": response,
        "response_region_groups": {"changed": [], "relatively_quiet": []},
        "relative_response_status": "not_resolved",
        "interpretation_type": "seismic_change_screen",
        "physical_registration": "unverified",
        "reservoir_interval": "not established from these measurements",
    }
    if sample_number < 2*min_region_traces+20:
        result["status"] = "insufficient_supported_traces"
        return result
    vals = response[eligible]
    center = float(np.median(vals))
    dispersion = float(np.median(np.abs(vals-center)))
    # Both absolute separation and neighborhood dispersion must be exceeded.
    delta = max(10.0, 1.5 * 1.4826 * dispersion)
    result["local_median_nrms_percent"] = round(center, 3)
    result["local_mad_nrms_percent"] = round(dispersion, 3)
    result["required_contrast_pp"] = round(delta, 3)
    for kind, mask, threshold in (
        ("changed", eligible & (response >= center+delta), center+delta),
        ("relatively_quiet", eligible & (response <= center-delta), center-delta),
    ):
        labels, number = label(mask, structure=STRUCTURE)
        for component_id in range(1, number+1):
            region = labels == component_id
            if int(region.sum()) < min_region_traces:
                continue
            record = _box_record(region, kind, response, support, i0, x0, s, inline_ids, crossline_ids, center, threshold)
            # An anomalous edge strip is weaker than an interior coherent patch.
            yy, xx = np.where(region)
            edge_fraction = float(np.mean((yy == 0) | (xx == 0) |
                                          (yy == response.shape[0]-1) |
                                          (xx == response.shape[1]-1)))
            record["tile_boundary_fraction"] = round(edge_fraction, 4)
            record["response_priority_index"] = round(
                abs(record["contrast_to_local_median_pp"]) * log1p(record["trace_count"]) *
                (1-0.75*edge_fraction) * record["valid_support_median"], 4)
            result["response_region_groups"][kind].append(record)
        result["response_region_groups"][kind].sort(
            key=lambda r: (-r["response_priority_index"], r["center_array"]))
        result["response_region_groups"][kind] = result["response_region_groups"][kind][:3]
    high = result["response_region_groups"]["changed"]
    low = result["response_region_groups"]["relatively_quiet"]
    if high and low:
        result["relative_response_status"] = "coherent_changed_and_quiet_regions"
    elif high:
        result["relative_response_status"] = "coherent_changed_region_only"
    elif low:
        result["relative_response_status"] = "coherent_relatively_quiet_region_only"
    result["status"] = "measured"
    return result


def _detector_context(finding):
    """Use source component-level detector counts, not invented probabilities."""
    raw = finding.get("detectors", finding.get("detector_flagged_voxels", {}))
    flagged = {}
    if isinstance(raw, dict):
        for method, record in raw.items():
            value = record.get("flagged_voxels", 0) if isinstance(record, dict) else record
            if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
                flagged[str(method)] = max(0, int(value))
    positive = [name for name, n in flagged.items() if n > 0]
    return {
        "component_detector_flag_counts": flagged,
        "flagging_models": positive,
        "n_models_with_flags": len(positive),
        "mean_detector_vote_fraction": (
            float(finding["mean_detector_vote_fraction"])
            if finding.get("mean_detector_vote_fraction") is not None else None
        ),
        "interpretation": (
            "Model flags identify statistical similarity, not validated fluid or reservoir classes."
            if positive else "No independent multivariate support is established at this component."
        ),
    }


def candidate_briefing(finding, result, *, source="full_field"):
    """Turn a candidate's measured local seismic map into a short decision record.

    A spatial response contrast is not evidence of movable hydrocarbons or an
    effective-sweep fraction. Unsupported drilling locations are NEVER emitted.
    """
    fid = str(finding["id"])
    position = list(finding.get("array_location", finding.get("representative", [])))
    if len(position) != 3:
        raise ValueError("Finding lacks source location")
    high = result["response_region_groups"]["changed"]
    low = result["response_region_groups"]["relatively_quiet"]
    edge = finding.get("edge_artifact_risk", finding.get("edge_risk", 0.5))
    if edge is None or not np.isfinite(edge):
        edge = 0.5
    edge = float(np.clip(edge, 0, 1))
    combined = (high[:1]+low[:1])
    local_evidence = sum((x["response_priority_index"] for x in combined), 0.0)
    score = round(local_evidence*(1-edge), 3)
    if high and low:
        finding_text = "Coherent stronger and quieter seismic-response patches occur near the selected source location."
        mechanism = "Production-linked change is one possibility; wavelet, timing and processing effects remain alternatives."
        focus = low[0]
        decision = "Investigate the quieter patch against reservoir quality, barriers and production/injection history before evaluating an infill or sidetrack."
    elif high:
        finding_text = "A coherent stronger 1995–2006 seismic-response patch was measured."
        mechanism = "Fluid/pressure response or acquisition/processing differences could explain the change."
        focus = high[0]
        decision = "Compare saturation/pressure models with seismic repeatability in this mapped patch."
    elif low:
        finding_text = "A relatively low-response patch exists, but there is no strong neighboring changed patch in the tested tile."
        mechanism = "A low response may indicate limited change, low sensitivity, or a processing difference."
        focus = low[0]
        decision = "Test reservoir presence and detectability before calling this patch poorly swept."
    else:
        finding_text = "No robust connected changed-versus-quieter response body was resolved in this tile."
        mechanism = "The measured field contrast does not define a distinct local compartment."
        focus = None
        decision = "Search other source-linked regions; do not infer a bypassed interval at this point."
    center = (focus or {}).get("center_segy") or finding.get("segy_location")
    return {
        "id": fid,
        "source": source,
        "source_array_location": position,
        "source_segy": finding.get("segy_location"),
        "location": center,
        "finding_text": finding_text,
        "why": mechanism,
        "next_decision": decision,
        "stronger_region": high[0] if high else None,
        "quieter_region": low[0] if low else None,
        "spatial_pattern": result["relative_response_status"],
        "local_trace_support": result["evaluated_traces"],
        "local_median_nrms_percent": result.get("local_median_nrms_percent"),
        "required_contrast_pp": result.get("required_contrast_pp"),
        "edge_risk": edge,
        "investigation_priority_index": score,
        "recovery_candidate_status": "reservoir_eligibility_unverified",
        "fluid_pathway": "not constrained by two seismic vintages alone",
        "sweep_efficiency": "not estimable without saturation/dynamic reservoir data",
        "bypassed_hydrocarbon": "hypothesis_only_no_positive_reservoir_fluid_tie",
        "drilling_location": None,
        "ml_evidence": _detector_context(finding),
        "evidence_ids": [fid+":seismic", fid+":spatial_region", fid+":detectors"],
    }
