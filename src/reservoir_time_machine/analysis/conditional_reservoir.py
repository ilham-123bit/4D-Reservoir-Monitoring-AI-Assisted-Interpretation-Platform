"""Bounded, conditional I-100 time-lapse evidence, not a reservoir or fluid classifier.

The well-to-grid coordinate frame and seismic time zero are hypotheses.  This
module never changes historical Stage 1/2B/3 products or emits a drilling
probability.  All measurements use existing, paired Baseline-grid arrays.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def conditional_scenario_key(candidate):
    """Return a stable identity for a well-marker/time-origin comparison.

    Historical evidence files used a per-well ``id`` that repeats across time
    origins.  Include the origin, source-grid location, and sample so widget
    state and comparisons cannot collide while preserving the historical ID.
    """
    well = str(candidate.get("well_id", "")).strip().casefold()
    if not well:
        raise ValueError("Conditional scenario requires a well identifier")
    location = candidate.get("array_location")
    if not isinstance(location, (list, tuple)) or len(location) != 3:
        raise ValueError("Conditional scenario requires a 3D array location")
    i, x, sample = (int(value) for value in location)
    origin = float(candidate["assumed_time_origin_ms"])
    if not np.isfinite(origin) or any(value < 0 for value in (i, x, sample)):
        raise ValueError("Conditional scenario location and origin must be valid")
    origin_token = f"m{abs(origin):g}" if origin < 0 else f"p{origin:g}"
    return (f"conditional-i100-{well}-origin-{origin_token}-"
            f"il{i}-xl{x}-s{sample}")


def load_conditional_i100_evidence(path, *, expected_data_identity=None,
                                    expected_shape=None, inline_ids=None,
                                    crossline_ids=None):
    """Load a checksummed, bounded I-100 result without touching seismic data."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("body"), dict):
        raise ValueError("Conditional I-100 evidence must contain a body envelope")
    body = document["body"]
    if body.get("schema") != "rtm-conditional-i100-evidence-1":
        raise ValueError("Unsupported conditional I-100 evidence schema")
    digest = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()
    if digest != document.get("sha256"):
        raise ValueError("Conditional I-100 evidence checksum mismatch")
    if expected_data_identity is not None and body.get("data_identity") != expected_data_identity:
        raise ValueError("Conditional I-100 evidence belongs to a different paired cache")
    shape = tuple(int(value) for value in body.get("source_shape", ()))
    if expected_shape is not None and shape != tuple(expected_shape):
        raise ValueError("Conditional I-100 evidence source shape does not match the loaded grid")
    axes = body.get("source_axes", {})
    if inline_ids is not None and axes.get("inline_ids") != [int(inline_ids[0]), int(inline_ids[-1])]:
        raise ValueError("Conditional I-100 inline axis does not match the loaded grid")
    if crossline_ids is not None and axes.get("crossline_ids") != [int(crossline_ids[0]), int(crossline_ids[-1])]:
        raise ValueError("Conditional I-100 crossline axis does not match the loaded grid")
    records, seen = [], set()
    for item in body.get("measurements", []):
        candidate, measured = item.get("candidate"), item.get("measurement")
        if not isinstance(candidate, dict) or not isinstance(measured, dict):
            raise ValueError("Conditional I-100 scenario is missing its candidate or measurements")
        key = conditional_scenario_key(candidate)
        if key in seen:
            raise ValueError("Duplicate conditional I-100 scenario identity")
        seen.add(key)
        location = tuple(int(value) for value in candidate["array_location"])
        if len(shape) != 3 or any(v < 0 or v >= shape[d] for d, v in enumerate(location)):
            raise ValueError("Conditional I-100 location is outside its recorded source grid")
        if tuple(measured.get("array_location", ())) != location:
            raise ValueError("Conditional I-100 measurement location disagrees with its candidate")
        observed = measured.get("observed", {})
        support = observed.get("support_fraction")
        valid_pairs = observed.get("valid_pairs")
        if (not isinstance(support, (int, float)) or not 0 <= support <= 1 or
                not isinstance(valid_pairs, int) or valid_pairs < 0):
            raise ValueError("Conditional I-100 paired support metadata is invalid")
        records.append({"scenario_key": key, "candidate": candidate, "measurement": measured})
    if body.get("status") == "bounded_conditional_scientific_test_completed" and not records:
        raise ValueError("Completed conditional I-100 evidence has no scenarios")
    return {"body": body, "sha256": digest, "records": records}


def conditional_follow_up_actions(scenario_results):
    """Rank next evidence-gathering actions, never recovery targets.

    The order prioritizes resolving assumptions that change which samples are
    compared, then investigates exceptional local-control behavior, then asks
    for independent spatial/registration validation. No action score is a
    geological probability.
    """
    rows = list(scenario_results)
    if not rows:
        return []
    values_by_well = {}
    sign_by_well = {}
    for candidate, result in rows:
        well = str(candidate.get("well_id", "unknown"))
        nrms = result.get("observed", {}).get("nrms_percent")
        signed = result.get("observed", {}).get("signed_difference_median")
        if isinstance(nrms, (int, float)) and np.isfinite(nrms):
            values_by_well.setdefault(well, []).append(float(nrms))
        if isinstance(signed, (int, float)) and np.isfinite(signed) and signed != 0:
            sign_by_well.setdefault(well, set()).add(int(np.sign(signed)))
    if not values_by_well:
        return []
    all_nrms = [value for group in values_by_well.values() for value in group]
    sign_flips = sorted(well for well, signs in sign_by_well.items() if len(signs) > 1)
    actions = [{
        "rank": 1,
        "id": "i100-resolve-sample-origin",
        "action": "Resolve absolute seismic sample origin and marker-to-seismic timing before assigning this interval a physical TWT.",
        "evidence": (f"Across the tested origin scenarios, measured NRMS spans "
                     f"{min(all_nrms):.2f}%–{max(all_nrms):.2f}%. "
                     + (f"Median signed difference changes sign for {', '.join(sign_flips)}."
                        if sign_flips else "")),
        "recovery_relevance": "No recovery target can be inferred until the sample mapping is established.",
    }]
    exceptional = sorted({
        str(candidate.get("well_id", "unknown"))
        for candidate, result in rows
        if result.get("response_vs_off_window_controls") ==
        "larger_change_than_local_off_window_comparisons"
    })
    if exceptional:
        actions.append({
            "rank": len(actions) + 1,
            "id": "i100-review-local-control-exception",
            "action": "Review the exceptional marker window with independent repeatability, wavelet and registration QC.",
            "evidence": (f"Only these tested well/origin cases exceeded the local off-window comparison: "
                         f"{', '.join(exceptional)}. The controls are not known unchanged reservoir samples."),
            "recovery_relevance": "QC follow-up only; not a fluid-change or drilling claim.",
        })
    actions.append({
        "rank": len(actions) + 1,
        "id": "i100-verify-well-grid-association",
        "action": "Validate the workbook-coordinate to SEG-Y spatial transform and obtain an independently tied horizon before reservoir interpretation.",
        "evidence": "Current well positions are conditional projections; CRS, units, datum and physical 4D registration are unresolved.",
        "recovery_relevance": "Reservoir association, sweep state and recovery chance remain unverified.",
    })
    return actions


def conditional_i100_locations(well_matrix, well_projection, inline_ids, crossline_ids,
                               sample_count, *, sample_interval_ms=2.0, origin_ms=0.0):
    """Construct explicitly conditional I-100 windows from same-well checkshots.

    The projected well position comes from a *candidate* common coordinate frame.
    A well is included only when projected onto the source grid and its source
    marker interpolation is numeric and lies inside the seismic sample range.
    """
    if not np.isfinite([sample_interval_ms, origin_ms]).all() or sample_interval_ms <= 0:
        raise ValueError("Finite positive seismic sample interval and finite origin required")
    il_axis = np.asarray(inline_ids, dtype=float)
    xl_axis = np.asarray(crossline_ids, dtype=float)
    if not il_axis.size or not xl_axis.size or int(sample_count) < 1:
        raise ValueError("Nonempty source axes and sample count are required")
    if np.unique(il_axis).size != il_axis.size or np.unique(xl_axis).size != xl_axis.size:
        raise ValueError("SEG-Y axes must not contain duplicate IDs")
    projected = {str(p.get("well_id", "")).casefold(): p
                 for p in well_projection.get("wells", [])}
    candidates = []
    for well in well_matrix:
        well_id = str(well.get("well_id", ""))
        placement = projected.get(well_id.casefold(), {})
        if placement.get("status") != "conditional_inside_header_axis_extent":
            continue
        try:
            il_id = float(placement["conditional_inline_id"])
            xl_id = float(placement["conditional_crossline_id"])
        except (ValueError, TypeError, KeyError):
            continue
        if not np.isfinite([il_id, xl_id]).all():
            continue
        i = int(np.abs(il_axis - il_id).argmin())
        x = int(np.abs(xl_axis - xl_id).argmin())
        # Do not snap a well located outside the surveyed grid to its boundary.
        if not (il_axis.min() <= il_id <= il_axis.max() and
                xl_axis.min() <= xl_id <= xl_axis.max()):
            continue
        relation = well.get("marker_checkshot_relationship", {})
        if relation.get("status") != "source_marker_to_checkshot_interpolation_only":
            continue
        for marker in relation.get("markers", []):
            name = str(marker.get("marker", "")).strip().upper().replace("_", "-")
            if name not in {"I-100", "I100"}:
                continue
            try:
                twt_ms = float(marker["interpolated_source_checkshot_twt_ms"])
            except (ValueError, TypeError, KeyError):
                continue
            if not np.isfinite(twt_ms):
                continue
            sample_float = (twt_ms - origin_ms) / sample_interval_ms
            s = int(np.floor(sample_float + 0.5))
            if not 0 <= s < int(sample_count):
                continue
            candidates.append({
                "id": f"conditional-i100-{well_id.casefold()}",
                "well_id": well_id,
                "marker": name,
                "source_twt_ms": twt_ms,
                "assumed_time_origin_ms": float(origin_ms),
                "sample_interval_ms": float(sample_interval_ms),
                "conditional_sample_float": float(sample_float),
                "array_location": [i, x, s],
                "segy_inline_id": int(il_axis[i]),
                "segy_crossline_id": int(xl_axis[x]),
                "fractional_well_inline_id": il_id,
                "fractional_well_crossline_id": xl_id,
                "status": "CONDITIONAL_NOT_A_VERIFIED_WELL_SEISMIC_TIE",
            })
            candidates[-1]["scenario_id"] = conditional_scenario_key(candidates[-1])
    return candidates


def _window_metrics(base, monitor, supported):
    valid = supported & np.isfinite(base) & np.isfinite(monitor)
    count = int(np.count_nonzero(valid))
    total = int(valid.size)
    if count < 2:
        return {"valid_pairs": count, "support_fraction": count / total if total else 0.0,
                "nrms_percent": None, "signed_difference_median": None,
                "zero_lag_correlation": None}
    b, m = np.asarray(base[valid], dtype=float), np.asarray(monitor[valid], dtype=float)
    db = m - b
    denominator = float(np.sqrt(np.mean(b*b)) + np.sqrt(np.mean(m*m)))
    nrms = 200.0 * float(np.sqrt(np.mean(db*db))) / denominator if denominator > 0 else None
    bc, mc = b - b.mean(), m - m.mean()
    corr_denominator = float(np.linalg.norm(bc) * np.linalg.norm(mc))
    corr = float(np.dot(bc, mc) / corr_denominator) if corr_denominator > 0 else None
    return {
        "valid_pairs": count, "support_fraction": count / total if total else 0.0,
        "nrms_percent": nrms, "signed_difference_median": float(np.median(db)),
        "zero_lag_correlation": corr,
    }


def evaluate_conditional_window(baseline, monitor, missing_trace_mask, candidate, *,
                                spatial_radius=2, half_window=10, max_lag_samples=4,
                                control_distance_samples=64):
    """Compute bounded paired-seismic response and local off-window comparisons.

    The best-lag diagnostic measures sensitivity to a possible time shift; it
    does not apply registration or modify either survey.  Off-window controls
    are comparison windows, NOT a verified unproduced/null reservoir population.
    """
    shape = tuple(np.shape(baseline))
    if len(shape) != 3 or tuple(np.shape(monitor)) != shape:
        raise ValueError("Baseline and Monitor must share a 3D paired grid")
    if tuple(np.shape(missing_trace_mask)) != shape[:2]:
        raise ValueError("Missing-trace grid does not match the paired data")
    i, x, s = [int(v) for v in candidate["array_location"]]
    if not all(0 <= v < shape[d] for d, v in enumerate((i, x, s))):
        raise ValueError("Candidate location outside source grid")
    if any(int(v) != v or v < 0 for v in
           (spatial_radius, half_window, max_lag_samples, control_distance_samples)):
        raise ValueError("Window settings must be nonnegative integers")
    if half_window < 1:
        raise ValueError("Time half-window must be positive")
    i0, i1 = max(i-spatial_radius, 0), min(i+spatial_radius+1, shape[0])
    x0, x1 = max(x-spatial_radius, 0), min(x+spatial_radius+1, shape[1])
    support2d = ~np.asarray(missing_trace_mask[i0:i1, x0:x1], dtype=bool)

    def at(center, lag=0):
        low, high = center-half_window, center+half_window+1
        if low < 0 or high + lag > shape[2] or low + lag < 0:
            return None
        b = np.asarray(baseline[i0:i1, x0:x1, low:high], dtype=float)
        m = np.asarray(monitor[i0:i1, x0:x1, low+lag:high+lag], dtype=float)
        return _window_metrics(b, m, np.broadcast_to(support2d[..., None], b.shape))

    raw = at(s)
    if raw is None:
        raise ValueError("Candidate time window extends outside the seismic record")
    lags = [(lag, at(s, lag)) for lag in range(-max_lag_samples, max_lag_samples+1)]
    eligible = [(lag, measure["nrms_percent"]) for lag, measure in lags
                if measure is not None and measure["nrms_percent"] is not None]
    best_lag, best_nrms = min(eligible, key=lambda row: (row[1], abs(row[0]))) if eligible else (None, None)
    controls = {str(offset): at(s+offset) for offset in
                (-control_distance_samples, control_distance_samples)}
    control_nrms = [v["nrms_percent"] for v in controls.values()
                    if v is not None and v["nrms_percent"] is not None]
    control_median = float(np.median(control_nrms)) if control_nrms else None
    current = raw["nrms_percent"]
    if raw["support_fraction"] < 0.8:
        response = "insufficient_paired_support_for_comparison"
    elif current is None:
        response = "unsupported_local_response"
    elif control_median is None:
        response = "no_valid_off_window_comparison"
    elif current > 1.5 * control_median:
        response = "larger_change_than_local_off_window_comparisons"
    elif current < (2.0/3.0) * control_median:
        response = "smaller_change_than_local_off_window_comparisons"
    else:
        response = "change_similar_to_local_off_window_comparisons"
    return {
        "id": candidate.get("scenario_id", conditional_scenario_key(candidate)),
        "well_id": candidate["well_id"],
        "array_location": [i, x, s],
        "segy_inline_id": candidate["segy_inline_id"],
        "segy_crossline_id": candidate["segy_crossline_id"],
        "source_twt_ms": candidate["source_twt_ms"],
        "assumed_time_origin_ms": candidate["assumed_time_origin_ms"],
        "time_window_samples": [s-half_window, s+half_window],
        "spatial_window_indices": [i0, i1, x0, x1],
        "observed": raw, "off_window_controls": controls,
        "off_window_control_nrms_median_percent": control_median,
        "best_local_lag_samples": best_lag,
        "best_lag_nrms_percent": best_nrms,
        "response_vs_off_window_controls": response,
        "eligibility": "conditional_seismic_comparison_not_recovery_target",
        "interpretation": (
            "This is a measured Baseline-to-Monitor difference near a proposed "
            "I-100 well-marker position. The well position and seismic time zero "
            "are assumptions. Differences alone do not establish swept oil, "
            "fluid type, pressure, remaining hydrocarbons, or a drilling target."
        ),
        "next_test": (
            "Compare other plausible sample origins and nearby locations; "
            "assess repeatability and an independent reservoir tie before "
            "assigning a fluid or development mechanism."
        ),
    }
