"""Paired-data, bounded spatial contrast for changed and relatively quiet regions.

This is a hypothesis screen, not a reservoir mask, flow model, hydrocarbon
classifier, or independently verified geological compartment interpretation.
Every output is calculated from the existing exact trace-key-paired seismic.
"""
from __future__ import annotations

import numpy as np


def _finite(value):
    return float(value) if value is not None and np.isfinite(value) else None


def _record(index, value, support, center):
    i, x = (int(v) for v in index)
    return {
        "array_location": [i, x, int(center[2])],
        "nrms_percent": _finite(value),
        "paired_support_fraction": _finite(support),
    }


def screen_local_compartments(
    baseline, monitor, missing_trace_mask, location, *,
    spatial_radius=12, half_window=10, core_radius=2,
    min_supported_samples=12, min_support_fraction=0.8,
):
    """Contrast local traces using the same sample window and grid.

    NRMS is calculated independently for every supported trace, avoiding
    a falsely low response caused by mixing changed and unchanged traces.
    The quiet location must have sufficient absolute baseline/monitor
    amplitude energy to make a zero-valued response meaningful.
    """
    shape = tuple(np.shape(baseline))
    if len(shape) != 3 or np.shape(monitor) != shape:
        raise ValueError("Expected matching (inline, crossline, sample) volumes")
    if tuple(np.shape(missing_trace_mask)) != shape[:2]:
        raise ValueError("Expected a 2D paired missing-trace mask")
    if len(location) != 3:
        raise ValueError("Location must include inline/crossline/sample indices")
    pos = [int(v) for v in location]
    if any(float(v) != n or n < 0 or n >= shape[d] for d, (v, n) in enumerate(zip(location, pos))):
        # This condition deliberately handles numeric location coercion below.
        pass
    if any(float(v) != pos[d] or pos[d] < 0 or pos[d] >= shape[d]
           for d, v in enumerate(location)):
        raise ValueError("Location lies outside the source grid")
    if not (2 <= spatial_radius <= 64 and 1 <= core_radius < spatial_radius
            and 1 <= half_window <= 256 and 1 <= min_supported_samples <= (2*half_window+1)
            and 0.0 < min_support_fraction <= 1.0):
        raise ValueError("Invalid bounded screening window or support requirement")
    ci, cx, cs = pos
    i0, i1 = max(0, ci-spatial_radius), min(shape[0], ci+spatial_radius+1)
    x0, x1 = max(0, cx-spatial_radius), min(shape[1], cx+spatial_radius+1)
    s0, s1 = max(0, cs-half_window), min(shape[2], cs+half_window+1)
    b = np.asarray(baseline[i0:i1, x0:x1, s0:s1], dtype=np.float64)
    m = np.asarray(monitor[i0:i1, x0:x1, s0:s1], dtype=np.float64)
    support2 = ~np.asarray(missing_trace_mask[i0:i1, x0:x1], dtype=bool)
    valid = np.isfinite(b) & np.isfinite(m) & support2[..., None]
    count = np.count_nonzero(valid, axis=-1)
    total = b.shape[-1]
    fraction = count / total
    eligible = (count >= min_supported_samples) & (fraction >= min_support_fraction)
    safe_count = np.maximum(count, 1)
    b0 = np.where(valid, b, 0.0)
    m0 = np.where(valid, m, 0.0)
    rb = np.sqrt(np.sum(b0*b0, axis=-1) / safe_count)
    rm = np.sqrt(np.sum(m0*m0, axis=-1) / safe_count)
    difference = np.sqrt(np.sum((m0-b0)**2, axis=-1) / safe_count)
    norm = rb + rm
    # Constant zero-amplitude pairs provide no relative difference information.
    scale = max(float(np.nanmedian(norm[eligible])) if np.any(eligible) else 0.0, 1e-12)
    eligible &= norm > scale * 1e-5
    nrms = np.full(norm.shape, np.nan, dtype=np.float64)
    np.divide(200.0*difference, norm, out=nrms, where=eligible)
    nrms[~eligible] = np.nan

    ii, xx = np.indices(nrms.shape)
    radial = np.maximum(np.abs(ii + i0-ci), np.abs(xx + x0-cx))
    inner = (radial <= core_radius) & eligible
    outer = (radial > core_radius) & eligible
    center_vals, outer_vals = nrms[inner], nrms[outer]
    ci_loc = (ci-i0, cx-x0)
    if len(center_vals) < 3 or len(outer_vals) < 12:
        return {
            "status": "insufficient_spatial_comparisons",
            "source_location": pos,
            "bounds": [i0, i1, x0, x1, s0, s1],
            "nrms_percent_map": nrms,
            "support_fraction_map": fraction,
            "evaluated_trace_count": int(np.count_nonzero(eligible)),
            "interpretation": "The selected location lacks enough supported neighboring traces for a spatial comparison.",
            "change_candidate": None,
            "quiet_candidate": None,
        }

    core_median = float(np.median(center_vals))
    ring_median = float(np.median(outer_vals))
    ring_mad = float(np.median(np.abs(outer_vals - ring_median)))
    # Descriptive contrast only; no confidence level or detector calibration.
    contrast = core_median - ring_median
    contrast_ratio = core_median / max(ring_median, 1e-9)
    positions = np.argwhere(outer)
    values = nrms[outer]
    high_position = positions[int(np.argmax(values))]
    low_position = positions[int(np.argmin(values))]
    hi_index = (int(high_position[0]+i0), int(high_position[1]+x0))
    lo_index = (int(low_position[0]+i0), int(low_position[1]+x0))
    high = _record(hi_index, np.max(values), fraction[tuple(high_position)], pos)
    low = _record(lo_index, np.min(values), fraction[tuple(low_position)], pos)
    center_nrms = float(nrms[ci_loc]) if eligible[ci_loc] else None

    if contrast > max(10.0, 2.0*ring_mad):
        pattern = "selected_region_more_responsive_than_neighborhood"
        summary = (
            "The selected seismic region has a stronger 1995–2006 paired "
            "response than nearby supported traces at the same sample indices."
        )
    elif contrast < -max(10.0, 2.0*ring_mad):
        pattern = "selected_region_less_responsive_than_neighborhood"
        summary = (
            "The selected seismic region is relatively quiet compared with "
            "nearby supported traces at the same sample indices."
        )
    else:
        pattern = "selected_region_similar_to_neighborhood"
        summary = (
            "The selected seismic region has a response similar to the "
            "nearby supported traces; there is no clear local contrast."
        )

    return {
        "status": "measured_local_spatial_contrast",
        "source_location": pos,
        "bounds": [i0, i1, x0, x1, s0, s1],
        "nrms_percent_map": nrms,
        "support_fraction_map": fraction,
        "center_nrms_percent": _finite(center_nrms),
        "core_median_nrms_percent": core_median,
        "surrounding_median_nrms_percent": ring_median,
        "spatial_nrms_contrast_percentage_points": contrast,
        "core_to_surrounding_nrms_ratio": contrast_ratio,
        "surrounding_mad_nrms_percent": ring_mad,
        "evaluated_trace_count": int(np.count_nonzero(eligible)),
        "core_trace_count": int(len(center_vals)),
        "surrounding_trace_count": int(len(outer_vals)),
        "pattern": pattern,
        "interpretation": summary,
        "change_candidate": high,
        "quiet_candidate": low,
        "quality": {
            "comparison": "same-source-sample-index spatial neighborhood",
            "source_vintages": ["1995 baseline", "2006 monitor"],
            "zero_energy_traces_excluded": True,
            "missing_and_nonfinite_samples_excluded": True,
            "time_origin": "unverified",
            "physical_registration": "unverified",
            "reservoir_interval": "not established by this screen",
        },
    }


def development_hypothesis(contrast: dict) -> dict:
    """Tie measured local pattern to explicit competing management hypotheses.

    Development *options* are returned, never sanctioned drilling plans.
    """
    status = contrast["status"]
    if status != "measured_local_spatial_contrast":
        return {
            "headline": "Spatial comparison not yet supported.",
            "production_response": "Not assessed",
            "poor_sweep": "Not assessed",
            "mechanisms": [],
            "development_options": [],
            "discipline_actions": {},
            "evidence_level": "unsupported",
        }
    pattern = contrast["pattern"]
    if pattern == "selected_region_more_responsive_than_neighborhood":
        headline = "Stronger local seismic response — investigate a production-affected compartment."
        focus = "Compare pressure/saturation-change scenarios with acquisition and wavelet alternatives."
        poor_sweep = "Nearby quieter traces are candidate comparison areas, not proved bypassed oil."
        options = [
            "Evaluate a sweep-redistribution or injector-pattern scenario if supported by verified injector history and mapped reservoir connectivity.",
            "Compare recompletion and infill alternatives in a reservoir model only after location, facies and pressure constraints are established.",
        ]
    elif pattern == "selected_region_less_responsive_than_neighborhood":
        headline = "Relatively quiet seismic region — investigate compartment connectivity and possible limited response."
        focus = "Test whether the lower response is real and detectable at a reservoir-quality interval."
        poor_sweep = "A possible low-response compartment, not a proven unswept hydrocarbon volume."
        options = [
            "Evaluate conditional infill-versus-recompletion cases if reservoir quality and a mapped barrier support compartment isolation.",
            "Compare targeted pressure surveillance with potential injector-pattern adjustment before choosing an intervention.",
        ]
    else:
        headline = "Comparable local seismic response — no distinct compartment response inferred."
        focus = "Assess whether response is spatially coherent and detectable within a tied reservoir interval."
        poor_sweep = "No evidence of a distinct low-response compartment at these sampled indices."
        options = ["Do not select a development location solely from this local seismic screen."]
    return {
        "headline": headline,
        "production_response": focus,
        "poor_sweep": poor_sweep,
        "mechanisms": [
            {"hypothesis": "Pressure or fluid saturation change",
             "status": "conditional",
             "positive": "Measured paired 1995–2006 seismic response and spatial contrast.",
             "contradiction": "No independently established reservoir interval, time origin or pressure/saturation-history correlation."},
            {"hypothesis": "Acquisition, registration or processing effect",
             "status": "credible_alternative",
             "positive": "Physical 4D registration is not independently verified.",
             "contradiction": "A spatially coherent and repeatable reservoir-tied response could reduce this alternative."},
            {"hypothesis": "Limited sweep / bypassed hydrocarbons",
             "status": "not_established",
             "positive": "Relatively quiet neighboring traces can be located for testing.",
             "contradiction": "Quiet seismic alone cannot identify movable oil, reservoir quality, or original saturation."},
        ],
        "development_options": options,
        "discipline_actions": {
            "Geophysics": "Test a reservoir-constrained response against local wavelet and repeatability alternatives, and check that the adjacent contrast survives sample-origin sensitivity.",
            "Geology": "Test whether a reservoir sand and a mapped barrier or channel connect the high- and low-response positions before assigning compartments.",
            "Petrophysics": "At eligible wells compare reservoir quality, fluid-indicator curve provenance and impedance sensitivity for pressure versus substitution scenarios.",
            "Reservoir engineering": "Compare modelled drainage and injection-pattern scenarios that could produce the observed spatial contrast; require well-history and pressure consistency.",
            "Production engineering": "Assess rate, water-cut, GOR and injection-history patterns for reservoir compartments linked to this region, if spatial association is validated.",
            "Drilling and well planning": "Evaluate conditional infill, sidetrack or recompletion concepts only after sand continuity, location, hazards and economic screening are available.",
        },
        "evidence_level": "paired_seismic_spatial_screening",
    }
