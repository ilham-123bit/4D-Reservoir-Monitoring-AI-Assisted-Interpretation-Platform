"""Compact finding-specific evidence panels; no geological cause is fabricated."""

import streamlit as st

from ..analysis.guided_report import page_report
from ..ai.grounding import finding_packet, explanation_identity
from ..settings import LLMSettings
from .coordinates import finding_header_location


def _result_text(finding, location):
    metrics = finding["metrics"]
    inline = location["inline_header_id"]
    crossline = location["crossline_header_id"]
    inline_text = f"Inline header {inline:,}" if inline is not None else f"Inline array index {location['inline_index']}"
    crossline_text = f"Crossline header {crossline:,}" if crossline is not None else f"Crossline array index {location['crossline_index']}"
    inline_bounds, crossline_bounds = finding["bounds"]
    extent = (f"array-index extent inline {inline_bounds[0]}–{inline_bounds[1]-1}, "
              f"crossline {crossline_bounds[0]}–{crossline_bounds[1]-1}")
    return (
        f"{inline_text} · {crossline_text} · sample index {location['sample_index']:,}; "
        f"{metrics['cells']:,} connected cells ({extent}); signed difference "
        f"{metrics['mean_signed_difference']:.4g}; absolute difference "
        f"{metrics['mean_absolute_difference']:.4g}; NRMS {metrics['mean_nrms_percent']:.2f}%; "
        f"representative Baseline {metrics['baseline_amplitude']:.4g} and Monitor {metrics['monitor_amplitude']:.4g}; "
        f"minimum paired support {metrics['minimum_paired_support']}."
    )


def _discussion_text(finding, discovery, wells, location, grid_shape, run_bounds):
    coverage = (
        f"Analysis coverage {discovery['completed_tiles']}/{discovery['planned_tiles']} tiles; ranks are local. "
        if "completed_tiles" in discovery else "Analysis coverage is not recorded. "
    )
    alternatives = "Acquisition, processing, positioning, scaling, noise, or residual misregistration may contribute."
    method_evidence = finding.get("method_evidence", {})
    method_support = []
    for name in ("statistical", "mahalanobis", "pca", "isolation_forest", "gmm"):
        evidence = method_evidence.get(name, {})
        support = int(evidence.get("inspection_support", 0))
        flagged = int(evidence.get("flagged_inspection_cells", 0))
        if support:
            label = "Isolation Forest" if name == "isolation_forest" else name.replace("_", " ").upper()
            method_support.append(f"{label} flagged {flagged}/{support} inspected component cells")
    detector_text = ("Detector inspection: " + "; ".join(method_support) + ". "
                     if method_support else
                     "Detector inspection had no eligible members within this component; corroboration is unavailable. ")
    limits = "Trace-key pairing only; physical 4D registration and sample time/depth are unverified."
    if finding["touches_tile_boundary"]:
        limits += " The component reaches its analysis-tile boundary; extent is truncated."
    elif finding["metrics"].get("members_adjacent_to_invalid_cells", 0):
        limits += " Some component cells adjoin invalid or missing support."
    tile_i0,tile_i1,tile_x0,tile_x1=run_bounds
    if tile_i0 == 0 or tile_i1 == grid_shape[0] or tile_x0 == 0 or tile_x1 == grid_shape[1]:
        inline_margin=min(location['inline_index'],grid_shape[0]-1-location['inline_index'])
        crossline_margin=min(location['crossline_index'],grid_shape[1]-1-location['crossline_index'])
        limits += (f" The analyzed tile reaches a survey-grid edge; this candidate is "
                   f"{inline_margin} inline and {crossline_margin} crossline indices from an edge.")
    well_limit = wells.get("missing_association_evidence", "")
    well_text = "Well-to-seismic association is unavailable until shared CRS, datum, and time-depth ties are verified."
    if well_limit:
        well_text += " Current well evidence lacks required association metadata."
    return f"{coverage}{detector_text}{limits} {alternatives} {well_text} This is a screening anomaly, not evidence of fluid movement or recovery."


def _action_text(finding, run_bounds, grid_shape):
    if finding["touches_tile_boundary"]:
        return "Extend the neighboring tile; verify repeatability and registration."
    if run_bounds[0] == 0 or run_bounds[1] == grid_shape[0] or run_bounds[2] == 0 or run_bounds[3] == grid_shape[1]:
        return "Test edge sensitivity, repeatability, and event alignment."
    return "Review repeatability, event alignment, and processing; establish a verified well tie."


def render_page_report(result, finding, wells, page, discovery, *, shared_interpretation=None,
                       inline_ids=(), crossline_ids=(), grid_shape=None):
    report = page_report(result, finding, wells, page, discovery,
                         shared_interpretation=shared_interpretation)
    st.markdown("### Finding interpretation")
    if not finding:
        st.markdown("**RESULT ·** No eligible candidate in completed analysis.")
        st.markdown("**DISCUSSION ·** Survey-wide conclusions are unavailable.")
        st.markdown("**ACTION ·** Extend validated analysis coverage.")
        return report

    location = finding_header_location(finding["representative"], inline_ids, crossline_ids,
                                       grid_shape=grid_shape)
    report["header_location"] = location
    report["result_text"] = _result_text(finding, location)
    report["discussion_text"] = _discussion_text(finding, discovery, wells, location,
                                                  tuple(grid_shape), tuple(result['bounds']))
    report["action_text"] = _action_text(finding, tuple(result['bounds']), tuple(grid_shape))
    packet = finding_packet(result, finding, well_context=wells)
    st.markdown("**RESULT ·** " + report["result_text"])
    st.markdown("**DISCUSSION ·** " + report["discussion_text"])
    st.markdown("**ACTION ·** " + report["action_text"])
    with st.expander("Next checks by discipline"):
        for action in report["discipline_actions"]:
            st.markdown(f"**{action['discipline']} ·** {action['action']}")
            st.caption(f"Evidence: {', '.join(action['evidence_ids'])} · {action['uncertainty']}")
    st.caption(f"Finding {finding['id']} · run {finding['run_id']} · preliminary tile-level evidence")

    try:
        key = explanation_identity(packet, LLMSettings.from_environment())
        cached = st.session_state.get("grounded_explanations", {}).get(key)
        if cached and cached["provenance"]["source"] == "validated_provider_selection":
            report["interpretation"] = cached
            report["source"] = "Gemma · cached validated evidence selection; not geological validation"
    except ValueError:
        pass
    st.caption("Interpretation source: " + report["source"])
    return report
