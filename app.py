"""Challenge 3 seismic surveillance dashboard; never substitutes generated data."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from reservoir_time_machine.attributes.difference import calculate_sample_attributes
from reservoir_time_machine.data_audit.audit import WELL_FILES
from reservoir_time_machine.data_ingest.manifest import challenge3_availability
from reservoir_time_machine.data_ingest.segy_loader import (
    load_challenge3_paired_volumes,
)
from reservoir_time_machine.detection.clustering import connected_zones
from reservoir_time_machine.detection.statistical import statistical_screen
from reservoir_time_machine.preprocessing.qc import quality_report
from reservoir_time_machine.settings import ApplicationSettings, LLMSettings
from reservoir_time_machine.visualization.animation import animation_figure
from reservoir_time_machine.visualization.dashboard_state import empty_dataset_state
from reservoir_time_machine.visualization.seismic_3d import volume_figure
from reservoir_time_machine.visualization.slices import (
    display_mask,
    display_slice,
    display_volume,
)
from reservoir_time_machine.visualization.theme import apply_theme, page_header, registration_warning, section


st.set_page_config(page_title="Reservoir Time Machine", page_icon="🌐", layout="wide")
apply_theme()
page_header("Reservoir Time Machine", "4D seismic monitoring · APGCE GeoHackathon 2026 · Challenge 3")

try:
    settings = ApplicationSettings.from_environment()
except (ValueError, OSError) as exc:
    st.error(f"Invalid application configuration: {exc}")
    st.stop()

availability = challenge3_availability(settings.challenge3_root)
if not availability["available"]:
    state = empty_dataset_state(availability["root"], availability["missing"])
    st.subheader("No input dataset available")
    st.info(state["message"])
    st.write(f"Configured data root: `{state['configured_root']}`")
    st.write("Expected inputs not found:")
    st.code("\n".join(state["missing_files"]) or "No required files were declared")
    st.markdown("Use the [dataset input contract](docs/data_contract.md) and the example configuration to prepare a validated input manifest.")
    st.caption("This is an empty application state. It contains no placeholder seismic, well data, anomaly map, or geological interpretation.")
    st.info("No analysis outputs are available. Provide authorized files, then run the read-only ingestion and geometry checks before processing.")
    st.stop()


@st.cache_resource(show_spinner="Loading paired Challenge 3 SEG-Y volumes…")
def cached_actual_pair(root: str, cache_dir: str | None):
    return load_challenge3_paired_volumes(
        root,
        output_dir=cache_dir,
    )


@st.cache_resource(max_entries=2)
def cached_quality_report(dataset_identity: str, survey_id: str, _volume):
    """Reuse a full-volume QC report for an unchanged cached survey."""
    return quality_report(_volume)


@st.cache_resource(max_entries=2)
def cached_sample_attributes(
    dataset_identity: str,
    sample_index: int,
    window_samples: int,
    _baseline,
    _monitor,
):
    """Reuse selected-sample attributes without hashing seismic arrays."""
    return calculate_sample_attributes(
        _baseline,
        _monitor,
        sample_index,
        window_samples=window_samples,
    )


@st.cache_resource(max_entries=4, show_spinner="Screening selected sample…")
def cached_sample_detection(dataset_identity: str, sample_index: int, window_samples: int,
                            threshold_sigma: float, min_size: int, _difference):
    """Cache statistical and connected-region output by data and scientific controls."""
    detection = statistical_screen(_difference, threshold_sigma=threshold_sigma)
    labels, zones = connected_zones(detection.mask, detection.score, min_size=min_size)
    return detection, labels, zones


@st.cache_resource(max_entries=2)
def cached_animation(dataset_identity: str, sample_index: int, overlay_threshold: float | None,
                     frame_duration_ms: int, _baseline_slice, _monitor_slice, _overlay,
                     _missing_trace_mask):
    return animation_figure(_baseline_slice, _monitor_slice, _overlay, max_axis=360,
                            missing_trace_mask=_missing_trace_mask, frame_duration_ms=frame_duration_ms)


def _step_sample(delta: int, maximum: int) -> None:
    st.session_state["shared_sample"] = min(maximum, max(0, st.session_state["shared_sample"] + delta))


def _reset_sample(maximum: int) -> None:
    st.session_state["shared_sample"] = maximum // 2


def _select_detection_region(region_key: str, zones) -> None:
    selected_id = st.session_state.get(region_key)
    selected = next((zone for zone in zones if zone.label == selected_id), None)
    if selected is not None:
        st.session_state["shared_inline"] = (selected.bounding_box[0][0] + selected.bounding_box[0][1]) // 2
        st.session_state["shared_crossline"] = (selected.bounding_box[1][0] + selected.bounding_box[1][1]) // 2


def _missing_overlay_trace(mask: np.ndarray, max_axis: int = 700,
                           x_name: str = "Crossline index", y_name: str = "Inline index"):
    """Show any missing trace within each display bin, with Baseline-grid indices."""
    mask = np.asarray(mask, dtype=bool)
    row_stride = max(1, int(np.ceil(mask.shape[0] / max_axis)))
    col_stride = max(1, int(np.ceil(mask.shape[1] / max_axis)))
    return go.Heatmap(
        z=np.where(display_mask(mask, max_axis=max_axis), 1.0, np.nan),
        x=np.arange(0, mask.shape[1], col_stride),
        y=np.arange(0, mask.shape[0], row_stride),
        colorscale=[[0, "rgba(0,0,0,0)"], [1, "#ded7d5"]],
        zmin=0, zmax=1, opacity=.9, showscale=False, name="Missing trace coverage",
        hovertemplate=f"Missing trace in display bin<br>{x_name} %{{x}}<br>{y_name} %{{y}}<extra></extra>",
    )


try:
   with st.spinner("Loading Baseline and Monitor volumes using trace-key pairing…"):
    baseline_volume, monitor_volume = cached_actual_pair(
        str(settings.challenge3_root),
        str(settings.cache_directory) if settings.cache_directory else None,
    )
except (ImportError, RuntimeError, FileNotFoundError, MemoryError, ValueError, OSError) as exc:
    st.error(f"Actual Challenge 3 data could not be loaded: {exc}")
    st.info(
        "Inspect the SEG-Y files, trace-key pairing and authorized cache directory "
        "on the HPC. Check RTM_CACHE_DIR permissions and available storage."
    )
    st.stop()

baseline, monitor = baseline_volume.data, monitor_volume.data

# The cache files are immutable inputs to this dashboard. Their paths,
# sizes and modification times form a stable identity without reading
# or hashing the multi-gigabyte seismic arrays.
cache_directory = (
    settings.cache_directory
    if settings.cache_directory is not None
    else settings.challenge3_root / ".reservoir_time_machine_cache" / "challenge3_paired"
)
identity_files = (
    cache_directory / "metadata.json",
    cache_directory / "baseline.npy",
    cache_directory / "monitor.npy",
)
identity_parts = [str(cache_directory.resolve())]
for identity_file in identity_files:
    stat = identity_file.stat()
    identity_parts.append(
        f"{identity_file.name}:{stat.st_size}:{stat.st_mtime_ns}"
    )
dataset_cache_identity = "|".join(identity_parts)
if baseline.shape != monitor.shape:
    st.error("Loaded baseline and monitor shapes differ. Processing is disabled until geometry is reviewed.")
    st.stop()
paired_grid_available = bool(
    monitor_volume.metadata.source_metadata.get(
        "paired_grid_available", False
    )
)

physical_4d_registration_verified = bool(
    monitor_volume.metadata.source_metadata.get(
        "physical_4d_registration_verified", False
    )
)

for state_key, maximum in (("shared_sample", baseline.shape[2] - 1),
                          ("shared_inline", baseline.shape[0] - 1),
                          ("shared_crossline", baseline.shape[1] - 1)):
    value = st.session_state.get(state_key, maximum // 2)
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = maximum // 2
    st.session_state[state_key] = min(maximum, max(0, value))

with st.sidebar:
    st.header("Interpretation workspace")
    page = st.radio("Workspace", ["Overview and QC", "3D explorer", "4D comparison",
                                   "Change detection", "Survey transition", "Well inventory", "Evidence explanation"],
                    label_visibility="collapsed")
    sample = st.slider("Sample Index", 0, baseline.shape[2]-1, key="shared_sample")
    inline = st.slider("Inline index", 0, baseline.shape[0]-1, key="shared_inline")
    crossline = st.slider("Crossline index", 0, baseline.shape[1]-1, key="shared_crossline")
    color_scale = st.selectbox("Seismic colour scale", ["RdBu", "Viridis", "Cividis", "Turbo"], index=0)
    st.caption("Selections are grid indices. Sample-domain origin and physical registration are unverified.")

if not availability["well_inventory_available"]:
    st.warning("The well-list spreadsheet is missing. Seismic-only views remain available; well context is not inferred.")

if not paired_grid_available:
    st.error("Paired Baseline/Monitor grid unavailable.")
    st.stop()

if not physical_4d_registration_verified:
    registration_warning()

if page == "Overview and QC":
    page_header("Survey overview & QC", "Survey geometry, data completeness, and amplitude quality at a glance.")
    base_qc = cached_quality_report(
        dataset_cache_identity,
        baseline_volume.metadata.survey_id,
        baseline,
    )
    monitor_qc = cached_quality_report(
        dataset_cache_identity,
        monitor_volume.metadata.survey_id,
        monitor,
    )
    section("Survey geometry")
    meta = monitor_volume.metadata.source_metadata
    inline_header_ids = meta.get("inline_ids", meta.get("inline_ids_selected", []))
    crossline_header_ids = meta.get("crossline_ids", meta.get("crossline_ids_selected", []))
    cols = st.columns(4)
    cols[0].metric("Baseline traces", f"{meta.get('baseline_trace_count', 1_240_941):,}")
    cols[1].metric("Monitor traces", f"{meta.get('monitor_trace_count', meta.get('paired_trace_count', 0) + meta.get('monitor_only_trace_count', 0)):,}")
    cols[2].metric("Common paired traces", f"{meta.get('paired_trace_count', 1_240_941):,}")
    cols[3].metric("Monitor-only traces", f"{meta.get('monitor_only_trace_count', 3_087):,}")
    cols = st.columns(4)
    cols[0].metric("Samples per trace", f"{baseline.shape[2]:,}")
    cols[1].metric("Grid dimensions", " × ".join(f"{n:,}" for n in baseline.shape))
    cols[2].metric("Inline IDs", f"{min(inline_header_ids):,}–{max(inline_header_ids):,}" if inline_header_ids else "Not recorded")
    cols[3].metric("Crossline IDs", f"{min(crossline_header_ids):,}–{max(crossline_header_ids):,}" if crossline_header_ids else "Not recorded")
    st.caption(f"Sample interval: {baseline_volume.metadata.sample_interval:g} {baseline_volume.metadata.sample_interval_units}. Array axes are (inline, crossline, sample); Sample Index is shown because its origin/domain is unknown.")
    section("Data quality")
    qc_cols = st.columns(2)
    for col, label, report in ((qc_cols[0], "Baseline", base_qc), (qc_cols[1], "Monitor", monitor_qc)):
        with col:
            st.markdown(f"**{label}**")
            metrics = st.columns(3)
            metrics[0].metric("Finite samples", f"{100*report['availability_fraction']:.3f}%")
            metrics[1].metric("Missing traces", f"{report['missing_trace_count']:,}")
            metrics[2].metric("Volume RMS", f"{report['amplitude_rms']:.4g}")
            st.caption(f"NaN {report['nan_sample_count']:,} · +Inf {report['positive_infinity_sample_count']:,} · −Inf {report['negative_infinity_sample_count']:,} · zero-energy traces {report['zero_energy_trace_count']:,}")
    section("Amplitude distribution · display sample")
    # Display sampling only; QC statistics above use the cached full-resolution reports.
    base_show = display_volume(baseline, max_axis=48)
    mon_show = display_volume(monitor, max_axis=48)
    hist = go.Figure()
    hist.add_histogram(x=base_show[np.isfinite(base_show)].ravel(), name="Baseline", opacity=.62, nbinsx=90)
    hist.add_histogram(x=mon_show[np.isfinite(mon_show)].ravel(), name="Monitor", opacity=.62, nbinsx=90)
    hist.update_layout(barmode="overlay", xaxis_title="Amplitude", yaxis_title="Display-sample count", height=390)
    st.plotly_chart(hist, use_container_width=True)
    section("Trace coverage")
    missing = baseline_volume.missing_trace_mask
    if missing is not None:
        coverage = display_mask(missing, max_axis=700)
        row_stride = max(1, int(np.ceil(missing.shape[0] / 700)))
        col_stride = max(1, int(np.ceil(missing.shape[1] / 700)))
        st.plotly_chart(go.Figure(go.Heatmap(z=coverage, x=np.arange(0, missing.shape[1], col_stride),
            y=np.arange(0, missing.shape[0], row_stride), colorscale=[[0,"#333033"],[1,"#ef5747"]],
            zmin=0,zmax=1,colorbar={"title":"Missing trace"})).update_layout(
                xaxis_title="Crossline index", yaxis_title="Inline index", height=540), use_container_width=True)
        st.caption("Coverage comes from the retained Baseline-grid mask. Each red display bin contains one or more absent traces; axes show the bin's starting grid indices, not geographic coordinates.")
    else:
        st.info("No missing-trace mask is available from the loaded survey metadata.")
    with st.expander("Advanced QC diagnostics"):
        st.write({"Baseline": base_qc, "Monitor": monitor_qc,
                  "trace_key_pairing_verified": True,
                  "physical_4d_registration_verified": physical_4d_registration_verified})
    st.caption("Trace RMS divided by amplitude standard deviation is a descriptive proxy, not a calibrated SNR.")
elif page == "3D explorer":
    page_header("3D seismic explorer", "Linked inline, crossline, and sample sections intersect at the shared selected location.")
    st.caption("Drag to rotate; choose Pan in the Plotly mode bar to pan; use scroll or mode bar controls to zoom and reset the camera. Planes are display-strided; missing trace bins are conservatively shown as gaps; source arrays and analytical data remain full resolution.")
    baseline_display, monitor_display = display_volume(baseline, 36), display_volume(monitor, 36)
    limit = max(float(np.nanpercentile(np.abs(baseline_display), 99)), float(np.nanpercentile(np.abs(monitor_display), 99)))
    left, right = st.columns(2)
    with left:
        st.plotly_chart(volume_figure(baseline, "Baseline · orthogonal sections", inline=inline,
            crossline=crossline, sample=sample, missing_trace_mask=baseline_volume.missing_trace_mask,
            colorscale=color_scale, value_range=(-limit, limit)),
            use_container_width=True, config={"scrollZoom": True, "displaylogo": False,
                "modeBarButtonsToAdd": ["resetCameraDefault3d"]})
    with right:
        st.plotly_chart(volume_figure(monitor, "Monitor · orthogonal sections", inline=inline,
            crossline=crossline, sample=sample, missing_trace_mask=monitor_volume.missing_trace_mask,
            colorscale=color_scale, value_range=(-limit, limit)),
            use_container_width=True, config={"scrollZoom": True, "displaylogo": False,
                "modeBarButtonsToAdd": ["resetCameraDefault3d"]})
    inline_ids = monitor_volume.metadata.source_metadata.get("inline_ids", monitor_volume.metadata.source_metadata.get("inline_ids_selected", []))
    crossline_ids = monitor_volume.metadata.source_metadata.get("crossline_ids", monitor_volume.metadata.source_metadata.get("crossline_ids_selected", []))
    st.metric("Selected coordinate", f"Inline {inline_ids[inline] if inline < len(inline_ids) else inline} · Crossline {crossline_ids[crossline] if crossline < len(crossline_ids) else crossline} · Sample Index {sample}")
    st.caption("Header IDs are shown when supplied; sample remains an array index because sample origin and physical domain are unverified.")
elif page == "4D comparison":
    page_header("4D comparison", "Compare both surveys at the same Baseline-grid location and selected sample.")
    attrs = cached_sample_attributes(
        dataset_cache_identity,
        sample,
        9,
        baseline,
        monitor,
    )
    color_limit = max(float(np.nanpercentile(np.abs(baseline[:, :, sample]), 99)),
                      float(np.nanpercentile(np.abs(monitor[:, :, sample]), 99)))
    view = st.radio("Synchronized section", ["Sample plane", "Inline section", "Crossline section"], horizontal=True)
    if view == "Sample plane":
        bview, mview, dview = baseline[:, :, sample], monitor[:, :, sample], attrs.signed_difference
        missing_view = baseline_volume.missing_trace_mask
        x_title, y_title = "Crossline index", "Inline index"
    elif view == "Inline section":
        bview, mview = baseline[inline, :, :].T, monitor[inline, :, :].T
        dview = mview - bview
        missing_view = (np.broadcast_to(np.asarray(baseline_volume.missing_trace_mask)[inline, :][None, :], bview.shape)
                        if baseline_volume.missing_trace_mask is not None else None)
        x_title, y_title = "Crossline index", "Sample Index"
    else:
        bview, mview = baseline[:, crossline, :].T, monitor[:, crossline, :].T
        dview = mview - bview
        missing_view = (np.broadcast_to(np.asarray(baseline_volume.missing_trace_mask)[:, crossline][None, :], bview.shape)
                        if baseline_volume.missing_trace_mask is not None else None)
        x_title, y_title = "Inline index", "Sample Index"
    shared_limit = max(float(np.nanpercentile(np.abs(bview), 99)), float(np.nanpercentile(np.abs(mview), 99)))
    difference_limit = float(np.nanpercentile(np.abs(dview), 99))
    difference_limit = difference_limit if np.isfinite(difference_limit) and difference_limit > 0 else 1.0
    section(f"{view} · Baseline | Monitor | Difference")
    panel = st.columns(3)
    for col, values, title, scale, limit_value in ((panel[0], bview, "BASELINE", color_scale, shared_limit),
            (panel[1], mview, "MONITOR", color_scale, shared_limit),
            (panel[2], dview, "MONITOR − BASELINE", "RdBu", difference_limit)):
        shown = display_slice(values, max_axis=700)
        row_stride = max(1, int(np.ceil(values.shape[0] / 700)))
        col_stride = max(1, int(np.ceil(values.shape[1] / 700)))
        xs = np.arange(0, values.shape[1], col_stride)
        ys = np.arange(0, values.shape[0], row_stride)
        section_fig = go.Figure(go.Heatmap(z=shown, x=xs, y=ys, colorscale=scale, zmin=-limit_value,
            zmax=limit_value, zmid=0, colorbar={"title": "Amplitude" if title != "MONITOR − BASELINE" else "Difference"}))
        if missing_view is not None:
            section_fig.add_trace(_missing_overlay_trace(missing_view, x_name=x_title, y_name=y_title))
        col.plotly_chart(section_fig.update_layout(
            title=title, xaxis_title=x_title, yaxis_title=y_title, height=540), use_container_width=True)
    if missing_view is not None:
        st.caption("Gray overlays mark display bins containing at least one missing paired-grid trace. Full-resolution attributes exclude the actual missing source cells; valid cells in the same visual bin remain included.")
    section("Selected point and quantitative attributes")
    base_amp, mon_amp = float(baseline[inline, crossline, sample]), float(monitor[inline, crossline, sample])
    values = st.columns(6)
    values[0].metric("Baseline amplitude", f"{base_amp:.5g}" if np.isfinite(base_amp) else "Missing")
    values[1].metric("Monitor amplitude", f"{mon_amp:.5g}" if np.isfinite(mon_amp) else "Missing")
    values[2].metric("Signed difference", f"{mon_amp-base_amp:.5g}" if np.isfinite(base_amp+mon_amp) else "Missing")
    values[3].metric("Baseline RMS · 9 samples", f"{attrs.rms_baseline[inline,crossline]:.5g}" if np.isfinite(attrs.rms_baseline[inline,crossline]) else "Missing")
    values[4].metric("Monitor RMS · 9 samples", f"{attrs.rms_monitor[inline,crossline]:.5g}" if np.isfinite(attrs.rms_monitor[inline,crossline]) else "Missing")
    values[5].metric("Windowed NRMS", f"{attrs.nrms[inline,crossline]:.2f}%" if np.isfinite(attrs.nrms[inline,crossline]) else "Missing")
    with st.expander("Attribute definitions and limits", expanded=True):
        st.markdown("**Signed difference** = Monitor − Baseline at the selected grid cell and sample. It retains sign; it does not identify a cause.")
        st.markdown("**Windowed NRMS** = 200 × RMS(Monitor − Baseline) / (RMS(Baseline) + RMS(Monitor)) over a centered 9-sample window, clipped at volume ends and calculated from finite paired samples. Higher values indicate greater local dissimilarity; NRMS is not a probability or registration correction.")
        st.markdown("**Limit:** equal grid keys establish correspondence only. Without physical 4D registration, differences remain preliminary screening evidence.")
elif page == "Change detection":
    page_header("Change detection", "Locate unusually large paired amplitude differences on the selected sample plane.")
    threshold_sigma = st.slider("MAD multiplier", 1.0, 8.0, 3.5, 0.5,
        help="Cutoff = median + multiplier × 1.4826 × median absolute deviation. The inherited 3.5 default is an uncalibrated high-tail screening heuristic, not a probability threshold. Larger multipliers flag fewer cells.")
    attrs = cached_sample_attributes(
        dataset_cache_identity,
        sample,
        9,
        baseline,
        monitor,
    )
    min_size = st.number_input("Minimum connected region size (grid cells)", min_value=1, max_value=1000, value=8,
        help="The inherited 8-cell default is an uncalibrated filter for very small 4-connected fragments. It is not derived from physical spacing or geological truth; smaller components are omitted from the region list.")
    detection, labels, zones = cached_sample_detection(dataset_cache_identity, sample, 9,
        float(threshold_sigma), int(min_size), attrs.absolute_difference)
    selected_zone = None
    if zones:
        data_key = hashlib.sha1(dataset_cache_identity.encode()).hexdigest()[:12]
        region_key = f"selected_region_{data_key}_{sample}_{threshold_sigma}_{min_size}"
        selected_zone_id = st.selectbox("Inspect connected region", [zone.label for zone in zones],
            format_func=lambda value: f"Region {value} · {next(z.voxel_count for z in zones if z.label == value)} cells",
            key=region_key, on_change=_select_detection_region, args=(region_key, zones))
        selected_zone = next(zone for zone in zones if zone.label == selected_zone_id)
    score_display = display_slice(detection.score, max_axis=700)
    mask_display = display_slice(detection.mask, max_axis=700)
    label_display = display_slice(labels, max_axis=700)
    row_stride = max(1, int(np.ceil(detection.score.shape[0] / 700)))
    col_stride = max(1, int(np.ceil(detection.score.shape[1] / 700)))
    image = go.Figure(
        go.Heatmap(
            z=score_display,
            x=np.arange(0, detection.score.shape[1], col_stride),
            y=np.arange(0, detection.score.shape[0], row_stride),
            colorscale="Magma", zmin=0,
            colorbar={"title": "Absolute difference"},
        )
    )
    if baseline_volume.missing_trace_mask is not None:
        image.add_trace(_missing_overlay_trace(baseline_volume.missing_trace_mask, max_axis=700))
    image.add_trace(go.Heatmap(z=np.where(mask_display, 1.0, np.nan),
        x=np.arange(0, detection.score.shape[1], col_stride),
        y=np.arange(0, detection.score.shape[0], row_stride),
        colorscale=[[0, "rgba(0,0,0,0)"], [1, "#f6e7df"]], zmin=0, zmax=1,
        opacity=.62, showscale=False, name="Flagged cells"))
    if selected_zone is not None:
        center_inline = (selected_zone.bounding_box[0][0] + selected_zone.bounding_box[0][1]) // 2
        center_crossline = (selected_zone.bounding_box[1][0] + selected_zone.bounding_box[1][1]) // 2
        image.add_trace(go.Scatter(x=[center_crossline], y=[center_inline], mode="markers",
            marker={"symbol": "x", "size": 13, "color": "#ff7047", "line": {"width": 3}},
            name="Selected region center"))
        image.add_trace(go.Heatmap(z=np.where(label_display == selected_zone.label, 1.0, np.nan),
            x=np.arange(0, detection.score.shape[1], col_stride),
            y=np.arange(0, detection.score.shape[0], row_stride),
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "#ff7047"]], zmin=0, zmax=1,
            opacity=.86, showscale=False, name=f"Selected region {selected_zone.label}"))
    st.plotly_chart(
        image.update_layout(
            title="Statistical screen · score is not probability",
            xaxis_title="Crossline index", yaxis_title="Inline index", height=600,
        ),
        use_container_width=True,
    )
    st.caption(
        f"Formula: median + {threshold_sigma:g} × 1.4826 × MAD; MAD is the median absolute deviation on the full-resolution absolute-difference plane. The multiplier changes statistical sensitivity, not geological truth."
    )
    st.metric("Retained connected regions", f"{len(zones):,}")
    st.metric("Flagged cells before region-size filter", f"{int(detection.mask.sum()):,}")
    st.caption("Connectivity is face-sharing (4-connected on this 2D sample plane). Gray display bins contain one or more missing traces; those source cells are excluded from screening, while valid cells in the same bin may contribute. Region sizes are source grid-cell counts; physical area is unavailable because spacing and CRS are not established.")
    score_values = score_display[np.isfinite(score_display)]
    if score_values.size:
        counts, edges = np.histogram(score_values, bins=60)
        diagnostics = st.columns(2)
        diagnostics[0].plotly_chart(go.Figure(go.Bar(x=(edges[:-1]+edges[1:])/2, y=counts,
            marker_color="#e34b3e")).update_layout(title="Absolute-difference distribution · display sample",
            xaxis_title="Absolute difference", yaxis_title="Sample count", height=340), use_container_width=True)
        if zones:
            diagnostics[1].plotly_chart(go.Figure(go.Histogram(x=[zone.voxel_count for zone in zones],
                nbinsx=40, marker_color="#f08b53")).update_layout(title="Retained region sizes",
                xaxis_title="Grid cells per region", yaxis_title="Region count", height=340), use_container_width=True)
    if selected_zone is not None:
        inline = (selected_zone.bounding_box[0][0] + selected_zone.bounding_box[0][1]) // 2
        crossline = (selected_zone.bounding_box[1][0] + selected_zone.bounding_box[1][1]) // 2
        st.markdown(f"**Region {selected_zone.label}** · {selected_zone.voxel_count:,} cells · center Inline {inline}, Crossline {crossline}, Sample Index {sample} · mean score {selected_zone.mean_strength:.4g} · maximum {selected_zone.max_strength:.4g}")
        point_values = st.columns(5)
        rb, rm = float(baseline[inline,crossline,sample]), float(monitor[inline,crossline,sample])
        point_values[0].metric("Baseline amplitude", f"{rb:.4g}" if np.isfinite(rb) else "Missing")
        point_values[1].metric("Monitor amplitude", f"{rm:.4g}" if np.isfinite(rm) else "Missing")
        point_values[2].metric("Difference", f"{rm-rb:.4g}" if np.isfinite(rb) and np.isfinite(rm) else "Missing")
        point_values[3].metric("Absolute difference", f"{attrs.absolute_difference[inline,crossline]:.4g}" if np.isfinite(attrs.absolute_difference[inline,crossline]) else "Missing")
        point_values[4].metric("Windowed NRMS", f"{attrs.nrms[inline,crossline]:.2f}%" if np.isfinite(attrs.nrms[inline,crossline]) else "Missing")
        evidence_cols = st.columns(3)
        bsection = baseline[inline, :, :].T
        msection = monitor[inline, :, :].T
        dsection = msection - bsection
        section_limit = max(float(np.nanpercentile(np.abs(bsection), 99)), float(np.nanpercentile(np.abs(msection), 99)))
        diff_limit = float(np.nanpercentile(np.abs(dsection), 99))
        diff_limit = diff_limit if np.isfinite(diff_limit) and diff_limit > 0 else 1.0
        missing_section = (np.broadcast_to(np.asarray(baseline_volume.missing_trace_mask)[inline, :][None, :],
                                           bsection.shape)
                           if baseline_volume.missing_trace_mask is not None else None)
        for col, values, name, scale, limit_value in ((evidence_cols[0], bsection, "Baseline evidence", color_scale, section_limit),
            (evidence_cols[1], msection, "Monitor evidence", color_scale, section_limit),
            (evidence_cols[2], dsection, "Monitor − Baseline", "RdBu", diff_limit)):
            section_values = display_slice(values, max_axis=500)
            evidence_fig = go.Figure(go.Heatmap(z=section_values, colorscale=scale,
                zmin=-limit_value, zmax=limit_value, zmid=0))
            if missing_section is not None:
                evidence_fig.add_trace(_missing_overlay_trace(missing_section, max_axis=500,
                    x_name="Crossline index", y_name="Sample Index"))
            col.plotly_chart(evidence_fig.update_layout(
                title=name + " · selected inline", xaxis_title="Crossline index", yaxis_title="Sample Index", height=390), use_container_width=True)
        if missing_section is not None:
            st.caption("Gray overlays identify display bins containing one or more missing traces; they do not imply the entire bin is missing.")
    else:
        st.info("No components meet the selected threshold and minimum size on this sample plane.")
    with st.expander("Statistical method and limits"):
        st.markdown("The screen flags finite absolute Monitor − Baseline amplitudes above the global median plus scaled MAD. The 1.4826 factor makes MAD comparable to standard deviation under a normal distribution; the inherited 3.5 multiplier is not calibrated to these surveys and does not produce probabilities.")
        st.markdown("The inherited 8-cell minimum suppresses very small face-connected fragments as an exploratory filter. It is not calibrated using physical bin spacing, noise, or geological truth, and it does not represent a geological minimum size. Review sensitivity at nearby settings; classifications are screening results, not geological truth.")
elif page == "Survey transition":
    page_header("Survey transition", "Navigate the selected sample plane and compare clearly labeled survey endpoints.")
    navigation = st.columns(4)
    navigation[0].button("Previous sample", on_click=_step_sample, args=(-1, baseline.shape[2]-1), use_container_width=True)
    navigation[1].button("Next sample", on_click=_step_sample, args=(1, baseline.shape[2]-1), use_container_width=True)
    navigation[2].button("Reset sample", on_click=_reset_sample, args=(baseline.shape[2]-1,), use_container_width=True)
    navigation[3].metric("Current Sample Index", f"{sample:,}")
    attrs = cached_sample_attributes(
        dataset_cache_identity,
        sample,
        9,
        baseline,
        monitor,
    )
    show_overlay = st.checkbox("Show detected anomaly overlay")
    overlay = None
    if show_overlay:
        threshold_sigma = st.slider("Overlay MAD multiplier", 1.0, 8.0, 3.5, 0.5, key="animation_threshold")
        overlay = statistical_screen(attrs.absolute_difference, threshold_sigma=threshold_sigma).mask
    speed = st.select_slider("Playback speed", options=["Slow", "Standard", "Fast"], value="Standard")
    frame_duration_ms = {"Slow": 360, "Standard": 180, "Fast": 80}[speed]
    fig = cached_animation(dataset_cache_identity, sample, threshold_sigma if show_overlay else None,
        frame_duration_ms,
        baseline[:, :, sample], monitor[:, :, sample], overlay,
        baseline_volume.missing_trace_mask[:, :])
    st.plotly_chart(fig, use_container_width=True)
    st.info("Play/Pause and the progress slider navigate named frames. Intermediate frames are display-only linear interpolation between Baseline and Monitor endpoints; they are not measured vintages or a continuous reservoir history.")
    st.caption("Gray overlay marks any missing trace within a display bin; anomaly overlay is independent and uses the selected robust-MAD screen.")
elif page == "Well inventory":
    page_header("Well inventory", "Inspect available well sources and source identifiers without assuming spatial alignment.")
    st.caption("Well coordinates, units, datums, CRS, and seismic correspondence are not inferred.")
    inventory = []
    for category, paths in WELL_FILES.items():
        for relative in paths:
            inventory.append({"data_type": category, "relative_path": relative,
                              "available": (settings.challenge3_root / relative).is_file()})
    inventory.insert(0, {"data_type": "well inventory workbook", "relative_path": "well_data/Challenge3_WellList_Information.xlsx",
                         "available": (settings.challenge3_root / "well_data/Challenge3_WellList_Information.xlsx").is_file()})
    st.dataframe(inventory, use_container_width=True)
    workbook_path = settings.challenge3_root / "well_data/Challenge3_WellList_Information.xlsx"
    if workbook_path.is_file():
        from reservoir_time_machine.data_ingest.well_information import inspect_well_information
        try:
            workbook_report = inspect_well_information(workbook_path)
            st.markdown("**Well-list workbook structure**")
            st.dataframe([{"sheet": sheet["sheet"], "source columns": ", ".join(sheet["columns"]),
                           "data rows": sheet["data_rows"]} for sheet in workbook_report["sheets"]], use_container_width=True)
        except (RuntimeError, OSError, ValueError) as exc:
            st.info(f"Workbook structure is unavailable: {type(exc).__name__}")
    available_tables = [row["relative_path"] for row in inventory if row["available"] and row["data_type"] in ("checkshots", "deviations", "markers")]
    if available_tables:
        selected_file = st.selectbox("Inspect source table", available_tables)
        from reservoir_time_machine.data_ingest.well_tables import read_well_table
        try:
            table = read_well_table(settings.challenge3_root / selected_file)
            st.caption(f"Source: {table.source_file}. Original headings and values are shown; no units are assigned by this reader.")
            st.dataframe([dict(zip(table.columns, row)) for row in table.rows], use_container_width=True)
        except (OSError, UnicodeError, ValueError) as exc:
            st.error(f"Could not read the selected table: {type(exc).__name__}")
    available_las = [row["relative_path"] for row in inventory if row["available"] and row["data_type"] == "las"]
    if available_las:
        selected_las = st.selectbox("Inspect LAS log metadata", available_las)
        from reservoir_time_machine.data_ingest.las_loader import inspect_las
        try:
            las_report = inspect_las(settings.challenge3_root / selected_las)
            st.markdown(f"**{las_report['well_id'] or 'Well identifier unavailable'}** · {las_report['provenance']} · {las_report['sample_count']:,} samples · depth reference {las_report['depth_mnemonic'] or 'unspecified'} ({las_report['depth_unit'] or 'unit unavailable'})")
            st.dataframe([{key: curve[key] for key in ("mnemonic", "unit", "description", "sample_count", "null_count")}
                          for curve in las_report["curves"]], use_container_width=True)
        except (RuntimeError, OSError, ValueError) as exc:
            st.info(f"LAS metadata is unavailable: {type(exc).__name__}")
    st.info("Well positions are not plotted because validated coordinates and a shared spatial reference have not been established.")
else:
    page_header("Evidence explanation", "Separate observed data, calculated attributes, statistical flags, interpretations, and hypotheses.")
    section("Selected location evidence")
    selected_base, selected_monitor = float(baseline[inline,crossline,sample]), float(monitor[inline,crossline,sample])
    if np.isfinite(selected_base) and np.isfinite(selected_monitor):
        st.write(f"**Observed data:** Inline {inline}, Crossline {crossline}, Sample Index {sample}. Baseline amplitude {selected_base:.5g}; Monitor amplitude {selected_monitor:.5g}.")
    else:
        st.write("**Observed data:** one or both survey samples are missing or non-finite at this grid location.")
    st.caption("Only a bounded typed evidence packet may be sent to the optional Gemma service after explicit confirmation. Model output remains unverified and for human review.")
    from reservoir_time_machine.llm.gemma_client import test_connection
    if st.button("Test configured Gemma connection"):
        st.json(test_connection(LLMSettings.from_environment()))
    attrs = cached_sample_attributes(
        dataset_cache_identity,
        sample,
        9,
        baseline,
        monitor,
    )
    threshold_sigma = st.slider("Evidence screen threshold", 1.0, 8.0, 3.5, 0.5, key="llm_threshold")
    detection, labels, zones = cached_sample_detection(dataset_cache_identity, sample, 9,
        float(threshold_sigma), 8, attrs.absolute_difference)
    if np.isfinite(attrs.absolute_difference[inline,crossline]):
        st.write(f"**Calculated attributes:** absolute difference {attrs.absolute_difference[inline,crossline]:.5g}; 9-sample windowed NRMS {attrs.nrms[inline,crossline]:.4g}%.")
    else:
        st.write("**Calculated attributes:** unavailable because paired samples are non-finite.")
    st.write(f"**Statistical detection:** {int(detection.mask.sum()):,} cells flagged by the global robust MAD screen; score is not a probability. Selected cell is {'flagged' if detection.mask[inline,crossline] else 'not flagged'}.")
    st.write("**Geological interpretation:** none is generated automatically. Registration, processing, acquisition, noise, and geological effects remain alternative explanations.")
    st.write("**Uncertainty:** physical 4D registration and independent voxel-level geological truth are unavailable.")
    from reservoir_time_machine.ai.interfaces import EvidenceItem, EvidencePacket, GemmaEvidenceProvider
    packet = EvidencePacket(
        run_id=f"dashboard-sample-{sample}",
        data_classification="unknown",
        observations=(EvidenceItem(
            statement=f"The requested display slice uses sample index {sample}; no absolute time/depth label is inferred here.",
            source="dataset sample-axis selection", status="measured",
        ),),
        detections=(EvidenceItem(
            statement=(f"Selected-sample absolute-difference screen flagged {int(detection.mask.sum())} cells "
                       f"at threshold {float(detection.diagnostics['threshold']):.6g}."),
            source="selected-sample statistical screen", status="algorithmic_detection",
            uncertainty="Screen score is uncalibrated and is not a probability.",
        ),),
        alternatives=("Acquisition, processing, noise, alignment, and geological changes may contribute.",),
        uncertainty=("No independent voxel-level geological truth is supplied.",),
        missing_evidence=("Validated survey registration", "Independent geological calibration"),
    )
    allow_transmission = st.checkbox(
        "I confirm this structured evidence is authorized for transmission to the configured Gemma endpoint."
    )
    if st.button("Ask configured Gemma endpoint to explain this evidence", disabled=not allow_transmission):
        provider = GemmaEvidenceProvider(LLMSettings.from_environment(), allow_restricted_data=allow_transmission)
        try:
            st.write(provider.explain(packet))
        except Exception as exc:
            st.error(f"Gemma request failed: {type(exc).__name__}: {exc}")
