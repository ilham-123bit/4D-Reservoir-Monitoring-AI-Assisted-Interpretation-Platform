"""Challenge 3 seismic surveillance dashboard; never substitutes generated data."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from reservoir_time_machine.attributes.difference import calculate_sample_attributes
from reservoir_time_machine.data_audit.audit import WELL_FILES
from reservoir_time_machine.data_ingest.manifest import challenge3_availability
from reservoir_time_machine.data_ingest.segy_loader import load_challenge3_pair
from reservoir_time_machine.detection.clustering import connected_zones
from reservoir_time_machine.detection.statistical import statistical_screen
from reservoir_time_machine.preprocessing.qc import quality_report
from reservoir_time_machine.settings import ApplicationSettings, LLMSettings
from reservoir_time_machine.visualization.animation import animation_figure
from reservoir_time_machine.visualization.dashboard_state import empty_dataset_state
from reservoir_time_machine.visualization.seismic_3d import volume_figure
from reservoir_time_machine.visualization.slices import display_volume


st.set_page_config(page_title="Reservoir Time Machine", page_icon="🌐", layout="wide")
st.markdown("""
<style>
.stApp {background:#101820;color:#e6edf3}
[data-testid="stMetric"] {background:#172631;padding:14px;border-radius:10px}
</style>
""", unsafe_allow_html=True)
st.title("Reservoir Time Machine")
st.caption("APGCE GeoHackathon 2026 · Challenge 3 · paired-vintage seismic surveillance")
st.caption("No dataset is bundled with this application. Scientific views appear only for validated supplied inputs and processing outputs.")
st.warning("Interpretation support only. Seismic anomalies do not independently prove depletion, fluid movement, or remaining hydrocarbons.")

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


@st.cache_resource(show_spinner="Validating SEG-Y geometry and loading actual Challenge 3 volumes…")
def cached_actual_pair(root: str, max_pair_memory_bytes: int, cache_dir: str | None):
    return load_challenge3_pair(root, max_pair_memory_bytes=max_pair_memory_bytes,
                                cache_directory=cache_dir)


try:
    with st.spinner("Checking baseline/monitor geometry and reading actual data…"):
        baseline_volume, monitor_volume = cached_actual_pair(
            str(settings.challenge3_root), settings.max_pair_memory_bytes,
            str(settings.cache_directory) if settings.cache_directory else None)
except (ImportError, RuntimeError, FileNotFoundError, MemoryError, ValueError, OSError) as exc:
    st.error(f"Actual Challenge 3 data could not be loaded: {exc}")
    st.info("Inspect the files and geometry on the authorized HPC. Configure `RTM_MAX_PAIR_MEMORY_BYTES` or an authorized `RTM_CACHE_DIR` when needed.")
    st.stop()

baseline, monitor = baseline_volume.data, monitor_volume.data
if baseline.shape != monitor.shape:
    st.error("Loaded baseline and monitor shapes differ. Processing is disabled until geometry is reviewed.")
    st.stop()
spatial_geometry_compatible = bool(monitor_volume.metadata.source_metadata.get("spatial_geometry_compatible", False))

with st.sidebar:
    st.header("Dataset")
    page = st.radio("Workspace", ["Overview and QC", "3D explorer", "4D comparison",
                                   "Change detection", "Survey transition", "Well inventory", "Evidence explanation"])
    sample = st.slider("Sample index", 0, baseline.shape[2]-1, baseline.shape[2]//2)
    color_scale = st.selectbox("Seismic colour scale", ["RdBu", "Viridis", "Cividis", "Turbo"])

if not availability["well_inventory_available"]:
    st.warning("The well-list spreadsheet is missing. Seismic-only views remain available; well context is not inferred.")

if not spatial_geometry_compatible:
    st.warning("Source/receiver coordinate headers differ or require alignment review. Only independent baseline/monitor inspection is enabled; paired attributes and change screens are blocked.")
    if page in {"4D comparison", "Change detection", "Survey transition", "Evidence explanation"}:
        st.error("Run the geometry audit and establish a validated registration before comparing these surveys.")
        st.stop()

if page == "Overview and QC":
    st.subheader("Dataset overview")
    base_qc, monitor_qc = quality_report(baseline), quality_report(monitor)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Survey A", "Baseline")
    c2.metric("Survey B", "Monitor")
    c3.metric("Array dimensions", " × ".join(map(str, baseline.shape)))
    c4.metric("Header sample interval", f"{baseline_volume.metadata.sample_interval:g} {baseline_volume.metadata.sample_interval_units}")
    st.caption(f"Standardized array convention: (inline, crossline, sample). Sample-axis domain is {baseline_volume.metadata.sample_axis_domain}; coordinates and CRS are not assigned by this application.")
    st.write({"trace_order_matches": True,
              "spatial_geometry_compatible": spatial_geometry_compatible,
              "coordinate_headers_match": monitor_volume.metadata.source_metadata.get("coordinate_headers_match")})
    left, right = st.columns(2)
    left.markdown("**Baseline metadata and QC**")
    left.json({"survey_id": baseline_volume.metadata.survey_id, "label": baseline_volume.metadata.label,
               "sample_units": baseline_volume.metadata.sample_units, **base_qc})
    right.markdown("**Monitor metadata and QC**")
    right.json({"survey_id": monitor_volume.metadata.survey_id, "label": monitor_volume.metadata.label,
                "sample_units": monitor_volume.metadata.sample_units, **monitor_qc})
    st.caption("Trace RMS divided by amplitude standard deviation is a descriptive proxy, not a calibrated SNR.")
elif page == "3D explorer":
    st.subheader("Interactive 3D seismic volumes")
    st.caption("Plotly renders downsampled display copies; input arrays remain unchanged. Rotate, zoom, and pan with the plot controls.")
    baseline_display, monitor_display = display_volume(baseline, 40), display_volume(monitor, 40)
    limit = max(float(np.nanpercentile(np.abs(baseline_display), 99)),
                float(np.nanpercentile(np.abs(monitor_display), 99)))
    left, right = st.columns(2)
    with left:
        st.plotly_chart(volume_figure(baseline, "Baseline", max_axis=40,
                                      colorscale=color_scale, value_range=(-limit, limit)), use_container_width=True)
    with right:
        st.plotly_chart(volume_figure(monitor, "Monitor", max_axis=40,
                                     colorscale=color_scale, value_range=(-limit, limit)), use_container_width=True)
    inline = st.slider("Inline", 0, baseline.shape[0]-1, baseline.shape[0]//2)
    crossline = st.slider("Crossline", 0, baseline.shape[1]-1, baseline.shape[1]//2)
    inline_ids = monitor_volume.metadata.source_metadata.get("inline_ids_selected", [])
    crossline_ids = monitor_volume.metadata.source_metadata.get("crossline_ids_selected", [])
    st.write({"selected_index_inline_crossline_sample": (inline, crossline, sample),
              "selected_inline_header_id": inline_ids[inline] if inline < len(inline_ids) else None,
              "selected_crossline_header_id": crossline_ids[crossline] if crossline < len(crossline_ids) else None,
              "baseline_amplitude": float(baseline[inline, crossline, sample]),
              "monitor_amplitude": float(monitor[inline, crossline, sample])})
    sample_plane = monitor[:, :, sample]
    inline_plane = monitor[inline, :, :]
    crossline_plane = monitor[:, crossline, :]
    slice_cols = st.columns(3)
    for col, values, title in zip(slice_cols, (inline_plane, crossline_plane, sample_plane),
                                  ("Monitor inline section", "Monitor crossline section", "Monitor sample plane")):
        col.plotly_chart(go.Figure(go.Heatmap(z=values, colorscale=color_scale, zmid=0)).update_layout(title=title),
                         use_container_width=True)
elif page == "4D comparison":
    st.subheader("Paired-vintage attributes at selected sample")
    attrs = calculate_sample_attributes(baseline, monitor, sample, window_samples=9)
    color_limit = max(float(np.nanpercentile(np.abs(baseline[:, :, sample]), 99)),
                      float(np.nanpercentile(np.abs(monitor[:, :, sample]), 99)))
    maps = [(baseline[:, :, sample], "Baseline"), (monitor[:, :, sample], "Monitor"),
            (attrs.signed_difference, "Signed difference"), (attrs.absolute_difference, "Absolute difference"),
            (attrs.nrms, "Windowed NRMS (%)")]
    cols = st.columns(5)
    for col, (values, title) in zip(cols, maps):
        kwargs = {"colorscale": color_scale}
        if title in ("Baseline", "Monitor", "Signed difference"):
            kwargs.update(zmin=-color_limit, zmax=color_limit, zmid=0)
        elif title == "Absolute difference":
            kwargs.update(zmin=0, zmax=float(np.nanpercentile(values, 99)))
        else:
            kwargs.update(zmin=0, zmax=200)
        col.plotly_chart(go.Figure(go.Heatmap(z=values, **kwargs)).update_layout(title=title), use_container_width=True)
    st.caption("NRMS measures dissimilarity. It does not identify a unique geological cause.")
elif page == "Change detection":
    st.subheader("Selected-sample anomaly screen")
    threshold_sigma = st.slider("Robust threshold (scaled MAD)", 1.0, 8.0, 3.5, 0.5)
    attrs = calculate_sample_attributes(baseline, monitor, sample, window_samples=9)
    detection = statistical_screen(attrs.absolute_difference, threshold_sigma=threshold_sigma)
    min_size = st.number_input("Minimum connected region size (cells)", min_value=1, max_value=1000, value=8)
    labels, zones = connected_zones(detection.mask, detection.score, min_size=int(min_size))
    image = go.Figure(go.Heatmap(z=detection.score, colorscale="Magma", colorbar={"title": "Difference strength"}))
    rows, cols = np.where(detection.mask)
    image.add_trace(go.Scatter(x=cols, y=rows, mode="markers", marker={"size": 5, "color": "cyan"}, name="Screened cells"))
    st.plotly_chart(image.update_layout(title="Flagged cells · score is not probability"), use_container_width=True)
    st.write(f"Detected components in selected sample: {len(zones)}")
    st.dataframe([zone.__dict__ for zone in zones], use_container_width=True)
elif page == "Survey transition":
    st.subheader("Baseline to monitor visual transition")
    attrs = calculate_sample_attributes(baseline, monitor, sample, window_samples=9)
    show_overlay = st.checkbox("Overlay selected-sample anomaly screen")
    overlay = None
    if show_overlay:
        threshold_sigma = st.slider("Overlay threshold (scaled MAD)", 1.0, 8.0, 3.5, 0.5, key="animation_threshold")
        overlay = statistical_screen(attrs.absolute_difference, threshold_sigma=threshold_sigma).mask
    st.plotly_chart(animation_figure(baseline[:, :, sample], monitor[:, :, sample], overlay), use_container_width=True)
    st.info("Intermediate animation frames are visual interpolation between the two actual endpoint sections, not measured vintages or verified fluid paths.")
elif page == "Well inventory":
    st.subheader("Well data availability")
    st.caption("This inventory reports source-file presence. It does not assign coordinates, depth references, or geological meaning.")
    inventory = []
    for category, paths in WELL_FILES.items():
        for relative in paths:
            inventory.append({"data_type": category, "relative_path": relative,
                              "available": (settings.challenge3_root / relative).is_file()})
    st.dataframe(inventory, use_container_width=True)
    st.info("Run the read-only data audit before integrating well coordinates, trajectories, checkshots, LAS curves, or markers.")
else:
    st.subheader("Evidence explanation (optional Gemma service)")
    st.caption("Only a bounded typed evidence packet is sent. No seismic samples or full volumes are transmitted. Model output is an unverified explanation for human review.")
    from reservoir_time_machine.llm.gemma_client import test_connection
    if st.button("Test configured Gemma connection"):
        st.json(test_connection(LLMSettings.from_environment()))
    attrs = calculate_sample_attributes(baseline, monitor, sample, window_samples=9)
    threshold_sigma = st.slider("Evidence screen threshold", 1.0, 8.0, 3.5, 0.5, key="llm_threshold")
    detection = statistical_screen(attrs.absolute_difference, threshold_sigma=threshold_sigma)
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
