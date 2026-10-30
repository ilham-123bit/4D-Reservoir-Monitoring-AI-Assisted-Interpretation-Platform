"""Interactive 3D scientific dashboard; scientific routines remain in package modules."""

import streamlit as st
import plotly.graph_objects as go
import numpy as np

from reservoir_time_machine.config import SyntheticConfig
from reservoir_time_machine.synthetic import generate_scenario
from reservoir_time_machine.attributes.difference import calculate_attributes
from reservoir_time_machine.detection.statistical import (difference_detector, nrms_detector,
    statistical_detector, spatial_detector)
from reservoir_time_machine.detection.clustering import connected_zones
from reservoir_time_machine.evaluation.metrics import evaluate_detection
from reservoir_time_machine.preprocessing.qc import quality_report
from reservoir_time_machine.visualization.seismic_3d import volume_figure
from reservoir_time_machine.visualization.slices import slice_triplet
from reservoir_time_machine.visualization.animation import animation_figure

st.set_page_config(page_title="Reservoir Time Machine 2.0", page_icon="🌐", layout="wide")
st.markdown("""
<style>
.stApp {background: #101820; color: #e6edf3} [data-testid="stMetric"] {background:#172631;padding:14px;border-radius:10px}
</style>
""", unsafe_allow_html=True)

st.title("◉ Reservoir Time Machine")
st.caption("3D paired-vintage seismic change explorer")
st.warning("SYNTHETIC — FOR DEVELOPMENT AND DEMONSTRATION ONLY. Scenario manipulations are not validated rock-physics simulations or evidence of remaining hydrocarbons.")


@st.cache_data(show_spinner="Generating reproducible synthetic seismic volumes…")
def cached_scenario(dimensions, seed, scenario, noise_std, shift, strength):
    cfg = SyntheticConfig(dimensions=tuple(dimensions), seed=int(seed), scenario=scenario,
                          noise_std=float(noise_std), maximum_time_shift_samples=float(shift),
                          change_strength=float(strength))
    return generate_scenario(cfg)


@st.cache_data(show_spinner="Calculating paired-vintage attributes…")
def cached_attributes(baseline, monitor):
    return calculate_attributes(baseline, monitor)


with st.sidebar:
    st.header("Synthetic project")
    view = st.radio("Workspace", ["Reservoir overview", "3D seismic explorer", "4D comparison",
                                    "Change detection", "Survey transition", "AI evaluation", "Opportunity explorer"])
    seed = st.number_input("Random seed", min_value=0, max_value=1_000_000, value=7)
    scenario_name = st.selectbox("Scenario", ["amplitude_increase", "amplitude_decrease", "weak_amplitude", "time_shift", "fluid_front",
                                                "unchanged", "acquisition_artifact", "noise_only"], index=0)
    inline_count = st.select_slider("Inline count", options=[32, 40, 48, 56], value=48)
    crossline_count = st.select_slider("Crossline count", options=[32, 40, 48, 56], value=48)
    sample_count = st.select_slider("Sample count", options=[64, 80, 96, 112], value=96)
    color_scale = st.selectbox("Seismic color scale", ["RdBu", "Viridis", "Cividis", "Turbo"])
    noise_std = st.slider("Random noise standard deviation", 0.0, 0.10, 0.035, 0.005)
    strength = st.slider("Synthetic change strength", 0.05, 0.80, 0.35, 0.05)
    shift = st.slider("Injected lag (samples)", 0.0, 4.0, 2.0, 0.5)

scenario = cached_scenario((inline_count, crossline_count, sample_count), seed, scenario_name, noise_std, shift, strength)
baseline, monitor = scenario.baseline.data, scenario.monitor.data
ni, nc, ns = baseline.shape

if view == "Reservoir overview":
    st.subheader("Meridian Field · synthetic paired-vintage overview")
    metrics = [quality_report(baseline), quality_report(monitor)]
    a, b, c, d = st.columns(4)
    a.metric("Survey A", "Year 1")
    b.metric("Survey B", "Year 11")
    c.metric("Volume", f"{ni} × {nc} × {ns}")
    d.metric("Available samples", f"{100*min(metrics[0]['availability_fraction'], metrics[1]['availability_fraction']):.1f}%")
    st.caption("Array axis order: inline × crossline × sample. No geographic coordinates are assigned.")
    left, right = st.columns(2)
    left.json(metrics[0])
    right.json(metrics[1])
    st.info(f"Scenario label: `{scenario_name}`. Survey gap is represented only by endpoint labels; no intermediate observations exist.")
elif view == "3D seismic explorer":
    st.subheader("3D seismic explorer")
    st.caption("Display cubes are downsampled copies. Analysis arrays remain at full resolution.")
    color_limit = max(float(np.nanpercentile(np.abs(baseline), 99)),
                      float(np.nanpercentile(np.abs(monitor), 99)))
    left, right = st.columns(2)
    with left: st.plotly_chart(volume_figure(baseline, "Survey A — Year 1", max_axis=40,
                                             value_range=(-color_limit, color_limit), colorscale=color_scale), use_container_width=True)
    with right: st.plotly_chart(volume_figure(monitor, "Survey B — Year 11", max_axis=40,
                                               value_range=(-color_limit, color_limit), colorscale=color_scale), use_container_width=True)
    i = st.slider("Inline index", 0, ni-1, ni//2, key="i3")
    x = st.slider("Crossline index", 0, nc-1, nc//2, key="x3")
    s = st.slider("Sample index", 0, ns-1, ns//2, key="s3")
    traces = slice_triplet(monitor, i, x, s)
    st.write({"inline/crossline/sample": (i, x, s), "amplitude": float(monitor[i, x, s])})
    slice_titles = ("Inline section", "Crossline section", "Sample slice")
    slice_cols = st.columns(3)
    for col, data, title in zip(slice_cols, traces, slice_titles):
        col.plotly_chart(go.Figure(go.Heatmap(z=data, colorscale="RdBu", zmid=0,
                                              zmin=-color_limit, zmax=color_limit)).update_layout(title=title),
                         use_container_width=True)
elif view == "4D comparison":
    st.subheader("Paired-vintage attributes")
    attrs = cached_attributes(baseline, monitor)
    i = st.slider("Inline index", 0, ni-1, ni//2, key="i4")
    x = st.slider("Crossline index", 0, nc-1, nc//2, key="x4")
    s = st.slider("Sample index", 0, ns-1, ns//2, key="s4")
    maps = [(baseline[:, :, s], "Survey A amplitude"), (monitor[:, :, s], "Survey B amplitude"),
            (attrs.signed_difference[:, :, s], "Signed difference"), (attrs.absolute_difference[:, :, s], "Absolute difference"),
            (attrs.nrms[:, :, s], "Windowed NRMS (%)")]
    cols = st.columns(len(maps))
    for col, (data, title) in zip(cols, maps):
        col.plotly_chart(go.Figure(go.Heatmap(z=data, colorscale="RdBu", zmid=0 if "difference" in title else None)).update_layout(title=title), use_container_width=True)
    st.caption(f"Selected voxel ({i}, {x}, {s}): A={baseline[i,x,s]:.4f}; B={monitor[i,x,s]:.4f}; Δ={attrs.signed_difference[i,x,s]:.4f}; NRMS={attrs.nrms[i,x,s]:.2f}%")
elif view == "Change detection":
    st.subheader("Change detection explorer")
    detector_kind = st.selectbox("Detector", ["Robust statistical", "Windowed NRMS"])
    if detector_kind == "Robust statistical":
        detected = statistical_detector(baseline, monitor)
    else:
        nrms_threshold = st.slider("NRMS threshold (%)", 5.0, 100.0, 35.0, 5.0)
        detected = nrms_detector(baseline, monitor, threshold=nrms_threshold)
    labels, zones = connected_zones(detected.mask, detected.score, min_size=16)
    sample = st.slider("Sample slice", 0, ns-1, ns//2, key="sd")
    fig = go.Figure(go.Heatmap(z=detected.score[:, :, sample], colorscale="Magma", colorbar={"title": "Strength"}))
    overlay_y, overlay_x = np.where(detected.mask[:, :, sample])
    fig.add_trace(go.Scatter(x=overlay_x, y=overlay_y, mode="markers", marker={"size": 4, "color": "cyan"}, name="Flagged voxels"))
    st.plotly_chart(fig.update_layout(title="Score strength (not calibrated probability) with detected voxels"), use_container_width=True)
    st.write(f"{len(zones)} connected zones (minimum 16 voxels)")
    st.dataframe([z.__dict__ for z in zones], use_container_width=True)
elif view == "Survey transition":
    st.subheader("Survey endpoint transition")
    sample = st.slider("Sample slice", 0, ns-1, ns//2, key="anim")
    show_overlay = st.checkbox("Overlay detected anomaly")
    overlay = None
    if show_overlay:
        overlay = statistical_detector(baseline, monitor).mask[:, :, sample]
    st.plotly_chart(animation_figure(baseline[:, :, sample], monitor[:, :, sample], overlay), use_container_width=True)
    st.info("Animation interpolates display values between the two endpoints. Frames between endpoints are not measured vintages and do not show verified fluid movement.")
elif view == "AI evaluation":
    st.subheader("Synthetic detector evaluation")
    st.caption("Ground truth is consulted here only after detector outputs have been produced.")
    truth = scenario.ground_truth.amplitude_change | scenario.ground_truth.time_shift | scenario.ground_truth.fluid_front
    import time
    rows = []
    for name, detector in [("Absolute difference", difference_detector), ("Windowed NRMS", nrms_detector),
                           ("Robust statistical", statistical_detector), ("Spatially coherent", spatial_detector)]:
        started = time.perf_counter()
        detected = detector(baseline, monitor)
        elapsed = time.perf_counter() - started
        scores = evaluate_detection(detected.mask, truth, runtime_seconds=elapsed)
        rows.append({"Detector": name, **scores.__dict__})
    st.dataframe(rows, use_container_width=True)
    st.caption("Scores are for this selected synthetic realization only; they are not field performance estimates or calibrated probabilities.")
else:
    st.subheader("Opportunity explorer · prepared extension point")
    st.info("No hydrocarbons or operational opportunities are estimated. Current evidence is synthetic change strength and scenario ground truth only.")
    st.write("Future inputs: validated geologic masks, authorized well context, uncertainty, and human-reviewed evidence.")
