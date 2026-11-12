"""Challenge 3 seismic surveillance dashboard; never substitutes generated data."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from reservoir_time_machine.attributes.difference import calculate_sample_attributes
from reservoir_time_machine.data_audit.audit import WELL_FILES
from reservoir_time_machine.data.readiness import assess_capabilities
from reservoir_time_machine.data_ingest.cache_identity import paired_cache_identity
from reservoir_time_machine.data_ingest.manifest import challenge3_availability
from reservoir_time_machine.data_ingest.segy_loader import (
    load_challenge3_paired_volumes,
)
from reservoir_time_machine.detection.clustering import connected_zones
from reservoir_time_machine.detection.statistical import statistical_screen
from reservoir_time_machine.settings import ApplicationSettings, LLMSettings
from reservoir_time_machine.ai.grounding import (stage3_finding_packet, interpret,
    GemmaFindingProvider, explanation_identity, transmission_allowed)
from reservoir_time_machine.analysis.preliminary import run_preliminary, VERSION as ML_VERSION
from reservoir_time_machine.visualization.preliminary import render_preliminary
from reservoir_time_machine.analysis.registry import load_registry
from reservoir_time_machine.analysis.full_field_registry import load_full_field_registry
from reservoir_time_machine.analysis.stage3_registry import load_stage3_registry
from reservoir_time_machine.analysis.recovery_intelligence import (
    default_investigation_id,
    investigation_records,
    select_investigation,
    reservoir_evolution_readiness,
)
from reservoir_time_machine.analysis.conditional_reservoir import (
    conditional_i100_locations, evaluate_conditional_window,
)
from reservoir_time_machine.analysis.technical_report import report_markdown
from reservoir_time_machine.visualization.presentation import select_entry,render_presentation
import json,re
from reservoir_time_machine.visualization.guided_report import render_page_report
from reservoir_time_machine.visualization.presentation_cache import (
    load_presentation,
    presentation_identity,
    presentation_cache_token,
)
from reservoir_time_machine.data_ingest.well_evidence import (
    acoustic_impedance_summary,
    well_evidence_summary,
    build_well_evidence_matrix,
)
from reservoir_time_machine.data_ingest.spatial_geometry import (
    conditionally_map_well_coordinates,
    sample_segy_header_geometry,
)
from reservoir_time_machine.visualization.animation import animation_figure
from reservoir_time_machine.visualization.dashboard_state import empty_dataset_state
from reservoir_time_machine.visualization.seismic_3d import volume_figure
from reservoir_time_machine.visualization.slices import (
    display_mask,
    display_coordinates,
    display_region,
    region_representative,
    display_slice,
    display_volume,
)
from reservoir_time_machine.visualization.coverage import coverage_figure
from reservoir_time_machine.visualization.coordinates import finding_header_location, header_id
from reservoir_time_machine.preprocessing.qc_artifact import load_qc_artifact
from reservoir_time_machine.visualization.theme import apply_theme, format_metric_count, page_header, registration_warning, section


st.set_page_config(page_title="Reservoir Time Machine", page_icon="🌐", layout="wide")
apply_theme()
app_header = st.empty()
with app_header.container():
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
    st.write("Expected inputs not found:")
    st.code("\n".join(state["missing_files"]) or "No required files were declared")
    st.markdown("Use the [dataset input contract](docs/data_contract.md) and the example configuration to prepare a validated input manifest.")
    st.caption("This is an empty application state. It contains no placeholder seismic, well data, anomaly map, or geological interpretation.")
    st.info("No analysis results are available until the required survey inputs are present.")
    st.stop()


@st.cache_resource(show_spinner="Loading paired Challenge 3 SEG-Y volumes…")
def cached_actual_pair(root: str, cache_dir: str | None, dataset_identity: str):
    return load_challenge3_paired_volumes(
        root,
        output_dir=cache_dir,
    )


@st.cache_resource(max_entries=1)
def cached_monitor_only_keys(path: str, dataset_identity: str):
    return np.load(path, mmap_mode="r", allow_pickle=False)


@st.cache_resource(max_entries=4)
def cached_tile_attributes(dataset_identity: str, run_id: str, sample_index: int,
                          bounds: tuple[int, int, int, int], _baseline_tile, _monitor_tile):
    return calculate_sample_attributes(_baseline_tile, _monitor_tile, sample_index, window_samples=9)


@st.cache_data(max_entries=2)
def cached_qc_artifact(path: str, dataset_identity: str, shape: tuple[int, int, int]):
    return load_qc_artifact(path, source_identity=dataset_identity, shape=shape)


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


@st.cache_resource(max_entries=8)
def cached_point_attributes(dataset_id,i,x,s,_baseline,_monitor):
    return calculate_sample_attributes(_baseline[i:i+1,x:x+1,:],_monitor[i:i+1,x:x+1,:],s,window_samples=9)

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


@st.cache_resource(max_entries=4, show_spinner="Fitting bounded preliminary reference models…")
def cached_preliminary(dataset_id, sample_index, bounds, analysis_version, artifact_root,
                       _baseline, _monitor, _missing):
    return run_preliminary(_baseline, _monitor, _missing, dataset_id=dataset_id,
                           sample=sample_index, bounds=bounds, artifact_root=artifact_root)


@st.cache_data(max_entries=2)
def cached_well_evidence(root, source_identity):
    return well_evidence_summary(root)


@st.cache_data(max_entries=2)
def cached_acoustic_impedance(root, source_identity):
    return acoustic_impedance_summary(root)


@st.cache_data(max_entries=2, show_spinner="Sampling SEG-Y headers for a conditional coordinate check…")
def cached_header_geometry(path, identity):
    return sample_segy_header_geometry(path, sample_count=1024)


@st.cache_data(max_entries=4)
def cached_well_las(path, source_identity):
    from reservoir_time_machine.data_ingest.las_loader import inspect_las
    return inspect_las(path)


@st.cache_resource(max_entries=2)
def cached_stage3_registry(path, package_token, shape, inline_ids, crossline_ids):
    return load_stage3_registry(path, expected_shape=shape,
                                inline_ids=inline_ids, crossline_ids=crossline_ids)


@st.cache_resource(max_entries=4)
def cached_finding_presentation(registry_id,run_id,finding_id,dataset_id,cache_root,cache_token):
    identity=presentation_identity(registry_id,run_id,finding_id,dataset_id)
    return load_presentation(cache_root,identity)


def _jump_to_finding(location):
    st.session_state['shared_inline'], st.session_state['shared_crossline'], st.session_state['shared_sample'] = location


def _select_stage3_finding(finding_id, location_by_id):
    """Keep the finding ID and all source-grid view coordinates synchronized."""
    location = location_by_id.get(finding_id)
    if location is None:
        return
    st.session_state["selected_finding_id"] = finding_id
    st.session_state["stage3_finding_selection"] = finding_id
    _jump_to_finding(location)


def _jump_to_stage3_selection(selection_key, location_by_id):
    selected_id = st.session_state.get(selection_key)
    _select_stage3_finding(selected_id, location_by_id)


def bounded_ml_evidence(allow_external=False):
    if full_field_registry and field_entry:
        st.markdown('### Completed full-volume screening findings')
        st.caption('Ranking is an investigation priority index, not a calibrated probability or geological classification.')
        rows=[]
        for item in full_field_registry['entries']:
            f=item['finding'];p=f['segy_location'];m=f['measurements']
            rows.append({'rank':item['rank'],'finding':item['id'],'inline ID':p['inline_id'],
                'crossline ID':p['crossline_id'],'sample index':p['sample_index'],
                'mean |difference|':m['mean_absolute_difference'],'mean NRMS (%)':m['mean_nrms_percent'],
                'edge risk':f['edge_artifact_risk'],'priority':f['investigation_priority_index']})
        st.dataframe(rows,use_container_width=True,hide_index=True)
        try:
            labels_path=Path(full_field_registry['body']['provenance']['component_labels_path'])
            labels=cached_component_labels(str(labels_path),labels_path.stat().st_size,
                labels_path.stat().st_mtime_ns,tuple(baseline.shape))
            plane=np.asarray(labels[:,:,field_finding['array_location'][2]])==int(field_finding['component_label'])
            shown=display_mask(plane,max_axis=700)
            row_stride=max(1,int(np.ceil(plane.shape[0]/700)));col_stride=max(1,int(np.ceil(plane.shape[1]/700)))
            fig=go.Figure(go.Heatmap(z=shown,x=np.asarray(crossline_header_ids)[::col_stride],
                y=np.asarray(inline_header_ids)[::row_stride],colorscale=[[0,'rgba(0,0,0,0)'],[1,'#31edc2']],
                zmin=0,zmax=1,showscale=False,name='Selected 3D component'))
            st.plotly_chart(fig.update_layout(template='plotly_dark',paper_bgcolor='#080d0e',plot_bgcolor='#101719',
                font={'color':'#fff'},title='Selected finding footprint · sample plane',
                xaxis_title='SEG-Y crossline ID',yaxis_title='SEG-Y inline ID',height=480),use_container_width=True,theme=None)
        except (OSError,ValueError,KeyError,TypeError):
            st.error('The validated component map could not be opened; finding measurements remain available.')
        st.write(field_finding['detectors'])
        st.info('Well evidence is contextual only; no well-to-seismic association is asserted.')
        return
    if guided_result is None:
        st.info('No completed registry is configured. Run the explicit bounded/scheduled analysis job; opening this page does not fit models.')
        return
    render_preliminary(guided_result, allow_external=allow_external, well_context=well_context)


def _step_sample(delta: int, maximum: int) -> None:
    st.session_state["shared_sample"] = min(maximum, max(0, st.session_state["shared_sample"] + delta))


def _reset_sample(maximum: int) -> None:
    st.session_state["shared_sample"] = maximum // 2


def _select_detection_region(region_key: str, zones, labels, score) -> None:
    selected_id = st.session_state.get(region_key)
    selected = next((zone for zone in zones if zone.label == selected_id), None)
    if selected is not None:
        location = region_representative(labels, selected, score)
        st.session_state["shared_inline"], st.session_state["shared_crossline"] = location


def _missing_overlay_trace(mask: np.ndarray, max_axis: int = 700,
                           x_name: str = "Crossline index", y_name: str = "Inline index", x_offset=0, y_offset=0):
    """Show any missing trace within each display bin, with Baseline-grid indices."""
    mask = np.asarray(mask, dtype=bool)
    row_stride = max(1, int(np.ceil(mask.shape[0] / max_axis)))
    col_stride = max(1, int(np.ceil(mask.shape[1] / max_axis)))
    return go.Heatmap(
        z=np.where(display_mask(mask, max_axis=max_axis), 1.0, np.nan),
        x=np.arange(0, mask.shape[1], col_stride)+x_offset,
        y=np.arange(0, mask.shape[0], row_stride)+y_offset,
        colorscale=[[0, "rgba(0,0,0,0)"], [1, "#ded7d5"]],
        zmin=0, zmax=1, opacity=.9, showscale=False, name="Missing trace coverage",
        hovertemplate=f"Missing trace in display bin<br>{x_name} %{{x}}<br>{y_name} %{{y}}<extra></extra>",
    )


def _render_finding_views(finding, *, baseline, monitor, missing_mask, inline_ids,
                          crossline_ids, dataset_identity, run_id="stage3-selection"):
    """Render bounded, synchronized slices around the selected component bounds."""
    il, xl, sample_index = map(int, finding["array_location"])
    extent = finding.get("extent", [[il, il], [xl, xl], [sample_index, sample_index]])
    i0 = max(0, int(extent[0][0]) - 24)
    i1 = min(baseline.shape[0], int(extent[0][1]) + 25)
    x0 = max(0, int(extent[1][0]) - 24)
    x1 = min(baseline.shape[1], int(extent[1][1]) + 25)
    s0 = max(0, int(extent[2][0]) - 24)
    s1 = min(baseline.shape[2], int(extent[2][1]) + 25)
    if i0 >= i1 or x0 >= x1 or not 0 <= sample_index < baseline.shape[2]:
        st.error("The selected finding extent is outside the loaded source grid.")
        return

    views = {
        "Sample plane": (baseline[i0:i1, x0:x1, sample_index],
                         monitor[i0:i1, x0:x1, sample_index],
                         missing_mask[i0:i1, x0:x1],
                         np.asarray(crossline_ids[x0:x1]), np.asarray(inline_ids[i0:i1]),
                         "SEG-Y Crossline ID", "SEG-Y Inline ID", "sample"),
        "Selected inline": (baseline[il, x0:x1, s0:s1].T,
                            monitor[il, x0:x1, s0:s1].T,
                            np.broadcast_to(missing_mask[il, x0:x1][None, :], (s1-s0, x1-x0)),
                            np.asarray(crossline_ids[x0:x1]), np.arange(s0, s1),
                            "SEG-Y Crossline ID", "Sample index", "inline"),
        "Selected crossline": (baseline[i0:i1, xl, s0:s1].T,
                               monitor[i0:i1, xl, s0:s1].T,
                               np.broadcast_to(missing_mask[i0:i1, xl][None, :], (s1-s0, i1-i0)),
                               np.asarray(inline_ids[i0:i1]), np.arange(s0, s1),
                               "SEG-Y Inline ID", "Sample index", "crossline"),
    }
    st.markdown(f"**SEG-Y IL {int(inline_ids[il])} · XL {int(crossline_ids[xl])} · sample index {sample_index}**")
    st.caption("All views use the selected finding. Source coordinates are preserved; sample origin/domain and physical registration are unverified.")
    tabs = st.tabs(list(views))
    for tab, (name, (bview, mview, missing_view, xs, ys, xlabel, ylabel, orientation)) in zip(tabs, views.items()):
        with tab:
            dview = np.subtract(mview, bview, out=np.full(np.shape(bview), np.nan),
                                where=np.isfinite(bview) & np.isfinite(mview))
            joint = np.concatenate((np.abs(bview[np.isfinite(bview)]),
                                    np.abs(mview[np.isfinite(mview)])))
            amp_limit = max(float(np.percentile(joint, 99)) if joint.size else 1.0, 1e-12)
            finite_diff = np.abs(dview[np.isfinite(dview)])
            diff_limit = max(float(np.percentile(finite_diff, 99)) if finite_diff.size else 1.0, 1e-12)
            if orientation == "sample":
                attrs = cached_tile_attributes(dataset_identity, run_id, sample_index,
                    (i0, i1, x0, x1), baseline[i0:i1, x0:x1, :], monitor[i0:i1, x0:x1, :])
                overlay_x0, overlay_x1 = int(crossline_ids[int(extent[1][0])]), int(crossline_ids[int(extent[1][1])])
                overlay_y0, overlay_y1 = int(inline_ids[int(extent[0][0])]), int(inline_ids[int(extent[0][1])])
            elif orientation == "inline":
                overlay_x0, overlay_x1 = int(crossline_ids[int(extent[1][0])]), int(crossline_ids[int(extent[1][1])])
                overlay_y0, overlay_y1 = int(extent[2][0]), int(extent[2][1])
            else:
                overlay_x0, overlay_x1 = int(inline_ids[int(extent[0][0])]), int(inline_ids[int(extent[0][1])])
                overlay_y0, overlay_y1 = int(extent[2][0]), int(extent[2][1])
            charts = st.columns(3)
            for col, values, title, limit, scale in (
                (charts[0], bview, "1995 Baseline", amp_limit, "RdBu"),
                (charts[1], mview, "2006 Monitor", amp_limit, "RdBu"),
                (charts[2], dview, "Monitor − Baseline", diff_limit, "RdBu"),
            ):
                shown = display_slice(values, max_axis=700)
                row_stride = max(1, int(np.ceil(values.shape[0] / 700)))
                col_stride = max(1, int(np.ceil(values.shape[1] / 700)))
                fig = go.Figure(go.Heatmap(z=shown, x=xs[::col_stride], y=ys[::row_stride],
                    colorscale=scale, zmin=-limit, zmax=limit,
                    zmid=0 if title == "Monitor − Baseline" else None,
                    colorbar={"title": "Difference" if title == "Monitor − Baseline" else "Amplitude"}))
                if title == "Monitor − Baseline":
                    fig.add_shape(type="rect", x0=overlay_x0, x1=overlay_x1,
                        y0=overlay_y0, y1=overlay_y1,
                        line={"color": "#31edc2", "width": 2})
                if missing_view is not None:
                    missing_display = display_mask(missing_view, max_axis=700)
                    fig.add_trace(go.Heatmap(z=np.where(missing_display, 1.0, np.nan),
                        x=xs[::col_stride], y=ys[::row_stride],
                        colorscale=[[0, "rgba(0,0,0,0)"], [1, "#c6c6c6"]],
                        zmin=0, zmax=1, opacity=.72, showscale=False,
                        name="Missing paired trace coverage"))
                fig.update_layout(template="plotly_dark", paper_bgcolor="#080d0e",
                    plot_bgcolor="#101719", font={"color": "white"}, title=title,
                    xaxis_title=xlabel, yaxis_title=ylabel, height=390)
                col.plotly_chart(fig, use_container_width=True, theme=None)
            if orientation == "sample":
                support = np.asarray(attrs.paired_sample_count)
                nrms = np.asarray(attrs.nrms)
                quality = st.columns(2)
                for col, values, title, maximum, palette in (
                    (quality[0], nrms, "9-sample NRMS (%)", None, "Viridis"),
                    (quality[1], support, "Paired support / 9", 9, "Viridis"),
                ):
                    col.plotly_chart(go.Figure(go.Heatmap(z=display_slice(values, max_axis=700),
                        x=np.asarray(crossline_ids[x0:x1])[::max(1, int(np.ceil(values.shape[1] / 700)))],
                        y=np.asarray(inline_ids[i0:i1])[::max(1, int(np.ceil(values.shape[0] / 700)))],
                        colorscale=palette, zmin=0, zmax=maximum,
                        colorbar={"title": title})).update_layout(template="plotly_dark",
                        paper_bgcolor="#080d0e", plot_bgcolor="#101719", font={"color": "white"},
                        title=title, xaxis_title="SEG-Y Crossline ID", yaxis_title="SEG-Y Inline ID",
                        height=330), use_container_width=True, theme=None)
                st.caption("NRMS and support are calculated on the displayed spatial crop only, from finite paired samples. The turquoise outline is the component bounding extent, not exact membership.")


cache_directory = (settings.cache_directory if settings.cache_directory is not None
                   else settings.challenge3_root / ".reservoir_time_machine_cache" / "challenge3_paired")
try:
    # Validate identity before resource lookup; changes to masks and source files invalidate reuse.
    dataset_cache_identity = paired_cache_identity(settings.challenge3_root, cache_directory)
    with st.spinner("Loading Baseline and Monitor volumes using trace-key pairing…"):
        baseline_volume, monitor_volume = cached_actual_pair(
            str(settings.challenge3_root), str(cache_directory), dataset_cache_identity,
        )
except (ImportError, RuntimeError, FileNotFoundError, MemoryError, ValueError, OSError) as exc:
    st.error(f"Actual Challenge 3 data could not be loaded: {exc}")
    st.info(
        "Check that both seismic surveys and sufficient cache storage are available."
    )
    st.stop()

baseline, monitor = baseline_volume.data, monitor_volume.data

if baseline.shape != monitor.shape:
    st.error("Loaded baseline and monitor shapes differ. Processing is disabled until geometry is reviewed.")
    st.stop()
paired_grid_available = bool(
    monitor_volume.metadata.source_metadata.get(
        "paired_grid_available", False
    )
)

capabilities = assess_capabilities(baseline_volume, monitor_volume)
physical_4d_registration_verified = capabilities["physical_4d_registration"]["available"]

for state_key, maximum in (("shared_sample", baseline.shape[2] - 1),
                          ("shared_inline", baseline.shape[0] - 1),
                          ("shared_crossline", baseline.shape[1] - 1)):
    value = st.session_state.get(state_key, maximum // 2)
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = maximum // 2
    st.session_state[state_key] = min(maximum, max(0, value))

if not paired_grid_available or not capabilities["trace_key_preliminary_screening"]["available"]:
    st.error("Paired Baseline/Monitor screening requirements are unresolved.")
    st.stop()
source_meta=monitor_volume.metadata.source_metadata
inline_header_ids=source_meta.get('inline_ids',source_meta.get('inline_ids_selected',[]))
crossline_header_ids=source_meta.get('crossline_ids',source_meta.get('crossline_ids_selected',[]))

# Load only completed, versioned results. Model fitting is never a startup side effect.
registry=None;registry_error=None;guided_result=None;guided_finding=None
registry_value=os.environ.get('RTM_FINDINGS_REGISTRY')
if registry_value is None:
    config=Path.home()/'.config/reservoir-time-machine/runtime.json'
    try:
        if config.exists():
            if config.stat().st_size>8000:raise ValueError('Runtime config too large')
            values=json.loads(config.read_text())
            if set(values)!={'findings_registry'} or not isinstance(values['findings_registry'],str):raise ValueError('Invalid runtime config')
            registry_value=values['findings_registry']
    except (OSError,ValueError,TypeError):registry_error='Invalid runtime registry configuration'
registry_path=Path(registry_value or str(Path.home()/'.reservoir_time_machine/analysis/registry.json')).expanduser()
presentation_cache_root=Path(os.environ.get(
    'RTM_PRESENTATION_CACHE',str(registry_path.parent/'presentations'))).expanduser()
@st.cache_resource(max_entries=2)
def cached_registry(path,identity,shape,payload_identity):
    return load_registry(path,identity,shape)
@st.cache_resource(max_entries=2)
def cached_full_field_registry(path,identity,shape,inline_ids,crossline_ids,payload_identity):
    return load_full_field_registry(path,dataset_identity=identity,shape=shape,
        inline_ids=inline_ids,crossline_ids=crossline_ids)
@st.cache_resource(max_entries=2)
def cached_component_labels(path,size,mtime_ns,shape):
    labels=np.load(path,mmap_mode='r',allow_pickle=False)
    if tuple(labels.shape)!=tuple(shape):raise ValueError('Full-field component map geometry mismatch')
    return labels

def select_full_field_entry(entries,offset=0):
    ids=[item['id'] for item in entries]
    if not ids:return
    if offset=='top':
        item=entries[0]
    else:
        current=st.session_state.get('registry_selection',ids[0]);index=ids.index(current) if current in ids else 0
        item=entries[(index+offset)%len(ids)]
    st.session_state['registry_selection']=item['id']
    st.session_state['selected_finding_id']=item['id']
    st.session_state['selected_preliminary_finding']={**item['finding'],'dataset_id':item['finding']['source_identity']}
    for key,value in zip(('shared_inline','shared_crossline','shared_sample'),item['finding']['array_location']):
        st.session_state[key]=value
if registry_path.is_file():
    try:
        if registry_path.stat().st_size>2_000_000:raise ValueError('Manifest too large')
        registry_bytes=registry_path.read_bytes();preview=json.loads(registry_bytes)
        run_ids=[r['run_id'] for r in preview['body']['runs']]
        if len(run_ids)>64 or any(not re.fullmatch('[a-f0-9]{64}',r) for r in run_ids):raise ValueError('Invalid manifest runs')
        artifact_stats=tuple((r,n,(registry_path.parent/'artifacts'/r/n).stat().st_size,
            (registry_path.parent/'artifacts'/r/n).stat().st_mtime_ns) for r in run_ids for n in ('complete.json','maps.npz'))
        registry=cached_registry(str(registry_path),dataset_cache_identity,tuple(baseline.shape),
            (hashlib.sha256(registry_bytes).hexdigest(),artifact_stats))
    except (ValueError,OSError,KeyError,TypeError) as exc:registry_error=type(exc).__name__
full_field_registry=None;full_field_registry_error=None;field_entry=None;field_finding=None
full_field_registry_value=os.environ.get('RTM_FULL_FIELD_REGISTRY')
if full_field_registry_value:
    field_path=Path(full_field_registry_value).expanduser().resolve()
    try:
        inline_axis=source_meta.get('inline_ids',source_meta.get('inline_ids_selected',[]))
        crossline_axis=source_meta.get('crossline_ids',source_meta.get('crossline_ids_selected',[]))
        if not field_path.is_file():
            raise FileNotFoundError('Configured full-field registry pointer is unavailable')
        if field_path.is_file():
            pointer_bytes=field_path.read_bytes()
            full_field_registry=cached_full_field_registry(str(field_path),dataset_cache_identity,
                tuple(baseline.shape),tuple(map(int,inline_axis)),tuple(map(int,crossline_axis)),
                (hashlib.sha256(pointer_bytes).hexdigest(),field_path.stat().st_size,field_path.stat().st_mtime_ns))
    except (OSError,ValueError,KeyError,TypeError) as exc:
        full_field_registry_error=type(exc).__name__
        full_field_registry=None
if full_field_registry and full_field_registry['entries']:
    registry=None;guided_result=None;guided_finding=None
    field_entries=full_field_registry['entries'];field_ids=[item['id'] for item in field_entries]
    if st.session_state.get('field_registry_identity')!=full_field_registry['id']:
        st.session_state['field_registry_identity']=full_field_registry['id']
        st.session_state['registry_selection']=field_ids[0]
    if st.session_state.get('registry_selection') not in field_ids:
        st.session_state['registry_selection']=field_ids[0]
    field_entry=next(item for item in field_entries if item['id']==st.session_state['registry_selection'])
    field_finding=field_entry['finding']
    for key,value in zip(('shared_inline','shared_crossline','shared_sample'),field_finding['array_location']):
        st.session_state[key]=value
elif full_field_registry is not None:
    st.success('Full-field results validated; no findings met the configured screening criteria.')
elif full_field_registry_error:
    st.error('Full-field findings failed validation. The prior accepted registry remains unchanged; any bounded results below are preliminary only.')
else:
    st.info('Full-field findings are not active. Any available bounded screening remains preliminary and is not a global ranking.')

# The Stage 3 assessment is a small checksummed interpretation package. Loading it
# validates only its JSON/CSV/PDF artifacts; no seismic or full-field score maps are
# re-read or recomputed during Streamlit navigation.
stage3_registry = None
stage3_registry_error = None
stage3_dir_value = os.environ.get("RTM_STAGE3_EVIDENCE_DIR")
if stage3_dir_value is None:
    default_stage3_dir = Path.home() / "rtm-scientific-qa" / "stage3-evidence"
    stage3_dir_value = str(default_stage3_dir) if default_stage3_dir.is_dir() else None
if stage3_dir_value:
    stage3_assessment_path = Path(stage3_dir_value).expanduser() / "stage3_assessment.json"
    try:
        manifest_path = stage3_assessment_path.parent / "stage3_manifest.sha256"
        package_token = tuple((p.name, p.stat().st_size, p.stat().st_mtime_ns)
                              for p in sorted(stage3_assessment_path.parent.iterdir()) if p.is_file())
        stage3_registry = cached_stage3_registry(
            str(stage3_assessment_path), package_token, tuple(baseline.shape),
            tuple(map(int, inline_header_ids)), tuple(map(int, crossline_header_ids)))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        stage3_registry_error = type(exc).__name__
stage3_investigations = investigation_records(stage3_registry) if stage3_registry else []
stage3_locations = {
    item["id"]: item["finding"]["array_location"] for item in stage3_investigations
}
selected_investigation = None
if stage3_investigations:
    stage3_ids = set(stage3_locations)
    selected_id = st.session_state.get("selected_finding_id")
    if selected_id not in stage3_ids:
        selected_id = default_investigation_id(stage3_investigations)
    st.session_state["selected_finding_id"] = selected_id
    if st.session_state.get("stage3_finding_selection") != selected_id:
        st.session_state["stage3_finding_selection"] = selected_id
    selected_investigation = select_investigation(stage3_registry, selected_id)
elif full_field_registry and field_entry:
    selected_investigation = {"id": field_finding["id"], "kind": "full_field_screening",
                              "finding": field_finding}

# A selected scientific record owns the synchronized source location. Manual
# navigation is available only on the separately named Expert diagnostics page.
if selected_investigation:
    selection_id = selected_investigation["id"]
    if st.session_state.get("last_synced_finding_id") != selection_id:
        finding = selected_investigation["finding"]
        location = finding.get("array_location", finding.get("representative"))
        if location is not None:
            _jump_to_finding(location)
        st.session_state["last_synced_finding_id"] = selection_id
if registry and registry['entries']:
    ids=[e['id'] for e in registry['entries']]
    pending=st.session_state.pop('pending_registry_selection',None)
    if pending in ids:
        st.session_state['registry_selection']=pending;select_entry(registry['entries'])
    elif st.session_state.get('registry_identity')!=registry['id']:
        st.session_state['registry_identity']=registry['id'];st.session_state['registry_selection']=ids[0];select_entry(registry['entries'])
    elif st.session_state.get('preliminary_choice') in ids:
        st.session_state['registry_selection']=st.session_state['preliminary_choice']
    entry=next(e for e in registry['entries'] if e['id']==st.session_state['registry_selection'])
    guided_result,guided_finding=entry['result'],entry['finding']
    selected_investigation = {"id": guided_finding["id"], "kind": "bounded_screening",
                              "finding": guided_finding}
    st.session_state["selected_finding_id"] = guided_finding["id"]
    if st.session_state.get("last_synced_finding_id") != guided_finding["id"]:
        location = (guided_finding["array_location"] if "array_location" in guided_finding
                    else guided_finding["representative"])
        _jump_to_finding(location)
        st.session_state["last_synced_finding_id"] = guided_finding["id"]
    if st.session_state['shared_sample']!=guided_finding['sample']:
        # Expert navigation may move off the finding; its report remains explicitly scoped.
        pass
if registry is None or not registry['entries']:
    st.session_state.pop('selected_preliminary_finding',None)
well_context=registry['body']['wells'] if registry else cached_well_evidence(str(settings.challenge3_root),
    tuple((str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in sorted((settings.challenge3_root/'well_data').rglob('*')) if p.is_file()))
discovery=registry['body']['coverage'] if registry else {'status':'No completed analysis registry'}
paired_missing=np.asarray(baseline_volume.missing_trace_mask)|np.asarray(monitor_volume.missing_trace_mask)
source_meta = monitor_volume.metadata.source_metadata
inline_header_ids = source_meta.get("inline_ids", source_meta.get("inline_ids_selected", []))
crossline_header_ids = source_meta.get("crossline_ids", source_meta.get("crossline_ids_selected", []))
try:
    monitor_only_keys = cached_monitor_only_keys(
        str(cache_directory / "monitor_only_keys.npy"), dataset_cache_identity
    )
except (OSError, ValueError):
    monitor_only_keys = None

qc_artifact_path = Path(os.environ.get(
    "RTM_QC_ARTIFACT",
    str(Path.home() / ".reservoir_time_machine/analysis/qc" /
        f"survey-qc-{dataset_cache_identity}.json"),
)).expanduser()
qc_artifact = None
qc_artifact_error = None
if qc_artifact_path.is_file():
    try:
        qc_artifact = cached_qc_artifact(str(qc_artifact_path), dataset_cache_identity, tuple(baseline.shape))
    except (OSError, ValueError, KeyError, TypeError):
        qc_artifact_error = "Survey QC artifact failed source, version, or checksum validation."

with st.sidebar:
    st.header("Workspace")
    page = st.radio("Workspace", ["Reservoir Command Center", "4D comparison", "3D discovery map", "3D explorer",
                                   "Recovery intelligence", "Change detection", "Overview and QC",
                                   "Survey transition", "Well inventory", "Evidence explanation",
                                   "Expert diagnostics"], index=0,
                    label_visibility="collapsed", key="workspace_page")
    if full_field_registry and full_field_registry['entries']:
        current=field_entry
        st.markdown(f"**Full-volume finding {current['rank']} / {len(full_field_registry['entries'])}**")
        nav=st.columns(2)
        nav[0].button('Previous finding',on_click=select_full_field_entry,args=(full_field_registry['entries'],-1),key='field_prev')
        nav[1].button('Next finding',on_click=select_full_field_entry,args=(full_field_registry['entries'],1),key='field_next')
        st.button('Return to top finding',on_click=select_full_field_entry,args=(full_field_registry['entries'],'top'),key='field_top')
    elif registry and registry['entries']:
        current = next((e for e in registry['entries'] if e['id'] == st.session_state.get('registry_selection')),
                       registry['entries'][0])
        st.markdown(f"**Finding {current['local_rank']} / {len(registry['entries'])}**")
        nav=st.columns(2)
        nav[0].button('Previous finding',on_click=select_entry,args=(registry['entries'],-1))
        nav[1].button('Next finding',on_click=select_entry,args=(registry['entries'],1))
        st.button('Return to automatic focus',on_click=select_entry,args=(registry['entries'],))
    if page == "Expert diagnostics":
        with st.expander("Expert-only manual grid navigation", expanded=True):
            sample = st.slider("Sample array index", 0, baseline.shape[2]-1, key="shared_sample")
            inline = st.slider("Inline array index", 0, baseline.shape[0]-1, key="shared_inline")
            crossline = st.slider("Crossline array index", 0, baseline.shape[1]-1, key="shared_crossline")
            color_scale = st.selectbox("Seismic colour scale", ["RdBu", "Viridis", "Cividis", "Turbo"], index=0)
            st.caption("Expert diagnostic mode only. Values are array indices; sample origin and physical registration are unverified.")
    else:
        inline = int(st.session_state.get("shared_inline", baseline.shape[0] // 2))
        crossline = int(st.session_state.get("shared_crossline", baseline.shape[1] // 2))
        sample = int(st.session_state.get("shared_sample", baseline.shape[2] // 2))
        color_scale = "RdBu"

if page != "4D comparison" and not availability["well_inventory_available"]:
    st.warning("The well-list spreadsheet is missing. Seismic-only views remain available; well context is not inferred.")

if not paired_grid_available or not capabilities["trace_key_preliminary_screening"]["available"]:
    st.error("Paired Baseline/Monitor screening requirements are unresolved.")
    st.stop()

if page != "4D comparison" and not physical_4d_registration_verified:
    registration_warning()

if full_field_registry and field_entry:
    finding=field_finding;record=field_entry['interpretation'];loc=finding['segy_location'];metrics=finding['measurements']
    st.markdown(f"## Full-volume finding {field_entry['rank']} · {finding['id']}")
    st.markdown(f"**Location** · Inline {loc['inline_id']} · Crossline {loc['crossline_id']} · Sample index {loc['sample_index']}")
    st.markdown(f"**RESULT** · Mean |difference| {metrics['mean_absolute_difference']:.4g} · NRMS {metrics['mean_nrms_percent']:.3g}% · {finding['component_voxels']:,} connected voxels")
    st.markdown(f"**DISCUSSION** · {record['discussion']}")
    st.markdown(f"**ACTION** · {record['actions'][0]['action']}")
    priority=finding.get('investigation_priority_index');priority_text=f"{priority:.3f}" if isinstance(priority,(int,float)) else 'N/A'
    st.caption(f"Priority {priority_text} (not probability) · edge risk {finding['edge_artifact_risk']:.2f} · physical registration unverified · valid coverage {full_field_registry['body']['coverage']['valid_voxels']:,}/{full_field_registry['body']['coverage']['processed_voxels']:,} voxels.")
    with st.expander('Well context, discipline actions and provenance'):
        st.markdown('\n'.join(f"- **{item['discipline']}:** {item['action']}" for item in record['actions']))
        st.write(field_entry['well_evidence'])
        st.write(record['limitations'])
        st.write(full_field_registry['body']['provenance'])

if guided_finding and [st.session_state[k] for k in ('shared_inline','shared_crossline','shared_sample')]!=guided_finding['representative']:
    st.caption('Advanced view coordinates differ from the report finding. Use Return to automatic focus to resynchronize. The report remains scoped to '+str(guided_finding['representative'])+'.')
if registry and 'source_identity' in well_context:
    current_well_sources=[{'path':str(p.relative_to(settings.challenge3_root)),'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}
        for p in sorted((settings.challenge3_root/'well_data').rglob('*')) if p.is_file()]
    if current_well_sources!=well_context['source_identity']:
        st.warning('Well sources changed after this completed analysis. The report retains its recorded source snapshot; refresh the offline registry before treating it as current well context.')

saved_finding = st.session_state.get('selected_preliminary_finding')
if saved_finding and saved_finding.get('dataset_id') == dataset_cache_identity:
    st.caption(f"Selected preliminary finding {saved_finding['id']} · source indices {saved_finding['representative']} · investigation priority only.")
    st.sidebar.button('Go to selected preliminary finding',on_click=_jump_to_finding,args=(saved_finding['representative'],))

@st.cache_data(max_entries=2)
def cached_technical_report(registry_id,_registry):
    return report_markdown(_registry)

if page not in ("Overview and QC", "Reservoir Command Center"):
    if registry:
        c=registry['body']['coverage']
        st.caption(f"Coverage {c['completed_tiles']}/{c['planned_tiles']} tiles · local ranking only")
        st.download_button('Download technical report',cached_technical_report(registry['id'],registry),file_name='reservoir-time-machine-report.md',mime='text/markdown')
    if registry is None or registry['body']['coverage']['pending_tiles']:
        with st.expander('Extend analysis'):
            st.caption('Run bounded analysis through the configured batch workflow. Opening this page does not start analysis.')
            template=Path(__file__).resolve().parent/'config/hpc/phase3_discovery.slurm'
            if template.is_file():st.download_button('Download bounded analysis template',template.read_text(),file_name='phase3_discovery.slurm')
    if guided_result is not None:
        with st.expander('Selected finding · result / discussion / action',expanded=True):
            render_page_report(guided_result,guided_finding,well_context,page,discovery,
                shared_interpretation=entry['interpretation'],inline_ids=inline_header_ids,
                crossline_ids=crossline_header_ids,grid_shape=baseline.shape[:2])
    else:
        st.info('No completed valid findings registry is available. Expert seismic views remain available; automatic ML analysis does not run on page load.')
        st.caption('Coverage: N/A · no completed findings registry is available.')

if page == "Reservoir Command Center":
    page_header("Angsi Field · 4D Reservoir Intelligence",
                "Baseline 1995 → Monitor 2006 · evidence-led surveillance and recovery investigation")

    # Run only bounded paired-array reads around source-local I-100 well markers.
    # This is a conditional physical-tie experiment, NOT a verified reservoir target.
    st.markdown("### I-100 reservoir-window hypothesis · actual 4D measurements")
    st.caption("Tested well marker and coordinate assumptions, not a mapped or verified reservoir horizon.")
    try:
        well_matrix = well_context.get("well_evidence_matrix") or build_well_evidence_matrix(well_context)
        source_baseline_file = settings.challenge3_root / "volumes" / "Challenge3_Baseline_FilteredScaled_FullStack.sgy"
        header_token = (str(source_baseline_file), source_baseline_file.stat().st_size,
                        source_baseline_file.stat().st_mtime_ns)
        baseline_header_geometry = cached_header_geometry(str(source_baseline_file), header_token)
        projected_wells = conditionally_map_well_coordinates(
            well_matrix, baseline_header_geometry,
            inline_ids=np.asarray(inline_header_ids), crossline_ids=np.asarray(crossline_header_ids))
        origin_cases = (-20.0, 0.0, 20.0)
        scenario_results = []
        for origin_ms in origin_cases:
            cases = conditional_i100_locations(
                well_matrix, projected_wells, inline_header_ids, crossline_header_ids,
                baseline.shape[2], sample_interval_ms=2.0, origin_ms=origin_ms)
            for case in cases:
                measured = evaluate_conditional_window(
                    baseline, monitor, paired_missing, case,
                    spatial_radius=2, half_window=10, max_lag_samples=4,
                    control_distance_samples=64)
                scenario_results.append((case, measured))
        if not scenario_results:
            st.info("No compatible source-local I-100 windows fall within the proposed Baseline header frame. No reservoir interval was invented.")
        else:
            st.warning("Conditional test only: Baseline header coordinates, workbook X/Y frame, sample-zero TWT, well-to-seismic tie and physical 4D registration are not independently verified.")
            def _display_number(value, places=1):
                return round(float(value), places) if value is not None and np.isfinite(value) else "—"
            st.dataframe([
                {"Well": case["well_id"], "Assumed sample origin (ms)": case["assumed_time_origin_ms"],
                 "IL": case["segy_inline_id"], "XL": case["segy_crossline_id"],
                 "Sample index": case["array_location"][2],
                 "Measured NRMS (%)": _display_number(result["observed"]["nrms_percent"]),
                 "Local off-window NRMS (%)": _display_number(result["off_window_control_nrms_median_percent"]),
                 "Best local lag (samples)": result["best_local_lag_samples"],
                 "NRMS after tested lag (%)": _display_number(result["best_lag_nrms_percent"]),
                 "Paired support (%)": _display_number(100 * result["observed"]["support_fraction"]),
                 "Comparison": result["response_vs_off_window_controls"]}
                for case, result in scenario_results
            ], hide_index=True, use_container_width=True)
            zero_origin = [(case, result) for case, result in scenario_results
                           if case["assumed_time_origin_ms"] == 0.0]
            if zero_origin:
                selected_well = st.selectbox(
                    "Inspect conditional I-100 well evidence",
                    [case["well_id"] for case, _ in zero_origin],
                    key="command_conditional_i100_well")
                focused_case, focused_result = next(
                    (case, result) for case, result in zero_origin
                    if case["well_id"] == selected_well)
                st.markdown("**Measured:** " + focused_result["interpretation"])
                st.markdown("**Next scientific test:** " + focused_result["next_test"])
                st.caption("The ±20 ms origin sensitivity cases above are not acquired vintages. Best local lag is a diagnostic, not a registered seismic correction. Off-window comparisons are not known unproduced reservoir controls.")
                with st.expander("Show 1995 / 2006 / 4D seismic at this conditional I-100 location"):
                    ci, cx, cs = focused_case["array_location"]
                    conditional_display = {
                        "array_location": [ci, cx, cs],
                        "extent": [[max(0, ci-2), min(baseline.shape[0]-1, ci+2)],
                                   [max(0, cx-2), min(baseline.shape[1]-1, cx+2)],
                                   [max(0, cs-10), min(baseline.shape[2]-1, cs+10)]]}
                    _render_finding_views(
                        conditional_display, baseline=baseline, monitor=monitor,
                        missing_mask=paired_missing, inline_ids=inline_header_ids,
                        crossline_ids=crossline_header_ids,
                        dataset_identity=dataset_cache_identity,
                        run_id="conditional-i100-" + selected_well)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        st.info(f"Bounded I-100 well/seismic experiment could not be completed: {type(exc).__name__}: {exc}")

    if stage3_registry:
        assessment = stage3_registry["assessment"]
        consensus = assessment.get("consensus", {})
        candidates = stage3_investigations
        selected = selected_investigation
        st.markdown("### WHAT IS HAPPENING IN THE RESERVOIR?")
        st.write("The completed paired survey arrays contain spatially varying seismic differences. These are observations on matched Baseline-grid trace keys; cause and physical 4D repeatability remain unverified.")
        st.markdown("### WHICH CHANGES MATTER MOST?")
        st.write("The archived Stage 2 top ten are MAD-only screening candidates. Three-detector consensus is limited and concentrated near the crossline survey boundary; it is not independent geological confirmation.")
        st.markdown("### WHERE SHOULD RECOVERY TEAMS INVESTIGATE NEXT?")
        st.write("No recovery opportunity is currently verified. The default focus is the highest-ranked QC follow-up action, which may resolve uncertainty before reservoir or production decisions.")
        metric = st.columns(4)
        metric[0].metric("Valid full-field positions", "1.604B", help="Completed Stage 1/2B report; finite paired positions.")
        metric[1].metric("Classical ML consensus voxels", f"{int(consensus.get('voxel_count', 0)):,}", help="Mahalanobis + PCA + GMM flags; Isolation Forest flags absent in this set.")
        metric[2].metric("Verified reservoir ties", "0", help="No finding has a verified well/horizon/sample association.")
        metric[3].metric("Recovery opportunities", f"{len(stage3_registry['recovery_opportunities'])}", help="Only independently supported opportunities qualify.")
        st.warning("Trace-key pairing is available on the Baseline grid. Physical 4D registration is not yet verified. Interpret paired attributes as preliminary screening only. Sample origin and well-to-seismic tie remain unresolved.")

        if selected:
            finding = selected["finding"]
            if selected["kind"] == "qc_follow_up":
                st.markdown(f"## Priority QC investigation {selected['queue_rank']} · {selected['id']}")
                st.caption("This ranks a validation action, not a hydrocarbon location or recovery probability.")
                st.markdown("**What happened?** " + str(selected["observation"]))
                st.markdown("**Why it matters:** the stored arrays differ numerically, but repeatability/registration and processing explanations must be resolved before geological attribution.")
                st.markdown(f"**Seismic observation confidence:** {selected['observation_confidence']}")
                st.markdown(f"**Geological interpretation confidence:** {selected['geological_confidence']}")
                if selected["alternatives"]:
                    st.markdown("**Alternative explanations:** " + " ".join(selected["alternatives"]))
                st.markdown("**Recommended next actions**")
                for discipline, action in selected["actions"].items():
                    st.markdown(f"- **{discipline.replace('_', ' ').title()}:** {action}")
                st.info(selected["recovery_chance_of_success"])
            else:
                st.markdown(f"## Selected historical screening record · {selected['id']}")
                st.caption("Historical Stage 2 order and Stage 3 QC evidence are retained; this is not a recovery-target ranking.")
                st.markdown("**Observed:** " + str(selected["observation"]))
                st.markdown(f"**Observation confidence:** {selected['observation_confidence']}")
                st.markdown(f"**Geological interpretation:** {selected['geological_confidence']}")
            action_ids = {item["finding"]["id"] for item in stage3_registry["qc_follow_up_actions"]}
            selectable = [item for item in candidates if item["id"] in action_ids]
            if selectable:
                st.markdown("#### Other ranked QC actions")
                buttons = st.columns(min(4, len(selectable)))
                for index, item in enumerate(selectable[:4]):
                    buttons[index].button(
                        f"Action {item['queue_rank']} · {item['id'][-6:]}",
                        key=f"command_action_{item['id']}",
                        on_click=_select_stage3_finding,
                        args=(item["id"], stage3_locations),
                        use_container_width=True,
                    )
            _render_finding_views(finding, baseline=baseline, monitor=monitor,
                missing_mask=paired_missing, inline_ids=inline_header_ids,
                crossline_ids=crossline_header_ids, dataset_identity=dataset_cache_identity,
                run_id=f"command-{selected['id']}")
        else:
            st.info("No validated investigation record is present. The homepage will not choose a midpoint or invent a target.")
        with st.expander("Eight-method and evidence status"):
            st.dataframe([
                {"Method": "Mahalanobis distance", "Angsi status": "Full-field score maps completed; historical top ten has zero flags"},
                {"Method": "PCA reconstruction", "Angsi status": "Full-field score maps completed; historical top ten has zero flags"},
                {"Method": "Isolation Forest", "Angsi status": "Full-field score maps completed; no threshold flags in the historical top ten or consensus set"},
                {"Method": "Gaussian Mixture Model", "Angsi status": "Full-field score maps completed; historical top ten has zero flags"},
                {"Method": "MAD screening", "Angsi status": "Completed statistical screen; all ten historical findings are MAD-only"},
                {"Method": "Bayesian evidence fusion", "Angsi status": "Code available; no validated Angsi likelihood calibration or posterior"},
                {"Method": "3D convolutional autoencoder", "Angsi status": "Architecture/trainer available; no trained or validated Angsi checkpoint"},
                {"Method": "Siamese 3D CNN", "Angsi status": "Architecture/trainer available; no trained or validated Angsi checkpoint"},
                {"Method": "Edge and coverage QC", "Angsi status": "Completed Stage 2/3 QC; consensus remains edge-sensitive"},
            ], use_container_width=True, hide_index=True)
    elif full_field_registry and field_finding:
        st.info("Validated full-field findings registry loaded. The Stage 3 QC action package is not configured; this record is shown as seismic screening, not recovery intelligence.")
        bounded_ml_evidence()
    elif registry and guided_result is not None:
        st.info("Only bounded preliminary results are available. They are not global priorities.")
        render_page_report(guided_result, guided_finding, well_context, page, discovery,
            shared_interpretation=entry["interpretation"], inline_ids=inline_header_ids,
            crossline_ids=crossline_header_ids, grid_shape=baseline.shape[:2])
    else:
        st.info("No completed, validated findings registry is configured. This page does not select arbitrary seismic slices or run full-volume analysis.")

elif page == "Overview and QC":
    page_header("Survey overview & QC", "Coverage, measured data quality, and preliminary finding evidence.")
    section("Scientific readiness")
    st.markdown(
        "- **Trace-key pairing · VERIFIED:** common Baseline/Monitor inline-crossline keys are paired; Monitor-only keys are excluded from paired attributes.\n"
        "- **Physical 4D registration · N/A:** trace-key pairing does not prove spatial or event alignment; validate geometry, timing, and repeatability.\n"
        "- **Sample time/depth · N/A:** the sample origin and domain are unresolved; sample labels remain array indices.\n"
        "- **Well tie · N/A:** shared CRS, datum, trajectory, and time-depth association are not established; validate these before assigning wells to findings.\n"
        "- **Ranking · LOCAL:** only completed tiles are comparable within their own analysis run; survey-wide priority is not established."
    )

    if registry:
        c=registry['body']['coverage']
        st.caption(f"Completed analysis: {c['completed_tiles']}/{c['planned_tiles']} tiles · {c['eligible_center_cells']:,} eligible sample centers · tile-local ranks")
        st.download_button('Download technical report',cached_technical_report(registry['id'],registry),file_name='reservoir-time-machine-report.md',mime='text/markdown')
    else:
        st.caption("Completed analysis coverage: N/A · no valid registry loaded.")

    section("Survey geometry and trace counts")
    missing = baseline_volume.missing_trace_mask
    exact_missing = int(np.asarray(missing).sum(dtype=np.int64)) if missing is not None else None
    grid_cells = int(baseline.shape[0] * baseline.shape[1])
    paired_count = grid_cells - exact_missing if exact_missing is not None else source_meta.get("paired_trace_count")
    monitor_only_count = (int(source_meta['monitor_only_trace_count'])
                          if monitor_only_keys is None and source_meta.get('monitor_only_trace_count') is not None
                          else int(len(monitor_only_keys)) if monitor_only_keys is not None else None)
    monitor_count = int(paired_count + monitor_only_count) if paired_count is not None else None
    source_pair_count = source_meta.get("paired_trace_count")
    source_baseline_count=source_meta.get('baseline_trace_count')
    source_monitor_count=source_meta.get('monitor_trace_count')
    source_monitor_only_count=source_meta.get('monitor_only_trace_count')
    count_mismatch = any((recorded is not None and calculated is not None and int(recorded)!=int(calculated))
        for recorded,calculated in ((source_pair_count,paired_count),(source_baseline_count,paired_count),
            (source_monitor_count,monitor_count),(source_monitor_only_count,monitor_only_count)))
    if count_mismatch:
        st.error("Cache trace-count metadata conflicts with the retained missing-trace mask; coverage interpretation is paused.")
    count_cols=st.columns(3)
    for col,label,count in zip(count_cols,("Paired Baseline-grid traces","Missing Baseline-grid cells","Monitor-only traces"),
                               (paired_count,exact_missing,monitor_only_count)):
        shown=format_metric_count(int(count)) if count is not None else "N/A"
        help_text=f"Exact count: {int(count):,}" if count is not None else "Exact count unavailable"
        col.metric(label,shown,help=help_text)
    cols = st.columns(3)
    cols[0].metric("Monitor traces", format_metric_count(monitor_count) if monitor_count is not None else "N/A",
                   help=f"Paired traces + Monitor-only traces = {monitor_count:,}" if monitor_count is not None else "Exact count unavailable")
    cols[1].metric("Inline header IDs", f"{min(inline_header_ids):,}–{max(inline_header_ids):,}" if len(inline_header_ids) else "N/A")
    cols[2].metric("Crossline header IDs", f"{min(crossline_header_ids):,}–{max(crossline_header_ids):,}" if len(crossline_header_ids) else "N/A")
    st.caption(f"Grid {baseline.shape[0]:,} inline × {baseline.shape[1]:,} crossline × {baseline.shape[2]:,} samples. Sample interval {baseline_volume.metadata.sample_interval:g} {baseline_volume.metadata.sample_interval_units}; physical sample origin/domain N/A.")

    section("Survey-wide Baseline / Monitor QC")
    if qc_artifact is None:
        if qc_artifact_error:
            st.error(qc_artifact_error)
        st.info("Full-survey finite-sample and amplitude QC: N/A. No matching versioned offline QC artifact is available; dashboard startup does not scan either seismic volume.")
        st.caption("Build this product with scripts/build_survey_qc.py in a scheduled SLURM allocation; the paired cache is read-only.")
    else:
        def render_qc_metrics(title, report):
            st.markdown(f"**{title}**")
            cols=st.columns(4)
            values=(('Finite sample support',f"{report['finite_voxels']:,} / {report['voxel_count']:,}"),
                    ('Amplitude range',f"{report['amplitude_min']:.4g} to {report['amplitude_max']:.4g}"),
                    ('Amplitude RMS',f"{report['amplitude_rms']:.4g}"),
                    ('Zero-energy traces',f"{report['zero_energy_trace_count']:,}"))
            for col,(label,value) in zip(cols,values):col.metric(label,value)
        render_qc_metrics("Baseline",qc_artifact['baseline'])
        render_qc_metrics("Monitor on paired Baseline grid",qc_artifact['monitor_on_paired_baseline_grid'])
        st.caption(f"Monitor-only traces are summarized separately: RMS {qc_artifact['monitor_only']['amplitude_rms']:.4g}. "
                   f"Paired finite-sample support: {qc_artifact['paired_sample_support']['valid_paired_samples']:,}/"
                   f"{qc_artifact['paired_sample_support']['potential_paired_samples']:,}. Source identity and artifact checksum verified.")
        st.caption("Amplitude RMS comparisons are descriptive only; no repeatability threshold or physical registration is established.")

    section("Completed-tile seismic attributes")
    if guided_result is not None:
        i0,i1,x0,x1=guided_result['bounds']
        attrs=cached_tile_attributes(dataset_cache_identity,guided_result['run_id'],guided_result['sample'],
            tuple(guided_result['bounds']),baseline[i0:i1,x0:x1,:],monitor[i0:i1,x0:x1,:])
        valid=np.asarray(guided_result['valid'],dtype=bool).copy()
        valid &= np.isfinite(attrs.signed_difference) & np.isfinite(attrs.absolute_difference) & np.isfinite(attrs.nrms)
        eligible=int(valid.sum())
        if eligible:
            charts=st.columns(3)
            panels=(("Signed Monitor − Baseline","Amplitude difference (dataset units)",attrs.signed_difference),
                    ("Absolute amplitude difference","Absolute difference (dataset units)",attrs.absolute_difference),
                    ("Windowed NRMS","NRMS (%)",attrs.nrms))
            for col,(title,x_title,values) in zip(charts,panels):
                fig=go.Figure(go.Histogram(x=np.asarray(values)[valid],nbinsx=36,
                    marker_color="#31edc2",name=title,showlegend=False))
                fig.update_layout(template="plotly_dark",paper_bgcolor="#080d0e",plot_bgcolor="#101719",
                    font={"color":"#ffffff","size":14},title=title,xaxis_title=x_title,
                    yaxis_title="Eligible tile cells",height=390,margin={"l":45,"r":12,"t":60,"b":75})
                col.plotly_chart(fig,use_container_width=True,theme=None)
            min_support=int(np.asarray(attrs.paired_sample_count)[valid].min())
            max_support=int(np.asarray(attrs.paired_sample_count)[valid].max())
            st.caption(f"Population: {eligible:,} eligible cells · sample index {guided_result['sample']} · source bounds inline {i0}:{i1}, crossline {x0}:{x1} · 9-sample centered RMS/NRMS window, clipped at sample-axis ends · finite paired support {min_support}–{max_support} samples. Amplitude units are unspecified in source metadata.")
        else:
            st.error("The selected completed tile has no finite paired attribute values for its QC plots.")
    else:
        st.info("Completed-tile signed difference, absolute difference, and NRMS: N/A until a valid analysis tile is loaded.")

    section("Trace coverage by SEG-Y header ID")
    if missing is None:
        st.error("Trace coverage figure unavailable: missing-trace mask is absent.")
    elif count_mismatch:
        st.error("Trace coverage figure withheld because mask and cache trace counts disagree.")
    elif monitor_only_keys is None:
        st.error("Trace coverage figure unavailable: Monitor-only trace-key coordinates could not be loaded.")
    else:
        try:
            coverage_fig,coverage_summary=coverage_figure(missing,inline_header_ids,crossline_header_ids,
                monitor_only_keys,max_axis=700)
            expected_monitor_only=source_meta.get('monitor_only_trace_count')
            if expected_monitor_only is not None and int(expected_monitor_only)!=coverage_summary['monitor_only_trace_count']:
                raise ValueError("Monitor-only key count disagrees with cache metadata")
            st.plotly_chart(coverage_fig,use_container_width=True,theme=None)
            st.caption(f"Exact source counts: {coverage_summary['paired_grid_trace_count']:,} paired grid cells · "
                f"{coverage_summary['missing_baseline_grid_count']:,} missing Baseline cells · "
                f"{coverage_summary['monitor_only_trace_count']:,} Monitor-only traces. "
                f"{coverage_summary['monitor_only_on_missing_baseline_cells']:,} Monitor-only keys coincide with missing Baseline cells; "
                f"{coverage_summary['monitor_only_trace_count']-coverage_summary['monitor_only_mapped_to_baseline_axes']:,} lie outside the Baseline header-axis grid. "
                "Red bins contain at least one missing Baseline trace; gold markers are separate Monitor-only keys. Exact counts are independent of display pooling.")
        except (TypeError,ValueError,RuntimeError) as exc:
            st.error(f"Trace coverage figure could not be constructed: {type(exc).__name__}. Exact trace counts above remain available.")

    if registry and guided_result is not None:
        section("Automatically detected candidates · local ranks")
        for item in registry['entries'][:5]:
            f=item['finding'];m=f['metrics'];loc=finding_header_location(f['representative'],inline_header_ids,crossline_header_ids,grid_shape=baseline.shape[:2])
            line=(f"Inline {loc['inline_header_id'] if loc['inline_header_id'] is not None else 'N/A'} · "
                  f"Crossline {loc['crossline_header_id'] if loc['crossline_header_id'] is not None else 'N/A'} · "
                  f"Sample index {loc['sample_index']:,}")
            st.markdown(f"**Local rank {item['local_rank']} · {line}**")
            st.caption(f"{m['cells']:,} connected cells · signed difference {m['mean_signed_difference']:.4g} · absolute difference {m['mean_absolute_difference']:.4g} · NRMS {m['mean_nrms_percent']:.2f}% · min support {m['minimum_paired_support']}.")
        with st.expander('Finding report · RESULT / DISCUSSION / ACTION',expanded=True):
            render_page_report(guided_result,guided_finding,well_context,page,discovery,
                shared_interpretation=entry['interpretation'],inline_ids=inline_header_ids,
                crossline_ids=crossline_header_ids,grid_shape=baseline.shape[:2])
    else:
        st.info("No eligible candidate is available from completed analysis coverage.")
    if registry is None or registry['body']['coverage']['pending_tiles']:
        with st.expander('Extend analysis'):
            st.caption('Run bounded analysis through the configured HPC workflow. Opening this page does not start analysis.')
            template=Path(__file__).resolve().parent/'config/hpc/phase3_discovery.slurm'
            if template.is_file():st.download_button('Download bounded analysis template',template.read_text(),file_name='phase3_discovery.slurm')
elif page == "3D discovery map":
    page_header("3D investigation map", "Click a validated finding marker to synchronize the selected investigation across the workspace.")
    st.caption("This map shows representative component locations and their bounding boxes in Baseline-grid array coordinates. It is not a geographic CRS or a reconstructed reservoir surface. Only the Stage 3 ten historical screening records are mapped; the full 260,401-candidate catalogue is not being plotted in the browser.")
    map_records = stage3_registry["historical_findings"] if stage3_registry else []
    if not map_records and full_field_registry:
        map_records = [item["finding"] for item in full_field_registry["entries"]]
    if map_records:
        action_ids = {item["finding"]["id"] for item in (stage3_registry or {}).get("qc_follow_up_actions", [])}
        locations = {item["id"]: item["array_location"] for item in map_records}
        custom = []
        colors = []
        symbols = []
        for item in map_records:
            array_location = item["array_location"]
            custom.append([
                item["id"], int(item.get("inline_id", inline_header_ids[array_location[0]])),
                int(item.get("crossline_id", crossline_header_ids[array_location[1]])),
                int(array_location[2]), int(item.get("component_voxels", 0)),
                float(item.get("edge_risk", item.get("edge_artifact_risk", 0.0))),
                int(item.get("detector_threshold_flagged_voxels", 0)),
            ])
            colors.append(float(item.get("edge_risk", item.get("edge_artifact_risk", 0.0))))
            symbols.append("diamond" if item["id"] in action_ids else "circle")
        fig = go.Figure(go.Scatter3d(
            x=[item["array_location"][1] for item in map_records],
            y=[item["array_location"][0] for item in map_records],
            z=[item["array_location"][2] for item in map_records],
            customdata=custom, mode="markers+text",
            text=[f"{item['rank']}" for item in map_records], textposition="top center",
            marker={"size": 8, "color": colors, "colorscale": "Turbo", "cmin": 0, "cmax": 1,
                    "colorbar": {"title": "Edge risk"}, "symbol": symbols,
                    "line": {"color": "#f4ffff", "width": 1}},
            hovertemplate=("%{customdata[0]}<br>SEG-Y IL %{customdata[1]} · XL %{customdata[2]}"
                           "<br>Sample index %{customdata[3]}<br>Component voxels %{customdata[4]}"
                           "<br>Edge risk %{customdata[5]:.2f}<br>Classical detector flags %{customdata[6]}"
                           "<extra></extra>"),
            name="Stage 3 finding locations"))
        ni, nx, ns = baseline.shape
        bounds = [(x, y, z) for x in (0, nx - 1) for y in (0, ni - 1) for z in (0, ns - 1)]
        boundary_edges = ((0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7))
        edge_x, edge_y, edge_z = [], [], []
        for first, second in boundary_edges:
            edge_x.extend((bounds[first][0], bounds[second][0], None))
            edge_y.extend((bounds[first][1], bounds[second][1], None))
            edge_z.extend((bounds[first][2], bounds[second][2], None))
        fig.add_trace(go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, mode="lines",
            line={"color": "#7b9293", "width": 2}, name="Baseline array envelope",
            hoverinfo="skip"))
        show_missing = st.checkbox("Overlay paired-grid bins masked by source validity", value=False,
                                   key="stage3_map_missing_coverage")
        if show_missing:
            sampled_missing = display_mask(paired_missing, max_axis=180)
            stride_i = int(np.ceil(ni / sampled_missing.shape[0]))
            stride_x = int(np.ceil(nx / sampled_missing.shape[1]))
            miss_i, miss_x = np.nonzero(sampled_missing)
            fig.add_trace(go.Scatter3d(
                x=np.minimum(miss_x * stride_x + stride_x // 2, nx - 1),
                y=np.minimum(miss_i * stride_i + stride_i // 2, ni - 1),
                z=np.full(miss_i.shape, int(st.session_state.get("shared_sample", 0))),
                mode="markers", marker={"size": 2, "color": "#c2c9c9", "opacity": .65},
                name="Display-binned masked paired-grid cells",
                hovertemplate="Masked Baseline-grid bin<br>array IL %{y} · array XL %{x}<extra></extra>"))
        selected = next((item for item in map_records
                         if item["id"] == st.session_state.get("selected_finding_id")), map_records[0])
        extent = selected.get("extent")
        if extent and len(extent) == 3:
            il0, il1 = map(int, extent[0]); xl0, xl1 = map(int, extent[1]); s0, s1 = map(int, extent[2])
            corners = [(x, y, z) for x in (xl0, xl1) for y in (il0, il1) for z in (s0, s1)]
            edges = ((0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7))
            bx, by, bz = [], [], []
            for a, b in edges:
                bx.extend((corners[a][0], corners[b][0], None))
                by.extend((corners[a][1], corners[b][1], None))
                bz.extend((corners[a][2], corners[b][2], None))
            fig.add_trace(go.Scatter3d(x=bx, y=by, z=bz, mode="lines",
                line={"color": "#31edc2", "width": 5}, name="Selected extent bounding box",
                hoverinfo="skip"))
        fig.update_layout(template="plotly_dark", paper_bgcolor="#080d0e", plot_bgcolor="#101719",
            font={"color": "white"}, height=640, margin={"l": 0, "r": 0, "t": 50, "b": 0},
            scene={"xaxis_title": "Crossline array index", "yaxis_title": "Inline array index",
                   "zaxis_title": "Sample array index", "aspectmode": "data"},
            legend={"orientation": "h"})
        event = st.plotly_chart(fig, use_container_width=True, theme=None,
            config={"scrollZoom": True, "displaylogo": False},
            on_select="rerun", selection_mode="points", key="stage3_discovery_map")
        points = getattr(getattr(event, "selection", None), "points", []) or []
        if points:
            point_index = points[-1].get("point_index", points[-1].get("pointIndex"))
            curve_index = points[-1].get("curve_number", points[-1].get("curveNumber", 0))
            if curve_index == 0 and isinstance(point_index, int) and 0 <= point_index < len(map_records):
                selected_id = map_records[point_index]["id"]
                _select_stage3_finding(selected_id, locations)
                selected = next(item for item in map_records if item["id"] == selected_id)
        chosen = selected
        st.markdown(f"**Selected {chosen['id']}** · SEG-Y IL {int(chosen.get('inline_id', inline_header_ids[chosen['array_location'][0]]))} · XL {int(chosen.get('crossline_id', crossline_header_ids[chosen['array_location'][1]]))} · sample index {int(chosen['array_location'][2])}")
        st.caption(f"Bounding extent only: inline {chosen['extent'][0]} · crossline {chosen['extent'][1]} · sample index {chosen['extent'][2]}. Exact component-membership voxels are not exposed by the validated Stage 3 package.")
        _render_finding_views(chosen, baseline=baseline, monitor=monitor,
            missing_mask=paired_missing, inline_ids=inline_header_ids,
            crossline_ids=crossline_header_ids, dataset_identity=dataset_cache_identity,
            run_id=f"3d-map-{chosen['id']}")
    else:
        st.info("No validated finding locations are available for a 3D map. No arbitrary voxel cloud is generated.")
elif page == "3D explorer":
    page_header("3D seismic explorer", "Linked inline, crossline, and sample sections intersect at the shared selected location.")
    if selected_investigation is None:
        st.info("No validated finding is selected. The application will not open arbitrary midpoint seismic slices. Load a completed finding registry or use Expert diagnostics for explicitly manual inspection.")
        st.stop()
    st.caption("Drag to rotate; choose Pan in the Plotly mode bar to pan; use scroll or mode bar controls to zoom and reset the camera. Planes are display-strided; missing trace bins are conservatively shown as gaps; source arrays and analytical data remain full resolution.")
    baseline_display, monitor_display = display_volume(baseline, 36), display_volume(monitor, 36)
    limit = max(float(np.nanpercentile(np.abs(baseline_display), 99)), float(np.nanpercentile(np.abs(monitor_display), 99)))
    left, right = st.columns(2)
    with left:
        st.plotly_chart(volume_figure(baseline, "Baseline · orthogonal sections", inline=inline,
            crossline=crossline, sample=sample, missing_trace_mask=baseline_volume.missing_trace_mask,
            colorscale=color_scale, value_range=(-limit, limit),finding_context=(guided_result,guided_finding) if guided_finding else None),
            use_container_width=True, config={"scrollZoom": True, "displaylogo": False,
                "modeBarButtonsToAdd": ["resetCameraDefault3d"]}, theme=None)
    with right:
        st.plotly_chart(volume_figure(monitor, "Monitor · orthogonal sections", inline=inline,
            crossline=crossline, sample=sample, missing_trace_mask=monitor_volume.missing_trace_mask,
            colorscale=color_scale, value_range=(-limit, limit),finding_context=(guided_result,guided_finding) if guided_finding else None),
            use_container_width=True, config={"scrollZoom": True, "displaylogo": False,
                "modeBarButtonsToAdd": ["resetCameraDefault3d"]}, theme=None)
    inline_ids = monitor_volume.metadata.source_metadata.get("inline_ids", monitor_volume.metadata.source_metadata.get("inline_ids_selected", []))
    crossline_ids = monitor_volume.metadata.source_metadata.get("crossline_ids", monitor_volume.metadata.source_metadata.get("crossline_ids_selected", []))
    coordinate_metrics = st.columns(3)
    inline_id=header_id(inline,inline_ids,expected_length=baseline.shape[0])
    crossline_id=header_id(crossline,crossline_ids,expected_length=baseline.shape[1])
    coordinate_metrics[0].metric("Inline header ID", f"{inline_id:,}" if inline_id is not None else "N/A",
                                 help=f"Array index: {inline:,}")
    coordinate_metrics[1].metric("Crossline header ID", f"{crossline_id:,}" if crossline_id is not None else "N/A",
                                 help=f"Array index: {crossline:,}")
    coordinate_metrics[2].metric("Sample index", f"{sample:,}")
    st.caption("Header IDs are shown when supplied; sample remains an array index because sample origin and physical domain are unverified.")
elif page == "4D comparison":
    app_header.empty()
    st.markdown("### 4D comparison · Reservoir Time Machine")
    registration_status = (
        "Physical 4D registration is not yet verified."
        if not physical_4d_registration_verified
        else "Physical 4D registration meets the declared capability requirements."
    )
    if not physical_4d_registration_verified:
        registration_warning()
    else:
        st.info(registration_status)
    st.caption("Differences do not establish a geological cause or remaining hydrocarbons.")
    if selected_investigation is None:
        st.info("No validated finding is selected. The application will not open an arbitrary survey midpoint. Load a completed finding registry or use Expert diagnostics for manual inspection.")
        st.stop()
    view = st.radio("Synchronized section", ["Sample plane", "Inline section", "Crossline section"], horizontal=True)
    st.caption("The selected investigation supplies the location. Array indices remain available in the evidence panel; sample origin and physical time/depth are unverified.")
    point_attrs=cached_point_attributes(dataset_cache_identity,inline,crossline,sample,baseline,monitor)
    i0,i1=max(0,inline-48),min(baseline.shape[0],inline+49)
    x0,x1=max(0,crossline-48),min(baseline.shape[1],crossline+49)
    s0,s1=max(0,sample-48),min(baseline.shape[2],sample+49)
    if view == "Sample plane":
        bview,mview=baseline[i0:i1,x0:x1,sample],monitor[i0:i1,x0:x1,sample]
        missing_view=paired_missing[i0:i1,x0:x1];x_title,y_title="SEG-Y Crossline ID","SEG-Y Inline ID"
        xs,ys=np.asarray(crossline_header_ids[x0:x1]),np.asarray(inline_header_ids[i0:i1])
    elif view == "Inline section":
        bview,mview=baseline[inline,x0:x1,s0:s1].T,monitor[inline,x0:x1,s0:s1].T
        missing_view=np.broadcast_to(paired_missing[inline,x0:x1],bview.shape);x_title,y_title="SEG-Y Crossline ID","Sample index"
        xs,ys=np.asarray(crossline_header_ids[x0:x1]),np.arange(s0,s1)
    else:
        bview,mview=baseline[i0:i1,crossline,s0:s1].T,monitor[i0:i1,crossline,s0:s1].T
        missing_view=np.broadcast_to(paired_missing[i0:i1,crossline],bview.shape);x_title,y_title="SEG-Y Inline ID","Sample index"
        xs,ys=np.asarray(inline_header_ids[i0:i1]),np.arange(s0,s1)
    dview=mview-bview
    finite=np.concatenate((np.abs(bview[np.isfinite(bview)]),np.abs(mview[np.isfinite(mview)])))
    shared_limit=max(float(np.percentile(finite,99)) if finite.size else 1.,1e-12)
    finite_difference=np.abs(dview[np.isfinite(dview)])
    difference_limit=max(float(np.percentile(finite_difference,99)) if finite_difference.size else 1.,1e-12)
    st.caption('Display context is bounded to at most 97×97 source cells around the selected point; RMS/NRMS use only its finite paired 9-sample trace window. No full-plane attributes are recomputed on page load.')
    panel = st.columns(3)
    for col, values, title, scale, limit_value in ((panel[0], bview, "BASELINE", color_scale, shared_limit),
            (panel[1], mview, "MONITOR", color_scale, shared_limit),
            (panel[2], dview, "MONITOR − BASELINE", "RdBu", difference_limit)):
        shown = display_slice(values, max_axis=700)
        row_stride = max(1, int(np.ceil(values.shape[0] / 700)))
        col_stride = max(1, int(np.ceil(values.shape[1] / 700)))
        section_fig = go.Figure(go.Heatmap(z=shown, x=xs, y=ys, colorscale=scale, zmin=-limit_value,
            zmax=limit_value, zmid=0, colorbar={"title": "Amplitude" if title != "MONITOR − BASELINE" else "Difference"}))
        if missing_view is not None:
            missing_display = display_mask(missing_view, max_axis=700)
            section_fig.add_trace(go.Heatmap(z=np.where(missing_display, 1.0, np.nan),
                x=xs[::col_stride], y=ys[::row_stride],
                colorscale=[[0, "rgba(0,0,0,0)"], [1, "#c6c6c6"]],
                zmin=0, zmax=1, opacity=.72, showscale=False,
                name="Missing paired trace coverage"))
        col.plotly_chart(section_fig.update_layout(
            title=title, xaxis_title=x_title, yaxis_title=y_title, height=540), use_container_width=True)
    inline_id = int(inline_header_ids[inline]); crossline_id = int(crossline_header_ids[crossline])
    section(f"Selected investigation · SEG-Y IL {inline_id} · XL {crossline_id} · Sample index {sample}")
    base_amp, mon_amp = float(baseline[inline, crossline, sample]), float(monitor[inline, crossline, sample])
    metric_data = (
        ("Baseline amplitude",f"{base_amp:.5g}" if np.isfinite(base_amp) else "N/A"),
        ("Monitor amplitude",f"{mon_amp:.5g}" if np.isfinite(mon_amp) else "N/A"),
        ("Signed difference",f"{mon_amp-base_amp:.5g}" if np.isfinite(base_amp+mon_amp) else "N/A"),
        ("Baseline RMS · 9 samples",f"{point_attrs.rms_baseline[0,0]:.5g}" if np.isfinite(point_attrs.rms_baseline[0,0]) else "N/A"),
        ("Monitor RMS · 9 samples",f"{point_attrs.rms_monitor[0,0]:.5g}" if np.isfinite(point_attrs.rms_monitor[0,0]) else "N/A"),
        ("Windowed NRMS",f"{point_attrs.nrms[0,0]:.2f}%" if np.isfinite(point_attrs.nrms[0,0]) else "N/A"),
    )
    for start in (0,3):
        row=st.columns(3)
        for col,(label,value) in zip(row,metric_data[start:start+3]):
            col.metric(label,value)
    st.caption(f"Window support at selected cell: {int(point_attrs.paired_sample_count[0,0])} finite paired samples; the requested 9-sample window is clipped at survey ends. Zero support is missing evidence, not zero energy.")
    with st.expander("Evidence · Seismic / Well / ML / AI", expanded=False):
        st.markdown("#### Seismic evidence")
        st.write("Available seismic sources: paired Baseline and Monitor amplitudes, signed difference, windowed RMS and NRMS. Missing selected-point values are shown explicitly in the metrics.")
        st.caption("RMS and NRMS always use the selected point's centered sample window, including when viewing an inline or crossline section.")
        st.caption("Baseline and Monitor share a symmetric 99th-percentile amplitude scale; difference uses its own symmetric scale. Plots are downsampled for display; metrics use full-resolution source cells.")
        if missing_view is not None:
            st.caption("Gray overlays mark display bins containing at least one missing paired-grid trace. Full-resolution attributes exclude the actual missing source cells; valid cells in the same visual bin remain included.")
        st.markdown("**Signed difference** = Monitor − Baseline at the selected grid cell and sample. It retains sign; it does not identify a cause.")
        st.markdown("**Windowed NRMS** = 200 × RMS(Monitor − Baseline) / (RMS(Baseline) + RMS(Monitor)) over a centered 9-sample window, clipped at volume ends and calculated from finite paired samples. Higher values indicate greater local dissimilarity; NRMS is not a probability or registration correction.")
        st.markdown("**Limit:** equal grid keys establish correspondence only. Without physical 4D registration, differences remain preliminary screening evidence.")
        st.caption(registration_status)
        st.markdown("#### Well evidence")
        if availability["well_inventory_available"]:
            st.write("Well-list spreadsheet available. A verified well-to-seismic tie for this selected point is unavailable; inspect source inventory in Well inventory.")
        else:
            st.write("Unavailable: well-list spreadsheet is missing. No well context is inferred.")
        st.caption("Well coordinates, units, datums, CRS and time-depth correspondence are not verified by this page.")
        st.markdown("#### ML evidence")
        if saved_finding and saved_finding.get('dataset_id') == dataset_cache_identity:
            st.write(f"Saved preliminary finding {saved_finding['id']} has bounded detector evidence in Change detection and Evidence explanation. It is not a validated reservoir prediction or calibrated probability for this point.")
        else:
            st.write("Unavailable on this page: no validated ML prediction or calibrated confidence is attached to the selected point. RMS and NRMS are seismic attributes, not ML probabilities.")
        st.markdown("#### AI evidence")
        st.write("Evidence explanation provides local grounded summaries for bounded findings. External Gemma interpretation requires explicit authorization there; this page makes no external request or geological inference.")

elif page == "Change detection":
    page_header("Change detection", "Completed full-field screening evidence and model disagreement.")
    if stage3_registry:
        st.caption("The original Stage 2 ranking is preserved as historical screening. It is not a recovery ranking.")
        st.markdown("- All ten archived priority findings are MAD-only.\n- None has threshold-flagged voxels from the four classical detectors.\n- 3,563 voxels have Mahalanobis + PCA + GMM flags; this set is edge-concentrated and not an independent geological confirmation.")
        st.dataframe([{"Stage 2 rank": item["rank"], "finding": item["id"],
            "SEG-Y IL": item["inline_id"], "SEG-Y XL": item["crossline_id"],
            "sample index": item["sample_index"], "MAD only": item["mad_only"],
            "edge risk": item["edge_risk"], "voxel count": item["component_voxels"]}
            for item in stage3_registry["historical_findings"]], use_container_width=True, hide_index=True)
        st.info("No full-field recovery inference is validated. Use Recovery intelligence for QC/surveillance actions; use Expert diagnostics only for bounded manual sensitivity checks.")
    elif full_field_registry:
        bounded_ml_evidence()
    elif registry:
        bounded_ml_evidence()
    else:
        st.info("No completed findings registry is available. No anomaly screening is started by this page.")

elif page == "Expert diagnostics":
    page_header("Expert diagnostics", "Manual array-index exploration and bounded sensitivity checks for specialist QC only.")
    st.warning("This diagnostic workspace is separate from the judge-facing command center. Coordinate sliders here are array indices, not physical map coordinates; sample origin and 4D registration remain unverified.")
    with st.expander('Advanced statistical screening'):
        expert=st.checkbox('Run expert full-sample screening',value=False,help='Evaluates the selected full-resolution sample plane; not part of automatic dashboard startup.')
    if not expert:
        st.caption('Automatic discovery and local ranking below use completed registry outputs. Full-sample numerical screening is optional expert work.')
        bounded_ml_evidence()
    else:
        with st.expander('Advanced statistical sensitivity',expanded=False):
            threshold_sigma = st.slider("MAD multiplier", 1.0, 8.0, 3.5, 0.5,
                help="Cutoff = median + multiplier × 1.4826 × median absolute deviation. The inherited 3.5 default is an uncalibrated high-tail screening heuristic, not a probability threshold. Larger multipliers flag fewer cells.")
        attrs = cached_sample_attributes(
            dataset_cache_identity,
            sample,
            9,
            baseline,
            monitor,
        )
        with st.expander('Advanced component filtering',expanded=False):
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
                key=region_key, on_change=_select_detection_region, args=(region_key, zones, labels, detection.score))
            selected_zone = next(zone for zone in zones if zone.label == selected_zone_id)
        score_display = display_slice(detection.score, max_axis=700)
        mask_display = display_mask(detection.mask, max_axis=700)
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
            center_inline, center_crossline = region_representative(labels, selected_zone, detection.score)
            image.add_trace(go.Scatter(x=[center_crossline], y=[center_inline], mode="markers",
                marker={"symbol": "x", "size": 13, "color": "#ff7047", "line": {"width": 3}},
                name="Strongest finite region member"))
            image.add_trace(go.Heatmap(z=np.where(display_region(labels, selected_zone.label, max_axis=700), 1.0, np.nan),
                x=np.arange(0, detection.score.shape[1], col_stride),
                y=np.arange(0, detection.score.shape[0], row_stride),
                colorscale=[[0, "rgba(0,0,0,0)"], [1, "#ff7047"]], zmin=0, zmax=1,
                opacity=.86, showscale=False, name=f"Selected region {selected_zone.label}"))
        st.plotly_chart(
            image.update_layout(
                title="Statistical screen · score is not probability",
                xaxis_title="Crossline index", yaxis_title="Inline index", height=600,
            ),
            use_container_width=True, theme=None,
        )
        st.caption(
            f"Formula: median + {threshold_sigma:g} × 1.4826 × MAD; MAD is the median absolute deviation on the full-resolution absolute-difference plane. The multiplier changes statistical sensitivity, not geological truth."
        )
        st.caption("Pooled overlays show presence within source-index bins, not exact boundaries or merged region identities. The selected-region layer pools its membership independently; other regions in the same bin remain distinct in the full-resolution analysis.")
        region_metrics = st.columns(2)
        region_metrics[0].metric("Retained connected regions", format_metric_count(len(zones)),
            help=f"Exact value: {len(zones):,}")
        flagged_count = int(detection.mask.sum())
        region_metrics[1].metric("Flagged cells before region-size filter", format_metric_count(flagged_count),
            help=f"Exact value: {flagged_count:,}")
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
            inline, crossline = region_representative(labels, selected_zone, detection.score)
            st.markdown(f"**Region {selected_zone.label}** · {selected_zone.voxel_count:,} cells · strongest finite member Inline {inline}, Crossline {crossline}, Sample Index {sample} · mean score {selected_zone.mean_strength:.4g} · maximum {selected_zone.max_strength:.4g}")
            point_values = st.columns(3)
            rb, rm = float(baseline[inline,crossline,sample]), float(monitor[inline,crossline,sample])
            point_values[0].metric("Baseline amplitude", f"{rb:.4g}" if np.isfinite(rb) else "Missing")
            point_values[1].metric("Monitor amplitude", f"{rm:.4g}" if np.isfinite(rm) else "Missing")
            point_values[2].metric("Difference", f"{rm-rb:.4g}" if np.isfinite(rb) and np.isfinite(rm) else "Missing")
            point_values[0].metric("Absolute difference", f"{attrs.absolute_difference[inline,crossline]:.4g}" if np.isfinite(attrs.absolute_difference[inline,crossline]) else "Missing")
            point_values[1].metric("Windowed NRMS", f"{attrs.nrms[inline,crossline]:.2f}%" if np.isfinite(attrs.nrms[inline,crossline]) else "Missing")
            point_values[2].metric("Paired support", f"{int(attrs.paired_sample_count[inline,crossline])}/9")
            st.caption(f"Window support at representative member: {int(attrs.paired_sample_count[inline, crossline])} finite paired samples out of the requested 9 (clipped at survey ends).")
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
                section_x, section_y = display_coordinates(values.shape, max_axis=500)
                evidence_fig = go.Figure(go.Heatmap(z=section_values, x=section_x, y=section_y, colorscale=scale,
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
        bounded_ml_evidence()

elif page == "Survey transition":
    page_header("Finding-focused survey comparison", "Automatically selected evidence · synchronized sample, inline and crossline sections.")
    if full_field_registry and field_entry:
        labels_path=Path(full_field_registry['body']['provenance']['component_labels_path'])
        labels=cached_component_labels(str(labels_path),labels_path.stat().st_size,
            labels_path.stat().st_mtime_ns,tuple(baseline.shape))
        selected_sample=field_finding['array_location'][2]
        member=np.asarray(labels[:,:,selected_sample])==int(field_finding['component_label'])
        fig=cached_animation(dataset_cache_identity,selected_sample,float(field_finding['component_label']),180,
            baseline[:,:,selected_sample],monitor[:,:,selected_sample],member,paired_missing)
        st.plotly_chart(fig,use_container_width=True,theme=None)
        st.caption('Animation interpolates the measured Baseline and Monitor display endpoints; intermediate frames are not acquired vintages. The turquoise overlay is the selected connected component at this sample index.')
    elif guided_finding:
        presentation_key=presentation_identity(registry['id'],guided_result['run_id'],guided_finding['id'],
            guided_result['provenance']['dataset_id'])
        presentation_token=presentation_cache_token(presentation_cache_root,presentation_key)
        fig=cached_finding_presentation(registry['id'],guided_result['run_id'],guided_finding['id'],
            guided_result['provenance']['dataset_id'],str(presentation_cache_root),presentation_token)
        if fig is None:
            st.info("Presentation artifact is not ready yet. The validated finding and measurements remain available in the report; offline presentation generation is still pending.")
        else:
            render_presentation(registry,entry,fig)
            st.info("The three upper panels switch between the actual Baseline and Monitor endpoints. The three lower panels keep Monitor − Baseline visible. No interpolated or invented vintages are shown.")
            st.caption("Gold crosses identify the same valid finding member in every section; gold shading preserves selected component membership. Gray cells are missing. Endpoint scales are fixed across surveys per section (99th-percentile absolute amplitude); difference panels have separate symmetric scales in the same amplitude units. Grid indices are not physical coordinates. The animation is a bounded local presentation, not a complete 3D change history.")
    else:
        st.info("No eligible connected ML finding is available in the automatically inspected tile; a focused interpretation animation is not fabricated. Review the report's screening coverage, support and missing well/registration evidence.")

elif page == "Well inventory":
    page_header("Wells & reservoir evidence", "Source well data inform context; seismic ties stay disabled until references are verified.")
    groups=well_context.get('by_well',{})
    well_keys=sorted(groups)
    evidence_matrix = well_context.get("well_evidence_matrix") or build_well_evidence_matrix(well_context)
    st.markdown("### Eight-well evidence matrix")
    matrix_rows=[]
    for item in evidence_matrix:
        families=item.get("curve_families",{})
        table_counts={category:sum(int(table.get("records",0)) for table in item.get(category,[]))
                      for category in ("markers","checkshots","deviations")}
        matrix_rows.append({
            "Well":item["well_id"], "Workbook X/Y":str(item.get("workbook_xy") or "not listed"),
            "Coordinate reference":"Unspecified; no SEG-Y association",
            "LAS sources":len(item.get("las_sources",[])),
            "GR":"; ".join(families.get("gamma_ray",[])) or "—",
            "Resistivity":"; ".join(families.get("resistivity",[])) or "—",
            "Sonic":"; ".join(families.get("sonic",[])) or "—",
            "Density / neutron":"; ".join(families.get("density",[])+families.get("neutron",[])) or "—",
            "Interpreted PHI / perm / saturation":"; ".join(families.get("porosity_interpreted",[])+families.get("permeability_interpreted",[])+families.get("saturation_or_volume_interpreted",[])) or "—",
            "Markers / checkshot / deviation records":f"{table_counts['markers']} / {table_counts['checkshots']} / {table_counts['deviations']}",
            "Well-to-seismic tie":"Unverified"})
    if matrix_rows:
        st.dataframe(matrix_rows,use_container_width=True,hide_index=True)
    st.caption("Curve mnemonics are reported as supplied. PHIE, PERM, SWE, VOIL and VGAS are interpreted/derived source curves, not independently verified fluid or reservoir truth. Workbook X/Y values are shown as source numbers only; units, CRS and datum are unspecified.")
    try:
        ai_identity = tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns)
                            for p in sorted((settings.challenge3_root / "well_data" / "las").glob("*.las")))
        ai_rows = cached_acoustic_impedance(str(settings.challenge3_root), ai_identity)
        st.markdown("### Sonic-density acoustic-impedance QC")
        display_rows = []
        for item in ai_rows:
            stats = item.get("acoustic_impedance_mrayl", {})
            display_rows.append({"Well": item["well_id"], "Status": item["status"], "Source curves":
                f"{item.get('sonic_curve') or '—'} × {item.get('density_curve') or '—'}",
                "Valid samples": f"{item.get('valid_samples', 0):,} / {item.get('total_samples', 0):,}",
                "AI median (MRayl)": round(stats["median"], 2) if stats else "—",
                "AI p10–p90 (MRayl)": f"{stats['p10']:.2f}–{stats['p90']:.2f}" if stats else "—"})
        st.dataframe(display_rows, use_container_width=True, hide_index=True)
        st.caption("P-impedance is calculated only from same-file sonic and density curves with compatible declared units, using valid paired depth samples. It is a log-derived rock-property indicator; it does not establish a synthetic seismic tie, fluid type, or production-related change.")
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        st.info(f"Sonic-density impedance summary unavailable: {exc}")
    st.info("Well-local marker/checkshot interpolation is available where both files declare compatible TVDSS (m) and TWT (ms). It is not a well-to-seismic tie. SEG-Y headers show 2 ms sampling and zero recording delay, but the textual headers do not explicitly declare a TWT sample domain; trace XY coordinates are zero and coordinate units are unset.")
    baseline_path = settings.challenge3_root / "volumes" / "Challenge3_Baseline_FilteredScaled_FullStack.sgy"
    monitor_path = settings.challenge3_root / "volumes" / "Challenge3_Monitor_FilteredScaled_FullStack.sgy"
    try:
        identity = (str(baseline_path), baseline_path.stat().st_size, baseline_path.stat().st_mtime_ns)
        header_geometry = cached_header_geometry(str(baseline_path), identity)
        monitor_identity = (str(monitor_path), monitor_path.stat().st_size, monitor_path.stat().st_mtime_ns)
        monitor_geometry = cached_header_geometry(str(monitor_path), monitor_identity)
        conditional_map = conditionally_map_well_coordinates(
            evidence_matrix, header_geometry,
            inline_ids=np.asarray(inline_header_ids), crossline_ids=np.asarray(crossline_header_ids))
        st.markdown("### Conditional workbook-coordinate / Baseline-header comparison")
        if conditional_map["status"] == "conditional_only":
            projected = {item["well_id"]: item for item in conditional_map["wells"]}
            rows = [{"Well": row["well_id"], "Conditional SEG-Y IL": round(projected[row["well_id"]]["conditional_inline_id"], 1)
                     if projected[row["well_id"]].get("conditional_inline_id") is not None else "—",
                     "Conditional SEG-Y XL": round(projected[row["well_id"]]["conditional_crossline_id"], 1)
                     if projected[row["well_id"]].get("conditional_crossline_id") is not None else "—",
                     "Within Baseline header axes": projected[row["well_id"]].get("inside_header_axis_extent", "—")}
                    for row in evidence_matrix]
            st.dataframe(rows, use_container_width=True, hide_index=True)
            monitor_fit = monitor_geometry.get("affine_geometry", {})
            monitor_r2 = monitor_fit.get("holdout_r2_x_y", [None, None])
            monitor_error = monitor_fit.get("holdout_error_coordinate_units_unknown", {}).get("p95")
            monitor_ok = all(value is not None and value >= 0.95 for value in monitor_r2)
            st.warning("This is a numerical consistency test only. The Baseline affine fit predicts sampled CDP headers well, and Well-2 is near the independent DUG location note; workbook and SEG-Y CRS, coordinate units, datum, well trajectory, and physical seismic tie remain unverified. Do not use these points as confirmed reservoir associations.")
            st.caption(f"Monitor sampled CDP affine holdout: R² X/Y {monitor_r2}; p95 residual {monitor_error if monitor_error is not None else 'unavailable'} in undeclared coordinate units; consistent sampled grid: {monitor_ok}. No Monitor well projection is performed.")
            st.caption(f"Baseline header-only sample: {header_geometry['sampled_trace_count']} traces; spatial holdout R² (X/Y) {header_geometry['affine_geometry']['holdout_r2_x_y']}; p95 holdout residual {header_geometry['affine_geometry']['holdout_error_coordinate_units_unknown']['p95']:.3f} in undeclared coordinate units. No amplitudes were loaded for this check.")
        else:
            st.info(f"Conditional coordinate comparison unavailable: {conditional_map.get('reason', 'header fit unavailable')}")
        conditional_marker_rows = []
        for item in evidence_matrix:
            relationship = item.get("marker_checkshot_relationship", {})
            for marker in relationship.get("markers", []):
                if str(marker.get("marker", "")).strip().upper() in {"I-100", "I100"}:
                    twt = float(marker["interpolated_source_checkshot_twt_ms"])
                    conditional_marker_rows.append({"Well": item["well_id"], "Marker": marker["marker"],
                        "Well-local interpolated checkshot TWT (ms)": round(twt, 2),
                        "Array sample if sample 0 = 0 ms and 2 ms interval": int(round(twt / 2))})
        if conditional_marker_rows:
            st.markdown("### Conditional I-100 marker/sample comparison")
            st.dataframe(conditional_marker_rows, use_container_width=True, hide_index=True)
            st.caption("The TWT values are well-local interpolations from each well's source-declared TVDSS/TWT checkshot. Sample values are a sensitivity case only (zero-origin assumption); the SEG-Y domain origin and marker-to-seismic tie are not verified.")
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        st.info(f"Header-only conditional coordinate comparison unavailable: {exc}")
    st.markdown('### Evidence by source well')
    if well_keys:
        selected_well=st.selectbox('Select source well',well_keys,key='well_evidence_selected')
        group=groups[selected_well]
        workbook_records=group.get('workbook_records',[])
        if workbook_records:
            record=workbook_records[0]
            coordinate=record.get('coordinate_evidence',{})
            st.markdown(f"**{record['well_id']}** · Workbook row {record['source_row']} · {coordinate.get('status','coordinate reference unavailable')}")
        st.caption('Source-ID grouping is not a verified spatial or reservoir-zone association.')

        logs=group.get('logs',[])
        if logs:
            st.markdown('#### Log evidence')
            log_by_name={item['source']:item for item in logs}
            selected_log_name=st.selectbox('LAS source',sorted(log_by_name),key='well_las_source')
            log_summary=log_by_name[selected_log_name]
            las_path=settings.challenge3_root/'well_data'/'las'/selected_log_name
            try:
                token=(str(las_path),las_path.stat().st_size,las_path.stat().st_mtime_ns)
                las_data=cached_well_las(str(las_path),token)
                curve_choices=[c['mnemonic'] for c in las_data['curves'] if c['mnemonic']!=las_data['depth_mnemonic']]
                preferred=[name for name in ('GR_P','GR','RT_P','RT','DENS_P','DENB','PHIE','SWE','DTC_P','DTCOM') if name in curve_choices]
                options=preferred or curve_choices
                if options:
                    curve_name=st.selectbox('Curve',options,key='well_las_curve')
                    curve=next(c for c in las_data['curves'] if c['mnemonic']==curve_name)
                    x=np.asarray(las_data['data'][curve_name],dtype=float)
                    depth=np.asarray(las_data['data'][las_data['depth_mnemonic']],dtype=float)
                    finite=np.isfinite(x)&np.isfinite(depth)
                    figure=go.Figure(go.Scatter(x=x[finite],y=depth[finite],mode='lines',line={'color':'#20e0c3','width':2}))
                    figure.update_layout(template='plotly_dark',paper_bgcolor='#080d12',plot_bgcolor='#080d12',
                        title=f"{curve_name} · {curve['unit'] or 'unit unavailable'} · {log_summary['provenance']}",
                        xaxis_title=f"{curve_name} ({curve['unit'] or 'unit unavailable'})",
                        yaxis_title=f"LAS depth axis ({las_data['depth_unit'] or 'unit unavailable'}; datum unverified)",
                        yaxis={'autorange':'reversed'},font={'color':'white'},height=540,margin={'l':60,'r':20,'t':55,'b':50})
                    st.plotly_chart(figure,use_container_width=True,theme=None)
                    st.caption(f"{int(finite.sum()):,}/{len(x):,} finite samples · {curve['evidence_kind']} · raw measurement status: {curve['raw_measurement_status']}")
                st.dataframe([{key:curve.get(key) for key in ('mnemonic','unit','description','sample_count','null_count','finite_count','finite_min','finite_median','finite_max','evidence_kind','raw_measurement_status','value_quality_flags')}
                              for curve in log_summary['curves']],use_container_width=True)
                st.caption("Finite ranges are source-file summaries, not validated reservoir distributions. Values flagged as outside declared fraction bounds or extreme finite interpreted values need source-encoding review before petrophysical use.")
            except (RuntimeError,OSError,ValueError,KeyError) as exc:
                st.warning(f"LAS curve view unavailable: {type(exc).__name__}")

        tables=group.get('tables',[])
        if tables:
            st.markdown('#### Formation, checkshot & deviation evidence')
            table_by_name={item['source']:item for item in tables}
            selected_table_name=st.selectbox('Source table',sorted(table_by_name),key='well_table_source')
            table=table_by_name[selected_table_name]
            st.caption(f"{table['category']} · {table['records']} source records · declared units {table['declared_units'] or 'N/A'} · QC: {table['quality']['status']}")
            if table['category']=='checkshots' and table['quality']['status']=='structurally_valid_declared_checkshot_pairs':
                rows=np.asarray([[float(v) for v in row[:2]] for row in table['raw_records']],dtype=float)
                figure=go.Figure(go.Scatter(x=rows[:,1],y=rows[:,0],mode='lines+markers',line={'color':'#f2c14e'}))
                figure.update_layout(template='plotly_dark',paper_bgcolor='#080d12',plot_bgcolor='#080d12',
                    title='Source checkshot pairs · not tied to seismic samples',xaxis_title='TWT (ms)',
                    yaxis_title='TVDSS (m; source declared)',yaxis={'autorange':'reversed'},
                    font={'color':'white'},height=430)
                st.plotly_chart(figure,use_container_width=True,theme=None)
            elif table['category']=='deviations' and table['quality']['status'].startswith('depth_survey_structurally_valid'):
                rows=np.asarray([[float(v) for v in row[:2]] for row in table['raw_records']],dtype=float)
                figure=go.Figure(go.Scatter(x=rows[:,0],y=rows[:,1],mode='lines+markers',line={'color':'#20e0c3'}))
                figure.update_layout(template='plotly_dark',paper_bgcolor='#080d12',plot_bgcolor='#080d12',
                    title='Source deviation depth survey · XY trajectory unavailable',xaxis_title='MDKB (m)',
                    yaxis_title='TVD (m; source declared)',yaxis={'autorange':'reversed'},font={'color':'white'},height=430)
                st.plotly_chart(figure,use_container_width=True,theme=None)
            elif table['category']=='markers' and table['ambiguous_rows']==0:
                columns=table['columns']; name_col=next((columns.index(x) for x in ('NAME','Marker','MARKER') if x in columns),None)
                depth_col=next((columns.index(x) for x in ('TVDSS','MDKB','TVD') if x in columns),None)
                if name_col is not None and depth_col is not None and table['units'][depth_col]:
                    try:
                        marker_names=[row[name_col] for row in table['raw_records']]
                        marker_depth=np.asarray([float(row[depth_col]) for row in table['raw_records']],dtype=float)
                        figure=go.Figure(go.Scatter(x=marker_names,y=marker_depth,mode='markers+text',text=marker_names,
                            textposition='top center',marker={'color':'#f2c14e','size':9}))
                        figure.update_layout(template='plotly_dark',paper_bgcolor='#080d12',plot_bgcolor='#080d12',
                            title='Source formation markers · seismic horizon tie unavailable',
                            xaxis_title='Source marker name',yaxis_title=f"{columns[depth_col]} ({table['units'][depth_col]}; datum/tie unresolved)",
                            yaxis={'autorange':'reversed'},font={'color':'white'},height=460)
                        st.plotly_chart(figure,use_container_width=True,theme=None)
                    except (TypeError,ValueError):
                        st.warning('Marker depth values are incomplete or non-numeric; original records are shown below.')
                        st.dataframe([dict(zip(table['columns'],row)) for row in table['raw_records']],use_container_width=True)
                else:
                    st.dataframe([dict(zip(table['columns'],row)) for row in table['raw_records']],use_container_width=True)
            else:
                columns=table['columns']
                st.dataframe([dict(zip(columns,row)) for row in table['raw_records']],use_container_width=True)
            st.caption('Marker and checkshot evidence remains contextual until CRS, datum, trajectory and seismic sample ties are verified.')
        else:
            st.info('N/A · no marker, checkshot or deviation file grouped under this source well.')
    else:
        st.info('N/A · no source well evidence is available.')

    selected_matrix = next((item for item in evidence_matrix if item.get("well_id")==selected_well),None) if well_keys else None
    if selected_matrix:
        deviation_relationship=selected_matrix.get("marker_deviation_relationship",{})
        st.markdown("#### Well-local marker/deviation relationship")
        if deviation_relationship.get("markers"):
            st.dataframe(deviation_relationship["markers"],use_container_width=True,hide_index=True)
            st.caption(deviation_relationship.get("limitation","This remains a source-well relationship, not a seismic tie."))
        else:
            st.info(deviation_relationship.get("reason","No compatible marker/deviation pair is available."))
        relationship=selected_matrix.get("marker_checkshot_relationship",{})
        st.markdown("#### Well-local marker/checkshot relationship")
        if relationship.get("markers"):
            st.dataframe(relationship["markers"],use_container_width=True,hide_index=True)
            st.caption(f"Linear interpolation only within source checkshot coverage {relationship['checkshot_tvdss_range_m']} m TVDSS / {relationship['checkshot_twt_range_ms']} ms TWT. {relationship.get('limitation','')}")
        else:
            st.info(relationship.get("reason","No compatible well-local marker/checkshot pair is available."))

    with st.expander('Association readiness'):
        readiness=well_context.get('association_assessment',{})
        st.markdown('**Well-to-seismic association: unavailable**')
        st.markdown('\n'.join(f'- {item}' for item in well_context.get('missing_association_evidence',[])))
        st.caption(readiness.get('reason','Required spatial and vertical references are unresolved.'))

    st.markdown('### Shared finding context')
    st.caption(f"{well_context['well_count']} workbook wells · {len(well_context['logs'])} LAS sources · {len(well_context['tables'])} marker/checkshot/deviation sources. Static logs do not measure temporal fluid or pressure change.")
    if well_context.get('errors'):
        st.warning(f"{len(well_context['errors'])} well source(s) could not be read; available evidence is preserved.")
    with st.expander('Advanced source inventory and reader diagnostics'):
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
                workbook_report = inspect_well_information(workbook_path, header_row=3, max_column=13)
                st.markdown("**Well-list workbook structure**")
                st.dataframe([{"sheet": sheet["sheet"], "source columns": ", ".join(sheet["columns"]),
                               "header row": sheet["header_row"], "data rows": sheet["data_rows"]} for sheet in workbook_report["sheets"]], use_container_width=True)
            except (RuntimeError, OSError, ValueError) as exc:
                st.info(f"Workbook structure is unavailable: {type(exc).__name__}")
        available_tables = [row["relative_path"] for row in inventory if row["available"] and row["data_type"] in ("checkshots", "deviations", "markers")]
        if available_tables:
            selected_file = st.selectbox("Inspect source table", available_tables)
            from reservoir_time_machine.data_ingest.well_tables import read_well_table
            try:
                table = read_well_table(settings.challenge3_root / selected_file)
                st.caption(f"Source: {table.source_file}. Source declarations and values are retained; missing trailing fields are blank. Units are reported only when explicitly declared in comments.")
                st.dataframe([dict(zip(table.columns, row)) for row in table.rows], use_container_width=True)
                st.caption(f"Header origin: {table.header_origin}; source units: {dict(zip(table.columns, table.units))}; variable-width source rows: {table.row_width_mismatch_count}. Declared columns: {table.declared_columns}. Short rows leave depth-column semantics unresolved; raw lines and source line numbers are preserved by the reader.")
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
        st.markdown('### Organized well-source evidence')
        st.write(well_context.get('by_well',{}))
        st.caption('Supporting marker, checkshot and deviation sources remain unassigned where an explicit well identifier/datum relationship is unresolved.')
        st.info("Well positions are not plotted because validated coordinates and a shared spatial reference have not been established.")

elif page == "Evidence explanation":
    page_header("Evidence explanation", "Separate observed data, calculated attributes, statistical flags, interpretations, and hypotheses.")
    section("Selected location evidence")
    selected_base, selected_monitor = float(baseline[inline,crossline,sample]), float(monitor[inline,crossline,sample])
    if np.isfinite(selected_base) and np.isfinite(selected_monitor):
        st.write(f"**Observed data:** Inline {inline}, Crossline {crossline}, Sample Index {sample}. Baseline amplitude {selected_base:.5g}; Monitor amplitude {selected_monitor:.5g}.")
    else:
        st.write("**Observed data:** one or both survey samples are missing or non-finite at this grid location.")
    st.caption("Only a bounded typed evidence packet may be sent to the optional Gemma service after explicit confirmation. Model output remains unverified and for human review.")
    bounded_ml_evidence(allow_external=True)
    with st.expander('Available well evidence and association eligibility'):
        st.write({'well_count':well_context['well_count'],'log_sources':len(well_context['logs']),
                  'table_sources':len(well_context['tables']),'association_available':False,
                  'missing_evidence':well_context['missing_association_evidence']})
        st.caption('Measured and interpreted curves are source context only. No candidate-to-well relationship is established.')
        st.write(well_context)
    if stage3_registry:
        st.markdown("### Saved full-field finding · deterministic evidence explanation")
        selected_stage3 = select_investigation(
            stage3_registry, st.session_state.get("selected_finding_id", stage3_investigations[0]["id"])
        )["finding"]
        packet = stage3_finding_packet(selected_stage3, assessment_id=stage3_registry["id"])
        try:
            llm_settings = LLMSettings.from_environment()
        except ValueError:
            llm_settings = LLMSettings()
            st.warning("Gemma settings are invalid; the deterministic explanation remains active.")
        explanation_key = explanation_identity(packet, llm_settings)
        explanations = st.session_state.setdefault("grounded_explanations", {})
        explanation = explanations.get(explanation_key) or interpret(packet)
        st.caption(f"Finding {selected_stage3['id']} · deterministic evidence selection · Stage 3 assessment {stage3_registry['id'][:16]}")
        for claim in explanation["claims"]:
            st.markdown(f"**{claim['category'].replace('_', ' ').title()}:** {claim['text']}")
            st.caption("Evidence IDs: " + ", ".join(claim["evidence_ids"]))
        eligible = bool(llm_settings.base_url and llm_settings.real_data_enabled)
        if not eligible:
            st.info("Gemma is not active: an approved endpoint and real-data sharing policy are not configured. No external request was made.")
        authorized = st.checkbox("I authorize sending this small typed finding record to the configured Gemma service.",
                                 key="stage3_gemma_consent", disabled=not eligible)
        if st.button("Ask Gemma to select evidence statements", key="stage3_gemma_request",
                     disabled=not (authorized and eligible)):
            if transmission_allowed(packet, llm_settings, authorized=authorized):
                result = interpret(packet, provider=GemmaFindingProvider(llm_settings), authorized=True)
                st.session_state["grounded_explanations"][explanation_key] = result
                explanation = result
                st.rerun()
            else:
                st.warning("Transmission policy did not authorize this request; no payload was sent.")
        st.caption("Gemma can select only constrained statements and exact supplied values. It does not validate geology; service output remains subject to human review.")

else:
    page_header("Recovery intelligence", "Evidence-led surveillance actions · no unsupported recovery targets or success probabilities.")
    if stage3_registry is None:
        if stage3_registry_error:
            st.error(f"The Stage 3 assessment did not pass checksum/schema/coordinate validation ({stage3_registry_error}).")
        else:
            st.info("No completed Stage 3 evidence package is configured. Existing seismic views remain available.")
    else:
        assessment = stage3_registry["assessment"]
        interpretation = assessment["candidate_interpretation"]
        stage2_review = assessment["stage2_top_ten_review"]
        consensus = assessment["consensus"]
        evolution = reservoir_evolution_readiness(well_context, assessment)
        section("What happened? · full-field screening, not a reservoir diagnosis")
        metric_cols = st.columns(4)
        metric_cols[0].metric("Stage 2 coverage", "2,040 / 2,040 chunks")
        candidate_voxels = (stage2_review["mad_only_voxels_in_candidate_union"]
                            + stage2_review["three_detector_consensus_voxels_in_candidate_union"])
        metric_cols[1].metric("Candidate voxels", f"{candidate_voxels:,}")
        metric_cols[2].metric("Multi-model consensus voxels", f"{consensus['voxel_count']:,}")
        metric_cols[3].metric("Verified reservoir / recovery targets", "0 / 0")
        st.info("Trace-key pairing is available on the Baseline grid. Physical 4D registration is not verified. Sample origin and well-to-seismic tie are unresolved. Findings are seismic screening evidence; they do not establish fluid movement or remaining hydrocarbons.")

        st.markdown("### Which changes matter? · first review the QC action queue")
        selection_ids = [item["id"] for item in stage3_investigations]
        selected_id = st.selectbox(
            "Investigation record", selection_ids,
            format_func=lambda ident: next(
                (f"QC action {item['queue_rank']} · {ident}" if item["kind"] == "qc_follow_up"
                 else f"Historical Stage 2 screening · {ident}"
                 for item in stage3_investigations if item["id"] == ident), ident),
            key="stage3_finding_selection", on_change=_jump_to_stage3_selection,
            args=("stage3_finding_selection", stage3_locations),
        )
        selected_record = select_investigation(stage3_registry, selected_id)
        selected = selected_record["finding"]
        if selected_record["kind"] == "qc_follow_up":
            st.caption("This queue prioritizes technical validation work. It is not a recovery-target ranking.")
        else:
            st.caption("Historical Stage 2 screening record; not a recovery-target ranking.")

        loc = selected["array_location"]
        il, xl, sample_idx = map(int, loc)
        il_id, xl_id = int(inline_header_ids[il]), int(crossline_header_ids[xl])
        extent = selected["extent"]
        st.markdown(f"**{selected_id}** · SEG-Y IL {il_id} · XL {xl_id} · array sample index {sample_idx} · {selected['component_voxels']:,} voxels")
        st.caption("Sample is an array index; the TWT/depth origin is not established. Bounds below are inclusive array-index extents.")

        i0, i1 = max(0, il - 48), min(baseline.shape[0], il + 49)
        x0, x1 = max(0, xl - 48), min(baseline.shape[1], xl + 49)
        bview = np.asarray(baseline[i0:i1, x0:x1, sample_idx])
        mview = np.asarray(monitor[i0:i1, x0:x1, sample_idx])
        dview = mview - bview
        finite_pair = np.concatenate((np.abs(bview[np.isfinite(bview)]), np.abs(mview[np.isfinite(mview)])))
        amp_scale = max(float(np.percentile(finite_pair, 99)) if finite_pair.size else 1.0, 1e-12)
        diff_finite = np.abs(dview[np.isfinite(dview)])
        diff_scale = max(float(np.percentile(diff_finite, 99)) if diff_finite.size else 1.0, 1e-12)
        fig_cols = st.columns(3)
        for col, values, title, palette, limit in (
            (fig_cols[0], bview, "1995 Baseline", "RdBu", amp_scale),
            (fig_cols[1], mview, "2006 Monitor", "RdBu", amp_scale),
            (fig_cols[2], dview, "Monitor − Baseline", "RdBu", diff_scale),
        ):
            heat = go.Figure(go.Heatmap(z=values, x=np.asarray(crossline_header_ids[x0:x1]),
                y=np.asarray(inline_header_ids[i0:i1]), colorscale=palette,
                zmin=-limit if title != "Monitor − Baseline" else -limit,
                zmax=limit, zmid=0 if title == "Monitor − Baseline" else None,
                colorbar={"title": "Amplitude" if title != "Monitor − Baseline" else "Difference"}))
            if int(extent[2][0]) <= sample_idx <= int(extent[2][1]):
                shape_x0, shape_x1 = max(x0, int(extent[1][0])), min(x1 - 1, int(extent[1][1]))
                shape_y0, shape_y1 = max(i0, int(extent[0][0])), min(i1 - 1, int(extent[0][1]))
                if shape_x0 <= shape_x1 and shape_y0 <= shape_y1:
                    heat.add_shape(type="rect", x0=int(crossline_header_ids[shape_x0]),
                        x1=int(crossline_header_ids[shape_x1]), y0=int(inline_header_ids[shape_y0]),
                        y1=int(inline_header_ids[shape_y1]), line={"color": "#32efc4", "width": 2})
            col.plotly_chart(heat.update_layout(template="plotly_dark", paper_bgcolor="#080d0e",
                plot_bgcolor="#101719", font={"color": "white"}, title=title,
                xaxis_title="SEG-Y Crossline ID", yaxis_title="SEG-Y Inline ID", height=390),
                use_container_width=True, theme=None)
        st.caption("Panels are a local sample-index slice centered on the selected component representative. Baseline and Monitor share a color range; the signed difference is zero-centered. The turquoise rectangle shows the component's bounding extent, not exact voxel membership.")
        local_attributes = cached_tile_attributes(dataset_cache_identity, "stage3-local-view", sample_idx,
            (i0, i1, x0, x1), baseline[i0:i1, x0:x1, :], monitor[i0:i1, x0:x1, :])
        attribute_panels = st.columns(2)
        for col, values, title, palette, limits in (
            (attribute_panels[0], local_attributes.nrms, "Local 9-sample NRMS (%)", "Viridis", None),
            (attribute_panels[1], local_attributes.paired_sample_count, "Paired sample support / 9", "Viridis", (0, 9)),
        ):
            shown_values = np.asarray(values)
            if shown_values.ndim == 3:
                shown_values = shown_values[:, :, sample_idx]
            heat = go.Figure(go.Heatmap(z=shown_values,
                x=np.asarray(crossline_header_ids[x0:x1]), y=np.asarray(inline_header_ids[i0:i1]),
                colorscale=palette, zmin=limits[0] if limits else None,
                zmax=limits[1] if limits else None, colorbar={"title": title}))
            col.plotly_chart(heat.update_layout(template="plotly_dark", paper_bgcolor="#080d0e",
                plot_bgcolor="#101719", font={"color": "white"}, title=title,
                xaxis_title="SEG-Y Crossline ID", yaxis_title="SEG-Y Inline ID", height=340),
                use_container_width=True, theme=None)
        st.caption("These are bounded 9-sample local attributes computed from finite paired source samples for the displayed crop only. They are not the stored full-field attribute volumes.")

        attrs = selected
        metric_cols = st.columns(5)
        metric_cols[0].metric("Mean signed Δ", f"{attrs['signed_difference_mean']:.4g}")
        metric_cols[1].metric("Mean |Δ|", f"{attrs['absolute_difference_mean']:.4g}")
        metric_cols[2].metric("Mean NRMS", f"{attrs['nrms_mean_percent']:.2f}%")
        metric_cols[3].metric("Min paired support", f"{attrs['min_paired_support']}/9")
        metric_cols[4].metric("Stage 3 capped index", f"{stage3_registry['stage3_indices'][selected_id]:.3f}")
        st.markdown("#### Why this is not yet a recovery target")
        st.write(f"The historical Stage 2 list is **MAD-only** for all ten entries; four classical detector threshold flags are zero for each. The Stage 3 support correction caps support at 1.0 and preserves the Stage 2 archive. The corrected index is still a screening index, not a probability. The consensus set contains {consensus['voxel_count']:,} Mahalanobis/PCA/GMM voxels and is concentrated near the high-crossline boundary; it does not establish independent geological confirmation.")
        st.markdown("#### What to investigate next")
        st.write(selected.get("geophysics_next_action", "Review registration, repeatability and acquisition/processing metadata."))
        st.write(selected.get("petrophysics_next_action", "No verified well or stratigraphic tie; establish coordinate, datum and sample mapping before correlation."))
        st.write(selected.get("reservoir_engineering_next_action", "Gather production, injection and pressure evidence only after a spatial/interval tie is verified."))
        with st.expander("Repeatability, detector and edge evidence"):
            st.json({"detector_threshold_flags": selected["detector_flagged_voxels"],
                     "edge_QC": selected["edge_qc"], "local_repeatability": selected["local_repeatability"],
                     "alternative_explanations": selected.get("plausible_alternatives", []),
                     "observation_confidence": selected.get("observation_confidence"),
                     "geological_interpretation_confidence": selected.get("geological_interpretation_confidence")})

        st.markdown("### Historical seismic discovery catalogue")
        st.caption("The original Stage 2 order is retained for audit. This table is not the recovery-intelligence ranking.")
        rows = []
        for item in stage3_registry["historical_findings"]:
            rows.append({"Stage 2 rank": item["rank"], "Finding": item["id"],
                "IL": item["inline_id"], "XL": item["crossline_id"], "Sample index": item["sample_index"],
                "Voxels": item["component_voxels"], "Mean NRMS %": item["nrms_mean_percent"],
                "MAD only": item["mad_only"], "Edge risk": item["edge_risk"],
                "Stage 3 index": stage3_registry["stage3_indices"][item["id"]]})
        st.dataframe(rows, use_container_width=True, hide_index=True)

        st.markdown("### Where is the next recovery opportunity?")
        st.info("Recovery Chance of Success: Not yet quantifiable. The source package has zero verified reservoir-interval associations and zero recovery opportunities supported by reservoir and production evidence. Current priorities are surveillance and data-validation actions, not drilling locations.")
        st.markdown("#### Reservoir evolution hypotheses: evidence gates")
        status_rows=[
            {"Question": "Production-response compartments", "Status": evolution["statuses"]["production_response"], "Current basis": "Full-field seismic screening exists; reservoir interval and verified repeatability tie do not."},
            {"Question": "Relatively limited-response / possible low-sweep compartments", "Status": evolution["statuses"]["limited_response_or_poor_sweep"], "Current basis": "No verified reservoir mask, detectability assessment, or production/injection context."},
            {"Question": "Depletion / fluid / pressure mechanism", "Status": evolution["statuses"]["fluid_or_pressure_mechanism"], "Current basis": "No mechanism-specific pressure, production, fluid-contact or validated rock-physics evidence."},
            {"Question": "Qualified recovery opportunities", "Status": str(evolution["statuses"]["development_opportunity_count"]), "Current basis": "None qualify in the checksummed Stage 3 assessment."},
        ]
        st.dataframe(status_rows,use_container_width=True,hide_index=True)
        st.caption(evolution["interpretation_rule"])
        st.markdown("#### Highest-value next evidence actions")
        st.dataframe(evolution["next_data_actions"],use_container_width=True,hide_index=True)
        st.markdown("#### Discipline-specific next actions")
        st.dataframe(evolution["discipline_actions"],use_container_width=True,hide_index=True)
        with st.expander("Why reservoir mechanisms remain unassigned"):
            st.write(evolution["limitations"])
        with st.expander("Method capability and validation status"):
            st.dataframe([
                {"Evidence component": "Mahalanobis / PCA / Isolation Forest / GMM", "Execution": "Completed full-field", "Validation boundary": "Computational outputs completed; historical top ten has no threshold flags from these four"},
                {"Evidence component": "MAD statistical screening", "Execution": "Completed full-field", "Validation boundary": "Screening only; no geological ground truth"},
                {"Evidence component": "3D edge / support QC", "Execution": "Completed in Stage 2/3", "Validation boundary": "Consensus detections remain edge-sensitive"},
                {"Evidence component": "Bayesian evidence fusion", "Execution": "Prototype code available", "Validation boundary": "No calibrated class likelihoods or posterior probability published"},
                {"Evidence component": "3D convolutional autoencoder", "Execution": "Architecture/training harness available", "Validation boundary": "No validated Angsi checkpoint or geological validation"},
                {"Evidence component": "Siamese 3D CNN", "Execution": "Architecture/training harness available", "Validation boundary": "No validated Angsi checkpoint or geological validation"},
            ], use_co