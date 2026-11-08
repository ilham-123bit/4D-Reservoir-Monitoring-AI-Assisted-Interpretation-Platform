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
from reservoir_time_machine.preprocessing.qc import quality_report
from reservoir_time_machine.settings import ApplicationSettings, LLMSettings
from reservoir_time_machine.analysis.preliminary import run_preliminary, VERSION as ML_VERSION
from reservoir_time_machine.visualization.preliminary import render_preliminary
from reservoir_time_machine.analysis.registry import load_registry
from reservoir_time_machine.analysis.technical_report import report_markdown
from reservoir_time_machine.visualization.presentation import select_entry,render_presentation
import json,re
from reservoir_time_machine.visualization.guided_report import render_page_report
from reservoir_time_machine.visualization.finding_animation import finding_animation
from reservoir_time_machine.data_ingest.well_evidence import well_evidence_summary
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


@st.cache_resource(max_entries=2)
def cached_finding_presentation(run_id,finding_id,_baseline,_monitor,_missing,_result,_finding):
    return finding_animation(_baseline,_monitor,_missing,_result,_finding)


def _jump_to_finding(location):
    st.session_state['shared_inline'], st.session_state['shared_crossline'], st.session_state['shared_sample'] = location


def bounded_ml_evidence(allow_external=False):
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
@st.cache_resource(max_entries=2)
def cached_registry(path,identity,shape,payload_identity):
    return load_registry(path,identity,shape)
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
    if st.session_state['shared_sample']!=guided_finding['sample']:
        # Expert navigation may move off the finding; its report remains explicitly scoped.
        pass
if registry is None or not registry['entries']:
    st.session_state.pop('selected_preliminary_finding',None)
well_context=registry['body']['wells'] if registry else cached_well_evidence(str(settings.challenge3_root),
    tuple((str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in sorted((settings.challenge3_root/'well_data').rglob('*')) if p.is_file()))
discovery=registry['body']['coverage'] if registry else {'status':'No completed analysis registry'}
paired_missing=np.asarray(baseline_volume.missing_trace_mask)|np.asarray(monitor_volume.missing_trace_mask)

with st.sidebar:
    st.header("Workspace")
    page = st.radio("Workspace", ["Overview and QC", "3D explorer", "4D comparison",
                                   "Change detection", "Survey transition", "Well inventory", "Evidence explanation"],
                    label_visibility="collapsed")
    if registry and registry['entries']:
        current = next((e for e in registry['entries'] if e['id'] == st.session_state.get('registry_selection')),
                       registry['entries'][0])
        st.markdown(f"**Finding {current['local_rank']} / {len(registry['entries'])}**")
        nav=st.columns(2)
        nav[0].button('Previous finding',on_click=select_entry,args=(registry['entries'],-1))
        nav[1].button('Next finding',on_click=select_entry,args=(registry['entries'],1))
        st.button('Return to automatic focus',on_click=select_entry,args=(registry['entries'],))
    if page != "4D comparison":
        with st.expander("Advanced Exploration", expanded=False):
            sample = st.slider("Sample Index", 0, baseline.shape[2]-1, key="shared_sample")
            inline = st.slider("Inline index", 0, baseline.shape[0]-1, key="shared_inline")
            crossline = st.slider("Crossline index", 0, baseline.shape[1]-1, key="shared_crossline")
            color_scale = st.selectbox("Seismic colour scale", ["RdBu", "Viridis", "Cividis", "Turbo"], index=0)
            st.caption("Selections are grid indices. Sample-domain origin and physical registration are unverified.")

if page != "4D comparison" and not availability["well_inventory_available"]:
    st.warning("The well-list spreadsheet is missing. Seismic-only views remain available; well context is not inferred.")

if not paired_grid_available or not capabilities["trace_key_preliminary_screening"]["available"]:
    st.error("Paired Baseline/Monitor screening requirements are unresolved.")
    st.stop()

if page != "4D comparison" and not physical_4d_registration_verified:
    registration_warning()

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
        render_page_report(guided_result,guided_finding,well_context,page,discovery,shared_interpretation=entry['interpretation'])
else:
    st.info('No completed valid findings registry is available. Expert seismic views remain available; automatic ML analysis does not run on page load.')
    st.caption('Coverage: N/A · no completed findings registry is available.')

if page == "Overview and QC":
    if registry:
        st.markdown('### Completed findings · local investigation ranks')
        for item in registry['entries'][:5]:
            f=item['finding'];m=f['metrics']
            st.markdown(f"**Local rank {item['local_rank']} · sample {f['sample']} · {f['id']}**")
            st.caption(f"Source indices {f['representative']} · {m['cells']} cells · mean NRMS {m['mean_nrms_percent']:.2f}% · minimum paired support {m['minimum_paired_support']}. Local tile priority only; other groups are not calibrated for comparison.")
        if len(registry['entries'])>5:st.caption('Five findings shown here; the registry selector and complete report include all completed findings.')
    page_header("Survey overview & QC", "Survey geometry, data completeness, and amplitude quality at a glance.")
    base_qc=None;monitor_qc=None
    section("Survey geometry")
    meta = monitor_volume.metadata.source_metadata
    inline_header_ids = meta.get("inline_ids", meta.get("inline_ids_selected", []))
    crossline_header_ids = meta.get("crossline_ids", meta.get("crossline_ids_selected", []))
    cols = st.columns(2)
    metric_slots = (cols[0], cols[0], cols[1], cols[1])
    for col,label,key in zip(metric_slots,('Baseline traces','Monitor traces','Common paired traces','Monitor-only traces'),
        ('baseline_trace_count','monitor_trace_count','paired_trace_count','monitor_only_trace_count')):
        count=meta.get(key)
        if key=='monitor_trace_count' and count is None and all(isinstance(meta.get(v),int) for v in ('paired_trace_count','monitor_only_trace_count')):
            count=meta['paired_trace_count']+meta['monitor_only_trace_count']
        display = format_metric_count(count) if isinstance(count,int) and not isinstance(count,bool) and count >= 0 else 'N/A'
        exact = f"Exact count: {count:,}" if isinstance(count,int) and not isinstance(count,bool) else 'Exact count unavailable'
        col.metric(label,display,help=exact)
    cols = st.columns(2)
    cols[0].metric("Samples per trace", format_metric_count(baseline.shape[2]), help=f"Exact value: {baseline.shape[2]:,}")
    cols[0].metric("Grid dimensions", " × ".join(f"{n:,}" for n in baseline.shape), help="Array order: inline × crossline × sample")
    cols[1].metric("Inline IDs", f"{min(inline_header_ids):,}–{max(inline_header_ids):,}" if inline_header_ids else "N/A")
    cols[1].metric("Crossline IDs", f"{min(crossline_header_ids):,}–{max(crossline_header_ids):,}" if crossline_header_ids else "N/A")
    st.caption(f"Sample interval: {baseline_volume.metadata.sample_interval:g} {baseline_volume.metadata.sample_interval_units}. Array axes are (inline, crossline, sample); Sample Index is shown because its origin/domain is unknown.")
    section("Data quality")
    st.info("Full-volume QC is not scanned on dashboard startup. Bounded run support and missing-data quality are recorded in the findings registry; survey-wide repeatability remains unverified.")
    section("Completed-tile difference distribution")
    if guided_result is not None:
        values=guided_result['scores']['statistical'][guided_result['valid']]
        st.plotly_chart(go.Figure(go.Histogram(x=values,nbinsx=60,marker_color='#31edc2')).update_layout(
            xaxis_title='Absolute Monitor − Baseline amplitude',yaxis_title='Eligible tile cells',height=350),use_container_width=True)
        st.caption('Loaded from the completed tile artifact. This is not a survey-wide amplitude distribution or repeatability calibration.')
    else:
        st.info('No completed-tile amplitude-change distribution is available. Global amplitude QC requires an offline product; the dashboard does not scan both cubes to create one.')
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
    with st.expander("Scientific capabilities and provenance"):
        st.write(capabilities)
        st.caption("Capabilities assess explicit evidence declarations, not independent certification. Cache identity includes all six cache files, source-file statistics, and small control-file hashes. Large source/cube SHA256 authentication remains unverified; physical and well registration are not inferred.")
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
            colorscale=color_scale, value_range=(-limit, limit),finding_context=(guided_result,guided_finding) if guided_finding else None),
            use_container_width=True, config={"scrollZoom": True, "displaylogo": False,
                "modeBarButtonsToAdd": ["resetCameraDefault3d"]})
    with right:
        st.plotly_chart(volume_figure(monitor, "Monitor · orthogonal sections", inline=inline,
            crossline=crossline, sample=sample, missing_trace_mask=monitor_volume.missing_trace_mask,
            colorscale=color_scale, value_range=(-limit, limit),finding_context=(guided_result,guided_finding) if guided_finding else None),
            use_container_width=True, config={"scrollZoom": True, "displaylogo": False,
                "modeBarButtonsToAdd": ["resetCameraDefault3d"]})
    inline_ids = monitor_volume.metadata.source_metadata.get("inline_ids", monitor_volume.metadata.source_metadata.get("inline_ids_selected", []))
    crossline_ids = monitor_volume.metadata.source_metadata.get("crossline_ids", monitor_volume.metadata.source_metadata.get("crossline_ids_selected", []))
    coordinate_metrics = st.columns(3)
    coordinate_metrics[0].metric("Inline", f"{inline_ids[inline] if inline < len(inline_ids) else inline:,}")
    coordinate_metrics[1].metric("Crossline", f"{crossline_ids[crossline] if crossline < len(crossline_ids) else crossline:,}")
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
    view = st.radio("Synchronized section", ["Sample plane", "Inline section", "Crossline section"], horizontal=True)
    with st.expander("Advanced Exploration",expanded=False):
        controls = st.columns(4)
        with controls[0]:
            sample = st.slider("Sample Index", 0, baseline.shape[2]-1, key="shared_sample")
        with controls[1]:
            inline = st.slider("Inline index", 0, baseline.shape[0]-1, key="shared_inline")
        with controls[2]:
            crossline = st.slider("Crossline index", 0, baseline.shape[1]-1, key="shared_crossline")
        with controls[3]:
            color_scale = st.selectbox("Seismic colour scale", ["RdBu", "Viridis", "Cividis", "Turbo"], index=0)
    st.caption("Selections are Baseline-grid array indices. Sample-domain origin is unverified; Sample Index is not physical time or depth.")
    point_attrs=cached_point_attributes(dataset_cache_identity,inline,crossline,sample,baseline,monitor)
    i0,i1=max(0,inline-48),min(baseline.shape[0],inline+49)
    x0,x1=max(0,crossline-48),min(baseline.shape[1],crossline+49)
    s0,s1=max(0,sample-48),min(baseline.shape[2],sample+49)
    if view == "Sample plane":
        bview,mview=baseline[i0:i1,x0:x1,sample],monitor[i0:i1,x0:x1,sample]
        missing_view=paired_missing[i0:i1,x0:x1];x_title,y_title="Crossline index","Inline index";x_origin,y_origin=x0,i0
    elif view == "Inline section":
        bview,mview=baseline[inline,x0:x1,s0:s1].T,monitor[inline,x0:x1,s0:s1].T
        missing_view=np.broadcast_to(paired_missing[inline,x0:x1],bview.shape);x_title,y_title="Crossline index","Sample Index";x_origin,y_origin=x0,s0
    else:
        bview,mview=baseline[i0:i1,crossline,s0:s1].T,monitor[i0:i1,crossline,s0:s1].T
        missing_view=np.broadcast_to(paired_missing[i0:i1,crossline],bview.shape);x_title,y_title="Inline index","Sample Index";x_origin,y_origin=i0,s0
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
        xs = np.arange(0, values.shape[1], col_stride)+x_origin
        ys = np.arange(0, values.shape[0], row_stride)+y_origin
        section_fig = go.Figure(go.Heatmap(z=shown, x=xs, y=ys, colorscale=scale, zmin=-limit_value,
            zmax=limit_value, zmid=0, colorbar={"title": "Amplitude" if title != "MONITOR − BASELINE" else "Difference"}))
        if missing_view is not None:
            section_fig.add_trace(_missing_overlay_trace(missing_view, x_name=x_title, y_name=y_title,x_offset=x_origin,y_offset=y_origin))
        col.plotly_chart(section_fig.update_layout(
            title=title, xaxis_title=x_title, yaxis_title=y_title, height=540), use_container_width=True)
    section(f"Selected point · Inline {inline} · Crossline {crossline} · Sample Index {sample}")
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
    page_header("Change detection", "Locate unusually large paired amplitude differences on the selected sample plane.")
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
            use_container_width=True,
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
    if guided_finding:
        fig=cached_finding_presentation(guided_result['run_id'],guided_finding['id'],baseline,monitor,paired_missing,guided_result,guided_finding)
        render_presentation(registry,entry,fig)
        st.info("The three upper panels switch between the actual Baseline and Monitor endpoints. The three lower panels keep Monitor − Baseline visible. No interpolated or invented vintages are shown.")
        st.caption("Gold crosses identify the same valid finding member in every section; gold shading preserves selected component membership. Gray cells are missing. Endpoint scales are fixed across surveys per section (99th-percentile absolute amplitude); difference panels have separate symmetric scales in the same amplitude units. Grid indices are not physical coordinates. The animation is a bounded local presentation, not a complete 3D change history.")
    else:
        st.info("No eligible connected ML finding is available in the automatically inspected tile; a focused interpretation animation is not fabricated. Review the report's screening coverage, support and missing well/registration evidence.")

elif page == "Well inventory":
    page_header("Well inventory", "Inspect available well sources and source identifiers without assuming spatial alignment.")
    st.markdown('### Evidence organized by source well identifier')
    st.caption('Workbook identity, LAS curves/units, depth intervals and null counts are source context. Marker/checkshot/deviation records remain unassigned where their well/datum association is unresolved.')
    groups=well_context.get('by_well',{})
    for key,group in sorted(groups.items()):
        with st.expander('Well source group · '+key):
            st.write(group['identity_rule']);st.write(group['workbook_records'])
            st.dataframe([{'source':log['source'],'depth_start':log.get('depth_start'),'depth_stop':log.get('depth_stop'),
                'depth_unit':log.get('depth_unit'),**curve} for log in group['logs'] for curve in log['curves']],use_container_width=True)
    st.write({'supporting_records':well_context.get('tables',[]),'source_errors':well_context.get('errors',[])})
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

else:
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
