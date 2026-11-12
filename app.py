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
from reservoir_time_machine.data_ingest.cache_identity import paired_cache_identity, paired_cache_provenance
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
    conditional_follow_up_actions, conditional_i100_locations, conditional_scenario_key,
    evaluate_conditional_window, load_conditional_i100_evidence,
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
def cached_mon