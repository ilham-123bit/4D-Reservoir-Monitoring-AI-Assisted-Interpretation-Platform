# Reservoir Time Machine

Reservoir Time Machine is a dataset-independent analytical engine and evidence interpretation platform for 4D seismic surveillance. The workspace currently has zero competition input files. The active application does not fabricate seismic, well, geological, or production data; tests use only small mathematical arrays and parser mocks.

The scientific array convention is `(inline, crossline, sample)`. SEG-Y header geometry is inspected before data are loaded into that convention. Actual trace coordinates and coordinate reference systems are not inferred or invented. See [architecture](docs/architecture.md) and the [Challenge 3 audit status](docs/challenge3_data_inventory.md).

## Environment and dataset configuration

Python 3.10 or later is supported; use the competition Python 3.12 environment on HPC. Install the app and optional actual-data/dashboard tools:

```bash
python -m pip install -e ".[dashboard,competition-data,test,lint,llm]"
```

When authorized inputs are supplied, point to their root. Never copy restricted input data to this public repository. The checked-in YAML is a template only and does not assert that any data exists:

```bash
export RTM_CHALLENGE3_ROOT='<authorized dataset root>'
```

On PowerShell, set `$env:RTM_CHALLENGE3_ROOT` to the authorized dataset location available on that machine. No site-specific input path is embedded in the repository. No competition input files are currently present in this workspace; rerun inventory and validation against authorized inputs when they are available.

## Inspect and run

The audit is read-only. Its report contains file presence and metadata summaries, not raw seismic samples or coordinate values:

```bash
reservoir-time-machine status
reservoir-time-machine audit --output /path/in/authorized/scratch/challenge3-audit.json
# or, from a source checkout:
python scripts/inspect_challenge3.py --root "$RTM_CHALLENGE3_ROOT"
```

Run the dashboard:

```bash
streamlit run app.py
```

If the baseline and monitor files are unavailable, the app displays a tested empty state and does not display substitute results. The dataset contract and supported formats are documented in [data_contract.md](docs/data_contract.md). SEG-Y reading uses the optional `segyio` dependency. Oversized data require an appropriate HPC memory limit or explicitly configured authorized scratch cache via `RTM_MAX_PAIR_MEMORY_BYTES` and `RTM_CACHE_DIR`.

Tests and lint:

```bash
pytest
ruff check .
```

Current implementations include SEG-Y header audit/loading adapters, well-file readers, QC summaries, paired-geometry validation, difference/NRMS attributes, bounded trace-lag registration, statistical screening, connected components, and a Gemma client. A provider-neutral evidence interface and versioned dataset/run/evidence schemas are also provided. No actual competition data are present; no algorithm has been validated against competition seismic. No trained AI model, geological opportunity ranking, or Gemma connectivity is claimed. See [Track B/C roadmap](docs/track_b_c_roadmap.md) for interfaces, limitations and gates.

## Preserved legacy prototype

The earlier synthetic prototype source and tests are preserved under `src/reservoir_time_machine/synthetic/`, related legacy evaluation modules, `legacy/`, and `tests/legacy_synthetic/`. They are excluded from the active app and default test collection. Do not use those modules for Challenge 3 results. Details are in [legacy module notes](docs/synthetic_data.md).

## Documentation

- [Dataset and analytical output contract](docs/data_contract.md)
- [Geometry, integrity, QC and repeatability definitions](docs/qc_and_validation.md)
- [Track B/C assessment and implementation roadmap](docs/track_b_c_roadmap.md)
- [Data inventory and availability](docs/challenge3_data_inventory.md)
- [Data dictionary](docs/challenge3_data_dictionary.md)
- [Scientific methodology](docs/scientific_methodology.md) and [processing assumptions](docs/processing_assumptions.md)
- [HPC workflow](docs/hpc_workflow.md), [Gemma integration](docs/llm_integration.md), and [security/data management](docs/security_and_data_management.md)
- [DUG capability and interoperability audit](docs/dug_insight_workflow.md)
- [Visualization architecture decision](docs/visualization_architecture_decision.md)
- [Scientific roadmap and prioritized backlog](docs/scientific_roadmap.md)
- [Development workflow and team ownership](docs/development_workflow.md)
- [Five-model detection and PyTorch architecture](docs/five_model_detection.md) and [scalable analytics](docs/scalable_analytics.md)

The optional PyTorch Siamese 3D CNN and nonlinear convolutional autoencoder target the authorized Python 3.12 GPU/Slurm environment; they are not installed or executed on Windows. Run the read-only HPC probe with `python scripts/validate_hpc_environment.py` inside the allocated GPU kernel. Training and inference entry points require private HPC data adapters and keep checkpoints, logs, and derived tiles under `RTM_HPC_OUTPUT_ROOT`. See the [HPC model workflow](docs/hpc_workflow.md) before running any workload.
