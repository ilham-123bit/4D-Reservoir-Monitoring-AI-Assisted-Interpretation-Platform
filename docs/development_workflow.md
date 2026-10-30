# Development workflow and team ownership

## Setup and execution

```bash
python -m pip install -e ".[dashboard,competition-data,test,lint,llm]"
export RTM_CHALLENGE3_ROOT='<authorized dataset root>'
reservoir-time-machine status
reservoir-time-machine audit --output "$TMPDIR/challenge3-audit.json"
pytest
ruff check .
streamlit run app.py
```

Use the authorized HPC paths for all restricted inputs and derived data. Keep unit tests limited to small in-memory mathematical arrays and parser mocks. Run actual-data integration checks only on HPC and store reports/derived cubes in authorized scratch. Do not commit restricted data or outputs.

## Five workstreams

| Workstream | Ownership | Shared interface |
|---|---|---|
| A | SEG-Y/well ingestion, geometry, manifest, QC | `SeismicVolume`, `SurveyMetadata`, `data_ingest/` |
| B | Preprocessing, registration, attributes | validated paired arrays and attribute functions |
| C | Conventional/ML detection, false-positive review, evaluation | detector result and metrics contracts |
| D | Well integration, geological alternatives, opportunity evidence | explicitly mapped well data and evidence records |
| E | Dashboard, visualization, Gemma, product integration | app page contracts and structured evidence schema |

### Person-level ownership and handoffs

| Person | Owns | Hands off | Required review |
|---|---|---|---|
| A — Data and QC | SEG-Y forensic audit, header/geometry adapters, well-file inventory and parsers, validity masks, manifest | Validated `SurveyMetadata`/`SeismicVolume`, data-availability and QC reports | B reviews geometry/axis assumptions; D reviews well provenance and missingness |
| B — 4D geophysics | Repeatability analysis, bounded registration, cross-equalization diagnostics, difference/NRMS/multi-scale attributes | Immutable processing products, configuration and registration-quality metrics | A reviews input compatibility; C reviews attribute suitability for detection |
| C — Detection and evaluation | Conventional baselines, local robust detectors, spatial coherence/components, ML experiment protocol, metrics and stability analyses | Detector contract with score semantics, masks, diagnostics, runtime and evaluation report | B reviews alignment/attribute dependence; team reviews false-positive interpretation |
| D — Well and interpretation | Well trajectory/reference mapping, checkshot/marker/log linkage, competing explanations and transparent evidence ranking | Provenance-aware well context and evidence records with unknowns/uncertainty explicit | A verifies source mappings; B/C verify attribute and score interpretation |
| E — Product and integration lead | Dashboard, linked visualization, animation, report export, Gemma adapter, integration coordination and release demo | Integrated user flow, evidence schema, launch/performance record and demo narrative | Every technical owner signs off the views of their results; E does not replace their methods |

Integration contract: modules exchange typed metadata/results and explicit masks; raw data remains in authorized storage. Agree schemas before parallel changes. Gemma consumes compact structured evidence, never full seismic arrays. DUG exports enter through validated file adapters, not private session parsing. Each owner supplies tests and a short actual-data validation record when authorized HPC execution becomes possible.

## Feature branch and review workflow

Each contributor works on a feature branch from the agreed base and owns their listed modules. Agree on shared data/attribute schemas before parallel implementation. Pull requests must describe evidence and assumptions, include small-array tests, avoid restricted files, and pass review by one scientific owner and one integration owner. Workstream E coordinates the release candidate; it does not reimplement A–D algorithms.

No GPU or Slurm job should be submitted without explicit user approval. Do not claim actual-data, DUG Insight, LLM, or GPU validation until it has been performed and recorded.
