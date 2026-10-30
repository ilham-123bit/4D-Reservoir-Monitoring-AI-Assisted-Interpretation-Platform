# Architecture and implementation status

## Active data path and zero-input behavior

`settings.py` obtains the authorized dataset root and memory settings from environment variables. `data_audit/` inventories expected inputs and extracts privacy-conscious metadata. `data_ingest/segy_loader.py` checks paired SEG-Y trace/sample geometry and standardizes data as `(inline, crossline, sample)`. `data_ingest/` also contains explicit spreadsheet, LAS, and text-table adapters. The `SeismicVolume` and `SurveyMetadata` types are shared by loading, processing, and display code.

The active UI (`app.py`) loads supplied baseline/monitor files only. If required inputs are missing, it shows a no-data state, lists missing paths, links to the data contract, and stops before creating charts or results. It does not import or call the legacy generator. The CLI supports `status` and read-only `audit` commands.

No competition inputs currently exist in the workspace. Previously copied incomplete volumes were deleted. Historical actual-data-derived audit artifacts are excluded from publication and are not active inputs or validation. Do not reconstruct or process from them.

## Scientific modules

Scalable chunked processing, memmapped analytical outputs, provenance, spatial transforms, and multi-attribute screening are documented in [scalable analytics](scalable_analytics.md). The common detector/evidence contract and five model statuses are in [five-model detection](five_model_detection.md).

Optional PyTorch 3D model builders and HPC-only training/inference orchestration are under `ai/torch_models.py` and `hpc/`. They import framework packages lazily and do not initialize or allocate a GPU in the Windows source-development environment.

- `preprocessing/`: quality statistics and optional, explicit normalization.
- `registration/`: bounded integer and fractional trace-lag diagnostics, exact trace-key reindexing, and explicitly parameterized affine grid resampling. Transform estimation and spatially regularized temporal warping remain incomplete.
- `data/validation.py`: streaming SEG-Y integrity, source hash/size/trace-count gates, explicit trace-key reindexing, coordinate checks, geometry compatibility, and machine-readable validation findings.
- `preprocessing/repeatability.py`: chunked global and local-tile waveform correlation/NRMS diagnostics over joint finite masks; no universal repeatability threshold is claimed.
- `attributes/`: signed/absolute/normalized difference, local RMS and windowed NRMS.
- `detection/`: statistical scoring and connected components. Scores are strengths, not calibrated probabilities.
- `visualization/`: display-only downsampling, volume and endpoint-animation figures.
- `llm/`: optional evidence-summary client, isolated from numerical processing.

## Dataset boundary

There is no current actual-data audit or actual-data processing. Current environment dependencies and module maturity are recorded in [Track B/C roadmap](track_b_c_roadmap.md). The formal input/output contract is in [data_contract.md](data_contract.md).

Prior synthetic generator/configuration/scenario/evaluation modules and their demo app/tests are retained for provenance but are not imported by the active app or package top-level API. See [legacy modules](synthetic_data.md).

## Product architecture decision

The strategic target is a hybrid DUG plus custom Python product. DUG is the preferred expert-led interpretation/QC companion if the competition installation and licences provide the relevant functions. Python remains the authoritative reproducible processing, detector, evaluation, evidence-ranking and integrated visualization layer. The current environment has not verified DUG installation, licence modules or session exports; see [DUG capability and integration audit](dug_insight_workflow.md) and [visualization architecture decision](visualization_architecture_decision.md).

The revised phases and implementation gates are in [scientific development roadmap](scientific_roadmap.md). This direction preserves the original ambition; it does not claim that advanced AI, actual-data interpretation, well integration or custom 3D performance has already been delivered.
