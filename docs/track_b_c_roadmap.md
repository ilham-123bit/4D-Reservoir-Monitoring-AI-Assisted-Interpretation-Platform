# Track B/C architecture and implementation roadmap

## Repository assessment (2026-10-30)

The workspace is the existing `main` checkout. No competition inputs are available. The working tree contains local implementation and documentation changes; preserve them. No data should be generated to fill the input gap. Historical actual-data-derived reports and figures are excluded from publication.

Useful existing components include SEG-Y/LAS/workbook/table adapters, metadata models, QC, difference/NRMS, bounded integer temporal registration, statistical detection, connected components, visualization helpers, a no-input stop state, CLI audit/status, and 21 passing math/mock tests. Incomplete items include formal manifest validation, a stable analytical output store, spatial correspondence/registration, scalable chunked attributes, well/geological joins, uncertainty-aware ranking, provider-neutral AI contracts, and actual-data validation. Legacy synthetic generator and synthetic tests remain isolated for provenance and must not run in the active dashboard or actual-data workflow.

The baseline environment is Python 3.11.7. NumPy, SciPy, pandas, ObsPy, pytest and Matplotlib are installed. Streamlit, Plotly, `segyio`, `lasio`, `openpyxl`, Ruff, JSON Schema validation libraries, PyVista/VTK, Zarr, and OpenAI client are absent. No dependencies were installed. This means core mathematical tests can run, while dashboard rendering, source adapters, schema validation by an external validator, DUG, and Gemma are not presently executable here.

## Layered architecture

```text
Authorized source files / DUG exports
          │
          ▼
1. Ingestion adapters ── source hashes, headers, raw provenance, validity masks
          │
          ▼
2. Canonical models ───── seismic volumes + geometry + wells + interpreted products
          │
          ▼
3. QC/quality gates ───── verified dimensions, sample domain, CRS, trace correspondence
          │
          ▼
4. Analytical engine ─── attributes → detectors → clusters → uncertainty/evidence
          │
          ▼
5. Run store ─────────── immutable run manifest + derived arrays/tables/figures
          ├──────────────────► 6. Visualization (Streamlit/Plotly; optional VTK/PyVista)
          ├──────────────────► 7. Evidence provider (Gemma or other; optional)
          └──────────────────► 8. Reports (JSON/CSV + figures/PDF after validation)
```

The analytical library must not import Streamlit, Plotly, DUG APIs, or an LLM client. Visualization consumes versioned output artifacts, not hidden module state. DUG integration is file-based and adapter-driven; no native session parsing or direct DUG dependency is assumed. Contract artifacts are `schemas/dataset-manifest.schema.json`, `schemas/analysis-run.schema.json`, `schemas/evidence-packet.schema.json`, and `docs/data_contract.md`.

## Track B — analytical engine backlog

1. **Manifest and ingestion gates:** implement schema validation; typed axis/CRS/sample-domain/missingness contracts; streaming SEG-Y metadata scan; correct integrity checks; loss-preserving adapters for SEG-Y/LAS/CSV/XLSX. Exit: mock fixtures and one authorized source subset with preserved metadata.
2. **Geometry and survey compatibility:** use explicit bin keys and verified navigation; report coverage, duplicates, missing traces, coordinate transform, and reject incompatible pairs. Exit: known geometry fixtures, negative mismatch tests, explicit acceptance report.
3. **Repeatability and registration:** windowed correlation/coherence, bounded integer then sub-sample estimation, spatial consistency, confidence/rejection masks. Keep original and corrected arrays traceable. Exit: known-lag math fixtures, edge/missing-data tests, no unrestricted warp.
4. **Attribute engine:** chunk-aware signed/absolute/normalized differences, local RMS/NRMS, phase/time shift where justified, optional filters with immutable originals. Exit: hand-computed formulas, zero energy, NaN, window-edge, chunk equivalence tests.
5. **Detection and validation:** conventional thresholds, local robust scores, multivariate evidence fusion, 3D components/clustering, stability checks. Exit: controlled small mathematical masks and negative controls; never describe them as geological benchmark performance.
6. **Well/geological association:** CRS/depth/time gated joins for wells, trajectories, checkshots, LAS, markers, horizons/faults. Exit: explicit missing-data behavior and hand-checkable coordinate/depth fixtures.
7. **Uncertainty/opportunity evidence:** evidence ledger separates measurement/detection/hypothesis/recommendation; transparent score contributions and abstention. Exit: bounds/order tests and explanations of missing support; no remaining-oil probability claim.
8. **Reproducible output store:** write run manifest and content-addressed artifact records, provenance and resource use; chunked scientific array format selected by HPC profiling. Exit: roundtrip/hash test on temporary mathematical arrays.

## Track C — visualization and AI backlog

1. Maintain graceful zero-data state and show the manifest/setup contract; do not show empty placeholder plots as observations.
2. Build synchronized baseline/monitor/attribute slice views driven by stored analysis products and true axis metadata; keep display downsampling separate from scientific arrays.
3. Add interactive 3D volume/cluster surfaces only after measuring target volume size and browser/VTK/GPU memory. Plotly is suitable for linked slice views; optional PyVista/VTK should be justified by measured rendering benefit.
4. Add endpoint animation with exactly two measured surveys; label interpolation and never depict it as measured vintages or a fluid path.
5. Add evidence-linked candidate inspection and report export; show input provenance, QC gates, scores, uncertainty, alternatives, and missing support.
6. Integrate the provider-neutral `EvidenceProvider`; use Gemma only for explanation of bounded evidence packets, explicitly gate restricted-data transmission, and never allow LLM output to overwrite measurements or detector outputs.

## Current implementation increment

- Dataset, run, and evidence JSON Schema contracts and an example YAML configuration template were added. They define required/optional/conditional fields, but no input data are embedded and external JSON Schema validation is not installed.
- `docs/data_contract.md` records supported file formats and their current adapter limitations.
- `ai/interfaces.py` adds typed evidence, status/classification fields, payload limits, a provider protocol, and a Gemma adapter. Restricted/confidential/unknown evidence is denied unless explicitly enabled; the dashboard requires an explicit authorization checkbox before submitting its bounded packet.
- `visualization/dashboard_state.py` makes the no-input presentation data pure/testable; the app now explains the empty state and links to the contract.
- Existing legacy source, generated artifacts, and old reports were not deleted.

## Validation gates and test strategy

### Unit tests (local, no reservoir fixtures)

- Exact difference/NRMS sign and formula, zero-energy, NaNs, window edges, and independent chunk equivalence.
- Registration using known integer/fractional shifts, noise, missing samples, lag limits, and rejection behavior.
- Geometry identity/mismatch, bin-key mapping, coordinate transform, axis/unit/domain requirements.
- Detector masks/components/metrics on small hand-constructed arrays; test no truth passed into detectors.
- Manifest/run/evidence schema validation; required versus unknown/missing fields; hash and provenance handling.
- Provider contract with fake providers, no network calls, payload limits, prompt evidence separation, restricted-data gating.
- Dashboard no-input state and figure-data preparation without Streamlit imports.
- Run-store round trips and hash checks using temporary small arrays only.

### Integration and acceptance tests (only when authorized data is supplied)

Read-only integrity/hash check; survey geometry/sample-domain gate; streamed/subset loading; DUG export/Python sample parity; full pipeline resource profile on approved compute; DUG visual QC; optional Gemma connectivity separately. No competition data should be used in public test fixtures, and no actual-data validation is claimed by the local unit suite.

## Dependency policy

Keep NumPy/SciPy as the numerical core. Keep Streamlit/Plotly, `segyio`, `lasio`, `openpyxl`, JSON Schema validation, OpenAI-compatible clients, Zarr, and PyVista/VTK optional by layer. Do not install them in this environment without authorization. A small in-memory JSON Schema contract check uses only the standard library; full schema validation should use an optional validator in CI/HPC. Avoid requiring GPU libraries until CPU profiling proves a workload benefits from acceleration.
