# Reservoir Time Machine
**APGCE GeoHackathon 2026 · Challenge 3 · Angsi Field**

An AI-assisted 4D seismic prototype comparing Angsi **1995 Baseline** and **2006 Monitor** surveys. It connects measured seismic response with source-linked investigation areas and conditional recovery decisions.

## The three questions

- **Where did the oil go?** Map production-compatible seismic differences and evaluate possible fluid/pressure mechanisms. Fluid type and paths are unconfirmed.
- **Which areas were swept?** Compare coherent stronger- and quieter-response regions; this does not measure volumetric sweep efficiency.
- **Where might hydrocarbons remain?** Screen quieter areas and potential compartments, subject to reservoir quality, detectability and well evidence. No drill-ready target has been established.

## Six-page app

| Page | What it shows |
| --- | --- |
| **Home** | Three questions, selected source-linked area, connected-response map and next decision |
| **01 Confidence & QC** | Pairing, coverage, uncertainty, model evidence and scientific readiness |
| **02 3D Reservoir** | Finding-linked seismic maps and interactive 3D exploration |
| **03 4D Evolution** | Baseline, Monitor, differences, NRMS, synchronized sections and animation |
| **04 Geology & Wells** | Eight-well inventory, curves, markers, checkshots and conditional ties |
| **05 Development Studio** | Ranked investigations, options, evidence gates and engineering roles |

**Ask Gemma about this view** is embedded throughout; the local evidence assistant works offline. Live Gemma requires approved service configuration and data-sharing consent.

## Actual Angsi results

- Full-field Stage 1 attributes, Stage 2B classical ML and a checksum-verified **private full-field registry** containing **10 seismic findings** are complete.
- Paired attributes use exact Baseline-grid trace keys. **3,087 Monitor-only traces are excluded**, retained separately for coverage diagnostics.
- Coherent changed/quieter regions are mapped from supported local NRMS using actual SEG-Y inline/crossline/sample indices. Missing and low-energy traces cannot become false quiet-zone targets.
- GMM, Isolation Forest, Mahalanobis and PCA **flagged zero voxels in all 10 priority components**; these MAD-only component findings lack independent multivariate confirmation.
- I-100 comparisons at Wells 5–7 use conditional timing assumptions, not verified reservoir ties.

**Limits:** Physical 4D registration, reservoir horizons, well-to-seismic time/depth ties, fluid type, flow paths, bypassed oil and drilling success are unverified. Detector rankings are not geological probabilities.

## Run / verify

```bash
python -m pip install -e '.[dashboard,competition-data,ml,llm,test]'
export RTM_CHALLENGE3_ROOT=./authorized/challenge3
export RTM_CACHE_DIR=./authorized/paired-cache
PYTHONPATH=src streamlit run app.py
```

```bash
PYTHONPATH=src python -m pytest -q
python scripts/check_public_docs.py
```

The existing authorized HPC deployment uses the original desktop launcher, autosync watcher and private registry pointer. `scripts/hpc/activate_full_field_registry.sh` publishes completed artifacts without rerunning ML.

**Submission scope:** source-linked change detection, explainable seismic-screening hypotheses and conditional development investigations—not a verified fluid-flow simulator, sweep fraction or approved well plan.

Only application source and tests are published. Organizer seismic/well data, private artifacts, credentials and runtime settings are excluded.

See [scientific and product acceptance](docs/rtm-product-contract.md).
