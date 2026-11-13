# Reservoir Time Machine
**APGCE GeoHackathon 2026 · Challenge 3 · Angsi Field**

An AI-assisted, source-linked application comparing Angsi's **1995 Baseline** and **2006 Monitor** 3D seismic surveys. The goal is to move from measured time-lapse response to understandable reservoir hypotheses and ranked **investigation** decisions.

## The three questions

- **Where did the oil go?** Investigate spatial 4D response and reservoir connectivity to develop competing fluid/pressure movement hypotheses. Fluid type and flow paths are **not confirmed** from the supplied evidence.
- **Which areas have been swept efficiently?** Contrast coherent stronger- and lower-response regions. This is a **seismic production-response screen**, not a measured volumetric sweep efficiency.
- **Where might hydrocarbons remain?** Map relatively quiet areas adjacent to changed areas and assess possible limited-drainage compartments. A quiet 4D signal alone does not identify bypassed oil, and no drill-ready target has been qualified.

## Six-page judging experience

| Page | Purpose |
| --- | --- |
| **Home — Reservoir Intelligence** | Three concise questions, selected source-linked finding, connected-region NRMS map, locations and conditional next decision |
| **01 — Confidence & QC** | Survey pairing, evidence coverage, registration uncertainty, model support and limitations |
| **02 — 3D Reservoir** | Interactive seismic sections, 3D map/explorer, selected regions and supporting geological context |
| **03 — 4D Reservoir Evolution** | Baseline, Monitor, difference, NRMS, comparative views and time-lapse visualizations |
| **04 — Geology & Wells** | Eight-well inventory, available logs, markers, checkshots and conditional well-to-seismic context |
| **05 — Development Studio** | Ranked **seismic investigation** areas, geological hypotheses, pre-drilling evidence gates and discipline-specific next decisions |

A context-aware **Ask Gemma about this view** panel is embedded across the six pages. Its bounded local evidence assistant works without an external model; live Gemma needs approved endpoint and data-sharing configuration.

## Implemented Angsi analysis

- Exact Baseline-grid trace-key pairing of the two surveys; **3,087 Monitor-only traces remain excluded from all paired 4D attributes** and available separately for coverage diagnostics.
- Completed full-field Stage 1 attributes and Stage 2B classical-ML screening. A checksum-verified **private full-field registry** was published on the HPC with **10 source-linked findings**. Results are loaded from private runtime configuration, not checked into GitHub.
- A coherent-region screen contrasts connected stronger and relatively quiet 1995–2006 seismic response within bounded, supported windows, using real SEG-Y inline/crossline and sample IDs. It excludes missing, nonfinite and low-energy traces and tests spatial separation against local variability.
- Stage 2B model outputs include GMM, Isolation Forest, Mahalanobis and PCA; **none flagged any voxels in the 10 published priority components**. Thus the component shortlist does **not** have independent multivariate-detector confirmation. The archival high-priority findings are MAD-only.
- Conditional I-100 well-window tests assess Wells 5–7 under three assumed time origins. These are sensitivity experiments, not nine confirmed reservoir discoveries.
- Candidate evidence, alternative physical/processing hypotheses and decision prerequisites are shown without converting detector scores into hydrocarbon probabilities.

**Scientific boundary:** Physical 4D registration, the absolute seismic time origin, verified reservoir horizons, well-to-seismic tie, fluid type, production-linked movement and remaining oil saturation are **not independently established**. No volumetric sweep fraction, calibrated recovery Chance of Success, proved bypassed-hydrocarbon body or drill-ready well has been claimed.

## Run and validate

Use only authorized datasets and an existing paired cache:

```bash
python -m pip install -e '.[dashboard,competition-data,ml,llm,test]'
export RTM_CHALLENGE3_ROOT=./authorized/challenge3
export RTM_CACHE_DIR=./authorized/paired-cache
PYTHONPATH=src streamlit run app.py
```

For the existing APGCE HPC deployment, the original desktop launcher and autosync watcher serve the canonical application. The private registry is referenced in `~/.config/reservoir-time-machine/runtime.json` under `full_field_registry`. Publication from completed artifacts uses `scripts/hpc/activate_full_field_registry.sh` with the verified H200 Slurm node constraint; it **does not rerun Stage 1/2B ML**.

```bash
PYTHONPATH=src python -m pytest -q
python scripts/check_public_docs.py
```

Merged source changes passed the repository's documentation, numerical smoke and candidate-release checks. Desktop/browser visual acceptance, runtime fingerprint and any live Gemma endpoint must be verified independently on the authorized HPC session; CI is not a substitute.

## Submission and data protection

This repository contains the application, test harnesses and scientific workflow code. It does **not** distribute organizer SEG-Y/LAS volumes, private derived seismic outputs, model checkpoints, credentials or private per-user runtime settings. Source-linked positions and the full candidate record remain within authorized HPC artifacts.

The prototype provides **measured change detection, candidate visualization and explainable conditional investigation**; it does not assert that seismic anomalies alone prove oil movement, sweep efficiency or drilling prospects.

See [the scientific and product acceptance contract](docs/rtm-product-contract.md) for the complete definitions and safeguards.
