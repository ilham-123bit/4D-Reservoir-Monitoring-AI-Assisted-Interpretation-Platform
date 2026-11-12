# Reservoir Time Machine

**APGCE GeoHackathon 2026 · Challenge 3**

Reservoir Time Machine compares the 1995 Angsi Baseline and 2006 Monitor seismic surveys. Its objective is to connect measured 4D differences with reservoir hypotheses, evidence quality, and recovery investigations—not to equate an anomaly with remaining hydrocarbons.

## Scientific status

Full-field Stage 1 seismic attributes, Stage 2B classical ML screening, and the checksummed Stage 3 review are complete. The archived ten leading findings are MAD-only. Multivariate consensus is concentrated near the survey boundary. Exact trace-key pairing is available on the Baseline grid; physical registration and reservoir ties are not independently verified. No drill-ready recovery target or calibrated drilling-success probability is established.

The eight-well inventory includes logs, markers, checkshots, and deviation information. Well-local I-100 marker times in Wells 5–7 allow **conditional** seismic-window comparisons under explicitly assumed coordinate and time origins. These tests quantify actual paired seismic response, nearby comparison windows, and small-lag sensitivity; they do **not** verify reservoir location, fluid movement, or sweep.

## Operator workflow

Use a validated existing cache and the authorized organizer inputs:

```bash
python -m pip install -e '.[dashboard,competition-data,ml,llm,test]'
export RTM_CHALLENGE3_ROOT=./authorized/challenge3
export RTM_CACHE_DIR=./authorized/paired-cache
export RTM_STAGE3_EVIDENCE_DIR=./authorized/stage3-evidence
PYTHONPATH=src streamlit run app.py
```

The **Reservoir Command Center** includes conditional I-100 evidence and preserves historical Stage 3 screening. Finding-directed seismic comparisons, NRMS, coverage, support, spatial displays, and animations use real cached surveys. Manual grid sliders are isolated in Expert diagnostics. Interpolated animation frames are visualizations, not acquired surveys.

For a versioned, bounded scientific result, run in an authorized environment:

```bash
PYTHONPATH=src python scripts/hpc/run_conditional_i100.py \
  --root ./authorized/challenge3 \
  --cache-dir ./authorized/paired-cache \
  --output ./private-results/conditional-i100.json
```

The output directory must be private. The run reads existing paired arrays; it never launches full-field processing.

## Validation

```bash
PYTHONPATH=src python -m unittest discover -s tests -q
python -m pytest -q
python scripts/check_public_docs.py
```

The repository includes classical detectors and experimental Bayesian, Autoencoder, and Siamese implementations. Neural smoke tests and uncalibrated model scores are not verified Angsi fluid classifications. Real-data interpretation requires scientific validation independent of model agreement.

## Reuse and privacy

New deployments require authorized datasets, correct survey geometry, existing or scheduled processing, and source-specific well references. Heavy computation runs outside Streamlit using configured resources and saved artifacts. This repository contains software and configuration templates, **not** organizer SEG-Y/LAS inputs, private derived data, checkpoints, secrets, or internal service endpoints.

The application differentiates observations, conditional hypotheses, and sanctioned development decisions. No hypothesis may be promoted to a drilling recommendation without adequate evidence.
