# Reservoir Time Machine

**AI-assisted 4D seismic screening and reservoir investigation support**
APGCE GeoHackathon 2026 · Challenge 3

Reservoir Time Machine compares paired Baseline and Monitor seismic surveys, screens for spatially coherent changes, and organizes evidence for geoscience and reservoir-engineering review. It distinguishes measured seismic changes, statistical detections, geological hypotheses, and recovery decisions.

## Angsi scientific status

The authorized Angsi run completed Stage 1 attributes and Stage 2B classical screening over the eligible Baseline grid. The separate Stage 3 assessment reports that the ten archived Stage 2 findings were MAD-only, while the detector-consensus set is concentrated near a survey edge. Physical 4D registration, the sample-time origin, and well-to-seismic ties remain unverified. There are currently **zero verified reservoir-interval associations and zero evidence-supported recovery opportunities**. The application therefore prioritizes QC and surveillance actions; it does not report a drilling target or a recovery probability.

Stage 2B outputs remain historical screening results. The Stage 3 interpretation package is checksummed and loaded independently; it does not rewrite Stage 2 rankings or recompute full-volume seismic/ML products during page navigation.

## Run with authorized Angsi data

Python 3.12 is the validated HPC runtime. Install the dashboard and data-reader extras in an appropriate environment:

```bash
python -m pip install -e '.[dashboard,competition-data,ml,llm]'
export RTM_CHALLENGE3_ROOT=/path/to/authorized/challenge3
export RTM_CACHE_DIR=/path/to/authorized/paired-cache
export RTM_STAGE3_EVIDENCE_DIR=/path/to/checksummed/stage3-evidence
streamlit run app.py
```

The input directory must contain the authorized organizer dataset. `RTM_CACHE_DIR` may point to an existing paired cache; no cache or analysis is included in this repository. `RTM_STAGE3_EVIDENCE_DIR` must contain the checksummed Stage 3 package. If inputs are missing or incompatible, the application shows an empty/error state instead of substituting synthetic Angsi results.

## Synthetic demonstration

The legacy synthetic dashboard is explicitly labeled synthetic and is for interface/method smoke demonstrations only. Its injected scenarios are not Angsi observations, geological labels, rock-physics simulations, or evidence of remaining hydrocarbons.

```bash
python -m pip install -e '.[dashboard]'
PYTHONPATH=src streamlit run legacy/synthetic_dashboard.py
```

## Tests

```bash
python -m pip install -e '.[test,dashboard,competition-data,ml]'
PYTHONPATH=src:tests python -m unittest discover -s tests -q
```

Some legacy tests use pytest fixtures and should also be run with `pytest` when the test extra is installed.

## HPC neural experiments

Bayesian evidence fusion, a nonlinear 3D convolutional autoencoder, and a shared-encoder Siamese 3D CNN have code paths. Bayesian output is a model-assumption likelihood ratio unless external calibration is independently verified. Neural example configurations use controlled synthetic perturbation/corruption targets on seismic texture; they are computational experiments, not geological ground truth. No validated Angsi neural checkpoints are included or implied.

Use the documented site-approved Slurm environment, private input/cache paths, and `scripts/hpc/submit_training.sh` only from the authorized HPC login environment. Example resource settings must be checked against current site policy. Do not run large-volume processing in the Streamlit process or on a login node. Training checkpoints and scientific outputs belong in authorized private storage and must not be committed here.

## Data and privacy

No proprietary SEG-Y volumes, paired caches, private result packages, model checkpoints, credentials, or HPC endpoint addresses belong in this repository. Configure authorized data and service paths through environment variables. Gemma integration is opt-in, accepts compact typed evidence only, and remains disabled until an approved compatible endpoint and data-sharing policy are configured.
