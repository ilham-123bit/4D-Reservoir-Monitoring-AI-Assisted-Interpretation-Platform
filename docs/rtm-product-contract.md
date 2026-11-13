# Reservoir Time Machine — scientific and submission contract
**APGCE GeoHackathon 2026 · Challenge 3 · Angsi Baseline 1995 / Monitor 2006**

## Product objective

Use actual paired time-lapse seismic and supplied well evidence to answer, as far as the evidence permits:

1. **Where did the oil potentially go?** Map production-compatible seismic changes; test physical mechanisms and plausible fluid/pressure connectivity without inventing a measured movement trajectory.
2. **Which reservoir areas were effectively swept?** Compare reservoir-associated coherent stronger- and lower-response areas; reserve quantitative volumetric sweep efficiency for validated dynamic and saturation evidence.
3. **Where might untapped hydrocarbons remain?** Combine demonstrable reservoir quality, spatial detectability, relatively limited response, compartment continuity/barriers and well/production constraints before describing a potential bypassed-hydrocarbon prospect.
4. **What should engineers investigate?** Rank actual source locations and distinguish exploration/appraisal hypotheses from drill-ready engineering decisions.

Two vintage snapshots constrain an endpoint comparison; any visualization between them is illustrative, not an observed annual fluid-flow history.

## Source and processing rules

- Match exactly by Baseline-grid trace keys. The **3,087 Monitor-only traces** must never enter paired 4D calculations; preserve them separately for edge/coverage diagnostics.
- Preserve valid/missing trace support, finite sample support, original SEG-Y IL/XL/sample identities, checksums and cache lineage.
- Completed Stage 1/Stage 2B source outputs must be reused where valid; do not rerun costly whole-volume processing on page navigation.
- A private, checksum-verified full-field registry is the source for the active 10-component seismic shortlist. Do not confuse the historical Stage 3 QC queue with geological prospects.
- MAD, PCA, GMM, Mahalanobis and Isolation Forest identify statistical characteristics, not fluid or geological truth. The 10 published components have **zero flagged voxels from each of the four multivariate detectors**; no independent model consensus supports them. Review their evidence and alternatives honestly.
- Missing, nonfinite, edge-sensitive and low-energy positions are not unswept-hydrocarbon signals; sample-time, reservoir-mask and survey-repeatability uncertainty must be carried into interpretation.

## Interpretation maturity

**Measured:** relative seismic changes, amplitudes, NRMS, correlation, finite support, candidate footprints and original source-grid identifiers.

**Conditional:** stronger/lower response patterns, possible depletion or pressure/saturation scenarios, geological barriers and limited-sweep hypotheses.

**Not established by this release:** independently verified physical 4D registration; absolute seismic time zero; well-to-seismic tie/reservoir top; reliable pressure or oil/gas/water identification; true fluid-flow paths; remaining hydrocarbon saturation; volumetric sweep percentages; calibrated recovery probability; sanctioned infill drilling target.

Every prominent interpretation should include a real source-grid location, observed measurement, mechanism(s), decisive alternative and a specific next decision. Do not pad Home or Development Studio with repetitive QC essays. Provide one concise scientific-readiness summary in Confidence & QC; retain appropriate contextual status elsewhere without hiding important uncertainty.

## Six pages — fixed presentation structure

1. **Home — Reservoir Intelligence** (unnumbered): *What changed and where? Why might it have changed? Where to investigate next?* A synchronized selected finding, local connected-response map and short decision.
2. **01 — Confidence & QC:** coverage, registration, well tie, model flags, detectability, edge risks and uncertainty evidence.
3. **02 — 3D Reservoir:** candidate-linked maps, 3D seismic/structural context and available well locations.
4. **03 — 4D Reservoir Evolution:** comparable Baseline and Monitor, difference, NRMS, source-linked inline/crossline/sample and bounded visual evolution.
5. **04 — Geology & Wells:** available well curves, checkshots, markers, conditional ties, impedance and geological evidence.
6. **05 — Development Studio:** source-grid candidate investigations, engineering options, assumptions and pre-drilling gates.

No extra top-level page for Gemma, Recovery Intelligence, Change Detection or Expert diagnostics. Existing specialized tools may remain subordinate sections. No manual coordinate sliders in the judge-facing workflow.

**Gemma:** a contextual, collapsible assistant on all six pages. Supply bounded evidence summaries with active page, finding, source locations and plot values. A local grounded fallback is not a live Gemma model. Live inference requires a verified approved endpoint, restricted-data permission and explicit consent; never transmit raw SEG-Y, LAS, secrets or private paths.

**Visual style:** black/dark background, bright white type and turquoise accents. Validate text/control contrast at rest (not only on hover). Keep plots readable and favor concise evidence-led conclusions.

## Acceptance and release

- Automated tests: Python compilation, full source tests, paired/missing trace invariants, numerical and false-oil safeguards, AppTest navigation/synchronization and privacy checks.
- HPC reality: published full-field registry checksum and source identity; synchronized canonical source and GitHub `main`; the original desktop launcher, Streamlit health, selected actual-data views and all six pages inspected on the authorized HPC session.
- CI green verifies software behavior; **it does not independently prove fluid type, bypassed oil, reservoir sweep or physical registration**.
- Preserve all private cache files, historical release artifacts and unrelated operator work. Never upload organizer data or confidential derived volumes to the public GitHub repository.
- Keep a tested, reproducible `main` as the submission source. Cleanup historical branches only after verifying that no unique current implementation is lost. Final release status must separate completed, conditional and blocked scientific/operational items.

**Final submission is a scientifically honest prototype:** completed 1995–2006 seismic difference analytics, coherent regional comparison, unsupervised ML screening, source-linked 3D/4D visualization and ranked investigation recommendations. It is **not** a validated hydrocarbon-flow simulator or an approved drilling plan.
