# Reservoir Time Machine — final product and acceptance contract

**Mission:** APGCE GeoHackathon 2026 Challenge 3. Interpret Angsi 1995 Baseline vs 2006 Monitor with paired seismic and eight-well source evidence. The primary user journey is **AI/ML detects → validates → selects → visualizes → interprets → recommends → reservoir team decides**. Do not substitute a generic QC dashboard for this product.

## Three required answers

1. **What is happening?** Show actual paired 4D attribute measurements, location, spatial extent, seismic character, relevant time/sample, support, and alternative explanations.
2. **Which changes matter?** Run completed whole-eligible-field statistical/ML evidence through repeatability, registration, missing-trace, survey-edge, and source-provenance gates. Prioritize 5–10 impactful scientific **investigations** by urgency, coherent support, decision value, uncertainty and recovery relevance—not raw amplitude or largest NRMS. Distinguish historical MAD-only ranks from independent detector support.
3. **Where is the next recovery opportunity?** Show a conditional geoscience hypothesis, evidence for and against, what a reservoir team should test next, and a clear unmet-evidence gate. Never label a drilling prospect, sweep state, bypassed oil, fluid, pressure mechanism, or calibrated chance of success as verified without independently adequate evidence.

## Scientific contract

- Pair Baseline and Monitor by exact common trace keys on the Baseline grid. Exclude all **3,087 Monitor-only traces** from paired attributes and preserve them only for separate edge/coverage diagnostics.
- Preserve completed full-field scientific output and checksummed lineage; never rerun Stage 1 or Stage 2B without reason and authorization.
- Always display: **"Trace-key pairing is available on the Baseline grid. Physical 4D registration is not yet verified. Interpret paired attributes as preliminary screening only."**
- No independently verified physical 4D registration, shared well/SEG-Y CRS or datum, well-to-seismic tie, absolute seismic sample origin, validated horizon/fault interpretation or production/injection/pressure response was provided in this release context. Do not silently substitute assumptions.
- I-100 source-local marker/checkshot scenarios for Wells 5–7 at assumed origins −20/0/+20 ms are nine **scenarios, not nine discoveries**. The real-data evaluation found 525/525 supported pairs in each, NRMS 14.18–40.10%, 0.92–0.99 correlation and origin-sensitive signed amplitudes; only Well-6 +20ms was just above its off-window threshold. Local controls are not verified unchanged reservoir controls.
- Historical full-field top ten are MAD-only; multi-detector consensus is concentrated near a survey edge. Bayesian, autoencoder and Siamese Angsi scientific outcomes remain unvalidated until data-specific training, calibration and acceptance.
- Segment every output into **Observed**, **Conditional interpretation**, **Alternatives**, **Confidence/limitations**, **Recovery relevance**, and **Next test**. Never turn an uncalibrated score into a recovery probability.

## Interface contract

- One existing Streamlit product, black background with bright white/high-contrast neon turquoise/green, large readable key figures.
- Default **Reservoir Command Center** answers the three questions above with a synchronized selected finding.
- Finding-directed **Baseline, Monitor and Monitor − Baseline** slices; shared comparative seismic amplitude scaling and centered difference palette; synchronized inline/crossline/sample, 4D comparison, 3D discovery/map, 3D explorer and selected evidence/report state. Source SEG-Y IDs and array indices must be labeled separately.
- No manual IL/XL/sample controls in the judge-facing workflow. Preserve them only in **Expert diagnostics**.
- Present well-top/checkshot/deviation, LAS curve families and unit-checked sonic-density impedance context when source data support it, with explicit non-association flags.
- Always offer a wide, usable **Gemma AI Copilot** conversational page, selected-finding context, evidence IDs, grounded explanations and human-readable follow-up decisions. A live model call requires configured and approved endpoint, explicit restricted-data authorization and bounded allowlisted evidence, never private raw SEG-Y/LAS/paths/keys. Provide a deterministic local fallback, labeled as **not Gemma**. Gemini/Gemma text is not geoscientific validation.
- Keep context synchronized on navigation; protect against stale finding/session identity, unverified source geometry and expensive full-field recomputation during interactions.
- Maintain a clear list of ML execution/capability statuses rather than presenting synthetic research models as validated Angsi production systems.

## Operations and release contract

- Canonical HPC application is the **only production checkout**. Existing systemd user service runs Streamlit on port **8501**; existing desktop icon opens it. Normal use should not require a persistent manual terminal.
- Work on reviewed development history and merge only an accepted, non-destructive release to `main`. Eventually **only the `main` remote branch** remains; review the unique old documentation commit before retiring it.
- The watcher currently auto-publishes allowlisted local source changes and has a separately modified remote-main fast-forward path. **That private watcher rollout/rollback is not yet integration-tested.** A GitHub merge can trigger automatic production deployment. Do not merge until deployment safety is proven.
- Never publish seismic/well data, private AI endpoints, raw generated scientific artifacts, model checkpoints, secrets, caches, per-user paths or HPC credentials. Stage only individually reviewed public source files. Protect the canonical checkout from destructive cleanup or overwriting local scientific assets.
- Test: full Python compilation, complete synthetic source suite, numerical/QC invariants, real Angsi bounded tests with checksums and edge exclusions, live-source Streamlit AppTest, actual desktop/browser visual acceptance, Gemma endpoint health and permitted question/reply when configured, false-answer/privacy gates, auto-deploy/rollback and post-release fingerprint/health.

## Completion definition

**Code ready** requires all GitHub checks green and source complete. **Science ready** requires actual Angsi proof for each asserted measurement and honest unresolved mechanisms. **Demo ready** requires verified synchronized pages and conversational UX against the real cache. **Production ready** requires successful guarded release through the original desktop icon plus tested rollback. Only then merge and retire other branches. If a gate cannot be satisfied, report it explicitly rather than claiming completion.
