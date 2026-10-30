# Scientific development roadmap and prioritized backlog

## Revised technical architecture

1. **Authorized data boundary:** environment-based dataset root; no raw competition data in Git or local unauthorized paths.
2. **Forensic audit and adapters:** read-only manifest and format inspection; SEG-Y, well inventory, LAS, checkshot, deviation, marker adapters preserving original identifiers and unknown metadata.
3. **Canonical data contracts:** survey geometry, axis/domain/units, source headers, validity masks, wells and trajectories with explicit CRS/datum/depth/time references. No coordinate or time-depth transformation without evidence.
4. **Scientific QC and pairing:** chunked volume stats; geometry and trace matching; repeatability diagnostics; excluded/degraded regions; explicit pair eligibility.
5. **Processing graph:** immutable originals; optional filtering/equalization; bounded spatial/temporal alignment; diagnostics and rejection criteria; multi-scale difference, RMS, NRMS, phase/time/amplitude and structure-aware attributes.
6. **Detector interface:** conventional baselines, robust local statistics, spatial coherence, multi-attribute anomaly scoring and clustering; scores not probability unless calibrated. Later self-supervised/unsupervised and paired neural methods only after justified controls and resources.
7. **Evaluation and interpretation:** stability/false-positive review, parameter sensitivity, independent well/geological evidence, alternatives and uncertainty. Opportunity scores explain evidence and missing support; never assert remaining hydrocarbons from seismic anomaly alone.
8. **Product layer:** custom interactive dashboard and evidence panels; DUG as independent interpretation/QC and validated file-exchange partner; Gemma explains structured evidence.

## Scientific phases

### Phase 0 — environment and DUG discovery

On the authorized workstation/HPC, verify the dataset, DUG build/session/licences, supported export formats, Python environment, GPU visibility and Slurm policy. Do not submit jobs or modify the supplied data during discovery. Deliver private factual audit and public non-sensitive capability status.

### Phase 1 — actual data foundation

Inspect both SEG-Y files and all available well files. Validate trace identifiers/order, coordinate headers/scalars/units, sample interval/domain, dimensions, missing/dead traces, amplitude distributions and pair geometry. Load a subset first; then memory profile full-volume access. Establish well-object provenance and unknown references.

### Phase 2 — repeatability and registration

Quantify trace correspondence and baseline-monitor repeatability before interpreting changes. Estimate bounded time/phase/amplitude mismatch on appropriate windows; use holdout diagnostics and spatial smoothness/rejection checks. Keep uncorrected and candidate-corrected outputs separate; avoid unrestricted warping.

### Phase 3 — robust 4D attributes

Create multi-scale signed/absolute difference, normalized difference, windowed NRMS, RMS/amplitude, local phase/time-lag and coherence attributes with explicit masks and physical units. Assess noise, edge effects and sensitivity. Compare selected metrics with DUG where available.

### Phase 4 — detection and AI

Benchmark global and local robust statistics, attribute fusion, spatially coherent components and unsupervised 3D methods. Use perturbation/negative controls, registration-error tests, parameter stability and expert review; there are no assumed voxel-level geological labels. Use HPC CPU/GPU only after profiling and approval. Train paired neural models only if data diversity, evaluation and time justify them.

### Phase 5 — geological evidence and decisions

Integrate wells only after verified transforms and curve/reference inspection. Separate measured attributes, detections, geological hypotheses, competing explanations, uncertainty and actions. Rank investigation priorities with transparent evidence weights and no reserve/oil-probability language unless independent support exists.

### Phase 6 — integrated experience

Deliver linked 3D/slice views, zone exploration, well/interpretation overlays, endpoint animation, evidence reports and presentation narrative. Test on competition hardware and against DUG reference picks. Preserve custom AI overlays and workflow even if DUG has overlapping views.

## Prioritized implementation backlog

| Priority | Task | Exit evidence |
|---:|---|---|
| 1 | Resolve source integrity and provenance for both local 2 GiB SEG-Y files | Authorized intact source copies or documented re-export; hashes and complete trace payloads verified; originals preserved |
| 2 | On-site DUG, licence, session, data and environment audit; verify geometry against authoritative session metadata | Version/modules/export matrix; source completeness, survey geometry, coordinate mapping and sample domain independently confirmed |
| 3 | Repair/extend adapters for actual survey key matching, masks and well-file edge cases | Parser coverage, known trace/sample spot checks, geometry checks, and explicit unsupported references; no silent coordinate assumptions |
| 4 | End-to-end QC and repeatability gate on validated complete inputs | Full-volume/chunked QC, masks, pair eligibility and reproducible reports on actual data |
| 5 | Registration and cross-equalization diagnostics | Bounded method comparison; correction/no-correction maps; rejection criteria and DUG comparison where available |
| 6 | Multi-scale 4D attributes with validity masks | Correct formulas/units, memory profile, actual-data output inspection and mathematical tests |
| 7 | Detector benchmark and spatial morphology | Conventional detector comparison; stability and false-positive review; scores explicitly uncalibrated |
| 8 | 3D AI experiment design and resource profile | Defined objective, controls, GPU/CPU benchmark plan, reproducible model/evaluation; proceed only if defensible |
| 9 | Well/marker relationship model | Verified CRS/depth/time mappings and provenance; missing coverage represented explicitly |
| 10 | Transparent candidate evidence/ranking | Auditable feature contributions, competing explanations, uncertainty and review workflow |
| 11 | Custom dashboard and Gemma integration | Launch on target environment; latency/memory results; evidence-only LLM; UI cross-checked with DUG |

## Validation criteria

- Never substitute synthetic reservoir-like inputs when actual data is missing.
- Every paired output records inputs, geometry compatibility, configuration and validity masks.
- Mathematical tests use small arrays/mocks; actual-data integration checks are separately labelled and executed only on HPC.
- Detection scores are not called probabilities without calibration and independent validation.
- Corrected and original surveys remain traceable; corrections require diagnostics and review.
- Well overlays require documented coordinate and depth/time transformations.
- DUG agreement is checked numerically on exported data, not inferred from screenshots.
- Performance reports include wall time, peak memory and display resolution; no GPU/Slurm use without the required approval.
