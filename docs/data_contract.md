# Dataset and analytical output contract

## Zero-input status

The application currently has no competition dataset configured or present in the workspace. No seismic, well, reservoir, anomaly, or production result is implied by the example configuration, schemas, or mathematical unit tests. `config/dataset.example.yaml` contains null/unknown geometry and paths that must be replaced only with authorized source metadata. The JSON schemas define interfaces; they do not certify input data.

The canonical machine-readable contracts are:

- [`schemas/dataset-manifest.schema.json`](../schemas/dataset-manifest.schema.json): source inventory, surveys, geometry, wells, and DUG products.
- [`schemas/analysis-run.schema.json`](../schemas/analysis-run.schema.json): run provenance, quality gates, and analytical artifacts.
- [`schemas/validation-report.schema.json`](../schemas/validation-report.schema.json): pass/warning/fail findings, blocking status, evidence metadata, and corrective action.
- [`schemas/evidence-packet.schema.json`](../schemas/evidence-packet.schema.json): compact, provenance-linked evidence for optional model explanation.

The current application loaders do not yet validate complete manifests against these JSON Schemas. Treat schema compliance as a required adapter gate before broad ingestion support is declared complete.

## Dataset manifest fields

| Field | Requirement | Contract |
|---|---|---|
| Schema version, dataset ID, classification | Required | Versioned; classification is public/restricted/confidential/unknown. Do not infer access permission from a file path. |
| Synthetic flag | Required | `false` for this no-synthetic development workflow. |
| Provenance | Required before processing | Source/provider, receipt time when known, authorized storage reference, original file size and SHA-256. Preserve unknown values as unknown. |
| Baseline and monitor | Required for paired 4D work | Each has a distinct ID, label, source file reference, acquisition date if supplied, axes, geometry status, CRS status, trace key/order, and processing history. |
| Seismic axes | Required | Canonical in-memory order `(inline, crossline, sample)`. Store identifier source/values, counts, units, sample interval, sample origin, sample domain (`two_way_time`, `one_way_time`, `depth`, `index`, `unknown`), and null/mask behavior. Never infer absolute time/depth from sample index alone. |
| CRS and transforms | Conditionally required | Required for map-space joins, wells, surfaces, and spatial registration. Record authority, units, axis order, transform definition, source/destination CRS, method, and evidence. If unknown, mark unknown and block spatial joins. |
| Survey correspondence | Required before paired processing | Record trace correspondence key, geometry checks, compatible dimensions/grid, rejected/missing bins, and any registration product with method, bounds, sign convention, validity, and QC. No array-index pairing without validation. |
| Well inventory | Optional at dataset level; each provided well must be represented | Preserve source well ID and each source file/status separately. Record coordinates and CRS, reference datum/depth, and coverage for LAS, deviation, checkshot, markers, production. Missing/not supplied/unknown remain explicit. |
| DUG products | Optional | Product type, path, file format, units/domain, source survey/geometry reference, export settings/version, processing history, integrity metadata, and completeness status. |
| QC and warnings | Required before scientific use | File integrity, finite/missing samples, duplicate/missing traces, dead traces, geometry compatibility, amplitude summaries, sample-domain issues, and warnings with evidence. QC failure blocks dependent operations. |

### Seismic geometry and time/depth conventions

The adapter must preserve the raw header values and record how they map to standardized axes. Store inline/crossline identifiers (not only ordinal array positions), trace sequence/order, coordinate scalars, coordinate units, source/group/CDP fields, dimensions, sampling interval and sample origin. Retain a validity mask for missing traces and invalid samples. Do not replace nonfinite values with zero for scientific computation.

Map coordinates require a defined CRS and units. A coordinate scalar without a declared coordinate unit is insufficient. Survey transforms must be explicit, reversible where feasible, and provenance-recorded. Baseline/monitor registration must report pre/post diagnostics and cannot conceal rejected or missing bins.

For sample axis values, require domain and interval unit. If the datum/reference, time convention, or origin is unknown, keep it unknown; do not label samples as TWT or convert well depth to seismic time by assumption.

## Accepted input formats and current adapter limits

| Input | Format | Current support/limit |
|---|---|---|
| Seismic volumes | SEG-Y (`.sgy`, `.segy`) | Optional `segyio` adapter exists and standardizes traces to `(inline,crossline,sample)`. A standard-library integrity check validates fixed-record boundaries before reading. Loading requires a verified SHA-256; paired processing additionally requires trusted completeness metadata, explicit sample-domain/origin metadata, and an explicit survey-geometry verification. The reader is not demonstrated on current competition inputs. |
| DUG volume exports | SEG-Y | Preferred documented interchange candidate; verify export configuration, headers, null traces, sampling, and geometry with a small authorized subset first. |
| DUG native volume | DUGIO | Proprietary/native; no Python reader is implemented or assumed. Export a validated SEG-Y or another documented open format from DUG. |
| DUG horizons/surfaces | Text/CSV, IL-CL-time/depth text, GEOH5 | Text/CSV can be handled through explicit schema adapters. GEOH5 support is not implemented; CRS, axis/domain, units, and export provenance remain mandatory. |
| DUG faults | CSV/text, IESX, GoCAD, GEOH5 | Only explicitly documented table adapters should be added first. IESX, GoCAD, and GEOH5 readers are not currently implemented. Preserve polyline/surface topology and CRS. |
| Attributes, time shift, QC | SEG-Y or documented tables | Import as a derived product with source volume IDs, attribute definition, units, window/filter settings, sign convention, and DUG build/tool provenance. A filename alone does not define an attribute. |
| Well logs | LAS 2.0/3.0 (`.las`) | Optional `lasio` reader exists. Preserve curve occurrence/index, mnemonic, unit, description, NULL sentinel, depth reference and source provenance; do not collapse duplicate mnemonics. |
| Well inventory | XLSX | Optional `openpyxl` reader exists. Column mapping must be explicit; preserve original identifiers and raw source fields. |
| Deviations/checkshots/markers | CSV/TSV/TXT | Generic loss-preserving table reader exists. Promote columns to typed trajectories/time-depth/markers only after schema, units, conventions, and datum are supplied. |
| Production/reservoir engineering | CSV/Parquet or documented exchange table | Not currently implemented. Requires well/time/unit definitions, provenance, missingness, and access classification before integration. |

The existence of a parser is not evidence that the format variant, actual source, or scientific relationship has been validated. DUG's native/session formats must not be reverse-engineered or treated as stable APIs.

## Analytical output contract

Each run has a unique `run_id`, timestamp, dataset manifest reference, software/version/revision, environment lock hash where available, full processing configuration, quality gates, output artifact list, runtime and peak-memory measurements. Every output artifact records relative path, format, size, SHA-256, units, axis order, source IDs, and processing provenance.

Recommended output organization (configured root only; do not write into source directories):

```text
<output-root>/runs/<run-id>/
├── run.json                 # conforms to analysis-run schema
├── qc/                      # JSON/CSV summaries and optional figures
├── attributes/              # chunked volume artifacts plus metadata
├── detections/              # masks/scores/components and detector configs
├── interpretation/          # evidence packets, reviewed explanations
└── reports/                 # derived figures and report exports
```

Small tabular metadata use JSON/CSV. Full-size arrays should use a chunked, self-describing format (Zarr or NetCDF/HDF5 after dependency and HPC benchmarking) with dimension coordinates, units, masks, chunk layout, compression, and provenance. NPY/NPZ is suitable only for small test arrays or explicitly bounded local artifacts; it is not the interchange contract for large cubes. GeoJSON is suitable only where valid geographic coordinates/CRS semantics exist; otherwise use index-coordinate tables or a documented grid-relative representation. DUG products should retain their source file unmodified and be referenced by hash.

## Evidence and interpretation contract

`schemas/evidence-packet.schema.json` requires separate measured observations, algorithmic detections, interpretation hypotheses, alternatives, uncertainty, missing evidence, and requested follow-up. Each item cites a source artifact and status. Uncalibrated detector strength must be labelled as such. A probability requires a calibration method and independent validation reference. Recommendations are investigation priorities and do not claim remaining hydrocarbons.

AI providers consume compact structured summaries only. They must not receive full seismic cubes or raw restricted files. `EvidencePacket` enforces the typed boundary and payload limit. Restricted/confidential/unknown inputs are blocked from the Gemma adapter unless an explicit opt-in is provided; provider availability is never required for scientific processing.
