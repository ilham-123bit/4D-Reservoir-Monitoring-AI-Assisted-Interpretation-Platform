# Geometry, integrity, and repeatability validation

## Report shape

Validation functions return machine-readable reports with an aggregate `pass`, `warning`, or `fail`, a `can_proceed` gate, metadata, and findings. Each finding includes a stable `check_id`, explanation, affected input(s), relevant metadata, recommended corrective action, and a `blocking` flag. The report structure is specified by [`validation-report.schema.json`](../schemas/validation-report.schema.json).

Warnings can be blocking when unknown metadata makes a comparison unsafe. A report with a warning is not automatically accepted; inspect `can_proceed` and each blocking finding. `validate_pair` preserves its exception-based calling interface but rejects when any check fails or blocks. Call `compare_volume_geometry` first when a serializable report is required.

## SEG-Y integrity

`inspect_segy_integrity` uses standard-library reads of the 3,200-byte textual header and 400-byte binary header. For supported fixed-length SEG-Y records, record size is `240 + samples_per_trace * bytes_per_sample`, with byte width selected from the declared SEG-Y format code. It honors a non-negative extended textual header count and calculates the first trace byte offset. Unknown/variable extended-header layouts and variable-length traces are blocking warnings rather than guessed offsets.

The check reports exact complete-record count, trailing header/sample bytes, first incomplete offset, and bytes needed to finish a partial record. It can compare a supplied expected total trace count, expected file size, and trusted expected SHA-256. Hashing streams bounded chunks and never writes the source. Size/time are checked for stability during inspection. A complete record boundary alone does not prove that no whole traces are missing; without trusted completeness metadata, that uncertainty is explicitly blocking for loading.

## Trace correspondence and geometry

Trace correspondence uses unique `(inline,crossline)` keys. Duplicate keys, zero IDs, empty key sets, or different key coverage fail. Equal key sets in different file order produce a warning and a monitor-index reordering map; comparison must reindex by keys, never file sequence numbers.

Volume geometry checks compare dimensions, axis labels/order, explicit inline/crossline ID arrays, sample interval and units, declared sample domain and sample units, sample origin and units, and explicit geometry-verification state. Unknown sample domain, origin, identifiers, or unverified survey geometry blocks paired calculations. A coordinate reference mismatch fails. Missing CRS is a non-blocking warning for verified bin-index analysis but disables map-space interpretation and well joins.

SEG-Y coordinate checks apply the signed coordinate scalar and compare X/Y at matched bins, while also comparing coordinate-unit codes. Undefined coordinate units with non-zero coordinates block spatial use. Matching coordinate headers do not establish a CRS; an authoritative CRS/navigation source remains necessary for map-space operations.

## Volume QC

`quality_report` returns finite-sample availability, NaN and signed-infinity counts, traces with non-finite samples, fully unavailable traces, zero-energy traces, amplitude range/RMS, trace RMS distribution, and a nested validation report. The amplitude-to-standard-deviation value is a descriptive proxy, not calibrated SNR.

`repeatability_report` processes along inline chunks and computes:

- Paired finite sample count and fraction using the joint finite mask.
- Global Pearson waveform correlation over all paired finite samples after global mean removal.
- Per-trace Pearson correlation after per-trace mean removal; zero-variance/insufficient traces have undefined correlation, not a fabricated zero.
- Global and per-trace NRMS: `200 * RMS(B-M) / (RMS(B)+RMS(M))` over paired finite samples. A zero denominator returns 0; no finite pairs fails.
- Local tile summaries over configurable inline/crossline tiles, including valid trace count, median trace correlation, and median trace NRMS.

These are descriptive repeatability metrics, not a universal pass/fail quality threshold. Genuine production-related change, acquisition differences, noise, phase, and residual mistie can all affect them. Local tile aggregates are diagnostics, not spatially calibrated thresholds. NaN/Inf pairs are excluded jointly and coverage is reported.

## Temporal registration

`estimate_temporal_registration` estimates bounded integer trace lags over pairwise finite overlap. It now rejects traces with insufficient overlap, zero energy, or a best correlation below configurable `min_correlation`. Rejected traces have `accepted=False`, NaN correlation if no estimate was possible, and a reason. Optional integer correction uses NaN edge padding; it never wraps or extrapolates edge values. This is not spatial registration, sub-sample registration, or a validated survey-level correction.

## Current validation limits

The unit tests use tiny constructed SEG-Y headers and mathematical arrays only. They validate arithmetic and software gates, not a competition dataset. The SEG-Y adapter and dashboard cannot be integration-validated until authorized inputs and optional reader/UI dependencies are available. Schema files are JSON-parsed in tests, but full JSON Schema semantic validation is pending an installed validator. No geological finding or actual-data performance is claimed.
