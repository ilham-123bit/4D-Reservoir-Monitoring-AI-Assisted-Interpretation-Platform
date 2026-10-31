# Baseline–Monitor trace coverage matching

## Purpose and scope

`data_ingest.trace_matching` matches trace headers by `(inline, crossline)`. It never assumes equal file trace indices. Common pairs are returned in Baseline order; Monitor-only and Baseline-only trace IDs and source indices remain separately represented. Duplicate keys are retained with all source indices and excluded from one-to-one pairing.

The prior audit facts supplied with the development task describe a Monitor coverage superset and a per-inline pattern. They are not remeasured or claimed by this code change: the current workspace has no competition SEG-Y inputs. Run the header audit in the authorized environment to reproduce counts and exact coordinates.

## Interfaces

- `match_trace_key_sequences(baseline_keys, monitor_keys)` is a pure, mathematical function returning a read-only mapping and coverage groups.
- `iter_segy_trace_keys(path)` reads only the inline/crossline trace-header fields.
- `iter_common_segy_trace_pairs(...)` lazily yields sample traces only for unique common keys, in Baseline order.
- `iter_monitor_only_segy_traces(...)` lazily yields the Monitor-only traces independently.
- `inspect_trace_coverage_files(...)` combines header matching with file sizes, integrity checks, optional streaming SHA-256, UTC timestamps, and software version.
- `write_trace_coverage_outputs(...)` writes JSON, Markdown, a common trace mapping CSV, and separate unmatched-coverage CSVs. It refuses to overwrite any existing output.

For every Baseline source index, `monitor_index_for_baseline` contains its matching Monitor source index or `-1` if it cannot be paired uniquely. The `common_*` arrays are the direct paired subset. `monitor_only_keys` and `monitor_only_trace_indices` retain Monitor-only data in Monitor order. These mappings are header-index metadata; they do not move or alter source samples.

## Run in authorized storage

Use complete input files and an output directory within authorized storage. This command scans headers, not trace sample payloads; the default SHA-256 calculation streams each full source file and can take substantial I/O. Skip it only when hash provenance is not required.

```bash
python scripts/audit_trace_coverage.py \
  --baseline '<authorized Baseline SEG-Y>' \
  --monitor '<authorized Monitor SEG-Y>' \
  --output-dir '<authorized output directory>' \
  --expected-baseline-traces '<trusted count>' \
  --expected-monitor-traces '<trusted count>'
```

The angle-bracketed arguments are placeholders, not literal values. Expected trace counts must come from a trusted inventory. Do not run a large scan on a login node or publish the resulting report: it contains source paths and trace identifiers. The command does not submit jobs or alter either input. Output files are created exclusively and an existing report is never replaced.

## Interpretation and use safeguards

Matching IDs establish only an index correspondence. They do not prove identical physical bin centres, CRS equivalence, spatial registration, phase/polarity/amplitude equivalence, or time alignment. The trace iterator also checks equal trace sample lengths, but it is not a substitute for the existing SEG-Y integrity and geometry validation gates.

The sample iterators require a mapping produced by file inspection with successful checks against trusted file integrity metadata (for example, a trusted expected SHA-256). Calculated hashes alone are recorded provenance and are not treated as trusted verification. Only consume `iter_common_segy_trace_pairs` for paired calculations after the independent geometry/sample-domain validation passes. Use the monitor-only iterator for coverage-boundary maps, RMS, finite/zero sample, or local quality checks. Monitor-only coverage is not automatically an edge effect, acquisition artefact, or geological feature.

The legacy full-volume `load_challenge3_pair` remains strict about its cube geometry and will reject mismatched grids rather than silently discarding traces. This new trace iterator is the memory-bounded key-matched route; it does not create an in-memory regularized cube or spatial transform.

## Tests and limits

Unit tests use small mathematical key sequences and mocked file readers only. They validate order preservation, unmatched-index retention, duplicate handling, per-inline grouping, non-overwrite behavior, and read-only source access. No actual competition SEG-Y file was opened in this development environment, and no production coverage count is asserted by the tests.
