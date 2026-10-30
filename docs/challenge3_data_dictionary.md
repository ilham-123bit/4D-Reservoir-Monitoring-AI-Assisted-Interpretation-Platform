# Challenge 3 standardized data dictionary

This document defines application-side fields. It does not assert that source files have been inspected or that every field is populated.

## Seismic volume

| Field | Meaning | Source/handling |
|---|---|---|
| `data` | 3D amplitude samples | Standardized array order `(inline, crossline, sample)` |
| `dimensions` | Number of indices on each axis | Measured from actual SEG-Y traces and sample axis |
| `inline`, `crossline` | Header identifiers | Preserved as identifiers; not geographic coordinates |
| `sample_interval` | Raw binary-header sample interval value | Preserved with its header-unit label; verify SEG-Y revision/project documentation |
| `sample_units`, `sample_axis_domain` | Time/depth meaning | Remain unverified until source documentation establishes domain and units |
| `missing_trace_mask` | Absent inline/crossline trace locations | Filled with NaN in standardized arrays, identified separately |
| `coordinate_scalar`, `coordinate_units` | SEG-Y coordinate header metadata | Inspected and preserved; no CRS or transformation assumed |

## Well data

- Well inventory: source workbook headings/rows and any standardized identifier or coordinate fields after explicit column mapping.
- Checkshot: original columns and rows retained; depth/time units, reference and one-way/two-way meaning remain unspecified until the source header/documentation supports them.
- Deviation: original columns/rows retained; MD, inclination, azimuth, coordinate convention and units are not guessed.
- LAS: one object per source file, with original/composite provenance, curve mnemonic, unit, description, depth index, null values and missing samples retained.
- Markers: original rows retained; marker names/depths are not assigned formation meaning without documented support.

Missing files/curves/columns remain explicitly missing. No absent well support information is synthesized.
