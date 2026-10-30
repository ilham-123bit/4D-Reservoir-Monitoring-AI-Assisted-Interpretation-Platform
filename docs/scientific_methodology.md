# Scientific methodology

## Implemented calculations

Out-of-core attributes, optional amplitude equalization, exact spatial key reindexing, and multi-attribute robust screening are documented in [scalable analytics](scalable_analytics.md). Outputs remain observed attributes or uncalibrated anomaly strengths; they do not establish geological cause.

For aligned baseline `B` and monitor `M`, signed difference is `M − B`; absolute difference is `|M − B|`. Normalized difference is `(M − B) / (0.5(|B| + |M|) + ε)`, with `ε=1e-8`. Local RMS and NRMS windows are calculated along the sample axis. The dashboard calculates attributes for a selected sample plane to avoid materializing multiple full-size attribute cubes.

Windowed NRMS follows `200 × RMS(B − M) / (RMS(B) + RMS(M))`. Zero/near-zero energy returns zero. Windows without valid paired samples return NaN. High NRMS means the vintages differ; it does not identify a unique geological cause.

QC reports dimensions, finite-sample availability, fully missing trace count, trace RMS spread, amplitude range/RMS, and a descriptive amplitude-to-standard-deviation proxy. It is not a calibrated signal-to-noise estimate.

## Registration and detection status

The current registration code estimates bounded integer trace lags by normalized cross-correlation, rejects insufficient/zero-energy/low-correlation traces, and can optionally apply NaN-padded non-wrapping integer shifts. It does not implement sub-sample shifts, spatial warping, or lateral consistency optimization. Registration has not been run on competition surveys.

The current anomaly screen applies median plus a scaled MAD threshold to an attribute map. Connected-component extraction can summarize regions. The dashboard's selected-sample detector is a screening tool and reports anomaly strength, not calibrated probability. No geological interpretation is inferred automatically.

## Validation boundary

No actual competition data are currently present in the workspace. Previously copied incomplete inputs were deleted; no findings from them are used as active inputs or validation. Numerical unit tests use small in-memory mathematical arrays or mocked headers and validate software behavior only. They do not validate seismic geology. SEG-Y/LAS/workbook readers, dashboard rendering, DUG and Gemma integrations have not been executed against current actual inputs. See the [Track B/C roadmap](track_b_c_roadmap.md) for validation gates.
