# Platform architecture

Reservoir Time Machine organizes its workflow around a shared finding: a candidate seismic response with a stable identifier, source-grid location, analysis scope, evidence and limitations.

## Data and analysis

Paired baseline and monitor volumes are validated before attributes are calculated. The canonical array order is `(inline, crossline, sample)`. Missing trace locations remain masked and are excluded from paired calculations. Data ingestion preserves source metadata for SEG-Y and available well tables/logs.

The attribute layer calculates signed and absolute differences, RMS and normalized RMS within explicit windows. Statistical screening and eligible classical detectors produce scores and masks with provenance. Connected components summarize spatially adjacent detections. Scores are not probabilities unless a separate calibration has been validated.

## Findings and interpretation

Completed analysis products are stored in a versioned registry that records data/configuration identity, analyzed tiles, coverage, model provenance and component membership. The dashboard loads completed products; opening pages or navigating findings does not trigger full-volume fitting. A shared local evidence ledger is used by the report and visualizations so the same finding retains its identity and coordinates.

Deterministic summaries distinguish measured observations, calculated attributes, detections, geological hypotheses and recommended investigations. An optional language-model client can interpret validated evidence packets when explicitly invoked and configured. It does not calculate seismic attributes or authorize a reservoir conclusion.

## Application views

The Streamlit application presents survey quality, linked 3D/grid context, baseline/monitor comparison, change findings, an automatic finding-directed storyboard, well evidence and a technical interpretation report. Display aggregation is separate from full-resolution analysis.

## Scientific boundaries

Trace-key correspondence is not physical 4D registration. Grid indices are not geographic coordinates, time or depth without verified metadata. Well-to-seismic relationships require validated coordinate reference systems and depth/time ties. When prerequisites are absent, the platform reports the association as unavailable.
