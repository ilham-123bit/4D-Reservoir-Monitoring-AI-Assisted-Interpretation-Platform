# Challenge 3 data inventory

The analysis workflow has been exercised with the Angsi Field Challenge 3 dataset. Source seismic and well files are not distributed in this repository.

## Data types used by the platform

| Data group | Role | Required metadata |
|---|---|---|
| Baseline and Monitor seismic | Paired 4D response comparison | Trace identifiers, sample axis, encoding, geometry, coverage and processing provenance |
| Well inventory | Well identity and declared attributes | Source identifiers, column meanings, units and coordinate reference where present |
| LAS logs | Measured well curves | Curve mnemonic, units, null value, depth reference and source provenance |
| Markers | Stratigraphic context | Marker name, depth reference, datum and declared units |
| Checkshots and deviations | Potential depth/time and trajectory support | Well identity, units, datum, coordinate reference and measurement conventions |

Availability and completeness can differ by survey or well. Missing files, curves, traces and metadata remain explicitly unavailable.

## Public result scope

Software validation includes bounded analysis and application rendering. The demonstrated analysis covers only a sparse subset of planned tiles and does not support survey-wide statistics, calibrated global ranking, or a validated reservoir conclusion. Results reflect the analyzed survey coverage. Physical 4D registration and well-to-seismic association have not been established.

The application reports file and metadata readiness from the configured local dataset. The public repository contains code and documentation, not the Challenge 3 seismic volumes or private analysis artifacts.
