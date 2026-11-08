# Quality control and validation

Quality checks determine whether measurements are suitable for paired calculations and how much support is available. They do not establish a geological cause.

## Survey checks

The ingestion workflow checks file readability, trace/sample counts, header identifiers, geometry, sample encoding, finite values and missing trace coverage. Baseline and Monitor trace keys are compared explicitly. Monitor-only traces and cells missing from the Baseline grid are retained in inventory rather than silently treated as paired measurements.

Pairing on the Baseline grid supports preliminary screening. Physical registration requires verified survey coordinates and an appropriate registration workflow; matching inline/crossline identifiers alone is insufficient.

## Attribute checks

Difference, absolute difference, RMS and NRMS calculations use documented axis order and valid paired samples. Missing support is masked. Windowed outputs include support counts; windows without valid measurements are unavailable rather than zero. Numerical regression tests compare selected-sample, chunked and reference calculations on controlled fixtures.

## Detection checks

Statistical thresholds, model scores, feature scaling and component rules are recorded with the analysis. Connected-region tests verify member identity and representative coordinates. A display mask should preserve isolated detections and missing coverage when aggregating cells. Detector agreement is descriptive and does not imply independent confirmation.

## Validation scope

Software tests, controlled numerical fixtures, bounded real-data checks, application rendering, browser interaction and geological validation are separate evidence classes. Passing a software test does not prove survey-wide accuracy or geological validity. Current public results reflect limited analyzed coverage; full-survey statistics and validated well ties are N/A.
