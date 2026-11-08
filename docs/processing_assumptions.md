# Processing assumptions

- The canonical array order is `(inline, crossline, sample)`; header identifiers are retained as indices.
- Baseline and Monitor pairing by trace key supports preliminary comparison but does not establish physical 4D registration.
- Missing traces and non-finite samples remain invalid support; they are not converted into measured zero amplitudes.
- Difference sign is Monitor minus Baseline. Absolute difference removes the sign and should be interpreted separately.
- RMS and NRMS are calculated over explicit sample windows using valid paired samples. Support counts accompany windowed results; unsupported windows remain unavailable.
- Attribute values depend on sample selection, window boundaries, finite-data masks and any explicitly configured preprocessing.
- Detection thresholds and reference populations are screening choices. Unless calibrated against suitable independent data, anomaly scores are not probabilities.
- Display downsampling must preserve flagged or missing cells conservatively. Full-resolution analysis remains separate from display aggregation.
- Sample indices are not seconds, two-way time, depth or stratigraphic intervals without verified origin, domain and units.
- Well associations require verified identifiers, CRS, datum and compatible depth/time references. Unverified ties are not generated.

These assumptions describe the current preliminary workflow. Survey-specific processing history and repeatability must be reviewed before interpreting an anomaly as a subsurface change.
