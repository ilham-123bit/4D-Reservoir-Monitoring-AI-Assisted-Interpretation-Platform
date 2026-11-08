# Scientific methodology

Reservoir Time Machine compares acquired baseline and monitor seismic responses and organizes candidate changes for expert review.

1. **Validate inputs.** Check survey headers, trace keys, sample axes, finite values, missing coverage and source metadata. Preserve data-quality masks.
2. **Establish comparability.** Compare geometry, acquisition and processing metadata. Trace-key matching permits preliminary screening; physical registration must be verified separately.
3. **Calculate attributes.** Compute signed and absolute amplitude differences and windowed RMS/NRMS over valid paired samples. Record window definition and support count.
4. **Screen for anomalies.** Apply robust statistical and eligible classical ML methods to documented features and reference samples. Preserve score meanings, model configuration and calibration status.
5. **Characterize components.** Group spatially adjacent candidate cells, retain exact membership and representative locations, and report spatial extent in grid indices.
6. **Integrate evidence.** Present seismic attributes, detector agreement/disagreement, coverage quality and available well information in a shared evidence record.
7. **Prioritize investigation.** Rank candidates for follow-up within the analyzed scope. Summaries identify alternatives and recommend verification work.

Observed amplitudes, calculated attributes, statistical/ML detections, geological hypotheses and engineering recommendations are kept distinct. A seismic anomaly alone cannot establish fluid movement, saturation, pressure, connectivity or remaining hydrocarbons. Well evidence is associated with a seismic finding only when spatial and depth/time references are validated.

The current demonstrated analysis is sparse and tile-bounded. It is not survey-wide, and the rankings are not globally calibrated.
