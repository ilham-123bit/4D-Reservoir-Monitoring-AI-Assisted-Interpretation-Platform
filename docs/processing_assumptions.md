# Processing assumptions and safeguards

- Paired samples are compared only after SEG-Y trace order, dimensions, sample interval, and coordinate header sequences pass compatibility checks.
- SEG-Y inline/crossline numbers are indices/identifiers, not map coordinates. No geographic CRS or well-to-seismic transform is assumed.
- Binary-header sample interval is exposed in seconds based on the SEG-Y interval convention; confirm revision and project documentation during HPC audit.
- Difference sign is monitor minus baseline. NRMS is `200 × RMS(B − M) / (RMS(B) + RMS(M))`; zero-energy windows return zero and windows without valid paired samples return NaN.
- Normalization is optional and off by default. Global or trace RMS equalization can suppress real amplitude changes; inspect diagnostics before use.
- Registration is bounded integer trace-lag cross-correlation only. It does not establish geological alignment and does not apply automatically in the dashboard.
- MAD thresholds and connected components are screening tools. Scores are not probabilities. High NRMS or a detected zone does not establish cause.
- Never equate amplitude/NRMS patterns with depletion, fluid movement, or remaining hydrocarbons without independent evidence.
- Actual Challenge 3 results remain untested until inspected and reviewed on HPC and visually compared in DUG Insight.
