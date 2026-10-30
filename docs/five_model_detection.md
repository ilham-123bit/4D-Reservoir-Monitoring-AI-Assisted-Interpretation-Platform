# Five-model detection and evidence architecture

## Shared contract

`detection.contracts.DetectionRequest` accepts paired 3D arrays and/or co-registered attribute maps with the canonical `(inline, crossline, sample)` shape, provenance, and processing configuration. `DetectionOutput` standardizes model ID/version, score map, mask, mathematical score meaning, calibration status, optional uncertainty, diagnostics, limitations, and missing evidence. Its compact summary reports detection extent in array indices and never serializes the full score map. These are analytical detections, not geological interpretations.

## Model status

| Model | Implementation status | Score meaning and limits |
|---|---|---|
| Siamese 3D CNN | Shared-encoder multi-scale PyTorch architecture and backend contract are implemented. It is designed for the provided HPC CUDA environment; PyTorch is absent locally and no weights/trainer execution has occurred. A mock backend verifies the interface. | Raw change-head logit. Untrained and uncalibrated. No supervised model is trained. HPC patch-batch inference is defined; overlap blending into a single survey grid remains an adapter responsibility. |
| Bayesian change | Implemented scalar Gaussian likelihood-ratio detector. | Log likelihood ratio with explicit null/change means, standard deviations, and prior. Entropy and model-conditional posterior are diagnostics under those assumptions, not calibrated probabilities. |
| Mahalanobis | Implemented fit/score detector using median/MAD feature scaling, optional coordinate-wise winsorization, covariance ridge, pseudoinverse, and finite-row rejection. | Squared distance in fitted feature space. Chi-square cutoff is only a Gaussian reference, not a calibrated false-alarm rate. Winsorization is not a high-breakdown robust covariance estimator. |
| Autoencoder | Implemented deterministic linear tied-weight PCA detector and added an optional nonlinear 3D convolutional PyTorch autoencoder architecture for HPC. The CNN has not been instantiated locally or trained. | Mean squared feature reconstruction residual; reference quantile is uncalibrated. CNN reconstruction maps require a justified reference population and separate calibration. |
| Unsupervised anomaly | Implemented optional scikit-learn Isolation Forest and Gaussian-mixture negative-log-likelihood detectors. | Isolation score or negative log likelihood; thresholds depend on contamination/reference sampling and are not probabilities. Spatial context must be included in input features or postprocessed. |

Scikit-learn is an optional `ml` extra. NumPy/SciPy implementations remain available without it. PyTorch and TensorFlow are supplied by the competition environment and are not added to the base package dependencies: installing generic wheels could conflict with the HPC's CUDA builds. PyTorch is the primary 3D model framework. TensorFlow is probed by the environment validator but no duplicate TensorFlow model implementation is maintained; consider an interoperability/export adapter only if a real TensorFlow-specific need is established.

Training/inference entry points, private adapter contract, TensorBoard, checkpoint and Slurm workflow are specified in [the HPC workflow](hpc_workflow.md). No model was trained and no live HPC GPU was inspected in this workspace.

## Evidence fusion

`ai.evidence_fusion.summarize_detection_agreement` reports mask intersections and Jaccard overlap without combining incomparable raw scores. `fuse_correlated_log_evidence` accepts consistently oriented evidence vectors plus null/change means and full covariance matrices. The joint multivariate likelihood handles model correlation; finite-model subsets use the corresponding covariance submatrix. Optional diagonal measurement-noise variances can represent repeatability/acquisition uncertainty in the same evidence units.

The fusion interface does not convert arbitrary detector outputs into evidence. A separately validated calibration transform must first map each output to a consistent evidence scale. A posterior map is omitted unless callers explicitly provide verified calibration status and a durable validation reference. Covariance/mean assumptions remain the caller's responsibility and must be estimated from independent, defensible data. No Angsi calibration or evidence fusion was run.

## Scientific boundaries

No actual Challenge 3 volumes or independent voxel labels are available. Therefore there is no model training on competition data, no claim of detection performance, and no geological validation. Detector scores do not identify fluid type, flow direction, sweep efficiency, or remaining hydrocarbons. Candidate interpretation requires survey-repeatability QC, independent geological evidence, and documented alternatives.

## Five-person work split

1. **Ingestion/QC:** validate SEG-Y headers, hashes, coordinate references, and repeatability inputs.
2. **4D processing:** spatial transforms, resampling diagnostics, fractional temporal registration, attributes, and chunk-store benchmarks.
3. **Detection/evaluation:** feature extraction, conventional baselines, all detector comparisons, and held-out calibration design.
4. **Geology/evidence:** connect validated detections to wells/markers and maintain explicit observation/hypothesis/recommendation separation.
5. **Dashboard/integration:** render score semantics, uncertainty, calibration, data gaps, detector agreement, and provenance; own model inference packaging.

## Next validation gate

Before actual-data interpretation: obtain complete authorized input files; verify geometry/CRS and sample axes; run QC; define stable/background reference regions with geophysical review; benchmark chunk I/O and detector resource use on scheduled compute; document independent calibration/validation sources; then compare outputs with conventional interpretation and DUG QC. No restricted data should be sent to external services without authorization.
