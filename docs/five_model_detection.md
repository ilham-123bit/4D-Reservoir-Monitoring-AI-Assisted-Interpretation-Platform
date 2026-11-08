# Statistical and machine-learning screening

The platform uses detector outputs to identify candidate seismic-response patterns for review. A score measures the behavior defined by its algorithm and reference data; it is not a probability of hydrocarbons or a confirmed geological interpretation.

## Available detector families

| Detector | Role | Interpretation limit |
|---|---|---|
| Mahalanobis distance | Multivariate deviation from a fitted reference population | Depends on feature scaling and reference covariance; not a calibrated false-alarm probability. |
| PCA reconstruction | Linear reconstruction residual across selected attributes | Highlights feature combinations poorly represented by the fitted reference. |
| Isolation Forest | Unsupervised isolation-based anomaly score | Sensitive to reference sampling and configuration; score is not a probability. |
| Gaussian Mixture Model | Negative log likelihood under a fitted mixture | Indicates fit relative to the chosen mixture, not reservoir significance. |
| Bayesian change score | Conditional comparison under specified distributions and priors | A probability interpretation requires defensible priors, likelihood assumptions and independent calibration. |
| Siamese 3D CNN | Neural comparison architecture for paired seismic patches | No validated trained checkpoint is available for reservoir prediction. Random or untrained outputs are not findings. |
| Nonlinear 3D autoencoder | Neural reconstruction architecture | No validated trained checkpoint or reference calibration is available. |

Classical detector scores can be combined with attribute and spatial evidence only when their meaning and support are compatible. Agreement among correlated detectors is not independent confirmation.

## Evidence supplied with a finding

A candidate retains its grid location, component membership, source tile, feature configuration, model version, valid-sample support, missing-data quality and detector scores. Findings are ranked for investigation priority within their documented analysis scope, not by geological probability.

## Readiness

Current bounded results use preliminary classical screening. Bayesian calibration and trained neural inference are unavailable. Full-survey comparative ranking requires representative analysis coverage and validated reference/calibration rules.
