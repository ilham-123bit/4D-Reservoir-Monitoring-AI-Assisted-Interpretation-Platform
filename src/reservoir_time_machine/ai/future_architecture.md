# AI architecture

## Implemented

The current active analytics are deterministic attribute and robust statistical screens. The generic `AnomalyDetectionResult` contract reports a score, binary mask and diagnostics. Scores are not calibrated probabilities. The optional Gemma client explains bounded JSON evidence summaries only; it is not a seismic detector.

## Actual-data ML work required

Before model development, inspect what labels and independent evidence the authorized dataset actually contains. The source description does not establish voxel-level labels for fluid movement, depletion, or bypassed hydrocarbons. Without such labels, supervised claims are unsupported.

Potential research includes unsupervised or self-supervised 3D feature representations, paired-vintage change encoders, and spatially coherent anomaly models. Compare any model against conventional attributes, test stability under acquisition/registration perturbations, assess false positives and runtime, and use held-out spatial regions or wells where defensible. Do not call an anomaly “missed oil” without independent geological and production evidence.

## Future interface

An eventual detector adapter should accept standardized baseline and monitor data, geometry metadata, optional calculated attributes and validity masks. It should return anomaly strength, mask, uncertainty/calibration metadata and diagnostics. No model weights, training run, GPU inference, or actual-data model result are currently implemented.

## HPC neural-model implementation status

The APGCE HPC Python 3.12 CUDA environment is the target for the optional shared-encoder Siamese 3D CNN and nonlinear convolutional autoencoder in `ai/torch_models.py`. PyTorch is the primary implementation framework; the provider-neutral `DetectionRequest`/`DetectionOutput` contract remains framework-independent. The autoencoder has no skip connections to constrain identity copying. Neither architecture has been instantiated or validated on a GPU in this local Windows workspace. The HPC training runner requires a private streaming dataset adapter, uses fixed-size patches, and rejects Siamese training unless defensible train and independent validation labels are configured. No synthetic data path is provided.

TensorFlow remains an environment-validated option, but a second model implementation is intentionally not duplicated without a justified requirement. Weights/checkpoints, training logs and derived scores remain on authorized HPC output storage.
