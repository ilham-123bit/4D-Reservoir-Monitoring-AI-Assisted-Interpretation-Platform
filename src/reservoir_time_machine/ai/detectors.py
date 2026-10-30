"""Five-model change-detection adapters using a shared evidence contract.

These methods produce detections or anomaly strengths. None asserts a geological cause.
The neural network architecture itself lives in ``ai.siamese`` and is untrained here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy.special import expit, logit
from scipy.stats import chi2

from ..detection.contracts import DetectionOutput, DetectionRequest


def _feature_cube(request: DetectionRequest, names: Sequence[str] | None = None) -> tuple[np.ndarray, tuple[str, ...]]:
    selected = tuple(names or sorted(request.attributes))
    if not selected or any(name not in request.attributes for name in selected):
        raise ValueError("requested model features are absent")
    cube = np.stack([np.asarray(request.attributes[name], dtype=np.float64) for name in selected], axis=-1)
    return cube, selected


def _output(request: DetectionRequest, model_id: str, score: np.ndarray, mask: np.ndarray, meaning: str,
            calibration: str, *, uncertainty: np.ndarray | None = None,
            diagnostics: dict[str, Any] | None = None, limitations: tuple[str, ...] = (),
            missing_evidence: tuple[str, ...] = ()) -> DetectionOutput:
    return DetectionOutput(model_id, "1.0", score, mask, meaning, calibration,
        request.input_provenance, limitations, request.processing_configuration,
        uncertainty, diagnostics or {}, missing_evidence)


@dataclass(frozen=True)
class GaussianChangeAssumptions:
    null_mean: float = 0.0
    null_std: float = 1.0
    change_mean: float = 2.0
    change_std: float = 1.0
    prior_change_probability: float = 0.05

    def __post_init__(self) -> None:
        if not np.isfinite((self.null_mean, self.null_std, self.change_mean, self.change_std,
                            self.prior_change_probability)).all():
            raise ValueError("Gaussian assumptions must be finite")
        if self.null_std <= 0 or self.change_std <= 0:
            raise ValueError("Gaussian standard deviations must be positive")
        if not 0 < self.prior_change_probability < 1:
            raise ValueError("prior probability must be in (0,1)")


class BayesianChangeDetector:
    """Scalar Gaussian likelihood-ratio change detector with explicit assumptions."""
    model_id = "bayesian_gaussian_change"

    def __init__(self, evidence_attribute: str, assumptions: GaussianChangeAssumptions = GaussianChangeAssumptions(),
                 *, decision_probability: float = 0.5) -> None:
        if not 0 < decision_probability < 1:
            raise ValueError("decision_probability must be in (0,1)")
        self.evidence_attribute, self.assumptions = evidence_attribute, assumptions
        self.decision_probability = decision_probability

    def detect(self, request: DetectionRequest) -> DetectionOutput:
        if self.evidence_attribute not in request.attributes:
            raise ValueError(f"missing Bayesian evidence attribute: {self.evidence_attribute}")
        x = np.asarray(request.attributes[self.evidence_attribute], dtype=np.float64)
        a = self.assumptions
        log_null = -0.5*((x-a.null_mean)/a.null_std)**2 - np.log(a.null_std) - 0.5*np.log(2*np.pi)
        log_change = -0.5*((x-a.change_mean)/a.change_std)**2 - np.log(a.change_std) - 0.5*np.log(2*np.pi)
        log_lr = log_change - log_null
        posterior_model_conditional = expit(log_lr + logit(a.prior_change_probability))
        posterior_model_conditional[~np.isfinite(x)] = np.nan
        entropy = -(posterior_model_conditional*np.log(np.maximum(posterior_model_conditional, 1e-12))
                    +(1-posterior_model_conditional)*np.log(np.maximum(1-posterior_model_conditional, 1e-12)))
        threshold_lr = logit(self.decision_probability) - logit(a.prior_change_probability)
        mask = np.isfinite(log_lr) & (log_lr >= threshold_lr)
        return _output(request, self.model_id, log_lr.astype(np.float32), mask,
            "log likelihood ratio under configured scalar Gaussian null/change models",
            "assumption_based_uncalibrated", uncertainty=entropy.astype(np.float32),
            diagnostics={"assumptions": a.__dict__, "decision_probability": self.decision_probability,
                         "threshold_log_likelihood_ratio": float(threshold_lr),
                         "model_conditional_posterior": posterior_model_conditional.astype(np.float32),
                         "posterior_is_calibrated": False, "score_is_probability": False},
            limitations=("Scalar Gaussian likelihoods may not describe seismic attribute distributions.",
                         "Posterior values are conditional on assumptions and are not validated/calibrated."),
            missing_evidence=("survey-specific null/change distributions", "independent validation labels"))


class MahalanobisDetector:
    """Regularized multivariate background-distance detector; missing voxels are excluded."""
    model_id = "mahalanobis_feature_distance"

    def __init__(self, *, regularization: float = 1e-3, threshold_squared_distance: float | None = None,
                 robust_winsor_z: float | None = 4.5) -> None:
        if regularization <= 0 or not np.isfinite(regularization):
            raise ValueError("regularization must be finite and positive")
        if threshold_squared_distance is not None and threshold_squared_distance <= 0:
            raise ValueError("threshold must be positive")
        if robust_winsor_z is not None and robust_winsor_z <= 0:
            raise ValueError("robust_winsor_z must be positive or None")
        self.regularization = float(regularization)
        self.threshold = threshold_squared_distance
        self.robust_winsor_z = robust_winsor_z
        self.fitted = False

    def fit(self, reference_features: np.ndarray, *, feature_names: Sequence[str] | None = None) -> "MahalanobisDetector":
        data = np.asarray(reference_features, dtype=np.float64)
        if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] < 1:
            raise ValueError("reference features must be a 2D samples-by-features matrix")
        valid = np.isfinite(data).all(axis=1)
        data = data[valid]
        if data.shape[0] < max(2, data.shape[1] + 1):
            raise ValueError("too few complete reference rows for covariance estimation")
        self.center = np.median(data, axis=0)
        mad = np.median(np.abs(data - self.center), axis=0) * 1.4826
        self.scale = np.where(mad > np.finfo(float).eps, mad, 1.0)
        standardized = (data - self.center) / self.scale
        if self.robust_winsor_z is not None:
            standardized = np.clip(standardized, -self.robust_winsor_z, self.robust_winsor_z)
        self.mean = standardized.mean(axis=0)
        covariance = np.atleast_2d(np.cov(standardized, rowvar=False, ddof=1))
        ridge = self.regularization * max(float(np.trace(covariance) / covariance.shape[0]), 1.0)
        self.covariance = covariance + np.eye(covariance.shape[0]) * ridge
        self.precision = np.linalg.pinv(self.covariance, hermitian=True)
        self.feature_names = tuple(feature_names or (f"feature_{i}" for i in range(data.shape[1])))
        if len(self.feature_names) != data.shape[1]:
            raise ValueError("feature_names length must match feature count")
        self.fitted = True
        return self

    def detect(self, request: DetectionRequest) -> DetectionOutput:
        if not self.fitted:
            raise RuntimeError("fit detector on a justified reference/background feature set first")
        cube, names = _feature_cube(request, self.feature_names)
        if cube.shape[-1] != self.center.size:
            raise ValueError("feature count differs from fitted covariance")
        valid = np.isfinite(cube).all(axis=-1)
        standardized = (cube - self.center) / self.scale - self.mean
        flat = np.where(valid[..., None], standardized, 0.0).reshape(-1, cube.shape[-1])
        distance = np.einsum("ij,jk,ik->i", flat, self.precision, flat).reshape(cube.shape[:-1])
        distance[~valid] = np.nan
        threshold = self.threshold if self.threshold is not None else float(chi2.ppf(0.99, cube.shape[-1]))
        mask = valid & (distance >= threshold)
        return _output(request, self.model_id, distance.astype(np.float32), mask,
            "squared Mahalanobis distance in robust-scaled feature space",
            "uncalibrated_reference_covariance", diagnostics={"feature_names": list(names),
                "regularization": self.regularization, "winsor_z": self.robust_winsor_z,
                "threshold_squared_distance": threshold, "threshold_reference": "chi-square 99th percentile; Gaussian assumption only",
                "covariance_condition_number": float(np.linalg.cond(self.covariance)), "score_is_probability": False},
            limitations=("Reference covariance must represent acceptable background variability.",
                         "Winsorized empirical covariance is not a fully robust high-breakdown estimator.",
                         "The chi-square cutoff is not calibrated for these data."),
            missing_evidence=("validated reference/background population", "survey-specific threshold calibration"))


class LinearAutoencoderDetector:
    """Deterministic linear autoencoder (PCA subspace) with reconstruction-error scoring."""
    model_id = "linear_autoencoder_reconstruction"

    def __init__(self, *, latent_dimensions: int, threshold_quantile: float = 0.99) -> None:
        if latent_dimensions < 1 or not 0.5 < threshold_quantile < 1:
            raise ValueError("latent_dimensions must be positive and threshold_quantile in (0.5,1)")
        self.latent_dimensions, self.threshold_quantile = latent_dimensions, threshold_quantile
        self.fitted = False

    def fit(self, reference_features: np.ndarray, *, feature_names: Sequence[str] | None = None) -> "LinearAutoencoderDetector":
        data = np.asarray(reference_features, dtype=np.float64)
        if data.ndim != 2 or not np.isfinite(data).all() or data.shape[0] < 2:
            raise ValueError("autoencoder reference must be a finite 2D samples-by-features matrix")
        if self.latent_dimensions >= data.shape[1]:
            raise ValueError("latent_dimensions must be smaller than feature count for a non-identity reconstruction")
        self.center = data.mean(axis=0)
        _, _, right = np.linalg.svd(data-self.center, full_matrices=False)
        self.components = right[:self.latent_dimensions]
        recon = self._reconstruct(data)
        errors = np.mean((data-recon)**2, axis=1)
        self.threshold = float(np.quantile(errors, self.threshold_quantile))
        self.feature_names = tuple(feature_names or (f"feature_{i}" for i in range(data.shape[1])))
        if len(self.feature_names) != data.shape[1]:
            raise ValueError("feature_names length must match feature count")
        self.fitted = True
        return self

    def _reconstruct(self, data: np.ndarray) -> np.ndarray:
        return self.center + ((data-self.center) @ self.components.T) @ self.components

    def detect(self, request: DetectionRequest) -> DetectionOutput:
        if not self.fitted:
            raise RuntimeError("fit the reconstruction model on a justified reference set first")
        cube, names = _feature_cube(request, self.feature_names)
        valid = np.isfinite(cube).all(axis=-1)
        flat = np.where(valid[..., None], cube, self.center).reshape(-1, cube.shape[-1])
        reconstructed = self._reconstruct(flat)
        error = np.mean((flat-reconstructed)**2, axis=1).reshape(cube.shape[:-1])
        error[~valid] = np.nan
        return _output(request, self.model_id, error.astype(np.float32), valid & (error >= self.threshold),
            "mean squared feature reconstruction residual", "reference_quantile_uncalibrated",
            diagnostics={"feature_names": list(names), "latent_dimensions": self.latent_dimensions,
                         "training_error_quantile": self.threshold_quantile, "threshold": self.threshold,
                         "architecture": "linear tied-weight PCA autoencoder", "score_is_probability": False},
            limitations=("Linear reconstruction does not model nonlinear seismic structure.",
                         "Reference residual quantile is not a calibrated false-alarm probability."),
            missing_evidence=("representative normal/reference training population", "independent calibration set"))


class IsolationForestDetector:
    """Optional scikit-learn Isolation Forest detector for attribute feature vectors."""
    model_id = "isolation_forest_attribute_detector"

    def __init__(self, *, contamination: float = 0.01, random_state: int = 0, n_estimators: int = 200) -> None:
        if not 0 < contamination <= 0.5 or n_estimators < 10:
            raise ValueError("contamination must be in (0,0.5] and n_estimators >= 10")
        self.contamination, self.random_state, self.n_estimators = contamination, random_state, n_estimators
        self.fitted = False

    def fit(self, reference_features: np.ndarray, *, feature_names: Sequence[str] | None = None) -> "IsolationForestDetector":
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError as exc:
            raise RuntimeError("Install the optional machine-learning dependency to use Isolation Forest") from exc
        data = np.asarray(reference_features, dtype=np.float32)
        if data.ndim != 2:
            raise ValueError("reference features must be 2D")
        data = data[np.isfinite(data).all(axis=1)]
        if data.shape[0] < 10:
            raise ValueError("at least 10 complete reference rows are required")
        self.estimator = IsolationForest(contamination=self.contamination, random_state=self.random_state,
                                         n_estimators=self.n_estimators, n_jobs=1).fit(data)
        self.feature_names = tuple(feature_names or (f"feature_{i}" for i in range(data.shape[1])))
        if len(self.feature_names) != data.shape[1]:
            raise ValueError("feature_names length must match feature count")
        self.fitted = True
        return self

    def detect(self, request: DetectionRequest) -> DetectionOutput:
        if not self.fitted:
            raise RuntimeError("fit Isolation Forest on a justified reference population first")
        cube, names = _feature_cube(request, self.feature_names)
        valid = np.isfinite(cube).all(axis=-1)
        flat = np.where(valid[..., None], cube, 0).reshape(-1, cube.shape[-1])
        score = -self.estimator.score_samples(flat).reshape(cube.shape[:-1])
        score[~valid] = np.nan
        threshold = -float(self.estimator.offset_)
        return _output(request, self.model_id, score.astype(np.float32), valid & (score >= threshold),
            "negative Isolation Forest score_samples; larger means more isolated",
            "contamination_cutoff_uncalibrated", diagnostics={"feature_names": list(names),
                "contamination": self.contamination, "threshold": threshold,
                "random_state": self.random_state, "score_is_probability": False},
            limitations=("Scores depend on reference sampling and contamination setting.",
                         "No spatial context is learned unless spatial features are supplied."),
            missing_evidence=("representative reference population", "independent threshold calibration"))


class GaussianMixtureDetector:
    """GMM negative log likelihood detector with a reference-quantile cutoff."""
    model_id = "gaussian_mixture_novelty"

    def __init__(self, *, components: int = 2, threshold_quantile: float = 0.99, random_state: int = 0) -> None:
        if components < 1 or not 0.5 < threshold_quantile < 1:
            raise ValueError("components must be positive and threshold_quantile in (0.5,1)")
        self.components, self.threshold_quantile, self.random_state = components, threshold_quantile, random_state
        self.fitted = False

    def fit(self, reference_features: np.ndarray, *, feature_names: Sequence[str] | None = None) -> "GaussianMixtureDetector":
        try:
            from sklearn.mixture import GaussianMixture
        except ImportError as exc:
            raise RuntimeError("Install the optional machine-learning dependency to use Gaussian mixtures") from exc
        data = np.asarray(reference_features, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError("reference features must be 2D")
        data = data[np.isfinite(data).all(axis=1)]
        if data.shape[0] < max(10, self.components * 3):
            raise ValueError("reference feature matrix is too small for the requested mixture")
        self.estimator = GaussianMixture(n_components=self.components, reg_covar=1e-6,
                                         random_state=self.random_state).fit(data)
        self.threshold = float(np.quantile(-self.estimator.score_samples(data), self.threshold_quantile))
        self.feature_names = tuple(feature_names or (f"feature_{i}" for i in range(data.shape[1])))
        if len(self.feature_names) != data.shape[1]:
            raise ValueError("feature_names length must match feature count")
        self.fitted = True
        return self

    def detect(self, request: DetectionRequest) -> DetectionOutput:
        if not self.fitted:
            raise RuntimeError("fit Gaussian mixture on a justified reference population first")
        cube, names = _feature_cube(request, self.feature_names)
        valid = np.isfinite(cube).all(axis=-1)
        flat = np.where(valid[..., None], cube, 0).reshape(-1, cube.shape[-1])
        score = -self.estimator.score_samples(flat).reshape(cube.shape[:-1])
        score[~valid] = np.nan
        return _output(request, self.model_id, score.astype(np.float32), valid & (score >= self.threshold),
            "negative Gaussian-mixture log likelihood; larger means less typical",
            "reference_quantile_uncalibrated", diagnostics={"feature_names": list(names),
                "components": self.components, "threshold_quantile": self.threshold_quantile,
                "threshold": self.threshold, "score_is_probability": False},
            limitations=("Gaussian mixture density depends on component count and reference population.",
                         "Negative log likelihood is not a posterior probability."),
            missing_evidence=("representative reference population", "independent calibration set"))
