"""Correlation-aware probabilistic fusion for explicitly characterized evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.special import expit, logit

from ..detection.contracts import DetectionOutput


@dataclass(frozen=True)
class EvidenceFusionResult:
    log_likelihood_ratio: np.ndarray
    posterior_probability: np.ndarray | None
    agreement_count: np.ndarray
    disagreement_count: np.ndarray
    valid_model_count: np.ndarray
    uncertainty: np.ndarray | None
    diagnostics: dict[str, object]


def summarize_detection_agreement(outputs: Sequence[DetectionOutput]) -> dict[str, object]:
    """Compare masks using overlap metrics without averaging scores or implying cause."""
    if len(outputs) < 2:
        raise ValueError("at least two model outputs are required")
    shape = np.shape(outputs[0].mask)
    if any(np.shape(item.mask) != shape for item in outputs):
        raise ValueError("all model outputs must share the same voxel grid")
    masks = [np.asarray(item.mask, dtype=bool) for item in outputs]
    pairwise = {}
    for first in range(len(masks)):
        for second in range(first + 1, len(masks)):
            intersection = int(np.count_nonzero(masks[first] & masks[second]))
            union = int(np.count_nonzero(masks[first] | masks[second]))
            pairwise[f"{outputs[first].model_id}__{outputs[second].model_id}"] = {
                "intersection_count": intersection, "union_count": union,
                "jaccard": intersection / union if union else 1.0,
            }
    unanimous = np.ones(shape, dtype=bool)
    any_detection = np.zeros(shape, dtype=bool)
    for mask in masks:
        unanimous &= mask
        any_detection |= mask
    return {"model_ids": [item.model_id for item in outputs],
            "detection_count_per_model": {item.model_id: int(mask.sum()) for item, mask in zip(outputs, masks)},
            "pairwise_mask_overlap": pairwise,
            "voxel_unanimous_count": int(np.count_nonzero(unanimous)),
            "voxel_any_count": int(np.count_nonzero(any_detection)),
            "interpretation": "detector agreement is algorithmic corroboration, not geological confirmation"}


def fuse_correlated_log_evidence(evidence: np.ndarray, *, null_mean: Sequence[float],
                                 change_mean: Sequence[float], null_covariance: np.ndarray,
                                 change_covariance: np.ndarray, prior_change_probability: float,
                                 measurement_noise_variance: Sequence[float] | None = None,
                                 calibration_verified: bool = False,
                                 calibration_reference: str | None = None) -> EvidenceFusionResult:
    """Fuse a correlated vector of model evidence with two multivariate Gaussian classes.

    ``evidence[..., model]`` must be a validated, consistently oriented evidence vector.
    Separate covariance matrices capture within-class model dependence. Missing evidence
    is marginalized by evaluating the matching covariance submatrix per finite pattern.
    A posterior is only emitted when external calibration has been explicitly verified.
    Otherwise only a model-assumption log-likelihood ratio is returned.
    """
    values = np.asarray(evidence, dtype=np.float64)
    mean0, mean1 = np.asarray(null_mean, dtype=np.float64), np.asarray(change_mean, dtype=np.float64)
    cov0, cov1 = np.asarray(null_covariance, dtype=np.float64), np.asarray(change_covariance, dtype=np.float64)
    count = mean0.size
    if values.ndim < 2 or values.shape[-1] != count or mean1.shape != (count,):
        raise ValueError("evidence must end in model axis matching both mean vectors")
    if cov0.shape != (count, count) or cov1.shape != (count, count):
        raise ValueError("class covariance matrices must match model count")
    if measurement_noise_variance is None:
        noise_variance = np.zeros(count, dtype=np.float64)
        noise_status = "not separately supplied; caller covariance must include repeatability/acquisition effects"
    else:
        noise_variance = np.asarray(measurement_noise_variance, dtype=np.float64)
        if noise_variance.shape != (count,) or not np.isfinite(noise_variance).all() or (noise_variance < 0).any():
            raise ValueError("measurement_noise_variance must be a non-negative finite vector with one entry per model")
        noise_status = "explicit diagonal repeatability/acquisition variance added to both class covariances"
    if not 0 < prior_change_probability < 1:
        raise ValueError("prior_change_probability must be in (0,1)")
    if calibration_verified and not calibration_reference:
        raise ValueError("verified calibration requires a durable independent validation reference")
    for covariance in (cov0, cov1):
        if not np.allclose(covariance, covariance.T, rtol=1e-8, atol=1e-10):
            raise ValueError("covariance matrices must be symmetric")
        if np.linalg.eigvalsh(covariance).min() <= 0:
            raise ValueError("covariance matrices must be positive definite; regularize explicitly")
    cov0 = cov0 + np.diag(noise_variance)
    cov1 = cov1 + np.diag(noise_variance)
    flat = values.reshape(-1, count)
    log_lr = np.full(flat.shape[0], np.nan, dtype=np.float64)
    agreement = np.zeros(flat.shape[0], dtype=np.uint16)
    disagreement = np.zeros(flat.shape[0], dtype=np.uint16)
    support_count = np.zeros(flat.shape[0], dtype=np.uint16)
    patterns, inverse = np.unique(np.isfinite(flat), axis=0, return_inverse=True)

    def log_density(samples, mean, covariance):
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0:
            raise ValueError("covariance is not positive definite")
        delta = samples - mean
        solved = np.linalg.solve(covariance, delta.T).T
        return -0.5 * (mean.size*np.log(2*np.pi) + logdet + np.einsum("ij,ij->i", delta, solved))

    for pattern_index, pattern in enumerate(patterns):
        selected_rows = np.flatnonzero(inverse == pattern_index)
        selected_features = np.flatnonzero(pattern)
        if not selected_features.size:
            continue
        samples = flat[np.ix_(selected_rows, selected_features)]
        sub_cov0 = cov0[np.ix_(selected_features, selected_features)]
        sub_cov1 = cov1[np.ix_(selected_features, selected_features)]
        log_lr[selected_rows] = (log_density(samples, mean1[selected_features], sub_cov1)
                                 - log_density(samples, mean0[selected_features], sub_cov0))
        signed = samples
        agreement[selected_rows] = (signed > 0).sum(axis=1)
        disagreement[selected_rows] = (signed < 0).sum(axis=1)
        support_count[selected_rows] = len(selected_features)
    shape = values.shape[:-1]
    log_lr = log_lr.reshape(shape)
    posterior = None
    entropy = None
    if calibration_verified:
        posterior = expit(log_lr + logit(prior_change_probability))
        entropy = -(posterior*np.log(np.maximum(posterior, 1e-12))
                    +(1-posterior)*np.log(np.maximum(1-posterior, 1e-12)))
    return EvidenceFusionResult(log_lr, posterior, agreement.reshape(shape), disagreement.reshape(shape),
        support_count.reshape(shape), entropy,
        {"model_count": count, "prior_change_probability": prior_change_probability,
         "null_covariance": cov0.tolist(), "change_covariance": cov1.tolist(),
         "dependence_handling": "joint multivariate Gaussian covariance; finite-feature submatrices for missing evidence",
         "measurement_noise_variance": noise_variance.tolist(), "repeatability_acquisition_handling": noise_status,
         "calibration_verified": calibration_verified, "calibration_reference": calibration_reference,
         "posterior_status": "validated_calibrated" if calibration_verified else "omitted_uncalibrated",
         "warning": "Evidence features must be consistently oriented and independently calibrated; detector score maps cannot be inserted without a validated calibration transform."})
