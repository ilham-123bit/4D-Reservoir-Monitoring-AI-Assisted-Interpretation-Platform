"""Mathematical model/interface tests; no seismic or geological fixtures are used."""

import numpy as np
import pytest
import json
from scipy.ndimage import shift
from scipy.special import expit

from reservoir_time_machine.ai.detectors import (
    BayesianChangeDetector, GaussianChangeAssumptions, GaussianMixtureDetector,
    IsolationForestDetector, LinearAutoencoderDetector, MahalanobisDetector,
)
from reservoir_time_machine.ai.evidence_fusion import fuse_correlated_log_evidence, summarize_detection_agreement
from reservoir_time_machine.ai.siamese import Siamese3DConfig, SiameseDetector
from reservoir_time_machine.detection.contracts import DetectionOutput, DetectionRequest


def _request(attributes, baseline=None, monitor=None):
    return DetectionRequest(attributes, baseline, monitor,
        metadata={"array_axes": ["inline", "crossline", "sample"]},
        input_provenance={"source_id": "mathematical-fixture"},
        processing_configuration={"fixture": True})


def test_siamese_input_output_contract_with_mock_inference_backend():
    class Backend:
        def predict(self, baseline, monitor):
            return monitor - baseline

    before = np.zeros((2, 3, 4), dtype=np.float32)
    after = before.copy()
    after[1, 2, 3] = 2
    output = SiameseDetector(Backend(), threshold=1).detect(_request({}, before, after))
    assert output.score.shape == before.shape
    assert output.mask.sum() == 1
    assert output.calibration_status == "untrained_or_uncalibrated"
    assert output.summary()["score_is_probability"] is False
    assert Siamese3DConfig().channels == (8, 16, 32)


def test_bayesian_gaussian_log_likelihood_ratio_matches_known_distribution():
    evidence = np.array([[[2.0, 0.0, np.nan]]], dtype=np.float32)
    assumptions = GaussianChangeAssumptions(null_mean=0, null_std=1, change_mean=2,
                                            change_std=1, prior_change_probability=0.5)
    output = BayesianChangeDetector("evidence", assumptions).detect(_request({"evidence": evidence}))
    assert output.score[0, 0, 0] == pytest.approx(2.0)
    assert output.diagnostics["model_conditional_posterior"][0, 0, 0] == pytest.approx(expit(2.0))
    assert np.isnan(output.score[0, 0, 2])
    assert output.calibration_status == "assumption_based_uncalibrated"
    json.dumps(output.summary())  # compact evidence summary excludes raw maps


def test_mahalanobis_regularized_distance_and_missing_voxel_contract():
    reference = np.array([[-2, -1], [-1, -2], [0, 0], [1, 2], [2, 1], [0.5, -0.5]], dtype=float)
    model = MahalanobisDetector(regularization=0.05, threshold_squared_distance=5).fit(
        reference, feature_names=("a", "b"))
    feature_a = np.zeros((1, 1, 3), dtype=float)
    feature_b = np.zeros_like(feature_a)
    feature_a[0, 0, 1], feature_b[0, 0, 1] = 20, -20
    feature_a[0, 0, 2] = np.nan
    output = model.detect(_request({"a": feature_a, "b": feature_b}))
    assert output.score[0, 0, 1] > output.score[0, 0, 0]
    assert output.mask[0, 0, 1]
    assert np.isnan(output.score[0, 0, 2])
    assert np.isfinite(output.diagnostics["covariance_condition_number"])


def test_linear_autoencoder_reconstruction_residual_contract():
    t = np.linspace(-1, 1, 20)
    reference = np.column_stack((t, 2*t, -t))
    model = LinearAutoencoderDetector(latent_dimensions=1).fit(reference, feature_names=("x", "y", "z"))
    features = np.zeros((1, 1, 2, 3), dtype=float)
    features[0, 0, 0] = [0.2, 0.4, -0.2]
    features[0, 0, 1] = [0.2, 4.0, -0.2]
    output = model.detect(_request({name: features[..., index] for index, name in enumerate(("x", "y", "z"))}))
    assert output.score[0, 0, 1] > output.score[0, 0, 0]
    assert output.diagnostics["architecture"] == "linear tied-weight PCA autoencoder"
    assert output.calibration_status == "reference_quantile_uncalibrated"


def test_unsupervised_isolation_forest_and_gmm_output_contracts():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(4)
    reference = rng.normal(size=(80, 2))
    model_input = np.zeros((2, 2, 3), dtype=float)
    model_input[0, 0] = [0.0, 0.2, 0.1]
    model_input[0, 1] = [0.0, -0.1, 0.1]
    model_input[1, 0] = [0.1, 0.0, 0.0]
    model_input[1, 1] = [20.0, 20.0, 20.0]
    request = _request({"f1": model_input, "f2": model_input * 0.7})
    isolation = IsolationForestDetector(contamination=0.05, random_state=3, n_estimators=30).fit(
        reference, feature_names=("f1", "f2"))
    gaussian = GaussianMixtureDetector(components=1, threshold_quantile=0.95, random_state=3).fit(
        reference, feature_names=("f1", "f2"))
    for output in (isolation.detect(request), gaussian.detect(request)):
        assert output.score.shape == model_input.shape
        assert output.model_id
        assert "uncalibrated" in output.calibration_status
        assert output.summary()["score_is_probability"] is False


def test_correlated_evidence_fusion_handles_dependence_missingness_and_disagreement():
    evidence = np.array([[[[2.0, 2.0], [2.0, -1.0], [np.nan, 1.0]]]])
    means0, means1 = (0.0, 0.0), (1.0, 1.0)
    correlated = np.array([[1.0, 0.8], [0.8, 1.0]])
    independent = np.eye(2)
    result = fuse_correlated_log_evidence(evidence, null_mean=means0, change_mean=means1,
        null_covariance=correlated, change_covariance=correlated, prior_change_probability=0.1)
    independent_result = fuse_correlated_log_evidence(evidence, null_mean=means0, change_mean=means1,
        null_covariance=independent, change_covariance=independent, prior_change_probability=0.1)
    assert result.posterior_probability is None  # uncalibrated: no probability is emitted
    assert not np.allclose(result.log_likelihood_ratio[..., :2], independent_result.log_likelihood_ratio[..., :2])
    assert result.agreement_count[0, 0, 0] == 2
    assert result.disagreement_count[0, 0, 1] == 1
    assert result.valid_model_count[0, 0, 2] == 1
    noisy = fuse_correlated_log_evidence(evidence[..., :2], null_mean=means0, change_mean=means1,
        null_covariance=correlated, change_covariance=correlated, prior_change_probability=0.1,
        measurement_noise_variance=(10.0, 10.0))
    assert abs(noisy.log_likelihood_ratio[0, 0, 0]) < abs(result.log_likelihood_ratio[0, 0, 0])
    calibrated = fuse_correlated_log_evidence(evidence, null_mean=means0, change_mean=means1,
        null_covariance=correlated, change_covariance=correlated, prior_change_probability=0.1,
        calibration_verified=True, calibration_reference="independent mathematical calibration fixture")
    assert calibrated.posterior_probability is not None
    assert not np.isnan(calibrated.posterior_probability[0, 0, 2])


def test_detector_agreement_summary_compares_masks_without_fusing_scores():
    shape = (1, 1, 3)
    a = np.array([[[True, True, False]]])
    b = np.array([[[True, False, True]]])
    one = DetectionOutput("one", "1", a.astype(float), a, "test score", "uncalibrated", {}, (), {})
    two = DetectionOutput("two", "1", b.astype(float), b, "test score", "uncalibrated", {}, (), {})
    report = summarize_detection_agreement((one, two))
    assert report["pairwise_mask_overlap"]["one__two"]["jaccard"] == pytest.approx(1/3)
    assert report["voxel_unanimous_count"] == 1
