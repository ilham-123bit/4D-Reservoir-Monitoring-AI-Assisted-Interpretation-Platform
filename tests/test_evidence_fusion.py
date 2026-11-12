"""Small tests for assumption-sensitive Bayesian evidence fusion."""
import unittest
import numpy as np

from reservoir_time_machine.ai.evidence_fusion import (
    fuse_correlated_log_evidence, summarize_detection_agreement,
)
from reservoir_time_machine.detection.contracts import DetectionOutput


class EvidenceFusionTests(unittest.TestCase):
    def test_uncalibrated_likelihood_ratio_does_not_publish_posterior(self):
        observations = np.array([[[[1.0, 1.2], [0.5, np.nan]]]], dtype=float)
        mean0 = np.array([0.0, 0.0]); mean1 = np.array([1.0, 1.0])
        covariance = np.array([[1.0, 0.5], [0.5, 1.0]])
        result = fuse_correlated_log_evidence(observations, null_mean=mean0,
            change_mean=mean1, null_covariance=covariance, change_covariance=covariance,
            prior_change_probability=0.1, calibration_verified=False)
        self.assertIsNone(result.posterior_probability)
        self.assertTrue(np.isfinite(result.log_likelihood_ratio).all())
        self.assertEqual(result.diagnostics["posterior_status"], "omitted_uncalibrated")

    def test_prior_and_calibration_are_not_implicitly_inferred(self):
        observations = np.ones((1, 1, 1, 1))
        args = dict(evidence=observations, null_mean=[0.0], change_mean=[1.0],
                    null_covariance=np.array([[1.0]]), change_covariance=np.array([[1.0]]),
                    calibration_verified=False)
        low = fuse_correlated_log_evidence(**args, prior_change_probability=0.01)
        high = fuse_correlated_log_evidence(**args, prior_change_probability=0.5)
        self.assertGreater(low.log_likelihood_ratio.item(), 0)
        self.assertAlmostEqual(low.log_likelihood_ratio.item(), high.log_likelihood_ratio.item())
        self.assertIsNone(low.posterior_probability)

    def test_detector_agreement_requires_matching_run_identity(self):
        def output(name, run):
            return DetectionOutput(name, "1", np.ones((1, 1, 1)), np.ones((1, 1, 1), bool),
                "uncalibrated score", "uncalibrated", {"dataset_id": "d", "run_id": run}, (), {})
        with self.assertRaisesRegex(ValueError, "incompatible"):
            summarize_detection_agreement([output("a", "run-a"), output("b", "run-b")])


if __name__ == "__main__":
    unittest.main()
