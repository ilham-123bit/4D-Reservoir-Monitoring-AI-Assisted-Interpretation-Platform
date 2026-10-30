"""Binary detection metrics; this module is the only core consumer of truth masks."""

import time

import numpy as np

from ..models import EvaluationMetrics


def evaluate_detection(prediction: np.ndarray, truth: np.ndarray, *, runtime_seconds: float = 0.0) -> EvaluationMetrics:
    pred, actual = np.asarray(prediction, bool), np.asarray(truth, bool)
    if pred.shape != actual.shape:
        raise ValueError("prediction and truth must have matching shapes")
    tp = int(np.count_nonzero(pred & actual))
    fp = int(np.count_nonzero(pred & ~actual))
    fn = int(np.count_nonzero(~pred & actual))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    union = tp + fp + fn
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return EvaluationMetrics(precision, recall, f1, tp / union if union else 1.0, fp, fn,
                             tp / int(actual.sum()) if actual.any() else 1.0, float(runtime_seconds))


def run_detector(detector, baseline: np.ndarray, monitor: np.ndarray, truth: np.ndarray):
    """Run detector without truth in its call signature, then evaluate separately."""
    start = time.perf_counter()
    result = detector(baseline, monitor)
    elapsed = time.perf_counter() - start
    from dataclasses import replace
    return result, replace(evaluate_detection(result.mask, truth), runtime_seconds=elapsed)
