"""Common detector protocol; scores are strengths unless calibrated explicitly."""

from typing import Protocol

import numpy as np

from ..models import AnomalyDetectionResult, SurveyMetadata


class ChangeDetector(Protocol):
    name: str

    def detect(self, baseline: np.ndarray, monitor: np.ndarray,
               metadata: SurveyMetadata | None = None) -> AnomalyDetectionResult: ...
