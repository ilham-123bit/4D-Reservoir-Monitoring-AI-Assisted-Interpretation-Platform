"""Typed contracts shared by generators, processing, evaluation, and displays."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import math

import numpy as np

ARRAY_AXES = ("inline", "crossline", "sample")
SYNTHETIC_LABEL = "SYNTHETIC — FOR DEVELOPMENT AND DEMONSTRATION ONLY"


@dataclass(frozen=True)
class SurveyMetadata:
    survey_id: str
    label: str
    dimensions: tuple[int, int, int]
    axis_labels: tuple[str, str, str] = ARRAY_AXES
    inline_units: str = "index"
    crossline_units: str = "index"
    sample_interval: float = 0.002
    sample_interval_units: str = "s"
    sample_units: str = "s"
    sample_axis_domain: str = "time"
    synthetic: bool = False
    random_seed: int | None = None
    generation_parameters: dict[str, Any] = field(default_factory=dict)
    source_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.dimensions) != 3 or any(d < 1 for d in self.dimensions):
            raise ValueError("metadata dimensions must contain three positive axes")
        if not math.isfinite(self.sample_interval) or self.sample_interval <= 0:
            raise ValueError("sample interval must be finite and positive")


@dataclass(frozen=True)
class SeismicVolume:
    data: np.ndarray
    metadata: SurveyMetadata
    missing_trace_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.data.ndim != 3:
            raise ValueError("volume data must use (inline, crossline, sample) axes")
        if tuple(self.data.shape) != self.metadata.dimensions:
            raise ValueError("metadata dimensions must match the volume")
        if self.metadata.axis_labels != ARRAY_AXES:
            raise ValueError(f"axis labels must be {ARRAY_AXES}")
        if self.missing_trace_mask is not None and self.missing_trace_mask.shape != self.data.shape[:2]:
            raise ValueError("missing trace mask must have shape (inline, crossline)")


@dataclass(frozen=True)
class GroundTruth:
    amplitude_change: np.ndarray
    time_shift: np.ndarray
    fluid_front: np.ndarray
    unchanged: np.ndarray
    acquisition_artifact: np.ndarray
    structural_fault: np.ndarray
    scenario_label: str


@dataclass(frozen=True)
class SyntheticScenario:
    baseline: SeismicVolume
    monitor: SeismicVolume
    ground_truth: GroundTruth
    parameters: dict[str, Any]
    label: str = SYNTHETIC_LABEL


@dataclass(frozen=True)
class ProcessingConfiguration:
    max_lag_samples: int = 4
    nrms_window: int = 9
    mad_threshold: float = 3.5
    min_cluster_size: int = 8
    normalization: str = "none"

    def __post_init__(self) -> None:
        if self.max_lag_samples < 0 or self.nrms_window < 1 or self.mad_threshold <= 0 or self.min_cluster_size < 1:
            raise ValueError("processing thresholds and windows must be positive; lag must be non-negative")
        if self.normalization not in {"none", "global_rms", "trace_rms"}:
            raise ValueError("unsupported normalization method")


@dataclass(frozen=True)
class RegistrationResult:
    shifts: np.ndarray
    correlation: np.ndarray
    registered_monitor: np.ndarray
    accepted: np.ndarray | None = None
    failure_reason: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ChangeAttributes:
    signed_difference: np.ndarray
    absolute_difference: np.ndarray
    normalized_difference: np.ndarray
    rms_baseline: np.ndarray
    rms_monitor: np.ndarray
    nrms: np.ndarray


@dataclass(frozen=True)
class AnomalyDetectionResult:
    score: np.ndarray
    mask: np.ndarray
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationMetrics:
    precision: float
    recall: float
    f1: float
    iou: float
    false_positive_count: int
    false_negative_count: int
    detection_coverage: float
    runtime_seconds: float


@dataclass(frozen=True)
class DetectedZone:
    label: int
    voxel_count: int
    bounding_box: tuple[tuple[int, int], ...]
    mean_strength: float
    max_strength: float
