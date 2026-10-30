"""Shared request/output contracts for heterogeneous change detectors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

import numpy as np


def _compact(value: Any) -> Any:
    """Keep summaries JSON-friendly without embedding full voxel maps."""
    if isinstance(value, np.ndarray):
        finite = value[np.isfinite(value)] if np.issubdtype(value.dtype, np.number) else value.reshape(-1)
        result = {"shape": list(value.shape), "dtype": str(value.dtype)}
        if finite.size and np.issubdtype(value.dtype, np.number):
            result.update({"finite_count": int(finite.size), "min": float(finite.min()), "max": float(finite.max())})
        return result
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _compact(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_compact(item) for item in value]
    return value


@dataclass(frozen=True)
class DetectionRequest:
    """Common model input: paired volumes and/or derived attribute maps."""
    attributes: Mapping[str, np.ndarray]
    baseline: np.ndarray | None = None
    monitor: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    input_provenance: Mapping[str, Any] = field(default_factory=dict)
    processing_configuration: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        arrays = [np.asanyarray(value) for value in self.attributes.values()]
        if self.baseline is not None:
            arrays.append(np.asanyarray(self.baseline))
        if self.monitor is not None:
            arrays.append(np.asanyarray(self.monitor))
        if self.baseline is not None and self.monitor is None or self.monitor is not None and self.baseline is None:
            raise ValueError("baseline and monitor must be supplied together")
        if self.baseline is not None and (np.ndim(self.baseline) != 3 or np.shape(self.baseline) != np.shape(self.monitor)):
            raise ValueError("paired input volumes must have matching 3D shapes")
        if arrays and (len(arrays[0].shape) != 3 or any(array.shape != arrays[0].shape for array in arrays)):
            raise ValueError("all supplied features and volumes must share one 3D grid shape")

    @property
    def shape(self) -> tuple[int, int, int]:
        if self.baseline is not None:
            return tuple(np.shape(self.baseline))
        if self.attributes:
            return tuple(np.shape(next(iter(self.attributes.values()))))
        raise ValueError("request contains no volumes or attributes")


@dataclass(frozen=True)
class DetectionOutput:
    """Structured detection evidence; score meaning and calibration are mandatory."""
    model_id: str
    model_version: str
    score: np.ndarray
    mask: np.ndarray
    score_meaning: str
    calibration_status: str
    input_provenance: Mapping[str, Any]
    limitations: tuple[str, ...]
    processing_configuration: Mapping[str, Any]
    uncertainty: np.ndarray | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    missing_evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        score, mask = np.asanyarray(self.score), np.asanyarray(self.mask)
        if score.ndim != 3 or score.shape != mask.shape:
            raise ValueError("detection score and mask must be matching 3D maps")
        if self.uncertainty is not None and np.shape(self.uncertainty) != score.shape:
            raise ValueError("uncertainty map must match detection score shape")
        if not self.model_id or not self.model_version or not self.score_meaning or not self.calibration_status:
            raise ValueError("model ID/version, score meaning, and calibration status are required")

    def summary(self) -> dict[str, Any]:
        mask = np.asarray(self.mask, dtype=bool)
        bounds = []
        if mask.any():
            for axis in range(mask.ndim):
                other_axes = tuple(index for index in range(mask.ndim) if index != axis)
                occupied = np.flatnonzero(mask.any(axis=other_axes))
                bounds.append([int(occupied[0]), int(occupied[-1])])
        else:
            bounds = None
        return {
            "model_id": self.model_id, "model_version": self.model_version,
            "score_meaning": self.score_meaning, "calibration_status": self.calibration_status,
            "dimensions": list(mask.shape), "detection_count": int(mask.sum()),
            "bounding_box_inclusive": bounds,
            "score_map_summary": _compact(np.asarray(self.score)),
            "input_provenance": _compact(dict(self.input_provenance)),
            "input_provenance_status": "supplied" if self.input_provenance else "missing",
            "processing_configuration": dict(self.processing_configuration),
            "limitations": list(self.limitations),
            "missing_evidence": list(self.missing_evidence) + ([] if self.input_provenance else ["input provenance not supplied"]),
            "array_axes": ["inline", "crossline", "sample"],
            "uncertainty_available": self.uncertainty is not None,
            "uncertainty_map_summary": _compact(np.asarray(self.uncertainty)) if self.uncertainty is not None else None,
            "diagnostics": _compact(dict(self.diagnostics)), "score_is_probability": False,
        }


class Detector(Protocol):
    model_id: str
    model_version: str

    def detect(self, request: DetectionRequest) -> DetectionOutput: ...
