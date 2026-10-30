"""Optional Siamese 3D CNN architecture and common detector adapter.

The network is an architecture scaffold only. It has not been trained and its raw
head output is not a probability. PyTorch is imported only when the builder is called.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from ..detection.contracts import DetectionOutput, DetectionRequest


@dataclass(frozen=True)
class Siamese3DConfig:
    in_channels: int = 1
    channels: tuple[int, ...] = (8, 16, 32)
    output_threshold: float = 0.0

    def __post_init__(self) -> None:
        if self.in_channels < 1 or not self.channels or any(value < 2 for value in self.channels):
            raise ValueError("Siamese channels must be positive and include at least one feature level")


class SiameseBackend(Protocol):
    def predict(self, baseline: np.ndarray, monitor: np.ndarray) -> np.ndarray: ...


class TrainableSiameseBackend(SiameseBackend, Protocol):
    """Future supervised/self-supervised trainer contract; no trainer is implemented here."""
    def fit(self, paired_patches: tuple[np.ndarray, np.ndarray], targets: np.ndarray, *,
            validation_pairs: tuple[np.ndarray, np.ndarray], validation_targets: np.ndarray,
            provenance: dict[str, Any], configuration: dict[str, Any]) -> dict[str, Any]: ...


class SiameseDetector:
    model_id = "siamese_3d_shared_encoder"
    model_version = "architecture-1"

    def __init__(self, backend: SiameseBackend, *, threshold: float = 0.0,
                 configuration: dict[str, Any] | None = None) -> None:
        self.backend = backend
        self.threshold = float(threshold)
        self.configuration = dict(configuration or {})

    def detect(self, request: DetectionRequest) -> DetectionOutput:
        if request.baseline is None or request.monitor is None:
            raise ValueError("Siamese detector requires paired baseline and monitor volumes")
        if not np.isfinite(request.baseline).all() or not np.isfinite(request.monitor).all():
            raise ValueError("Siamese backend requires finite paired inputs; apply an explicit validity-mask strategy first")
        score = np.asarray(self.backend.predict(request.baseline, request.monitor), dtype=np.float32)
        if score.shape != request.shape:
            raise ValueError("Siamese backend output must match the input voxel grid")
        mask = np.isfinite(score) & (score >= self.threshold)
        return DetectionOutput(self.model_id, self.model_version, score, mask,
            "raw Siamese change-head logit/score (not probability)", "untrained_or_uncalibrated",
            request.input_provenance,
            ("No trained weights or validated labels are supplied.",
             "Whole-volume inference is not memory tiled; use overlap-aware patches for large volumes.",
             "The network response is not a geological classification."),
            {**request.processing_configuration, **self.configuration, "threshold": self.threshold},
            diagnostics={"feature_design": "shared multi-scale encoder; absolute and signed feature differences",
                         "training_status": "architecture_only_untrained", "score_is_probability": False},
            missing_evidence=("independent voxel labels", "survey-specific calibration", "repeatability/QC gating"))


def build_torch_siamese_3d(config: Siamese3DConfig):
    """Build the shared-weight multi-scale Torch model; requires optional PyTorch."""
    from .torch_models import build_siamese_3d
    return build_siamese_3d(in_channels=config.in_channels, channels=config.channels)


class TorchSiameseBackend:
    """Minimal full-volume inference adapter for the optional Torch architecture."""
    def __init__(self, model, *, device: str = "cpu") -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is not installed") from exc
        self.torch = torch
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()

    def predict(self, baseline: np.ndarray, monitor: np.ndarray) -> np.ndarray:
        torch = self.torch
        before = torch.as_tensor(np.asarray(baseline), dtype=torch.float32, device=self.device)[None, None]
        after = torch.as_tensor(np.asarray(monitor), dtype=torch.float32, device=self.device)[None, None]
        with torch.inference_mode():
            output = self.model(before, after)
        return output[0, 0].detach().cpu().numpy()
