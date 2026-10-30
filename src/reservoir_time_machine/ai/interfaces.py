"""Provider-neutral, compact evidence contracts for optional interpretation models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol
import json
import math


EVIDENCE_STATUSES = {"measured", "algorithmic_detection", "hypothesis", "recommendation"}
DATA_CLASSIFICATIONS = {"public", "restricted", "confidential", "unknown"}


@dataclass(frozen=True)
class EvidenceItem:
    statement: str
    source: str
    status: str
    score: float | None = None
    score_semantics: str = "not_applicable"
    uncertainty: str | None = None
    supporting_output: str | None = None
    calibration_method: str | None = None
    validation_reference: str | None = None

    def __post_init__(self) -> None:
        if not self.statement.strip() or not self.source.strip():
            raise ValueError("evidence statement and source are required")
        if self.status not in EVIDENCE_STATUSES:
            raise ValueError(f"status must be one of {sorted(EVIDENCE_STATUSES)}")
        if self.score_semantics not in {"not_applicable", "uncalibrated_strength", "calibrated_probability"}:
            raise ValueError("unsupported score semantics")
        if self.score is not None and (
            isinstance(self.score, bool) or not isinstance(self.score, (int, float)) or not math.isfinite(self.score)
        ):
            raise ValueError("score must be a finite number")
        if self.score_semantics == "calibrated_probability" and (
            self.score is None or not 0.0 <= self.score <= 1.0
            or not self.calibration_method or not self.validation_reference
        ):
            raise ValueError("calibrated probability requires a [0,1] score, method, and independent validation reference")

    def to_dict(self) -> dict[str, Any]:
        return {
            "statement": self.statement,
            "source": self.source,
            "status": self.status,
            "score": self.score,
            "score_semantics": self.score_semantics,
            "uncertainty": self.uncertainty,
            "supporting_output": self.supporting_output,
            "calibration_method": self.calibration_method,
            "validation_reference": self.validation_reference,
        }


@dataclass(frozen=True)
class EvidencePacket:
    run_id: str
    data_classification: str = "unknown"
    observations: tuple[EvidenceItem, ...] = ()
    detections: tuple[EvidenceItem, ...] = ()
    interpretation_hypotheses: tuple[EvidenceItem, ...] = ()
    alternatives: tuple[str, ...] = ()
    uncertainty: tuple[str, ...] = ()
    missing_evidence: tuple[str, ...] = ()
    requested_output: tuple[str, ...] = field(default=(
        "observations", "detections", "hypotheses", "alternatives", "uncertainty", "follow_up"
    ))

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        if self.data_classification not in DATA_CLASSIFICATIONS:
            raise ValueError(f"data_classification must be one of {sorted(DATA_CLASSIFICATIONS)}")
        required_output = ("observations", "detections", "hypotheses", "alternatives", "uncertainty", "follow_up")
        if self.requested_output != required_output:
            raise ValueError("requested_output must require evidence, alternatives, uncertainty, and follow-up")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "run_id": self.run_id,
            "data_classification": self.data_classification,
            "observations": [item.to_dict() for item in self.observations],
            "detections": [item.to_dict() for item in self.detections],
            "interpretation_hypotheses": [item.to_dict() for item in self.interpretation_hypotheses],
            "alternatives": list(self.alternatives),
            "uncertainty": list(self.uncertainty),
            "missing_evidence": list(self.missing_evidence),
            "requested_output": list(self.requested_output),
        }

    def to_json(self, *, max_bytes: int = 24_000) -> str:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        try:
            rendered = json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("evidence packet must contain finite JSON-compatible summaries only") from exc
        if len(rendered.encode("utf-8")) > max_bytes:
            raise ValueError(f"evidence packet exceeds configured limit of {max_bytes} bytes")
        return rendered


class EvidenceProvider(Protocol):
    """Optional model provider. Implementations receive compact structured evidence only."""

    provider_name: str

    def explain(self, packet: EvidencePacket) -> str:
        """Return a human-reviewable explanation; never produce numeric detections."""


class GemmaEvidenceProvider:
    """Adapter for the existing OpenAI-compatible Gemma client."""

    provider_name = "gemma_openai_compatible"

    def __init__(self, settings=None, *, allow_restricted_data: bool = False, max_payload_bytes: int = 24_000):
        self.settings = settings
        self.allow_restricted_data = allow_restricted_data
        self.max_payload_bytes = max_payload_bytes

    def explain(self, packet: EvidencePacket) -> str:
        if packet.data_classification != "public" and not self.allow_restricted_data:
            raise PermissionError(
                "AI provider transmission is disabled for restricted, confidential, or unknown data unless explicitly authorized"
            )
        packet_json = packet.to_json(max_bytes=self.max_payload_bytes)
        from .gemma_client import explain_evidence

        return explain_evidence(json.loads(packet_json), self.settings, max_payload_bytes=self.max_payload_bytes)
