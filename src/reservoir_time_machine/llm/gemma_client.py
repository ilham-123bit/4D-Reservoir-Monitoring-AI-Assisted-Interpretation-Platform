"""Optional OpenAI-compatible Gemma client for concise structured evidence only."""

import json
from typing import Any

from ..settings import LLMSettings


def _client(settings: LLMSettings):
    if not settings.base_url:
        raise RuntimeError(
            "No LLM service endpoint is configured. Set RTM_LLM_BASE_URL in the authorized runtime environment."
        )
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Gemma requests require the optional dependency: pip install .[llm]") from exc
    return OpenAI(api_key=settings.api_key or "not-needed", base_url=settings.base_url,
                  timeout=settings.timeout_seconds)


def test_connection(settings: LLMSettings | None = None) -> dict[str, Any]:
    """Check configured OpenAI-compatible model-list endpoint; does not imply model quality."""
    settings = settings or LLMSettings.from_environment()
    if not settings.base_url:
        return {"connected": False, "configured_model": settings.model,
                "error": "No LLM service endpoint is configured."}
    try:
        response = _client(settings).models.list()
        available = [item.id for item in response.data]
        return {"connected": True, "configured_model": settings.model,
                "configured_model_listed": settings.model in available,
                "available_model_count": len(available), "base_url": settings.base_url}
    except Exception as exc:  # provider-specific network/protocol exceptions
        return {"connected": False, "configured_model": settings.model,
                "base_url": settings.base_url, "error": f"{type(exc).__name__}: {exc}"}


def explain_evidence(evidence: dict[str, Any], settings: LLMSettings | None = None,
                     *, max_payload_bytes: int = 24_000) -> str:
    """Explain a bounded JSON evidence object; arrays/volumes are rejected by JSON encoding."""
    settings = settings or LLMSettings.from_environment()
    payload = json.dumps(evidence, ensure_ascii=False, allow_nan=False)
    if len(payload.encode("utf-8")) > max_payload_bytes:
        raise ValueError(f"Evidence exceeds {max_payload_bytes} bytes; pass selected summaries, not seismic cubes.")
    if not settings.base_url:
        raise RuntimeError(
            "No LLM service endpoint is configured. Set RTM_LLM_BASE_URL in the authorized runtime environment."
        )
    response = _client(settings).chat.completions.create(
        model=settings.model,
        messages=[{"role": "system", "content": (
            "Explain only the supplied seismic and well evidence. Separate observations, algorithmic detections, "
            "interpretations, alternatives, uncertainty, and suggested follow-up. Never infer remaining oil as fact. "
            "State when evidence is insufficient. Do not claim probabilities unless present in the evidence.")},
                  {"role": "user", "content": payload}],
        temperature=0.2,
    )
    return response.choices[0].message.content or "The model returned no explanation text."
