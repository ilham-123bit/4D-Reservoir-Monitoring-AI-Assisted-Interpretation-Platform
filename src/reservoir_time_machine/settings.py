"""Environment-based paths and service configuration; contains no secrets."""

from dataclasses import dataclass
import os
from pathlib import Path


DEFAULT_CHALLENGE3_ROOT = Path(".")


@dataclass(frozen=True)
class ApplicationSettings:
    challenge3_root: Path
    max_pair_memory_bytes: int = 2_000_000_000
    cache_directory: Path | None = None

    @classmethod
    def from_environment(cls) -> "ApplicationSettings":
        root = Path(os.environ.get("RTM_CHALLENGE3_ROOT", str(DEFAULT_CHALLENGE3_ROOT))).expanduser()
        max_bytes = int(os.environ.get("RTM_MAX_PAIR_MEMORY_BYTES", "2000000000"))
        cache_value = os.environ.get("RTM_CACHE_DIR")
        if max_bytes <= 0:
            raise ValueError("RTM_MAX_PAIR_MEMORY_BYTES must be positive")
        return cls(root, max_bytes, Path(cache_value).expanduser() if cache_value else None)


@dataclass(frozen=True)
class LLMSettings:
    # Service locations are deployment-specific and must be supplied at runtime.
    base_url: str | None = None
    model: str = "google/gemma-4-26B-A4B-it"
    api_key: str = "not-needed"
    timeout_seconds: float = 30.0

    @classmethod
    def from_environment(cls) -> "LLMSettings":
        base_url = os.environ.get("RTM_LLM_BASE_URL", "").strip() or None
        return cls(base_url=base_url,
                   model=os.environ.get("RTM_LLM_MODEL", cls.model),
                   api_key=os.environ.get("RTM_LLM_API_KEY", "not-needed"),
                   timeout_seconds=float(os.environ.get("RTM_LLM_TIMEOUT", "30")))
