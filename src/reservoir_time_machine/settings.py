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
    # Deployment values only: no private credential or endpoint is embedded.
    base_url: str | None = None
    model: str = "google/gemma-4-26B-A4B-it"
    api_key: str = "not-needed"
    timeout_seconds: float = 45.0
    max_tokens: int = 2048
    structured_mode: str = "json_schema"
    policy_confirmed: bool = False
    allow_challenge3: bool = False
    policy_reference: str = ""
    lock_directory: str | None = None

    def __post_init__(self):
        import math
        from urllib.parse import urlsplit
        if not isinstance(self.policy_confirmed,bool) or not isinstance(self.allow_challenge3,bool):
            raise ValueError('LLM policy flags must be booleans')
        if isinstance(self.timeout_seconds,bool) or not math.isfinite(self.timeout_seconds) or not 1 <= self.timeout_seconds <= 120:
            raise ValueError("LLM timeout must be between 1 and 120 seconds")
        if isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, int) or not 128 <= self.max_tokens <= 4096:
            raise ValueError("LLM max_tokens must be between 128 and 4096")
        if self.structured_mode not in {"json_schema", "json_object"} or not self.model.strip():
            raise ValueError("Invalid model or structured-output mode")
        if self.base_url:
            url = urlsplit(self.base_url)
            if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query or url.fragment:
                raise ValueError("LLM base URL must be HTTP(S), without embedded credentials/query")

    @property
    def real_data_enabled(self):
        return self.policy_confirmed and self.allow_challenge3 and bool(self.policy_reference.strip())

    def configuration_id(self):
        import hashlib, json
        # API keys are neither science provenance nor cache key material.
        values = {key:getattr(self,key) for key in ('base_url','model','max_tokens','structured_mode',
                   'timeout_seconds','policy_confirmed','allow_challenge3','policy_reference')}
        return hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest()

    @classmethod
    def from_environment(cls) -> "LLMSettings":
        def flag(name):
            value = os.environ.get(name, '0').strip().lower()
            if value not in {'0','1','false','true'}:
                raise ValueError(f'{name} must be 0/1 or false/true')
            return value in {'1','true'}
        return cls(base_url=os.environ.get("RTM_LLM_BASE_URL", "").strip() or None,
                   model=os.environ.get("RTM_LLM_MODEL", cls.model),
                   api_key=os.environ.get("RTM_LLM_API_KEY", "not-needed"),
                   timeout_seconds=float(os.environ.get("RTM_LLM_TIMEOUT", "45")),
                   max_tokens=int(os.environ.get("RTM_LLM_MAX_TOKENS", "2048")),
                   structured_mode=os.environ.get('RTM_LLM_STRUCTURED_MODE','json_schema'),
                   policy_confirmed=flag('RTM_LLM_DATA_POLICY_CONFIRMED'),
                   allow_challenge3=flag('RTM_LLM_ALLOW_CHALLENGE3'),
                   policy_reference=os.environ.get('RTM_LLM_DATA_POLICY_REFERENCE',''),
                   lock_directory=os.environ.get('RTM_LLM_LOCK_DIR') or None)
