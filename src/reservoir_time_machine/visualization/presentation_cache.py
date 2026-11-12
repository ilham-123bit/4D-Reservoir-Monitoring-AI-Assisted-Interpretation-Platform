"""Atomic, identity-checked cache for offline-built finding presentations.

Presentation JSON is a derived display artifact. It is kept outside the source
tree and seismic cache; stale or incomplete artifacts are ignored safely.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

import plotly.io as pio

SCHEMA = "rtm-finding-presentation-1"
VERSION = "finding-animation-1"
_TOKEN = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")


def presentation_identity(registry_id: str, run_id: str, finding_id: str,
                          dataset_id: str) -> dict[str, str]:
    values = {"registry_id": registry_id, "run_id": run_id,
              "finding_id": finding_id, "dataset_id": dataset_id,
              "renderer": VERSION}
    if any(not isinstance(value, str) or not _TOKEN.fullmatch(value)
           for value in values.values()):
        raise ValueError("Presentation identity contains an invalid token")
    return values


def _target(root: str | Path, identity: dict[str, str]) -> Path:
    unresolved = Path(root).expanduser()
    if unresolved.is_symlink():
        raise ValueError("Presentation cache root cannot be a symlink")
    root = unresolved.resolve()
    if ".reservoir_time_machine_cache" in root.parts:
        raise ValueError("Presentation artifacts cannot be written inside the seismic cache")
    key = hashlib.sha256(json.dumps(identity, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()
    return root / f"{key}.json"


def presentation_cache_token(root: str | Path, identity: dict[str, str]) -> tuple:
    """Small cache key that changes atomically when an offline artifact appears."""
    try:
        info = _target(root, identity).stat()
        return (True, info.st_size, info.st_mtime_ns, info.st_ino)
    except OSError:
        return (False,)


def save_presentation(fig, root: str | Path, identity: dict[str, str]) -> Path:
    """Serialize a finished figure and atomically publish its checksum envelope."""
    target = _target(root, identity)
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if target.parent.is_symlink() or target.parent.stat().st_uid != os.getuid():
        raise PermissionError("Presentation cache directory must be user-owned and not redirected")
    os.chmod(target.parent, 0o700)
    figure_json = fig.to_json(validate=True, pretty=False, remove_uids=True)
    figure_value = json.loads(figure_json)
    figure_bytes = json.dumps(figure_value, sort_keys=True, separators=(",", ":"),
                              allow_nan=False).encode("utf-8")
    envelope = {"schema": SCHEMA, "identity": identity,
                "figure_sha256": hashlib.sha256(figure_bytes).hexdigest(),
                "figure": figure_value}
    payload = json.dumps(envelope, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8") + b"\n"
    fd, temporary_name = tempfile.mkstemp(prefix=".presentation-", suffix=".tmp",
                                           dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, target)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    return target


def load_presentation(root: str | Path, identity: dict[str, str]):
    """Load a completed presentation, returning None for absent/stale/corrupt data."""
    target = _target(root, identity)
    try:
        if target.is_symlink() or target.stat().st_size > 8_000_000:
            return None
        envelope = json.loads(target.read_text(encoding="utf-8"))
        if envelope.get("schema") != SCHEMA or envelope.get("identity") != identity:
            return None
        figure_json = json.dumps(envelope["figure"], sort_keys=True,
                                 separators=(",", ":"), allow_nan=False)
        if hashlib.sha256(figure_json.encode("utf-8")).hexdigest() != envelope.get("figure_sha256"):
            return None
        return pio.from_json(figure_json, skip_invalid=False)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def get_or_build_presentation(root: str | Path, identity: dict[str, str], builder):
    """Use a verified disk figure or build and atomically publish a replacement."""
    cached = load_presentation(root, identity)
    if cached is not None:
        return cached, True
    figure = builder()
    save_presentation(figure, root, identity)
    return figure, False
