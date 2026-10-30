"""Pure dashboard state preparation, testable without Streamlit or seismic inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def empty_dataset_state(root: str | Path, missing_files: list[str] | tuple[str, ...]) -> dict[str, Any]:
    """Describe an empty data state without implying any measurement or interpretation."""
    missing = tuple(str(path) for path in missing_files)
    return {
        "state": "no_input_data",
        "configured_root": str(Path(root).expanduser()),
        "missing_files": list(missing),
        "analysis_enabled": False,
        "geological_findings": [],
        "message": "No input data are available. No seismic analysis or geological findings are being displayed.",
        "required_next_step": "Provide authorized input files that satisfy the documented dataset manifest contract.",
    }
