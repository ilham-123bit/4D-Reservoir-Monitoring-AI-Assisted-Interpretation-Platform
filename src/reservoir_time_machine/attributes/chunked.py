"""Out-of-core paired attributes, chunked over inline/crossline with full sample traces."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from ..models import ChangeAttributes
from .difference import calculate_attributes


_FIELDS = ("signed_difference", "absolute_difference", "normalized_difference",
           "rms_baseline", "rms_monitor", "nrms")


def calculate_attributes_chunked(baseline: np.ndarray, monitor: np.ndarray, output_directory: str | Path, *,
                                 inline_chunk: int = 8, crossline_chunk: int = 8,
                                 window_samples: int = 9, epsilon: float = 1e-8,
                                 output_dtype: str | np.dtype = "float32") -> ChangeAttributes:
    """Calculate sample-window attributes without holding full result cubes in RAM.

    Sample windows remain whole within each spatial tile, so no temporal halo is needed.
    Inputs are read-only from this function's perspective; output files are exclusive.
    Spatial sample-independent attributes are chunk-equivalent to ``calculate_attributes``.
    """
    before, after = np.asanyarray(baseline), np.asanyarray(monitor)
    if before.ndim != 3 or before.shape != after.shape:
        raise ValueError("paired inputs must be matching 3D volumes")
    if inline_chunk < 1 or crossline_chunk < 1:
        raise ValueError("spatial chunk sizes must be positive")
    target = Path(output_directory)
    target.mkdir(parents=True, exist_ok=True)
    paths = {name: target / f"{name}.npy" for name in _FIELDS}
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite analytical outputs: {', '.join(existing)}")
    outputs = {name: np.lib.format.open_memmap(path, mode="w+", dtype=output_dtype, shape=before.shape)
               for name, path in paths.items()}
    try:
        ni, nx, _ = before.shape
        for i0 in range(0, ni, inline_chunk):
            for x0 in range(0, nx, crossline_chunk):
                core = (slice(i0, min(ni, i0 + inline_chunk)),
                        slice(x0, min(nx, x0 + crossline_chunk)), slice(None))
                attrs = calculate_attributes(before[core], after[core],
                                             window_samples=window_samples, epsilon=epsilon)
                for name in _FIELDS:
                    outputs[name][core] = getattr(attrs, name)
        for output in outputs.values():
            output.flush()
    except Exception:
        for output in outputs.values():
            output.flush()
        raise
    return ChangeAttributes(**outputs)
