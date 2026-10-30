"""Band-limited seismic wavelets."""

import numpy as np


def ricker(sample_offsets: np.ndarray, peak_frequency_hz: float = 30.0, sample_interval: float = 0.002) -> np.ndarray:
    """Evaluate a zero-phase Ricker wavelet at sample offsets (seconds)."""
    if peak_frequency_hz <= 0 or sample_interval <= 0:
        raise ValueError("frequency and sample interval must be positive")
    t = np.asarray(sample_offsets, dtype=float) * sample_interval
    term = (np.pi * peak_frequency_hz * t) ** 2
    return (1.0 - 2.0 * term) * np.exp(-term)
