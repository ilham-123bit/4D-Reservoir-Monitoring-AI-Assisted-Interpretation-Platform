"""Display-only downsampling and orthogonal slice preparation."""

import numpy as np


def display_volume(volume: np.ndarray, max_axis: int = 48) -> np.ndarray:
    data = np.asarray(volume)
    if data.ndim != 3 or max_axis < 1:
        raise ValueError("expected 3D volume and positive display limit")
    strides = tuple(max(1, int(np.ceil(axis / max_axis))) for axis in data.shape)
    return data[::strides[0], ::strides[1], ::strides[2]].copy()


def slice_triplet(volume: np.ndarray, inline: int, crossline: int, sample: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if volume.ndim != 3:
        raise ValueError("volume must be 3D")
    ni, nc, ns = volume.shape
    if not (0 <= inline < ni and 0 <= crossline < nc and 0 <= sample < ns):
        raise IndexError("slice index is outside the volume")
    return volume[inline, :, :].copy(), volume[:, crossline, :].copy(), volume[:, :, sample].copy()
