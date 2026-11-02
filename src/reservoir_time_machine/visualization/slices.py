"""Display-only downsampling and orthogonal slice preparation."""

import numpy as np


def display_slice(values: np.ndarray, max_axis: int = 600) -> np.ndarray:
    """Return a display-only 2D view reduced to at most max_axis per axis."""
    data = np.asarray(values)
    if data.ndim != 2 or max_axis < 1:
        raise ValueError("expected a 2D array and positive display limit")
    row_stride = max(1, int(np.ceil(data.shape[0] / max_axis)))
    col_stride = max(1, int(np.ceil(data.shape[1] / max_axis)))
    return data[::row_stride, ::col_stride]


def display_mask(mask: np.ndarray, max_axis: int = 600) -> np.ndarray:
    """Conservatively reduce a boolean mask: a display cell is true if any source cell is true."""
    data = np.asarray(mask, dtype=bool)
    if data.ndim != 2 or max_axis < 1:
        raise ValueError("expected a 2D mask and positive display limit")
    row_stride = max(1, int(np.ceil(data.shape[0] / max_axis)))
    col_stride = max(1, int(np.ceil(data.shape[1] / max_axis)))
    out_rows = (data.shape[0] + row_stride - 1) // row_stride
    out_cols = (data.shape[1] + col_stride - 1) // col_stride
    padded = np.zeros((out_rows * row_stride, out_cols * col_stride), dtype=bool)
    padded[:data.shape[0], :data.shape[1]] = data
    return padded.reshape(out_rows, row_stride, out_cols, col_stride).any(axis=(1, 3))


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
