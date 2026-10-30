"""Explicit coordinate-aware affine mapping between seismic survey grids."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

from ..data.validation import ValidationFinding, ValidationReport


@dataclass(frozen=True)
class AffineGridTransform:
    """Map target physical coordinates (x,y) to continuous source (inline,crossline) indices.

    The 2x3 matrix is explicitly supplied from a documented survey transform. It is
    never estimated from seismic amplitudes. The matrix must include any CRS conversion
    needed to express target coordinates in the source grid's physical reference.
    """
    target_xy_to_source_index: tuple[tuple[float, float, float], tuple[float, float, float]]
    source_coordinate_reference: str
    target_coordinate_reference: str
    transform_reference: str

    def __post_init__(self) -> None:
        matrix = np.asarray(self.target_xy_to_source_index, dtype=np.float64)
        if matrix.shape != (2, 3) or not np.isfinite(matrix).all():
            raise ValueError("affine transform must be a finite 2x3 matrix")
        if abs(float(np.linalg.det(matrix[:, :2]))) < 1e-12:
            raise ValueError("affine transform spatial matrix must be invertible")
        if not self.source_coordinate_reference.strip() or not self.target_coordinate_reference.strip():
            raise ValueError("source and target coordinate references are required")
        if not self.transform_reference.strip():
            raise ValueError("a durable transform provenance reference is required")


def resample_volume_to_target_grid(source: np.ndarray, target_xy: np.ndarray, transform: AffineGridTransform,
                                   output_path: str | Path, *, sample_chunk: int = 16,
                                   interpolation_order: int = 1,
                                   minimum_valid_support: float = 0.999) -> tuple[np.memmap, ValidationReport]:
    """Resample spatial traces onto a target grid, preserving the sample axis.

    Only nearest-neighbor (0) and linear (1) interpolation are accepted. NaNs are
    interpolated using a separate validity weight and outputs below the required
    support are NaN. This is a derived volume; no source values are changed. The
    resulting amplitude/waveform interpolation must still be QC'd before 4D use.
    """
    values = np.asanyarray(source)
    coordinates = np.asarray(target_xy, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("source volume must be 3D (inline,crossline,sample)")
    if coordinates.ndim != 3 or coordinates.shape[-1] != 2 or not np.isfinite(coordinates).all():
        raise ValueError("target_xy must be finite (target_inline,target_crossline,2) coordinates")
    if sample_chunk < 1 or interpolation_order not in (0, 1):
        raise ValueError("sample_chunk must be positive and interpolation order must be 0 or 1")
    if not 0 < minimum_valid_support <= 1:
        raise ValueError("minimum_valid_support must be in (0,1]")
    matrix = np.asarray(transform.target_xy_to_source_index, dtype=np.float64)
    homogeneous = np.concatenate((coordinates, np.ones((*coordinates.shape[:2], 1))), axis=2)
    mapped = np.einsum("ab,ijb->aij", matrix, homogeneous)
    target_shape = (*coordinates.shape[:2], values.shape[2])
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite resampled volume: {destination}")
    output = np.lib.format.open_memmap(destination, mode="w+", dtype=np.result_type(values.dtype, np.float32),
                                       shape=target_shape)
    for start in range(0, values.shape[2], sample_chunk):
        stop = min(values.shape[2], start + sample_chunk)
        for sample in range(start, stop):
            plane = np.asarray(values[:, :, sample], dtype=np.float32)
            finite = np.isfinite(plane)
            numerator = map_coordinates(np.where(finite, plane, 0.0), mapped,
                                        order=interpolation_order, mode="constant", cval=0.0,
                                        prefilter=False)
            support = map_coordinates(finite.astype(np.float32), mapped,
                                      order=interpolation_order, mode="constant", cval=0.0,
                                      prefilter=False)
            valid = support >= minimum_valid_support
            output[:, :, sample] = np.divide(numerator, support, out=np.full(support.shape, np.nan, np.float32),
                                             where=valid & (support > 0))
        output.flush()
    finding = ValidationFinding("affine_spatial_resampling", "warning",
        "Source samples were spatially resampled using the explicitly supplied affine mapping; this is a correction, not an observed 4D change.",
        ("source_volume", "target_grid"),
        {"source_shape": list(values.shape), "target_shape": list(target_shape),
         "source_coordinate_reference": transform.source_coordinate_reference,
         "target_coordinate_reference": transform.target_coordinate_reference,
         "transform_reference": transform.transform_reference,
         "interpolation_order": interpolation_order, "minimum_valid_support": minimum_valid_support,
         "sample_chunk": sample_chunk},
        "Inspect geometry, interpolation footprint, amplitude/waveform fidelity, and residual misties before calculating 4D attributes.")
    return output, ValidationReport((finding,))
