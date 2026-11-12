"""Bounded, header-only SEG-Y coordinate diagnostics.

This module evaluates whether CDP headers support an affine map from source
inline/crossline IDs to stored coordinates. It cannot assign a CRS, datum, or
physical Baseline/Monitor registration from header values alone.
"""
from __future__ import annotations

import numpy as np


def fit_inline_crossline_affine(inline, crossline, x, y, *, holdout_fraction=0.2):
    """Fit a deterministic coordinate transform and score held-out IL bands.

    Returns only numerical geometry diagnostics. Coordinate units and CRS are
    reported as unknown unless separately established by authoritative metadata.
    The held-out split is by inline bands, not random neighboring traces.
    """
    arrays = [np.asarray(v, dtype=np.float64).reshape(-1) for v in (inline, crossline, x, y)]
    if len({a.size for a in arrays}) != 1 or arrays[0].size < 12:
        raise ValueError("at least 12 equal-length coordinate samples are required")
    il, xl, xx, yy = arrays
    finite = np.isfinite(il) & np.isfinite(xl) & np.isfinite(xx) & np.isfinite(yy)
    il, xl, xx, yy = il[finite], xl[finite], xx[finite], yy[finite]
    if il.size < 12 or np.unique(il).size < 3 or np.unique(xl).size < 3:
        raise ValueError("coordinate sample does not span enough inline/crossline geometry")
    bands = np.unique(il)
    # Hold out every fifth distinct inline band, keeping spatially coherent lines
    # together. Fall back to the last band if the sample set is very small.
    held_bands = bands[::max(2, int(round(1 / holdout_fraction)))]
    if held_bands.size == 0 or held_bands.size >= bands.size:
        held_bands = bands[-1:]
    test = np.isin(il, held_bands)
    train = ~test
    design = np.column_stack((il, xl, np.ones(il.size)))
    if train.sum() < 6 or test.sum() < 3:
        raise ValueError("spatial holdout leaves too few training or validation samples")
    coefficients, _, rank, singular = np.linalg.lstsq(design[train], np.column_stack((xx[train], yy[train])), rcond=None)
    if rank < 3:
        raise ValueError("sampled SEG-Y geometry is rank deficient for a 2D affine fit")
    residual = np.column_stack((xx[test], yy[test])) - design[test] @ coefficients
    distance = np.linalg.norm(residual, axis=1)
    axis = np.column_stack((xx[test], yy[test]))
    ss_res = np.sum(residual ** 2, axis=0)
    ss_tot = np.sum((axis - axis.mean(axis=0)) ** 2, axis=0)
    r2 = [float(1 - ss_res[i] / ss_tot[i]) if ss_tot[i] else None for i in range(2)]
    return {
        "sample_count": int(il.size), "train_count": int(train.sum()), "spatial_holdout_count": int(test.sum()),
        "held_out_inline_bands": [float(v) for v in held_bands],
        "coefficient_rows_inline_crossline_intercept_columns_x_y": coefficients.tolist(),
        "design_rank": int(rank), "design_condition_number": float(singular[0] / singular[-1]),
        "holdout_r2_x_y": r2,
        "holdout_error_coordinate_units_unknown": {
            "median": float(np.median(distance)), "p95": float(np.percentile(distance, 95)),
            "maximum": float(np.max(distance)),
            "rmse_x": float(np.sqrt(np.mean(residual[:, 0] ** 2))),
            "rmse_y": float(np.sqrt(np.mean(residual[:, 1] ** 2))),
        },
        "interpretation_limit": "A good affine fit describes sampled header geometry only; it does not identify CRS, physical units, well tie, or 4D registration.",
    }


def sample_segy_header_geometry(path, *, sample_count=2048, endian="big"):
    """Read a deterministic spread of trace headers without reading amplitudes."""
    if isinstance(sample_count, bool) or int(sample_count) < 12:
        raise ValueError("sample_count must be at least 12")
    try:
        import segyio
    except ImportError as exc:
        raise RuntimeError("Header geometry sampling requires segyio") from exc
    with segyio.open(str(path), "r", ignore_geometry=True, endian=endian) as segy:
        count = int(segy.tracecount)
        indices = np.unique(np.linspace(0, count - 1, min(count, int(sample_count)), dtype=np.int64))
        fields = ("INLINE_3D", "CROSSLINE_3D", "CDP_X", "CDP_Y", "SourceX", "SourceY", "SourceGroupScalar",
                  "CoordinateUnits", "DelayRecordingTime", "TRACE_SAMPLE_COUNT", "TRACE_SAMPLE_INTERVAL")
        values = {name: np.empty(indices.size, dtype=np.int64) for name in fields}
        for out_index, trace_index in enumerate(indices):
            header = segy.header[int(trace_index)]
            for name in fields:
                values[name][out_index] = int(header[getattr(segyio.TraceField, name)])
        scalar = values["SourceGroupScalar"]
        scale = np.where(scalar > 0, scalar, np.where(scalar < 0, 1.0 / np.abs(scalar), 1.0))
        x = values["CDP_X"].astype(np.float64) * scale
        y = values["CDP_Y"].astype(np.float64) * scale
        nonzero_xy = (values["CDP_X"] != 0) | (values["CDP_Y"] != 0)
        usable = nonzero_xy & (values["INLINE_3D"] != 0) & (values["CROSSLINE_3D"] != 0)
        report = {
            "path_basename": str(path).rsplit("/", 1)[-1], "trace_count": count,
            "sampled_trace_count": int(indices.size),
            "coordinate_unit_code_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(values["CoordinateUnits"], return_counts=True))},
            "coordinate_scalar_code_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(scalar, return_counts=True))},
            "nonzero_cdp_xy_fraction_in_sample": float(nonzero_xy.mean()),
            "zero_source_xy_fraction_in_sample": float(np.mean((values["SourceX"] == 0) & (values["SourceY"] == 0))),
            "trace_sample_interval_us_values": sorted(int(v) for v in np.unique(values["TRACE_SAMPLE_INTERVAL"])),
            "trace_sample_count_values": sorted(int(v) for v in np.unique(values["TRACE_SAMPLE_COUNT"])),
            "recording_delay_ms_values": sorted(int(v) for v in np.unique(values["DelayRecordingTime"])),
            "scaled_cdp_coordinate_extents_units_unverified": {
                "x_min": float(x[usable].min()) if usable.any() else None,
                "x_max": float(x[usable].max()) if usable.any() else None,
                "y_min": float(y[usable].min()) if usable.any() else None,
                "y_max": float(y[usable].max()) if usable.any() else None,
            },
        }
        if usable.sum() >= 12:
            report["affine_geometry"] = fit_inline_crossline_affine(
                values["INLINE_3D"][usable], values["CROSSLINE_3D"][usable], x[usable], y[usable])
        else:
            report["affine_geometry"] = {"status": "unavailable", "reason": "Fewer than 12 sampled traces have nonzero CDP X/Y and valid IL/XL."}
        report["crs_status"] = "unverified"
        report["sample_domain_status"] = "unverified_as_TWT"
        return report


def conditionally_map_well_coordinates(well_matrix, geometry_report, *, inline_ids, crossline_ids):
    """Project workbook X/Y through a sampled affine header fit as a hypothesis.

    This is deliberately named conditional: numerical agreement does not
    establish a CRS, units, datum, well trajectory, or seismic tie.
    """
    affine = geometry_report.get("affine_geometry", {})
    coefficients = np.asarray(
        affine.get("coefficient_rows_inline_crossline_intercept_columns_x_y", []),
        dtype=np.float64,
    )
    if coefficients.shape != (3, 2) or not np.isfinite(coefficients).all():
        return {"status": "unavailable", "reason": "A finite rank-3 affine SEG-Y header fit is required.", "wells": []}
    transform = coefficients[:2, :].T
    if abs(float(np.linalg.det(transform))) < 1e-12:
        return {"status": "unavailable", "reason": "The sampled affine map is not invertible.", "wells": []}
    output = []
    intercept = coefficients[2, :]
    il_lo, il_hi = float(np.min(inline_ids)), float(np.max(inline_ids))
    xl_lo, xl_hi = float(np.min(crossline_ids)), float(np.max(crossline_ids))
    for item in well_matrix:
        xy = item.get("workbook_xy")
        row = {"well_id": item.get("well_id"), "workbook_xy_source_values": xy}
        try:
            x, y = (float(value) for value in xy)
            if not np.isfinite([x, y]).all():
                raise ValueError
        except (TypeError, ValueError):
            row.update({"status": "no_numeric_workbook_xy", "conditional_inline_id": None, "conditional_crossline_id": None})
        else:
            il, xl = np.linalg.solve(transform, np.asarray([x, y]) - intercept)
            inside = il_lo <= il <= il_hi and xl_lo <= xl <= xl_hi
            row.update({
                "status": "conditional_inside_header_axis_extent" if inside else "conditional_outside_header_axis_extent",
                "conditional_inline_id": float(il), "conditional_crossline_id": float(xl),
                "inside_header_axis_extent": bool(inside),
            })
        row["interpretation_limit"] = "Conditional numeric mapping only; workbook/SEG-Y CRS, units, datum, well trajectory and seismic tie are not verified."
        output.append(row)
    return {"status": "conditional_only", "wells": output,
            "input_geometry_holdout_r2_x_y": affine.get("holdout_r2_x_y"),
            "input_geometry_holdout_error": affine.get("holdout_error_coordinate_units_unknown"),
            "interpretation_limit": "Do not use these projected locations as verified well-to-seismic ties or for reservoir association."}
