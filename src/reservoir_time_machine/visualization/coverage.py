"""Trace coverage figures with verified source-header coordinates."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .slices import display_coordinates


def _integer_axis(values, expected: int, name: str) -> np.ndarray:
    axis = np.asarray(values)
    if (axis.ndim != 1 or len(axis) != expected or axis.dtype.kind not in "iu"
            or len(np.unique(axis)) != expected):
        raise ValueError(f"{name} must be a unique integer axis matching the Baseline grid")
    return axis.astype(np.int64, copy=False)


def _pooled_missing_counts(mask: np.ndarray, max_axis: int) -> np.ndarray:
    row_stride = max(1, int(np.ceil(mask.shape[0] / max_axis)))
    col_stride = max(1, int(np.ceil(mask.shape[1] / max_axis)))
    rows = (mask.shape[0] + row_stride - 1) // row_stride
    cols = (mask.shape[1] + col_stride - 1) // col_stride
    padded = np.zeros((rows * row_stride, cols * col_stride), dtype=np.uint8)
    padded[:mask.shape[0], :mask.shape[1]] = mask
    return padded.reshape(rows, row_stride, cols, col_stride).sum(axis=(1, 3), dtype=np.uint32)


def coverage_figure(missing_mask, inline_ids, crossline_ids, monitor_only_keys, *, max_axis=700):
    """Build a downsampled Baseline-grid footprint plus separate Monitor-only keys.

    Heatmap value 0 means paired Baseline-grid trace(s) are represented in the bin;
    value 1 means the bin contains at least one missing Baseline trace. Exact trace
    totals are returned separately and never inferred from display bins.
    """
    mask = np.asarray(missing_mask)
    if mask.ndim != 2 or mask.dtype.kind != "b" or not mask.size:
        raise ValueError("missing_mask must be a nonempty two-dimensional Boolean array")
    if isinstance(max_axis, bool) or not isinstance(max_axis, int) or max_axis < 1:
        raise ValueError("max_axis must be a positive integer")
    inline_axis = _integer_axis(inline_ids, mask.shape[0], "inline_ids")
    crossline_axis = _integer_axis(crossline_ids, mask.shape[1], "crossline_ids")
    keys = np.asarray(monitor_only_keys)
    if keys.ndim != 2 or keys.shape[1] != 2 or keys.dtype.kind not in "iu":
        raise ValueError("monitor_only_keys must be an integer (n, 2) array")
    if len(np.unique(keys, axis=0)) != len(keys):
        raise ValueError("monitor_only_keys must be unique")

    pooled_missing = _pooled_missing_counts(mask, max_axis)
    shown = (pooled_missing > 0).astype(np.uint8)
    x_index, y_index = display_coordinates(mask.shape, max_axis=max_axis)
    if shown.shape != (len(y_index), len(x_index)):
        raise RuntimeError("Coverage values and display coordinates are misaligned")
    row_stride = max(1, int(np.ceil(mask.shape[0] / max_axis)))
    col_stride = max(1, int(np.ceil(mask.shape[1] / max_axis)))
    represented_rows = np.minimum(row_stride, mask.shape[0] - np.arange(len(y_index)) * row_stride)
    represented_cols = np.minimum(col_stride, mask.shape[1] - np.arange(len(x_index)) * col_stride)
    represented = represented_rows[:, None] * represented_cols[None, :]
    custom = np.stack((pooled_missing, represented), axis=-1)

    missing_count = int(mask.sum(dtype=np.int64))
    total_grid_cells = int(mask.size)
    paired_count = total_grid_cells - missing_count
    inline_lookup = {int(value): index for index, value in enumerate(inline_axis)}
    crossline_lookup = {int(value): index for index, value in enumerate(crossline_axis)}
    mapped = sum(int(i) in inline_lookup and int(x) in crossline_lookup for i, x in keys)
    coincident = sum(
        int(i) in inline_lookup and int(x) in crossline_lookup
        and bool(mask[inline_lookup[int(i)], crossline_lookup[int(x)]])
        for i, x in keys
    )

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=shown,
        x=crossline_axis[x_index],
        y=inline_axis[y_index],
        customdata=custom,
        zmin=0,
        zmax=1,
        colorscale=[[0, "#17685c"], [0.499, "#17685c"], [0.5, "#e6534b"], [1, "#e6534b"]],
        colorbar={"title": "Baseline-grid bin", "tickmode": "array", "tickvals": [0, 1],
                  "ticktext": ["paired present", "contains missing"]},
        name="Baseline-grid coverage",
        hovertemplate=("Crossline header ID %{x}<br>Inline header ID %{y}<br>"
                       "Missing traces in bin: %{customdata[0]}<br>"
                       "Source cells represented: %{customdata[1]}<extra></extra>"),
    ))
    if len(keys):
        fig.add_trace(go.Scattergl(
            x=keys[:, 1], y=keys[:, 0], mode="markers",
            marker={"color": "#ffd166", "size": 5, "symbol": "triangle-up",
                    "line": {"color": "#0b1112", "width": 0.5}},
            name=f"Monitor-only keys ({len(keys):,})",
            customdata=np.arange(len(keys), dtype=np.int32),
            hovertemplate=("Monitor-only trace key<br>Crossline header ID %{x}<br>"
                           "Inline header ID %{y}<extra></extra>"),
        ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#080d0e",
        plot_bgcolor="#101719",
        font={"color": "#ffffff", "size": 15},
        xaxis={"title": "Crossline header ID", "showgrid": False,
               "range": [min(int(crossline_axis.min()), int(keys[:, 1].min())) if len(keys) else int(crossline_axis.min()),
                         max(int(crossline_axis.max()), int(keys[:, 1].max())) if len(keys) else int(crossline_axis.max())]},
        yaxis={"title": "Inline header ID", "showgrid": False,
               "range": [max(int(inline_axis.max()), int(keys[:, 0].max())) if len(keys) else int(inline_axis.max()),
                         min(int(inline_axis.min()), int(keys[:, 0].min())) if len(keys) else int(inline_axis.min())]},
        height=590,
        margin={"l": 70, "r": 40, "t": 35, "b": 65},
        legend={"orientation": "h", "y": 1.03, "x": 0},
    )
    return fig, {
        "paired_grid_trace_count": paired_count,
        "missing_baseline_grid_count": missing_count,
        "monitor_only_trace_count": int(len(keys)),
        "monitor_only_mapped_to_baseline_axes": int(mapped),
        "monitor_only_on_missing_baseline_cells": int(coincident),
        "display_bins": int(shown.size),
        "display_bins_containing_missing": int(np.count_nonzero(shown)),
        "source_grid_shape": [int(v) for v in mask.shape],
    }
