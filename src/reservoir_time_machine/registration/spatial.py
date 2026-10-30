"""Exact trace-key spatial correspondence; no geometric interpolation is inferred."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np

from ..data.validation import ValidationFinding, ValidationReport
from ..models import SeismicVolume


def reindex_volume_to_axes(volume: SeismicVolume, target_inline_ids: Sequence[int],
                           target_crossline_ids: Sequence[int]) -> tuple[SeismicVolume, ValidationReport]:
    """Reorder traces by explicit bin IDs and fill absent target bins with NaN.

    This is exact trace-key correspondence only. It does not transform coordinates,
    interpolate traces, or establish that two differently numbered grids overlap.
    """
    source_inline = volume.metadata.source_metadata.get("inline_ids_selected")
    source_crossline = volume.metadata.source_metadata.get("crossline_ids_selected")
    inputs = (volume.metadata.survey_id,)
    if source_inline is None or source_crossline is None:
        return volume, ValidationReport((ValidationFinding(
            "trace_axis_ids", "fail", "Source inline/crossline identifiers are required for reindexing.", inputs, {},
            "Load header-derived identifiers or provide verified axis IDs; do not pair ordinal traces.", True),))
    source_inline, source_crossline = tuple(map(int, source_inline)), tuple(map(int, source_crossline))
    target_inline, target_crossline = tuple(map(int, target_inline_ids)), tuple(map(int, target_crossline_ids))
    if (len(source_inline) != volume.data.shape[0] or len(source_crossline) != volume.data.shape[1]
            or len(set(source_inline)) != len(source_inline) or len(set(source_crossline)) != len(source_crossline)
            or not target_inline or not target_crossline
            or len(set(target_inline)) != len(target_inline) or len(set(target_crossline)) != len(target_crossline)):
        return volume, ValidationReport((ValidationFinding(
            "trace_axis_ids", "fail", "Axis identifiers are duplicate, empty, or inconsistent with the source array shape.", inputs,
            {"source_shape": list(volume.data.shape), "source_inline_count": len(source_inline),
             "source_crossline_count": len(source_crossline)},
            "Resolve trace-key metadata before reindexing.", True),))
    dtype = np.result_type(volume.data.dtype, np.float32)
    aligned = np.full((len(target_inline), len(target_crossline), volume.data.shape[2]), np.nan, dtype=dtype)
    source_i = {value: index for index, value in enumerate(source_inline)}
    source_x = {value: index for index, value in enumerate(source_crossline)}
    missing: list[tuple[int, int]] = []
    target_missing = np.zeros((len(target_inline), len(target_crossline)), dtype=bool)
    for i, inline_id in enumerate(target_inline):
        for x, crossline_id in enumerate(target_crossline):
            if inline_id not in source_i or crossline_id not in source_x:
                missing.append((inline_id, crossline_id))
                target_missing[i, x] = True
            else:
                src_i, src_x = source_i[inline_id], source_x[crossline_id]
                source_is_missing = volume.missing_trace_mask is not None and volume.missing_trace_mask[src_i, src_x]
                if source_is_missing:
                    missing.append((inline_id, crossline_id))
                    target_missing[i, x] = True
                else:
                    aligned[i, x] = volume.data[src_i, src_x]
    findings = [ValidationFinding("trace_reindex", "warning" if missing else "pass",
        "Trace samples were mapped by explicit inline/crossline identifiers without interpolation." if not missing
        else "Trace samples were reindexed, but one or more target bins have no valid source trace.", inputs,
        {"source_shape": list(volume.data.shape), "target_shape": list(aligned.shape),
         "source_order_changed": source_inline != target_inline or source_crossline != target_crossline,
         "missing_target_bin_count": len(missing), "missing_target_bin_examples": [list(item) for item in missing[:10]]},
        "" if not missing else "Review missing target bins and propagate the validity mask; do not treat filled NaNs as observations.",
        bool(missing))]
    new_metadata = replace(volume.metadata, dimensions=aligned.shape,
        source_metadata={**volume.metadata.source_metadata, "inline_ids_selected": target_inline,
                         "crossline_ids_selected": target_crossline,
                         "spatial_reindexing": "exact_bin_key_mapping_no_interpolation"})
    return replace(volume, data=aligned, metadata=new_metadata, missing_trace_mask=target_missing), ValidationReport(tuple(findings))
