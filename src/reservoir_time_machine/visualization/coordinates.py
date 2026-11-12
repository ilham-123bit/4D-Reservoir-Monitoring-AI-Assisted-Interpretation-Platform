"""Map source-array indices to explicit SEG-Y header identifiers."""

from __future__ import annotations

import numpy as np


def header_id(index: int, identifiers, *, expected_length: int | None = None) -> int | None:
    """Return the header ID at a verified array-axis position, otherwise None."""
    axis = np.asarray(identifiers)
    if (axis.ndim != 1 or axis.dtype.kind not in "iu" or not len(axis)
            or len(np.unique(axis)) != len(axis)
            or (expected_length is not None and len(axis) != expected_length)):
        return None
    if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)):
        return None
    if not 0 <= int(index) < len(axis):
        return None
    return int(axis[int(index)])


def finding_header_location(representative, inline_ids, crossline_ids, *, grid_shape=None):
    if len(representative) != 3:
        raise ValueError("finding representative must be (inline index, crossline index, sample index)")
    inline_index, crossline_index, sample_index = representative
    return {
        "inline_index": int(inline_index),
        "inline_header_id": header_id(inline_index, inline_ids,
                                      expected_length=grid_shape[0] if grid_shape else None),
        "crossline_index": int(crossline_index),
        "crossline_header_id": header_id(crossline_index, crossline_ids,
                                          expected_length=grid_shape[1] if grid_shape else None),
        "sample_index": int(sample_index),
        "sample_physical_value": None,
        "sample_physical_domain": None,
    }
