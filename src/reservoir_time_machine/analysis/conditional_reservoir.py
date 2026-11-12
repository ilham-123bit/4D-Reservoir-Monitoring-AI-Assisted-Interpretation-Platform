"""Bounded, conditional I-100 time-lapse evidence, not a reservoir or fluid classifier.

The well-to-grid coordinate frame and seismic time zero are hypotheses.  This
module never changes historical Stage 1/2B/3 products or emits a drilling
probability.  All measurements use existing, paired Baseline-grid arrays.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def conditional_scenario_key(candidate):
    """Return a stable identity for a well-marker/time-origin comparison.

    Historical evidence files used a per-well ``id`` that repeats across time
    origins.  Include the origin, source-grid location, and sample so widget
    state and comparisons cannot collide while preserving the historical ID.
    """
    well = str(candidate.get("well_id", "")).strip().casefold()
    if not well:
        raise ValueError("Conditional scenario requires a well identifier")
    location = candidate.get("array_location")
    if not isinstance(location, (list, tuple)) or len(location) != 3:
        raise ValueError("Conditional scenario requires a 3D array location")
    i, x, sample = (int(value) for value in location)
    origin = float(candidate["assumed_time_origin_ms"])
    if not np.isfinite(origin) or any(value < 0 for value in (i, x, sample)):
        raise ValueError("Conditional scenario location and origin must be valid")
    origin_token = f"m{abs(origin):g}" if origin < 0 else f"p{origin:g}"
    return (f"conditional-i100-{well}-origin-{origin_token}-"
            f"il{i}-xl{x}-s{sample}")


def load_conditional_i100_evidence(path, *, expected_data_identity=None,
                                    expected_shape=None, inline_ids=None,
                                    crossline_ids=None):
    """Load a checksummed, bounded I-100 result without touching seismic data."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("body"), dict):
        raise ValueError("Conditional I-100 evidence must contain a body envelope")
    body = document["body"]
    if body.get("schema") != "rtm-conditional-i100-evidence-1":
        raise ValueError("Unsupported conditional I-100 evidence schema")
    digest = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()
    if digest != document.get("sha256"):
        raise ValueError("Conditional I-100 evidence checksum mismatch")
    if expected_data_identity is not None and body.get("data_identity") != expected_data_identity:
        raise ValueError("Conditional I-100 evidence belongs to a different paired cache")
    shape = tuple(int(value) for value in body.get("source_shape", ()))
    if expected_shape is not None and shape != tuple(expected_shape):
        raise ValueError("Conditional I-100 evidence source shape does not match the loaded grid")
    axes = body.get("source_axes", {})
    if inline_ids is not None and axes.get("inline_ids") != [int(inline_ids[0]), int(inline_ids[-1])]:
        raise ValueError("Conditional I-100 inline axis does not match the loaded grid")
    if crossline_ids is not None and axes.get("crossline_ids") != [int(crossline_ids[0]), int(crossline_ids[-1])]:
        raise ValueError("Conditional I-100 crossline axis does not match the loaded grid")
    records, seen = [], set()
    for item in body.get("measurements", []):
        candidate, measured = item.get("candidate"), item.get("measurement")
        if not isinstance(candidate, dict) or not isinstance(measured, dict):
            raise ValueError("Conditional I-100 scenario is missing its candidate or measurements")
        key = conditional_scenario_key(candidate)
        if key in seen:
            raise ValueError("Duplicate conditional I-100 scenario identity")
        seen.add(key)
        location = tuple(int(value) for value in candidate["array_location"])
        if len(shape) != 3 or any(v < 0 or v >= shape[d] for d, v in enumerate(location)):
            raise ValueError("Conditional I-100 location is outside its recorded source grid")
        if tuple(measured.get("array_location", ())) != location:
            raise ValueError("Conditional I-100 measurement location disagrees with its candidate")
        observed = measured.get("observed", {})
        support = observed.get("support_fraction")
        valid_pairs = observed.get("valid_pairs")
        if (not isinstance(support, (int, float)) or not 0 <= support <= 1 or
                not isinstance(valid_pairs, int) or valid_pairs < 0):
            raise ValueError("Conditional I-100 paired support metadata is invalid")
        records.append({"scenario_key": key, "candidate": candidate, "measurement": measured})
    if body.get("status") == "bounded_conditional_scientific_test_completed" and not records:
        raise ValueError("Completed conditional I-100 evidence has no scenarios")
    return {"body": bo