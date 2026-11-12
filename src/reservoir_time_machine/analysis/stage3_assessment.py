"""Post-process Stage 2 findings without mutating their published ranking."""

from __future__ import annotations

import math


def normalized_support_fraction(valid_support: float, required_support: float) -> float:
    """Return a bounded support fraction for a Stage 3 assessment.

    Stage 2 records may contain ratios above one when observed support exceeds
    the configured minimum. Such excess is not extra confidence, so Stage 3
    caps the fraction while leaving the archived Stage 2 record untouched.
    """
    valid = float(valid_support)
    required = float(required_support)
    if not math.isfinite(valid) or not math.isfinite(required) or required <= 0 or valid < 0:
        raise ValueError("support values must be finite, nonnegative, and required > 0")
    return min(1.0, valid / required)


def capped_stage3_priority(finding: dict) -> float:
    """Recalculate the existing weighted index with support capped at 1.

    This is a parallel Stage 3 value, not a replacement for Stage 2's score.
    It does not add geological evidence or make the index a probability.
    """
    parts = finding["priority_components"]
    support = min(1.0, max(0.0, float(parts["support_fraction"])))
    base = (
        0.35 * float(parts["change_strength"])
        + 0.25 * float(parts["component_extent"])
        + 0.25 * float(parts["detector_vote_fraction"])
        + 0.15 * support
    )
    factors = (
        float(finding.get("survey_edge_factor", 1.0)),
        float(finding.get("unoccupied_grid_factor", 1.0)),
        float(finding.get("monitor_only_coverage_factor", 1.0)),
        float(finding.get("edge_gradient_factor", 0.75)),
    )
    if any(not math.isfinite(factor) or not 0 <= factor <= 1 for factor in factors):
        raise ValueError("Stage 3 QC factors must be finite values in [0,1]")
    return math.prod(factors) * base
