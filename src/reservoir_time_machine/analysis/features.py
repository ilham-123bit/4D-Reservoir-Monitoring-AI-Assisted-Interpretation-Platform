"""Explicit paired attributes, support and feature order for bounded sample tiles."""
from dataclasses import dataclass
import hashlib
import json
import numpy as np
from ..attributes.difference import calculate_sample_attributes

FEATURE_NAMES = ('normalized_difference', 'nrms_fraction', 'log_rms_ratio', 'relative_absolute_difference')
VERSION = 'preliminary-features-2'


@dataclass(frozen=True)
class FeatureTile:
    attributes: object
    features: np.ndarray
    valid: np.ndarray
    blocks: np.ndarray
    origin: tuple[int, int]
    sample: int
    window: int
    available_window: int
    content_sha256: str


def extract_features(baseline, monitor, missing, *, sample, bounds, window=9, min_support_fraction=.8):
    if (baseline.ndim != 3 or baseline.shape != monitor.shape or np.shape(missing) != baseline.shape[:2]
            or not 0 < min_support_fraction <= 1):
        raise ValueError('Matching paired volumes, trace mask and positive support fraction required')
    i0, i1, x0, x1 = bounds
    if not (0 <= i0 < i1 <= baseline.shape[0] and 0 <= x0 < x1 <= baseline.shape[1]):
        raise ValueError('Tile bounds outside input')
    if (i1-i0)*(x1-x0) > 16384:
        raise ValueError('Preliminary tile budget is at most 16384 grid cells')
    attrs = calculate_sample_attributes(baseline[i0:i1, x0:x1], monitor[i0:i1, x0:x1], sample,
                                        window_samples=window)
    lo, hi = max(0, sample-window//2), min(baseline.shape[2], sample+(window+1)//2)
    available = hi-lo
    energy = attrs.rms_baseline+attrs.rms_monitor
    with np.errstate(invalid='ignore', divide='ignore'):
        values = np.stack((attrs.normalized_difference, attrs.nrms/200,
                           np.log((attrs.rms_monitor+1e-8)/(attrs.rms_baseline+1e-8)),
                           attrs.absolute_difference/(energy+1e-8)), axis=-1)
    valid = (~np.asarray(missing[i0:i1, x0:x1], bool) & np.isfinite(values).all(axis=-1)
             & (attrs.paired_sample_count >= int(np.ceil(available*min_support_fraction)))
             & (energy > 1e-8))
    values[~valid] = np.nan
    # Global 8x8 blocks: train=0/1, threshold reference=2, inspection=3.
    ii, xx = np.meshgrid(np.arange(i0, i1), np.arange(x0, x1), indexing='ij')
    blocks = ((ii//8+2*(xx//8)) % 4).astype(np.uint8)
    h = hashlib.sha256(json.dumps([VERSION, FEATURE_NAMES, bounds, sample, window,
                                   min_support_fraction]).encode())
    # Raw bounded samples are hashed too: relative attributes alone could hide
    # a common amplitude rescaling while reported raw measurements changed.
    for a in (baseline[i0:i1,x0:x1,lo:hi], monitor[i0:i1,x0:x1,lo:hi],
              np.asarray(missing[i0:i1,x0:x1]), values, valid, attrs.paired_sample_count):
        h.update(np.ascontiguousarray(a).tobytes())
    return FeatureTile(attrs, values, valid, blocks, (i0, x0), sample, window, available, h.hexdigest())


def robust_scaling(reference):
    if reference.ndim != 2 or not np.isfinite(reference).all():
        raise ValueError('Scaling reference must contain finite rows')
    center = np.median(reference, axis=0)
    mad = 1.4826*np.median(np.abs(reference-center), axis=0)
    std = np.std(reference, axis=0)
    scale = np.where(mad > 1e-8, mad, np.where(std > 1e-8, std, 1.0))
    return center, scale, std > 1e-8
