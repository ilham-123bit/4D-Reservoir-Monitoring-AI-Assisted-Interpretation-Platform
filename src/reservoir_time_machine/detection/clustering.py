"""Connected-component extraction for 3D binary anomaly screens."""

import numpy as np
from scipy.ndimage import find_objects, generate_binary_structure, label
from pathlib import Path
from itertools import product
from typing import Iterator

from ..models import DetectedZone
from ..processing.chunks import Chunk, iter_chunks


def connected_zones(mask: np.ndarray, score: np.ndarray | None = None, *, min_size: int = 8,
                    connectivity: int = 1) -> tuple[np.ndarray, list[DetectedZone]]:
    flags = np.asarray(mask, dtype=bool)
    if flags.ndim not in (2, 3) or min_size < 1 or connectivity not in (1, 2, 3):
        raise ValueError("expected 2D/3D mask, positive min_size, connectivity 1..3")
    structure = generate_binary_structure(flags.ndim, min(connectivity, flags.ndim))
    labels, count = label(flags, structure=structure)
    values = np.asarray(score, float) if score is not None else flags.astype(float)
    if values.shape != flags.shape:
        raise ValueError("score and mask shapes must match")
    zones = []
    filtered = np.zeros(flags.shape, dtype=np.int32)
    next_label = 1
    for original in range(1, count + 1):
        points = np.argwhere(labels == original)
        if len(points) < min_size:
            continue
        filtered[labels == original] = next_label
        strength = values[labels == original]
        lo, hi = points.min(axis=0), points.max(axis=0)
        zones.append(DetectedZone(next_label, len(points), tuple((int(a), int(b)) for a, b in zip(lo, hi)),
                                  float(np.nanmean(strength)), float(np.nanmax(strength))))
        next_label += 1
    return filtered, zones


class _DisjointSet:
    """Union-find table for provisional labels created independently per chunk."""

    def __init__(self) -> None:
        self.parent = [0]

    def add(self) -> int:
        value = len(self.parent)
        self.parent.append(value)
        return value

    def find(self, value: int) -> int:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            value, self.parent[value] = self.parent[value], root
        return root

    def union(self, first: int, second: int) -> None:
        a, b = self.find(first), self.find(second)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def _positive_offsets(connectivity: int) -> tuple[tuple[int, int, int], ...]:
    offsets = []
    for offset in product((-1, 0, 1), repeat=3):
        manhattan = sum(abs(value) for value in offset)
        if not manhattan or manhattan > connectivity:
            continue
        first_nonzero = next(value for value in offset if value)
        if first_nonzero > 0:
            offsets.append(offset)
    return tuple(offsets)


def _offset_slices(core: tuple[slice, ...], shape: tuple[int, ...], offset: tuple[int, ...]):
    source, target = [], []
    for part, size, delta in zip(core, shape, offset):
        start = max(part.start, 0, -delta)
        stop = min(part.stop, size, size - delta)
        if start >= stop:
            return None
        source.append(slice(start, stop))
        target.append(slice(start + delta, stop + delta))
    return tuple(source), tuple(target)


def connected_zones_chunked(mask: np.ndarray, output_directory: str | Path, *,
                            chunk_shape: tuple[int, int, int] = (32, 32, 64),
                            score: np.ndarray | None = None, min_size: int = 8,
                            connectivity: int = 1) -> tuple[np.memmap, list[DetectedZone]]:
    """Label 3D components out-of-core and reconcile connectivity across chunk seams.

    Local labels are provisional. Face/edge/corner adjacency according to the selected
    connectivity is reconciled with union-find, then labels are rewritten to globally
    consistent compact IDs. Label output is a new NPY memmap; input arrays are unchanged.
    """
    flags = np.asanyarray(mask)
    if flags.ndim != 3 or min_size < 1 or connectivity not in (1, 2, 3):
        raise ValueError("mask must be 3D, min_size positive, and connectivity 1..3")
    values = np.asanyarray(score) if score is not None else None
    if values is not None and values.shape != flags.shape:
        raise ValueError("score and mask shapes must match")
    # Validate chunk shape before creating outputs.
    first_chunk = next(iter_chunks(flags.shape, chunk_shape))
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    label_path = output_dir / "component_labels.npy"
    if label_path.exists():
        raise FileExistsError(f"Refusing to overwrite component labels: {label_path}")
    labels = np.lib.format.open_memmap(label_path, mode="w+", dtype=np.int64, shape=flags.shape)
    dsu = _DisjointSet()
    stats: dict[int, dict[str, object]] = {}
    chunks: Iterator[Chunk] = iter_chunks(flags.shape, chunk_shape)
    for chunk in chunks:
        local, count = label(np.asarray(flags[chunk.core], dtype=bool),
                             structure=generate_binary_structure(3, connectivity))
        if not count:
            labels[chunk.core] = 0
            continue
        counts = np.bincount(local.ravel(), minlength=count + 1)
        score_block = np.asarray(values[chunk.core], dtype=np.float64) if values is not None else None
        ids = np.arange(1, count + 1, dtype=np.int64)
        mapping = np.zeros(count + 1, dtype=np.int64)
        for local_id in ids:
            mapping[local_id] = dsu.add()
        labels[chunk.core] = mapping[local]
        slices = find_objects(local, max_label=count)
        local_flat = local.ravel()
        if score_block is None:
            score_sums = counts.astype(np.float64)
            score_counts = counts.astype(np.int64)
            score_maxima = np.where(counts > 0, 1.0, -np.inf)
        else:
            score_flat = score_block.ravel()
            finite_scores = np.isfinite(score_flat)
            score_counts = np.bincount(local_flat[finite_scores], minlength=count + 1)
            score_sums = np.bincount(local_flat[finite_scores], weights=score_flat[finite_scores], minlength=count + 1)
            score_maxima = np.full(count + 1, -np.inf, dtype=np.float64)
            np.maximum.at(score_maxima, local_flat[finite_scores], score_flat[finite_scores])
        for local_id, local_slice in enumerate(slices, start=1):
            global_id = int(mapping[local_id])
            bounds = tuple((int(core.start + region.start), int(core.start + region.stop - 1))
                           for core, region in zip(chunk.core, local_slice))
            stats[global_id] = {"size": int(counts[local_id]), "bounds": bounds,
                                "score_sum": float(score_sums[local_id]),
                                "score_count": int(score_counts[local_id]),
                                "score_max": float(score_maxima[local_id]) if np.isfinite(score_maxima[local_id]) else 0.0}
    labels.flush()

    offsets = _positive_offsets(connectivity)
    for chunk in iter_chunks(flags.shape, chunk_shape):
        for offset in offsets:
            regions = _offset_slices(chunk.core, flags.shape, offset)
            if regions is None:
                continue
            source_slice, target_slice = regions
            source_labels = np.asarray(labels[source_slice])
            target_labels = np.asarray(labels[target_slice])
            joined = (source_labels > 0) & (target_labels > 0) & (source_labels != target_labels)
            if joined.any():
                pairs = np.unique(np.stack((source_labels[joined], target_labels[joined]), axis=1), axis=0)
                for first, second in pairs:
                    dsu.union(int(first), int(second))

    aggregate: dict[int, dict[str, object]] = {}
    for component_id, component in stats.items():
        root = dsu.find(component_id)
        existing = aggregate.get(root)
        if existing is None:
            aggregate[root] = {**component}
        else:
            existing["size"] = int(existing["size"]) + int(component["size"])
            existing["bounds"] = tuple((min(a[0], b[0]), max(a[1], b[1]))
                                       for a, b in zip(existing["bounds"], component["bounds"]))
            existing["score_sum"] = float(existing["score_sum"]) + float(component["score_sum"])
            existing["score_count"] = int(existing["score_count"]) + int(component["score_count"])
            existing["score_max"] = max(float(existing["score_max"]), float(component["score_max"]))
    retained = {root: data for root, data in aggregate.items() if int(data["size"]) >= min_size}
    compact = {root: index + 1 for index, root in enumerate(sorted(retained))}
    root_lookup = np.arange(len(dsu.parent), dtype=np.int64)
    for value in range(1, len(dsu.parent)):
        root_lookup[value] = compact.get(dsu.find(value), 0)
    for chunk in iter_chunks(flags.shape, chunk_shape):
        local_labels = np.asarray(labels[chunk.core])
        labels[chunk.core] = root_lookup[local_labels]
    labels.flush()
    zones = []
    for root in sorted(retained):
        record = retained[root]
        count = int(record["score_count"])
        zones.append(DetectedZone(compact[root], int(record["size"]), record["bounds"],
                                  float(record["score_sum"]) / count if count else 0.0,
                                  float(record["score_max"])))
    return labels, zones
