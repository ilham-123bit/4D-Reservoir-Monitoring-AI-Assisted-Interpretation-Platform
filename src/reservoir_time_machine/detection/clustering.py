"""Connected-component extraction for 3D binary anomaly screens."""

import numpy as np
from scipy.ndimage import find_objects, generate_binary_structure, label
from pathlib import Path
from itertools import product
import tempfile

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
    if count == 0:
        return np.zeros(flags.shape, dtype=np.int32), []
    # Aggregate once over the grid. The former per-component full-grid argwhere
    # scan scaled with component count and became very slow for noisy screens.
    component = labels.ravel()
    flat_values = values.ravel()
    sizes = np.bincount(component, minlength=count + 1)
    finite = np.isfinite(flat_values)
    score_counts = np.bincount(component[finite], minlength=count + 1)
    score_sums = np.bincount(component[finite], weights=flat_values[finite], minlength=count + 1)
    maxima = np.full(count + 1, -np.inf, dtype=float)
    np.maximum.at(maxima, component[finite], flat_values[finite])
    bounds = find_objects(labels, max_label=count)
    keep = (sizes >= min_size)
    keep[0] = False
    lookup = np.zeros(count + 1, dtype=np.int32)
    kept_ids = np.flatnonzero(keep)
    lookup[kept_ids] = np.arange(1, len(kept_ids) + 1, dtype=np.int32)
    filtered = lookup[labels]
    zones = []
    for new_id, original in enumerate(kept_ids, start=1):
        region = bounds[int(original) - 1]
        region_bounds = tuple((int(part.start), int(part.stop - 1)) for part in region)
        n_scores = int(score_counts[original])
        mean_score = float(score_sums[original] / n_scores) if n_scores else float("nan")
        max_score = float(maxima[original]) if n_scores else float("nan")
        zones.append(DetectedZone(new_id, int(sizes[original]), region_bounds, mean_score, max_score))
    return filtered, zones


_ZONE_DTYPE = np.dtype([
    ("label", "<u8"), ("voxel_count", "<u8"), ("bounds", "<i4", (3, 2)),
    ("mean_strength", "<f8"), ("max_strength", "<f8"),
])


class _DiskBackedZones:
    """Lazy sequence of component records stored in a compact NPY metadata file."""

    def __init__(self, path: Path, count: int):
        self.path = Path(path)
        self.count = int(count)

    def __len__(self):
        return self.count

    def __iter__(self):
        if self.count == 0:
            return
        records = np.load(self.path, mmap_mode="r", allow_pickle=False)
        for row in records:
            yield DetectedZone(int(row["label"]), int(row["voxel_count"]),
                tuple(tuple(map(int, pair)) for pair in row["bounds"]),
                float(row["mean_strength"]), float(row["max_strength"]))

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(self.count)
            return [self[item] for item in range(start, stop, step)]
        index = int(index)
        if index < 0:
            index += self.count
        if not 0 <= index < self.count:
            raise IndexError(index)
        records = np.load(self.path, mmap_mode="r", allow_pickle=False)
        row = records[index]
        return DetectedZone(int(row["label"]), int(row["voxel_count"]),
            tuple(tuple(map(int, pair)) for pair in row["bounds"]),
            float(row["mean_strength"]), float(row["max_strength"]))

    def __eq__(self, other):
        if isinstance(other, (list, tuple)):
            return list(self) == list(other)
        return NotImplemented


def _find_many(parent: np.memmap, values: np.ndarray) -> np.ndarray:
    """Path-compress a bounded vector of provisional IDs in disk-backed storage."""
    roots = np.asarray(values, dtype=np.uint32).copy()
    while roots.size:
        parents = np.asarray(parent[roots], dtype=np.uint32)
        grandparents = np.asarray(parent[parents], dtype=np.uint32)
        moving = grandparents != roots
        if not moving.any():
            return roots
        parent[roots[moving]] = grandparents[moving]
        roots = grandparents
    return roots


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
                            connectivity: int = 1,
                            max_components: int | None = None,
                            component_summary: dict | None = None,
                            component_capacity: int | None = None) -> tuple[np.memmap, _DiskBackedZones]:
    """Group 3D candidates using chunk labels and disk-backed union/stat tables.

    Small local components are omitted from provisional labels only when they cannot
    reach a chunk seam. Seam-connected fragments are retained until global unions
    finish; ``min_size`` is applied only to globally resolved components. The exact
    input candidate mask is never modified. A small optional ``max_components`` is
    retained for callers/tests that explicitly request a guard; field-scale callers
    should use the bounded disk-backed tables without that arbitrary ceiling.
    """
    flags = np.asanyarray(mask)
    if (flags.ndim != 3 or min_size < 1 or connectivity not in (1, 2, 3)
            or (max_components is not None and
                (isinstance(max_components, bool) or max_components < 1))):
        raise ValueError("mask must be 3D, min_size positive, and connectivity 1..3")
    values = np.asanyarray(score) if score is not None else None
    if values is not None and values.shape != flags.shape:
        raise ValueError("score and mask shapes must match")
    first_chunk = next(iter_chunks(flags.shape, chunk_shape))
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    label_path = output_dir / "component_labels.npy"
    metadata_path = output_dir / "component_metadata.npy"
    if label_path.exists() or metadata_path.exists():
        raise FileExistsError(f"Refusing to overwrite component outputs in {output_dir}")
    capacity = int(component_capacity) if component_capacity is not None else int(np.count_nonzero(flags))
    if capacity < 0 or capacity > np.iinfo(np.uint32).max:
        raise ValueError("candidate component capacity must fit uint32 labels")
    # The number of provisional labels cannot exceed the number of candidate voxels.
    # These work tables are file-backed, so memory use is bounded by chunk buffers.
    table_capacity = max(1, capacity)
    label_dtype = (np.uint32 if int(np.prod(flags.shape, dtype=np.uint64)) <= np.iinfo(np.uint32).max
                   else np.uint64)
    labels = np.lib.format.open_memmap(label_path, mode="w+", dtype=label_dtype, shape=flags.shape)
    summary = {"input_candidate_voxels": 0, "input_provisional_components": 0,
               "interior_subminimum_components": 0, "interior_subminimum_voxels": 0,
               "retained_local_large_components": 0,
               "retained_seam_subminimum_fragments": 0,
               "postmerge_subminimum_components": 0, "postmerge_subminimum_voxels": 0,
               "subminimum_size_histogram": {}}
    work_temp = tempfile.TemporaryDirectory(prefix=".cc-work-", dir=output_dir)
    work = Path(work_temp.name)
    # uint32 is sufficient because each provisional component contains at least
    # one candidate voxel and capacity is bounded by candidate_voxels.
    parent = np.lib.format.open_memmap(work / "parent.npy", mode="w+",
        dtype=np.uint32, shape=(table_capacity + 1,))
    stats_dtype = np.dtype([("size", "<u4"), ("minimum", "<u4", (3,)),
        ("maximum", "<u4", (3,)), ("score_sum", "<f8"), ("score_count", "<u4"),
        ("score_max", "<f8"), ("minimum_linear_index", "<u8")])
    stats = np.lib.format.open_memmap(work / "stats.npy", mode="w+",
        dtype=stats_dtype, shape=(table_capacity + 1,))
    parent[0] = 0
    next_id = 0
    structure = generate_binary_structure(3, connectivity)
    try:
        for chunk in iter_chunks(flags.shape, chunk_shape):
            local, local_count = label(np.asarray(flags[chunk.core], dtype=bool), structure=structure)
            counts = np.bincount(local.ravel(), minlength=local_count + 1)
            summary["input_candidate_voxels"] += int(counts[1:].sum())
            summary["input_provisional_components"] += int(local_count)
            if not local_count:
                labels[chunk.core] = 0
                continue

            # Only components that might merge across a seam are retained when
            # locally smaller than min_size. This cannot remove a global component:
            # connectivity across chunks requires an actual cross-seam neighbor.
            if connectivity == 1:
                crossing = []
                core = chunk.core
                for axis in range(3):
                    if core[axis].start > 0:
                        face = [slice(None)] * 3
                        face[axis] = 0
                        neighbor = [core[dim] for dim in range(3)]
                        neighbor[axis] = slice(core[axis].start - 1, core[axis].start)
                        neighbor = np.asarray(flags[tuple(neighbor)]).squeeze(axis=axis)
                        linked = local[tuple(face)]
                        crossing.append(linked[(linked > 0) & np.asarray(neighbor, dtype=bool)])
                    if core[axis].stop < flags.shape[axis]:
                        face = [slice(None)] * 3
                        face[axis] = -1
                        neighbor = [core[dim] for dim in range(3)]
                        neighbor[axis] = slice(core[axis].stop, core[axis].stop + 1)
                        neighbor = np.asarray(flags[tuple(neighbor)]).squeeze(axis=axis)
                        linked = local[tuple(face)]
                        crossing.append(linked[(linked > 0) & np.asarray(neighbor, dtype=bool)])
                seam_ids = np.unique(np.concatenate(crossing)) if crossing else np.empty(0, dtype=np.int32)
            else:
                border = np.concatenate((local[0, :, :].ravel(), local[-1, :, :].ravel(),
                    local[:, 0, :].ravel(), local[:, -1, :].ravel(),
                    local[:, :, 0].ravel(), local[:, :, -1].ravel()))
                seam_ids = np.unique(border)
                seam_ids = seam_ids[seam_ids > 0]
            keep = counts >= min_size
            keep[0] = False
            keep[seam_ids] = True
            summary["retained_local_large_components"] += int(np.count_nonzero(counts[1:] >= min_size))
            summary["retained_seam_subminimum_fragments"] += int(
                np.count_nonzero(counts[seam_ids] < min_size))
            dropped = np.flatnonzero((~keep)[1:]) + 1
            if dropped.size:
                dropped_sizes = counts[dropped]
                summary["interior_subminimum_components"] += int(dropped.size)
                summary["interior_subminimum_voxels"] += int(dropped_sizes.sum())
                histogram = summary["subminimum_size_histogram"]
                for size, number in zip(*np.unique(dropped_sizes, return_counts=True)):
                    key = str(int(size))
                    histogram[key] = histogram.get(key, 0) + int(number)
            retained_ids = np.flatnonzero(keep)
            retained_ids = retained_ids[retained_ids > 0]
            if max_components is not None and next_id + retained_ids.size > max_components:
                raise RuntimeError(
                    f"retained provisional 3D component count exceeds explicitly requested limit "
                    f"{max_components} after safe subminimum compaction; candidate mask is unchanged")
            if next_id + retained_ids.size > table_capacity:
                raise RuntimeError("provisional component count exceeded candidate-voxel capacity")
            mapping = np.zeros(local_count + 1, dtype=np.uint32)
            new_ids = np.arange(next_id + 1, next_id + retained_ids.size + 1, dtype=np.uint32)
            mapping[retained_ids] = new_ids
            if new_ids.size:
                parent[new_ids] = new_ids
                # Initialize assigned records explicitly rather than relying on
                # sparse-file zero-fill. IDs are contiguous, so scalar broadcast
                # avoids allocating per-component Python objects or a second
                # large statistics array.
                stats[int(new_ids[0]):int(new_ids[-1]) + 1] = np.zeros((), dtype=stats_dtype)
            labels[chunk.core] = mapping[local]
            next_id += int(retained_ids.size)
        labels.flush()
        parent.flush()
        if summary["input_candidate_voxels"] != capacity:
            raise ValueError("component_capacity does not match the exact candidate-mask voxel count")

        # Reconcile face/edge/corner pairs with a bounded disk-backed union-find.
        for chunk in iter_chunks(flags.shape, chunk_shape):
            for offset in _positive_offsets(connectivity):
                regions = _offset_slices(chunk.core, flags.shape, offset)
                if regions is None:
                    continue
                source_slice, target_slice = regions
                left = np.asarray(labels[source_slice])
                right = np.asarray(labels[target_slice])
                joined = (left > 0) & (right > 0) & (left != right)
                if not joined.any():
                    continue
                pairs = np.unique(np.stack((left[joined], right[joined]), axis=1), axis=0)
                for first, second in pairs:
                    a, b = int(first), int(second)
                    while parent[a] != a:
                        parent[a] = parent[parent[a]]
                        a = int(parent[a])
                    while parent[b] != b:
                        parent[b] = parent[parent[b]]
                        b = int(parent[b])
                    if a != b:
                        parent[max(a, b)] = min(a, b)

        parent.flush()
        root_map = np.lib.format.open_memmap(work / "root-map.npy", mode="w+",
            dtype=np.uint32, shape=(table_capacity + 1,))
        root_map[:] = 0
        component_ids = np.arange(1, next_id + 1, dtype=np.uint32)
        # Resolve IDs in bounded batches, caching the final root in the map.
        for start in range(0, len(component_ids), 250_000):
            ids = component_ids[start:start + 250_000]
            roots = _find_many(parent, ids)
            root_map[ids] = roots
        root_map.flush()

        # Stream component voxel statistics by globally resolved root. No Python
        # object is retained per provisional or final component.
        for chunk in iter_chunks(flags.shape, chunk_shape):
            block = np.asarray(labels[chunk.core])
            positive = np.flatnonzero(block.ravel())
            if not positive.size:
                continue
            provisional = block.ravel()[positive]
            provisional_ids, inverse = np.unique(provisional, return_inverse=True)
            roots = np.asarray(root_map[provisional_ids], dtype=np.uint32)
            unique_roots, root_id_inverse = np.unique(roots, return_inverse=True)
            root_inverse = root_id_inverse[inverse]
            coordinates = np.unravel_index(positive, block.shape)
            absolute_coordinates = np.column_stack(tuple(
                np.asarray(coordinates[axis], dtype=np.uint32) + chunk.core[axis].start
                for axis in range(3)))
            local_min = np.full((len(unique_roots), 3), np.iinfo(np.uint32).max, dtype=np.uint32)
            local_max = np.zeros((len(unique_roots), 3), dtype=np.uint32)
            for axis in range(3):
                axis_min = np.full(len(unique_roots), np.iinfo(np.uint32).max, dtype=np.uint32)
                axis_max = np.zeros(len(unique_roots), dtype=np.uint32)
                np.minimum.at(axis_min, root_inverse, absolute_coordinates[:, axis])
                np.maximum.at(axis_max, root_inverse, absolute_coordinates[:, axis])
                local_min[:, axis] = axis_min
                local_max[:, axis] = axis_max
            counts = np.bincount(root_inverse, minlength=len(unique_roots)).astype(np.uint32)
            old_sizes = np.asarray(stats["size"][unique_roots], dtype=np.uint32)
            stats["size"][unique_roots] = old_sizes + counts
            for axis in range(3):
                old_min = np.asarray(stats["minimum"][unique_roots, axis], dtype=np.uint32)
                old_max = np.asarray(stats["maximum"][unique_roots, axis], dtype=np.uint32)
                stats["minimum"][unique_roots, axis] = np.where(
                    old_sizes == 0, local_min[:, axis], np.minimum(old_min, local_min[:, axis]))
                stats["maximum"][unique_roots, axis] = np.where(
                    old_sizes == 0, local_max[:, axis], np.maximum(old_max, local_max[:, axis]))
            flat_scores = (np.asarray(values[chunk.core], dtype=np.float64).ravel()[positive]
                           if values is not None else np.ones(positive.size, dtype=np.float64))
            finite = np.isfinite(flat_scores)
            if finite.any():
                score_sums = np.bincount(root_inverse[finite], weights=flat_scores[finite],
                                          minlength=len(unique_roots))
                score_counts = np.bincount(root_inverse[finite], minlength=len(unique_roots)).astype(np.uint32)
                score_max = np.full(len(unique_roots), -np.inf, dtype=np.float64)
                np.maximum.at(score_max, root_inverse[finite], flat_scores[finite])
                old_score_count = np.asarray(stats["score_count"][unique_roots], dtype=np.uint32)
                stats["score_sum"][unique_roots] += score_sums
                stats["score_count"][unique_roots] = old_score_count + score_counts
                old_score_max = np.asarray(stats["score_max"][unique_roots], dtype=np.float64)
                stats["score_max"][unique_roots] = np.where(old_score_count == 0, score_max,
                                                              np.maximum(old_score_max, score_max))
            linear = (absolute_coordinates[:, 0].astype(np.uint64) * flags.shape[1] * flags.shape[2]
                      + absolute_coordinates[:, 1].astype(np.uint64) * flags.shape[2]
                      + absolute_coordinates[:, 2])
            local_linear = np.full(len(unique_roots), np.iinfo(np.uint64).max, dtype=np.uint64)
            np.minimum.at(local_linear, root_inverse, linear)
            old_linear = np.asarray(stats["minimum_linear_index"][unique_roots], dtype=np.uint64)
            stats["minimum_linear_index"][unique_roots] = np.where(
                old_sizes == 0, local_linear, np.minimum(old_linear, local_linear))

        parent_roots = 0
        retained_count = 0
        postmerge_small = 0
        postmerge_small_voxels = 0
        histogram = summary["subminimum_size_histogram"]
        component_labels = np.lib.format.open_memmap(work / "component-label-map.npy", mode="w+",
            dtype=label_dtype, shape=(table_capacity + 1,))
        component_labels[:] = 0
        for start in range(1, next_id + 1, 250_000):
            stop = min(next_id + 1, start + 250_000)
            ids = np.arange(start, stop, dtype=np.uint32)
            is_root = np.asarray(parent[ids] == ids)
            root_ids = ids[is_root]
            if not root_ids.size:
                continue
            parent_roots += int(root_ids.size)
            sizes = np.asarray(stats["size"][root_ids], dtype=np.uint32)
            small = sizes < min_size
            if small.any():
                small_sizes = sizes[small]
                postmerge_small += int(small.sum())
                postmerge_small_voxels += int(small_sizes.sum())
                for size, number in zip(*np.unique(small_sizes, return_counts=True)):
                    key = str(int(size))
                    histogram[key] = histogram.get(key, 0) + int(number)
            keep_ids = root_ids[~small]
            if keep_ids.size:
                stable_ids = np.asarray(stats["minimum_linear_index"][keep_ids], dtype=np.uint64) + 1
                if stable_ids.max(initial=0) > np.iinfo(label_dtype).max:
                    raise ValueError("stable component label exceeds output label dtype capacity")
                component_labels[keep_ids] = stable_ids.astype(label_dtype, copy=False)
                retained_count += int(keep_ids.size)
        component_labels.flush()

        if retained_count:
            if metadata_path.exists():
                raise FileExistsError(f"Refusing to overwrite component metadata: {metadata_path}")
            metadata = np.lib.format.open_memmap(metadata_path, mode="w+", dtype=_ZONE_DTYPE,
                                                  shape=(retained_count,))
            row = 0
            for start in range(1, next_id + 1, 250_000):
                stop = min(next_id + 1, start + 250_000)
                ids = np.arange(start, stop, dtype=np.uint32)
                is_root = np.asarray(parent[ids] == ids)
                root_ids = ids[is_root]
                if not root_ids.size:
                    continue
                sizes = np.asarray(stats["size"][root_ids], dtype=np.uint32)
                keep_ids = root_ids[sizes >= min_size]
                if not keep_ids.size:
                    continue
                order = np.argsort(np.asarray(stats["minimum_linear_index"][keep_ids]), kind="stable")
                keep_ids = keep_ids[order]
                count = len(keep_ids)
                metadata[row:row + count]["label"] = component_labels[keep_ids]
                metadata[row:row + count]["voxel_count"] = stats["size"][keep_ids]
                metadata[row:row + count]["bounds"][:, :, 0] = stats["minimum"][keep_ids]
                metadata[row:row + count]["bounds"][:, :, 1] = stats["maximum"][keep_ids]
                score_counts = np.asarray(stats["score_count"][keep_ids], dtype=np.uint32)
                score_sums = np.asarray(stats["score_sum"][keep_ids], dtype=np.float64)
                metadata[row:row + count]["mean_strength"] = np.divide(score_sums, score_counts,
                    out=np.zeros(count, dtype=np.float64), where=score_counts > 0)
                metadata[row:row + count]["max_strength"] = stats["score_max"][keep_ids]
                row += count
            metadata.flush()
        else:
            np.save(metadata_path, np.empty(0, dtype=_ZONE_DTYPE), allow_pickle=False)

        for chunk in iter_chunks(flags.shape, chunk_shape):
            block = np.asarray(labels[chunk.core])
            unique_ids, inverse = np.unique(block, return_inverse=True)
            roots = np.asarray(root_map[unique_ids], dtype=np.uint32)
            labels[chunk.core] = component_labels[roots][inverse].reshape(block.shape)
        labels.flush()

        summary["postmerge_subminimum_components"] = postmerge_small
        summary["postmerge_subminimum_voxels"] = postmerge_small_voxels
        summary["total_components"] = summary["interior_subminimum_components"] + parent_roots
        summary["catalogued_components"] = retained_count
        summary["retained_provisional_components"] = next_id
        summary["global_resolved_components"] = parent_roots
        summary["temporary_table_capacity"] = table_capacity
        if component_summary is not None:
            component_summary.update(summary)
    finally:
        # The public label map and compact metadata are persistent outputs; large
        # union/stat scratch maps are temporary and are removed on success or error.
        work_temp.cleanup()
    return labels, _DiskBackedZones(metadata_path, retained_count)
