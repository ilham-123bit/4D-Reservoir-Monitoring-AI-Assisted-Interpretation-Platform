"""Read-only paired-cache identity; no full-volume hashing or cache creation."""
import hashlib
import json
from pathlib import Path

CACHE_FILES = ('metadata.json', 'baseline.npy', 'monitor.npy', 'missing_trace_mask.npy',
               'monitor_only_traces.npy', 'monitor_only_keys.npy')
SMALL_FILES = {'metadata.json', 'missing_trace_mask.npy', 'monitor_only_keys.npy'}
ANALYSIS_VERSION = 'phase1-attributes-support-1'
PORTABLE_PROVENANCE_VERSION = 'rtm-paired-cache-provenance-2'


def paired_cache_identity(root, cache_directory):
    """Fingerprint all cache/source stat identities and small control-file contents.

    Large arrays use path/size/mtime/ctime/inode, not content authentication.
    Missing caches fail closed in the dashboard rather than rebuilding inputs.
    """
    from ..data_audit.audit import BASELINE, MONITOR
    root, cache = Path(root).resolve(), Path(cache_directory).resolve()
    records = []
    for path in [*(cache / name for name in CACHE_FILES), root / BASELINE, root / MONITOR]:
        stat = path.stat()
        record = [str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino]
        if path.parent == cache and path.name in SMALL_FILES:
            record.append(hashlib.sha256(path.read_bytes()).hexdigest())
        records.append(record)
    return hashlib.sha256(json.dumps([ANALYSIS_VERSION, records], sort_keys=True).encode()).hexdigest()


def paired_cache_provenance(cache_directory):
    """Return a reproducible, non-authenticating cache fingerprint and details.

    Deliberately excludes inode and ctime, which can vary between filesystem
    clients. It hashes the small metadata/occupancy/key files and records stable
    path, size, mtime, NPY shape and dtype for large arrays. Processed chunk
    hashes remain the content-level verification for the data actually read.
    This function never hashes a full seismic cube or writes to the cache.
    """
    import numpy as np

    cache = Path(cache_directory).expanduser().resolve()
    records = []
    required_files = {"metadata.json", "baseline.npy", "monitor.npy", "missing_trace_mask.npy"}
    for name in CACHE_FILES:
        path = cache / name
        if not path.exists():
            if name in required_files:
                raise FileNotFoundError(f"required paired-cache file is missing: {name}")
            records.append({"name": name, "present": False})
            continue
        stat = path.stat()
        item = {"name": name, "path": str(path), "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns, "present": True}
        if name in {"baseline.npy", "monitor.npy", "missing_trace_mask.npy"}:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            item.update(shape=list(array.shape), dtype=str(array.dtype))
        if name in SMALL_FILES:
            item["content_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append(item)
    details = {"schema": PORTABLE_PROVENANCE_VERSION, "cache_directory": str(cache),
               "identity_scope": "metadata/control-file SHA256 plus large-file stat fingerprint; not full-array authentication",
               "files": records,
               "content_verification": "processed chunk SHA256 records authenticate the input blocks actually analyzed"}
    payload = json.dumps(details, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest(), details


def validate_cache_index_metadata(metadata, missing, monitor_only_keys):
    """Bounded structural checks; do not assert source authentication or registration."""
    import numpy as np
    shape = tuple(metadata['shape'])
    axes = []
    for key, size in [('inline_ids', shape[0]), ('crossline_ids', shape[1])]:
        ids = metadata[key]
        if (len(ids) != size or any(isinstance(x, bool) or not isinstance(x, int) for x in ids)
                or any(a >= b for a, b in zip(ids, ids[1:]))):
            raise ValueError(f'{key} must be unique increasing integer identifiers')
        axes.append({value: index for index, value in enumerate(ids)})
    if metadata['paired_trace_count'] != int(np.count_nonzero(~missing)):
        raise ValueError('Paired trace count disagrees with grid occupancy')
    keys = np.asarray(monitor_only_keys)
    if len(np.unique(keys, axis=0)) != len(keys) or np.any(keys == 0):
        raise ValueError('Monitor-only keys must be unique nonzero identifiers')
    for inline, crossline in keys:
        i, x = axes[0].get(int(inline)), axes[1].get(int(crossline))
        if i is not None and x is not None and not missing[i, x]:
            raise ValueError('Monitor-only key overlaps a paired grid cell')
