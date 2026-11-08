"""Read-only paired-cache identity; no full-volume hashing or cache creation."""
import hashlib
import json
from pathlib import Path

CACHE_FILES = ('metadata.json', 'baseline.npy', 'monitor.npy', 'missing_trace_mask.npy',
               'monitor_only_traces.npy', 'monitor_only_keys.npy')
SMALL_FILES = {'metadata.json', 'missing_trace_mask.npy', 'monitor_only_keys.npy'}
ANALYSIS_VERSION = 'phase1-attributes-support-1'


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
