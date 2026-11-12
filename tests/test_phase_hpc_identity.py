import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from reservoir_time_machine.data_ingest.cache_identity import CACHE_FILES, paired_cache_provenance


class PortableCacheIdentityTests(unittest.TestCase):
    def make_cache(self, root):
        cache = Path(root) / "paired"
        cache.mkdir()
        (cache / "metadata.json").write_text('{"shape":[2,3,9]}')
        np.save(cache / "baseline.npy", np.arange(54, dtype=np.float32).reshape(2, 3, 9))
        np.save(cache / "monitor.npy", np.arange(54, dtype=np.float32).reshape(2, 3, 9) + 1)
        np.save(cache / "missing_trace_mask.npy", np.zeros((2, 3), dtype=bool))
        (cache / "monitor_only_traces.npy").write_bytes(b"monitor-only-fixture")
        np.save(cache / "monitor_only_keys.npy", np.empty((0, 2), dtype=np.int32))
        return cache

    def test_identity_is_repeatable_and_inode_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = self.make_cache(tmp)
            first, details = paired_cache_provenance(cache)
            self.assertEqual(first, paired_cache_provenance(cache)[0])
            self.assertEqual(details["schema"], "rtm-paired-cache-provenance-2")
            path = cache / "baseline.npy"
            stat = path.stat()
            replacement = cache / "replacement.npy"
            replacement.write_bytes(path.read_bytes())
            os.utime(replacement, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            replacement.replace(path)
            self.assertEqual(first, paired_cache_provenance(cache)[0])

    def test_large_array_fingerprint_is_not_content_authentication(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = self.make_cache(tmp)
            first, details = paired_cache_provenance(cache)
            path = cache / "baseline.npy"
            stat = path.stat()
            replacement = cache / "replacement.npy"
            np.save(replacement, np.full((2, 3, 9), -1, dtype=np.float32))
            os.utime(replacement, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            replacement.replace(path)
            self.assertEqual(first, paired_cache_provenance(cache)[0])
            self.assertIn("not full-array authentication", details["identity_scope"])
            self.assertIn("processed chunk SHA256", details["content_verification"])

    def test_control_file_content_change_changes_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = self.make_cache(tmp)
            first, _ = paired_cache_provenance(cache)
            mask = np.load(cache / "missing_trace_mask.npy")
            mask[0, 1] = True
            np.save(cache / "missing_trace_mask.npy", mask)
            second, _ = paired_cache_provenance(cache)
            self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
