import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

SCRIPT = Path(__file__).parents[1] / "scripts/hpc/process_full_field.py"
SPEC = importlib.util.spec_from_file_location("rtm_process_full_field", SCRIPT)
PROCESSOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROCESSOR)


class BoundedHpcPilotTests(unittest.TestCase):
    def test_cli_processes_and_records_absolute_pilot_region(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "paired"
            output = root / "derived"
            cache.mkdir(mode=0o700)
            rng = np.random.default_rng(42)
            baseline = rng.normal(size=(8, 9, 20)).astype(np.float32)
            monitor = baseline.copy()
            monitor[2:6, 3:8, 5:15] += 0.5
            np.save(cache / "baseline.npy", baseline)
            np.save(cache / "monitor.npy", monitor)
            np.save(cache / "missing_trace_mask.npy", np.zeros((8, 9), dtype=bool))
            (cache / "metadata.json").write_text(json.dumps({"shape": [8, 9, 20]}))
            bounds = [2, 6, 3, 8, 5, 15]
            args = ["--cache", str(cache), "--output", str(output), "--chunk", "2", "3", "4",
                    "--window", "3", "--min-free-gib", "0", "--pilot-bounds", *map(str, bounds)]
            with mock.patch.dict(os.environ, {"SLURM_JOB_ID": "synthetic-test"}):
                self.assertEqual(PROCESSOR.main(args), 0)
            runs = list(output.glob("attributes-*/complete.json"))
            self.assertEqual(len(runs), 1)
            body = json.loads(runs[0].read_text())["body"]
            self.assertEqual(body["configuration"]["source_bounds"], bounds)
            self.assertEqual(body["configuration"]["source_domain_shape"], [8, 9, 20])
            self.assertEqual(body["configuration"]["shape"], [4, 5, 10])
            self.assertEqual(body["configuration"]["source_identity_details"]["schema"],
                             "rtm-full-field-input-identity-2")
            self.assertIn("paired_cache_sha256",
                          body["configuration"]["source_identity_details"])
            self.assertEqual(body["status"], "complete")
            rms = np.load(runs[0].parent / "rms_baseline.npy", mmap_mode="r")
            self.assertTrue(np.isnan(rms[:, :, 0]).all())
            self.assertTrue(np.isfinite(rms[:, :, 1:-1]).all())

    def test_invalid_source_bounds_are_rejected(self):
        from reservoir_time_machine.processing.full_field import run_full_field_attributes
        values = np.ones((2, 3, 5), dtype=np.float32)
        mask = np.zeros((2, 3), dtype=bool)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "source_bounds"):
                run_full_field_attributes(values, values, mask, temporary,
                    source_identity="a" * 64, source_bounds=(0, 2, 0, 3, 0, 4))


if __name__ == "__main__":
    unittest.main()
