from __future__ import annotations

import subprocess
import unittest
from unittest.mock import patch

from reservoir_time_machine.hpc.preflight import _query


class HpcPreflightTests(unittest.TestCase):
    @patch("reservoir_time_machine.hpc.preflight.subprocess.run", side_effect=FileNotFoundError)
    def test_missing_scheduler_command_is_reported_without_failure(self, _run):
        self.assertEqual(_query(["sinfo"]), {"available": False, "reason": "command unavailable"})

    @patch("reservoir_time_machine.hpc.preflight.subprocess.run", side_effect=subprocess.TimeoutExpired(["sinfo"], 8))
    def test_scheduler_timeout_is_bounded_and_reported(self, run):
        result = _query(["sinfo"], timeout=8)
        self.assertFalse(result["available"])
        self.assertIn("timed out", result["reason"])
        self.assertEqual(run.call_args.kwargs["timeout"], 8)


if __name__ == "__main__":
    unittest.main()
