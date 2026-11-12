import json
import tempfile
import unittest
from pathlib import Path

import plotly.graph_objects as go

from reservoir_time_machine.visualization.presentation_cache import (
    get_or_build_presentation,
    load_presentation,
    presentation_identity,
    save_presentation,
)


class FindingPresentationCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "presentations"
        self.identity = presentation_identity("a" * 64, "b" * 64,
                                              "finding-test-1", "c" * 64)
        self.figure = go.Figure(go.Heatmap(z=[[0.0, 1.0], [2.0, 3.0]]))
        self.figure.frames = [go.Frame(name="Baseline", data=[go.Heatmap(z=[[0, 1]])])]

    def tearDown(self):
        self.temp.cleanup()

    def test_round_trip_preserves_traces_and_frames(self):
        path = save_presentation(self.figure, self.root, self.identity)
        loaded = load_presentation(self.root, self.identity)
        self.assertIsNotNone(loaded)
        self.assertEqual(len(loaded.data), 1)
        self.assertEqual(loaded.frames[0].name, "Baseline")
        self.assertEqual(path.parent.stat().st_mode & 0o777, 0o700)

    def test_build_is_skipped_after_cache_publication(self):
        calls = []
        builder = lambda: (calls.append(True) or self.figure)
        first, first_hit = get_or_build_presentation(self.root, self.identity, builder)
        second, second_hit = get_or_build_presentation(self.root, self.identity, builder)
        self.assertFalse(first_hit)
        self.assertTrue(second_hit)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(first.data), len(second.data))

    def test_identity_change_and_checksum_corruption_are_rejected(self):
        target = save_presentation(self.figure, self.root, self.identity)
        changed = {**self.identity, "registry_id": "d" * 64}
        self.assertIsNone(load_presentation(self.root, changed))
        envelope = json.loads(target.read_text())
        envelope["figure"]["data"][0]["z"][0][0] = 999
        target.write_text(json.dumps(envelope))
        self.assertIsNone(load_presentation(self.root, self.identity))

    def test_invalid_identity_and_seismic_cache_destination_are_rejected(self):
        with self.assertRaises(ValueError):
            presentation_identity("../registry", "b" * 64, "finding-1", "c" * 64)
        protected = Path(self.temp.name) / ".reservoir_time_machine_cache" / "presentations"
        with self.assertRaises(ValueError):
            save_presentation(self.figure, protected, self.identity)


if __name__ == "__main__":
    unittest.main()
