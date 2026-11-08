import unittest
from pathlib import Path

from reservoir_time_machine.visualization.theme import format_metric_count


class MetricFormattingTests(unittest.TestCase):
    def test_counts_are_compact_without_ellipsis(self):
        self.assertEqual(format_metric_count(1_240_941), "1.24M")
        self.assertEqual(format_metric_count(43_251), "43.3k")
        self.assertEqual(format_metric_count(1_301), "1,301")
        self.assertNotIn("...", format_metric_count(1_240_941))

    def test_invalid_counts_are_rejected(self):
        for value in (-1, True, 1.5, "100"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                format_metric_count(value)

    def test_responsive_typography_targets_are_present(self):
        css = (Path(__file__).parents[1] / "src/reservoir_time_machine/visualization/theme.py").read_text()
        self.assertIn("clamp(2.625rem, 3.1vw, 3.25rem)", css)  # 42–52 px
        self.assertIn("clamp(1.5rem, 1.7vw, 1.75rem)", css)  # 24–28 px
        self.assertIn("clamp(1.75rem, 2.2vw, 2.1rem)", css)  # 28–34 px
        self.assertIn("text-overflow:clip", css)


if __name__ == "__main__":
    unittest.main()
