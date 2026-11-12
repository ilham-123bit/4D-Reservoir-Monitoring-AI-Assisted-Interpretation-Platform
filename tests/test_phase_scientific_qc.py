"""Synthetic safeguards for scientific QC, coverage, and finding labels."""

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from reservoir_time_machine.preprocessing.qc import quality_report
from reservoir_time_machine.preprocessing.qc_artifact import (
    build_qc_artifact,
    load_qc_artifact,
    paired_sample_support,
    write_qc_artifact,
)
from reservoir_time_machine.visualization.coordinates import finding_header_location, header_id
from reservoir_time_machine.visualization.coverage import coverage_figure
from reservoir_time_machine.visualization.guided_report import _discussion_text, _result_text


class ScientificQcTests(unittest.TestCase):
    def test_missing_trace_mask_excludes_finite_fill_values_but_retains_zero_energy(self):
        volume = np.array([[[0.0, 0.0], [1.0, 1.0]], [[2.0, 2.0], [999.0, 999.0]]])
        valid = np.array([[True, True], [True, False]])
        report = quality_report(volume, inline_chunk=1, valid_trace_mask=valid)
        self.assertEqual(report["finite_voxels"], 6)
        self.assertEqual(report["voxel_count"], 6)
        self.assertEqual(report["source_voxel_count"], 8)
        self.assertEqual(report["excluded_masked_trace_count"], 1)
        self.assertEqual(report["missing_trace_count"], 1)
        self.assertEqual(report["zero_energy_trace_count"], 1)
        self.assertAlmostEqual(report["amplitude_max"], 2.0)

    def test_paired_sample_support_excludes_missing_cells(self):
        baseline = np.ones((2, 3, 4), dtype=np.float32)
        monitor = baseline * 2
        missing = np.zeros((2, 3), dtype=bool)
        missing[1, 2] = True
        # Finite placeholders at a masked cell must not become paired support.
        support = paired_sample_support(baseline, monitor, missing, inline_chunk=1)
        self.assertEqual(support["paired_grid_trace_count"], 5)
        self.assertEqual(support["potential_paired_samples"], 20)
        self.assertEqual(support["valid_paired_samples"], 20)
        self.assertEqual(support["complete_paired_traces"], 5)

    def test_qc_artifact_is_versioned_checksummed_and_read_only_scoped(self):
        baseline = np.ones((2, 3, 5), dtype=np.float32)
        monitor = baseline * 1.5
        missing = np.zeros((2, 3), dtype=bool)
        missing[1, 2] = True
        baseline[1, 2] = 500
        monitor[1, 2] = 900
        monitor_only = np.array([[4, 3, 2, 1, 0]], dtype=np.float32)
        artifact = build_qc_artifact(baseline, monitor, missing, monitor_only,
                                     source_identity="a" * 64,
                                     source_metadata={"paired_trace_count": 5,
                                                      "monitor_only_trace_count": 1},
                                     inline_chunk=1)
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "protected-cache"
            cache.mkdir()
            target = Path(tmp) / "private-qc" / "qc.json"
            write_qc_artifact(artifact, target, protected_cache=cache)
            loaded = load_qc_artifact(target, source_identity="a" * 64, shape=(2, 3, 5))
            self.assertEqual(loaded["trace_counts"]["paired_trace_count"], 5)
            self.assertEqual(loaded["trace_counts"]["missing_baseline_grid_count"], 1)
            self.assertEqual(loaded["paired_sample_support"]["valid_paired_samples"], 25)
            self.assertEqual(loaded["baseline"]["amplitude_max"], 1.0)
            self.assertNotEqual(loaded["monitor_only"]["amplitude_rms"], loaded["monitor_on_paired_baseline_grid"]["amplitude_rms"])
            with self.assertRaises(FileExistsError):
                write_qc_artifact(artifact, target, protected_cache=cache)
            with self.assertRaises(ValueError):
                load_qc_artifact(target, source_identity="b" * 64, shape=(2, 3, 5))
            with self.assertRaises(ValueError):
                write_qc_artifact(artifact, cache / "bad.json", protected_cache=cache)

    def test_coverage_figure_has_aligned_serializable_layers_and_exact_counts(self):
        missing = np.zeros((5, 7), dtype=bool)
        missing[0, 0] = True
        missing[1, 1] = True
        missing[4, 6] = True
        inlines = np.arange(344, 349, dtype=np.int32)
        crosslines = np.arange(1949, 1956, dtype=np.int32)
        keys = np.array([[344, 1949], [348, 1955], [9999, 9999]], dtype=np.int32)
        fig, stats = coverage_figure(missing, inlines, crosslines, keys, max_axis=3)
        self.assertEqual(len(fig.data), 2)
        self.assertEqual(fig.data[0].z.dtype, np.uint8)
        self.assertEqual(fig.data[0].z.shape, (len(fig.data[0].y), len(fig.data[0].x)))
        self.assertEqual(fig.data[0].customdata.shape[:2], fig.data[0].z.shape)
        self.assertEqual(len(fig.data[1].x), len(keys))
        self.assertEqual(stats["paired_grid_trace_count"], 32)
        self.assertEqual(stats["missing_baseline_grid_count"], 3)
        self.assertEqual(stats["monitor_only_trace_count"], 3)
        self.assertEqual(stats["monitor_only_mapped_to_baseline_axes"], 2)
        self.assertEqual(stats["monitor_only_on_missing_baseline_cells"], 2)
        serialized = fig.to_json()
        self.assertIn("Crossline header ID", serialized)
        self.assertIn("Monitor-only keys (3)", serialized)

    def test_actual_cache_coverage_counts_and_axes_when_requested(self):
        cache_value = os.environ.get("RTM_TEST_CACHE_DIR")
        if not cache_value:
            self.skipTest("Set RTM_TEST_CACHE_DIR for bounded read-only cache validation")
        cache = Path(cache_value)
        metadata = json.loads((cache / "metadata.json").read_text())
        missing = np.load(cache / "missing_trace_mask.npy", mmap_mode="r", allow_pickle=False)
        keys = np.load(cache / "monitor_only_keys.npy", mmap_mode="r", allow_pickle=False)
        _, stats = coverage_figure(missing, metadata["inline_ids"], metadata["crossline_ids"], keys)
        self.assertEqual(stats["paired_grid_trace_count"], metadata["paired_trace_count"])
        self.assertEqual(stats["missing_baseline_grid_count"], metadata["missing_grid_cell_count"])
        self.assertEqual(stats["monitor_only_trace_count"], metadata["monitor_only_trace_count"])
        self.assertEqual(stats["monitor_only_on_missing_baseline_cells"], 3016)
        self.assertEqual(stats["monitor_only_trace_count"] - stats["monitor_only_mapped_to_baseline_axes"], 71)

    def test_header_ids_are_not_confused_with_source_indices(self):
        inline_ids = np.arange(344, 1373, dtype=np.int32)
        crossline_ids = np.arange(1949, 3197, dtype=np.int32)
        result = finding_header_location([262, 1236, 650], inline_ids, crossline_ids,
                                         grid_shape=(1029, 1248))
        self.assertEqual(result["inline_index"], 262)
        self.assertEqual(result["inline_header_id"], 606)
        self.assertEqual(result["crossline_index"], 1236)
        self.assertEqual(result["crossline_header_id"], 3185)
        self.assertEqual(result["sample_index"], 650)
        self.assertIsNone(result["sample_physical_value"])
        self.assertIsNone(header_id(262, inline_ids[:-1], expected_length=1029))

    def test_finding_discussion_uses_evidence_limits_not_placeholder_causes(self):
        finding = {"representative": [262, 1236, 650], "metrics": {
            "cells": 8, "mean_signed_difference": 48207.45,
            "mean_absolute_difference": 48207.45, "mean_nrms_percent": 118.59,
            "minimum_paired_support": 9, "members_adjacent_to_invalid_cells": 0,
            "baseline_amplitude": -34148.2, "monitor_amplitude": 24036.37},
            "bounds": [[261, 264], [1235, 1237]],
            "method_evidence": {"statistical": {"inspection_support": 0, "flagged_inspection_cells": 0}},
            "touches_tile_boundary": False}
        location = {"inline_index": 262, "inline_header_id": 606,
                    "crossline_index": 1236, "crossline_header_id": 3185,
                    "sample_index": 650}
        result = _result_text(finding, location)
        discussion = _discussion_text(finding, {"completed_tiles": 2, "planned_tiles": 28},
                                      {"missing_association_evidence": ["CRS", "time-depth tie"]},
                                      location, (1029, 1248), (256, 320, 1216, 1248))
        self.assertIn("Inline header 606", result)
        self.assertIn("NRMS 118.59%", result)
        self.assertIn("array-index extent inline 261–263, crossline 1235–1236", result)
        for forbidden in ("cause N/A", "Registration N/A", "seismic tie N/A"):
            self.assertNotIn(forbidden, result + discussion)
        self.assertIn("2/28 tiles", discussion)
        self.assertIn("no eligible members", discussion)
        self.assertIn("Acquisition, processing", discussion)
        self.assertIn("not evidence of fluid movement or recovery", discussion)


if __name__ == "__main__":
    unittest.main()
