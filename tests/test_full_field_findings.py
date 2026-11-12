from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from reservoir_time_machine.analysis.full_field_findings import (
    _verify_streamed_envelope, build_full_field_findings)
from reservoir_time_machine.detection.clustering import connected_zones, connected_zones_chunked


class FullFieldFindingTests(unittest.TestCase):
    def test_many_interior_specks_are_summarized_without_losing_candidate_mask(self):
        mask = np.zeros((102, 102, 102), dtype=bool)
        mask[1:101:2, 1:101:2, 1:101:2] = True
        expected = int(mask.sum())
        summary = {}
        with tempfile.TemporaryDirectory() as temp:
            labels, zones = connected_zones_chunked(mask, Path(temp) / "many-specks",
                chunk_shape=mask.shape, min_size=16, connectivity=1, max_components=1000,
                component_summary=summary)
            self.assertEqual(zones, [])
            self.assertEqual(int(mask.sum()), expected)
            self.assertEqual(int(np.count_nonzero(labels)), 0)
            self.assertEqual(summary["input_candidate_voxels"], expected)
            self.assertEqual(summary["input_provisional_components"], expected)
            self.assertEqual(summary["interior_subminimum_components"], expected)
            self.assertEqual(summary["interior_subminimum_voxels"], expected)
            self.assertEqual(summary["total_components"], expected)
            self.assertEqual(summary["subminimum_size_histogram"], {"1": expected})
            self.assertEqual(labels.dtype, np.uint32)

    def test_small_fragments_cross_chunk_seam_before_extent_filter(self):
        mask = np.zeros((8, 5, 5), dtype=bool)
        mask[2:6, 2, 2] = True
        summary = {}
        with tempfile.TemporaryDirectory() as temp:
            labels, zones = connected_zones_chunked(mask, Path(temp) / "seam-fragments",
                chunk_shape=(4, 5, 5), min_size=4, connectivity=1, max_components=10,
                component_summary=summary)
            self.assertEqual(len(zones), 1)
            self.assertEqual(zones[0].voxel_count, 4)
            # Component labels are stable source-linear IDs, not compact ranks.
            self.assertEqual(np.unique(labels[2:6, 2, 2]).tolist(), [63])
            self.assertEqual(summary["input_provisional_components"], 2)
            self.assertEqual(summary["catalogued_components"], 1)
            self.assertEqual(summary["retained_seam_subminimum_fragments"], 2)
            self.assertEqual(summary["retained_local_large_components"], 0)

    def test_more_than_one_hundred_thousand_components_use_disk_backed_catalogue(self):
        mask = np.indices((64, 64, 64)).sum(axis=0) % 2 == 0
        expected_count = int(mask.sum())
        summary = {}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            labels, zones = connected_zones_chunked(mask, root / "many-components",
                chunk_shape=(16, 16, 16), min_size=1, connectivity=1,
                component_capacity=expected_count, component_summary=summary)
            _, reference_zones = connected_zones(mask, min_size=1, connectivity=1)
            self.assertGreater(expected_count, 100_000)
            self.assertEqual(len(zones), expected_count)
            self.assertEqual(len(reference_zones), expected_count)
            self.assertIsInstance(labels, np.memmap)
            self.assertNotIsInstance(zones, list)
            stable_ids = (np.arange(mask.size, dtype=np.uint32).reshape(mask.shape) + 1) * mask
            np.testing.assert_array_equal(labels, stable_ids)
            self.assertTrue(all(zone.voxel_count == 1 for zone in reference_zones))
            self.assertEqual(summary["catalogued_components"], expected_count)
            self.assertEqual(summary["input_candidate_voxels"], expected_count)
            self.assertEqual(np.count_nonzero(labels), expected_count)
            self.assertTrue((root / "many-components" / "component_metadata.npy").is_file())

    def test_compacted_chunk_components_match_full_volume_reference(self):
        rng = np.random.default_rng(204)
        mask = rng.random((17, 15, 13)) > 0.89
        full_labels, full_zones = connected_zones(mask, min_size=3, connectivity=1)
        summary = {}
        with tempfile.TemporaryDirectory() as temp:
            chunk_labels, chunk_zones = connected_zones_chunked(mask, Path(temp) / "equivalence",
                chunk_shape=(5, 4, 6), min_size=3, connectivity=1, max_components=1000,
                component_summary=summary)
            self.assertEqual(sorted(zone.voxel_count for zone in chunk_zones),
                             sorted(zone.voxel_count for zone in full_zones))
            self.assertEqual(int(np.count_nonzero(chunk_labels)),
                             int(np.count_nonzero(full_labels)))
            for zone in chunk_zones:
                component_voxels = chunk_labels == zone.label
                self.assertTrue(component_voxels.any())
                self.assertEqual(len(np.unique(full_labels[component_voxels])), 1)
                self.assertNotEqual(int(full_labels[component_voxels][0]), 0)

    def test_component_membership_and_ids_are_chunk_shape_invariant(self):
        rng = np.random.default_rng(771)
        mask = rng.random((19, 17, 15)) > 0.86
        # An explicit component crossing multiple chunk faces.
        mask[7:13, 7, 7] = True
        mask[10, 7:12, 7] = True
        expected_labels, expected_zones = connected_zones(mask, min_size=3, connectivity=1)
        with tempfile.TemporaryDirectory() as temp:
            outputs = []
            for index, shape in enumerate(((5, 6, 4), (8, 7, 9))):
                labels, zones = connected_zones_chunked(mask, Path(temp) / str(index),
                    chunk_shape=shape, min_size=3, connectivity=1,
                    component_capacity=int(mask.sum()))
                outputs.append(np.asarray(labels).copy())
                self.assertEqual(sorted(zone.voxel_count for zone in zones),
                                 sorted(zone.voxel_count for zone in expected_zones))
            np.testing.assert_array_equal(outputs[0], outputs[1])
            np.testing.assert_array_equal(outputs[0] > 0, expected_labels > 0)

    def test_component_explosion_hits_safe_limit_before_unbounded_union_bookkeeping(self):
        mask = np.indices((8, 8, 8)).sum(axis=0) % 2 == 0
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(RuntimeError, "explicitly requested limit 2"):
                connected_zones_chunked(mask, Path(temp) / "components",
                    chunk_shape=(8, 8, 8), min_size=1, max_components=2)

    def test_cross_chunk_3d_component_gets_member_representative_and_header_ids(self):
        shape = (8, 8, 8)
        rng = np.random.default_rng(12)
        absolute = rng.uniform(0.01, 0.08, size=shape).astype(np.float32)
        signed = absolute * np.where(rng.random(shape) > 0.5, 1, -1)
        nrms = rng.uniform(5, 30, size=shape).astype(np.float32)
        support = np.full(shape, 9, dtype=np.uint8)
        component = [(2, 4, 4), (3, 4, 4), (4, 4, 4), (5, 4, 4)]
        for pos in component:
            absolute[pos] = 0.4
            signed[pos] = -0.4
            nrms[pos] = 72
        absolute[6, 6, 6] = 0.5
        signed[6, 6, 6] = 0.5
        nrms[6, 6, 6] = 60
        flag = np.zeros(shape, dtype=np.uint8)
        for pos in component:
            flag[pos] = 1
        flag[6, 6, 6] = 1
        model_outputs = {name: {"score": absolute.copy(), "flag": flag.copy()} for name in ("mahalanobis", "pca")}
        attributes = {"absolute_difference": absolute, "signed_difference": signed,
                      "nrms": nrms, "paired_sample_count": support,
                      "rms_baseline": np.ones(shape, dtype=np.float32),
                      "rms_monitor": np.ones(shape, dtype=np.float32)}
        missing = np.zeros(shape[:2], dtype=bool)
        missing[0, 0] = True
        missing[1, 4] = True
        baseline = rng.normal(size=shape).astype(np.float32)
        monitor = baseline + signed
        with tempfile.TemporaryDirectory() as temp:
            result = build_full_field_findings(attributes, model_outputs, missing,
                rng.uniform(0.01, 0.08, size=256), Path(temp) / "findings",
                source_identity=hashlib.sha256(b"source").hexdigest(),
                inline_ids=np.arange(344, 344 + shape[0]),
                crossline_ids=np.arange(1949, 1949 + shape[1]), min_region_voxels=4,
                monitor_only_keys=np.asarray([[345, 1953]], dtype=np.int32),
                survey_amplitudes={"baseline": baseline, "monitor": monitor},
                chunk_shape=(3, 3, 3), max_findings=10)
            self.assertEqual(result["finding_count"], 1)
            self.assertEqual(result["candidate_count"], 2)
            catalogue = json.loads((Path(temp) / "findings" / "candidate_catalogue.json").read_text())
            self.assertEqual(catalogue["body"]["candidate_count"], 2)
            self.assertEqual(hashlib.sha256(json.dumps(catalogue["body"], sort_keys=True,
                separators=(",", ":"), allow_nan=False).encode()).hexdigest(), catalogue["sha256"])
            self.assertTrue(_verify_streamed_envelope(Path(temp) / "findings" / "candidate_catalogue.json",
                catalogue["sha256"], result["source_identity"], "rtm-full-field-candidate-catalogue-1"))
            self.assertTrue(any(c["component_voxels"] == 1 and not c["priority_eligible"]
                                for c in catalogue["body"]["candidates"]))
            finding = result["findings"][0]
            self.assertEqual(finding["component_voxels"], 4)
            self.assertEqual(finding["segy_location"]["inline_id"], 344 + finding["array_location"][0])
            self.assertEqual(finding["segy_location"]["crossline_id"], 1949 + finding["array_location"][1])
            labels = np.load(result["component_labels"], mmap_mode="r")
            self.assertEqual(int(labels[tuple(finding["array_location"])]), finding["component_label"])
            region = finding["extent_in_array_indices"]
            self.assertTrue(region[0][0] <= finding["array_location"][0] <= region[0][1])
            self.assertEqual(finding["status"], "preliminary 3D change candidate; physical registration and geological cause unverified")
            self.assertNotIn("probability", finding["priority_components"])
            self.assertEqual(finding["edge_metrics"]["status"], "computed")
            self.assertIn("change_to_survey_edge_gradient_ratio_by_axis", finding["edge_metrics"])
            self.assertGreaterEqual(
                finding["distance_to_nearest_unoccupied_grid_cell"]["minimum_grid_cells"], 0)
            self.assertEqual(result["monitor_only_coverage"]["mapped_baseline_axis_positions"], 1)
            self.assertFalse(result["monitor_only_coverage"]["used_in_paired_calculations"])
            self.assertGreater(finding["monitor_only_neighborhood_fraction"], 0)
            self.assertGreater(finding["edge_artifact_risk"], 0)

    def test_fully_survey_edge_dominated_candidate_is_quarantined_and_zero_is_allowed(self):
        shape = (8, 8, 8)
        absolute = np.full(shape, 0.05, dtype=np.float32)
        signed = np.full(shape, 0.01, dtype=np.float32)
        nrms = np.full(shape, 10, dtype=np.float32)
        support = np.full(shape, 9, dtype=np.uint8)
        component = [(0, 3, 3), (0, 3, 4), (0, 3, 5), (0, 4, 5)]
        flags = np.zeros(shape, dtype=np.uint8)
        for pos in component:
            absolute[pos], signed[pos], nrms[pos], flags[pos] = 0.6, 0.6, 80, 1
        outputs = {name: {"score": absolute.copy(), "flag": flags.copy()}
                   for name in ("mahalanobis", "pca")}
        attributes = {"absolute_difference": absolute, "signed_difference": signed,
            "nrms": nrms, "paired_sample_count": support,
            "rms_baseline": np.ones(shape, dtype=np.float32),
            "rms_monitor": np.ones(shape, dtype=np.float32)}
        baseline = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        monitor = baseline + signed
        with tempfile.TemporaryDirectory() as temp:
            result = build_full_field_findings(attributes, outputs, np.zeros(shape[:2], bool),
                np.linspace(0.01, 0.08, 256), Path(temp) / "findings",
                source_identity=hashlib.sha256(b"edge-only").hexdigest(),
                inline_ids=np.arange(400, 408), crossline_ids=np.arange(900, 908),
                min_region_voxels=4, survey_amplitudes={"baseline": baseline, "monitor": monitor})
            self.assertEqual(result["priority_discovery_count"], 0)
            self.assertEqual(result["findings"], [])
            self.assertEqual(result["quarantined_qc_count"], 1)
            qc_catalogue = json.loads((Path(temp) / "findings" / "qc_catalogue.json").read_text())
            self.assertIn("survey boundary", qc_catalogue["body"]["items"][0]["reason_codes"][0])
            self.assertTrue(_verify_streamed_envelope(Path(temp) / "findings" / "qc_catalogue.json",
                qc_catalogue["sha256"], result["source_identity"], "rtm-full-field-qc-catalogue-1"))
            self.assertTrue((Path(temp) / "findings" / "candidate_catalogue.json").is_file())
            self.assertTrue((Path(temp) / "findings" / "qc_catalogue.json").is_file())

    def test_displays_at_most_ten_and_catalogues_remaining_eligible_components(self):
        shape = (20, 20, 20)
        rng = np.random.default_rng(41)
        absolute = rng.uniform(0.01, 0.08, size=shape).astype(np.float32)
        signed = absolute.copy()
        nrms = rng.uniform(5, 30, size=shape).astype(np.float32)
        support = np.full(shape, 9, dtype=np.uint8)
        flags = np.zeros(shape, dtype=np.uint8)
        component_count = 12
        starts = [(i, x, s) for i in (2, 8, 14) for x in (2, 8)
                  for s in (2, 8, 14)]
        for i, x, s in starts[:component_count]:
            for sample in range(s, s + 4):
                absolute[i, x, sample] = 0.6
                signed[i, x, sample] = 0.6
                nrms[i, x, sample] = 80
                flags[i, x, sample] = 1
        outputs = {name: {"score": absolute.copy(), "flag": flags.copy()}
                   for name in ("mahalanobis", "pca")}
        attributes = {"absolute_difference": absolute, "signed_difference": signed,
            "nrms": nrms, "paired_sample_count": support,
            "rms_baseline": np.ones(shape, dtype=np.float32),
            "rms_monitor": np.ones(shape, dtype=np.float32)}
        baseline = rng.normal(size=shape).astype(np.float32)
        monitor = baseline + signed
        with tempfile.TemporaryDirectory() as temp:
            result = build_full_field_findings(attributes, outputs, np.zeros(shape[:2], bool),
                rng.uniform(0.01, 0.08, 256), Path(temp) / "findings",
                source_identity=hashlib.sha256(b"twelve-components").hexdigest(),
                inline_ids=np.arange(344, 344 + shape[0]),
                crossline_ids=np.arange(1949, 1949 + shape[1]), min_region_voxels=4,
                survey_amplitudes={"baseline": baseline, "monitor": monitor})
            self.assertEqual(result["candidate_count"], component_count)
            self.assertEqual(result["priority_eligible_count"], component_count)
            self.assertEqual(result["priority_discovery_count"], 10)
            self.assertEqual(len(result["findings"]), 10)
            catalogue = json.loads((Path(temp) / "findings" / "candidate_catalogue.json").read_text())
            statuses = [candidate["catalogue_status"]
                        for candidate in catalogue["body"]["candidates"]]
            self.assertEqual(statuses.count("priority_discovery"), 10)
            self.assertEqual(statuses.count("additional_valid_candidate_not_in_top_display_limit"), 2)

    def test_requires_header_axes_and_caps_findings_at_ten(self):
        shape = (4, 4, 4)
        attributes = {"absolute_difference": np.ones(shape), "signed_difference": np.ones(shape),
                      "nrms": np.ones(shape), "paired_sample_count": np.ones(shape, dtype=np.uint8)}
        outputs = {"a": {"score": np.ones(shape), "flag": np.ones(shape, dtype=np.uint8)}}
        with tempfile.TemporaryDirectory() as temp, self.assertRaisesRegex(ValueError, "header axes"):
            build_full_field_findings(attributes, outputs, np.zeros(shape[:2], bool), np.arange(32.),
                Path(temp) / "x", source_identity=hashlib.sha256(b"x").hexdigest(),
                inline_ids=[1], crossline_ids=list(range(4)))


if __name__ == "__main__":
    unittest.main()
