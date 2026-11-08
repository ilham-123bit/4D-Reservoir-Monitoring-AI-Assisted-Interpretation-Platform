"""Phase 1 regressions; unittest also supports execution without pytest."""
import unittest
import numpy as np
from reservoir_time_machine.detection.clustering import connected_zones
from reservoir_time_machine.visualization.slices import (
    display_coordinates, display_mask, display_region, display_slice, region_representative,
)


class SpatialEvidenceTests(unittest.TestCase):
    def test_isolated_and_shared_bin_regions_preserved_independently(self):
        labels = np.zeros((11, 13), dtype=np.int32)
        labels[1, 1], labels[1, 2], labels[10, 12] = 1, 2, 3
        self.assertEqual(display_slice(labels > 0, 4).sum(), 0)
        self.assertEqual(display_mask(labels > 0, 4).sum(), 2)
        self.assertTrue(display_region(labels, 1, 4)[0, 0])
        self.assertTrue(display_region(labels, 2, 4)[0, 0])
        self.assertEqual(display_region(labels, 3, 4).sum(), 1)
        self.assertEqual(labels[1, 2], 2)

    def test_layers_share_source_coordinates_including_partial_bins(self):
        values = np.arange(143).reshape(11, 13)
        x, y = display_coordinates(values.shape, 4)
        shown = display_slice(values, 4)
        np.testing.assert_array_equal(shown, values[np.ix_(y, x)])
        self.assertEqual(shown.shape, display_mask(values > 0, 4).shape)
        np.testing.assert_array_equal(x, [0, 4, 8, 12])
        np.testing.assert_array_equal(y, [0, 3, 6, 9])

    def test_representative_inside_nonconvex_region_and_finite(self):
        mask = np.zeros((7, 7), dtype=bool)
        mask[1:6, 1] = True
        mask[5, 1:6] = True
        score = np.ones(mask.shape)
        score[1, 1] = np.nan
        score[5, 4] = 8
        labels, zones = connected_zones(mask, score, min_size=1)
        location = region_representative(labels, zones[0], score)
        self.assertEqual(location, (5, 4))
        self.assertEqual(labels[location], zones[0].label)
        self.assertFalse(mask[3, 3])
        with self.assertRaises(ValueError):
            region_representative(labels, zones[0], np.full(mask.shape, np.nan))


class AttributeTests(unittest.TestCase):
    def test_missing_zero_and_incomplete_support(self):
        from reservoir_time_machine.attributes.difference import calculate_attributes
        b = np.zeros((1, 3, 5)); m = b.copy()
        b[0, 0] = np.nan
        b[0, 2] = [1, 2, np.nan, 4, 5]
        m[0, 2] = [2, 4, 6, np.inf, 10]
        a = calculate_attributes(b, m, window_samples=3)
        for field in ['rms_baseline', 'rms_monitor', 'nrms']:
            self.assertTrue(np.isnan(getattr(a, field)[0, 0]).all())
            np.testing.assert_array_equal(getattr(a, field)[0, 1], 0)
        np.testing.assert_array_equal(a.paired_sample_count[0, 0], 0)
        np.testing.assert_array_equal(a.paired_sample_count[0, 1], [2, 3, 3, 3, 2])
        np.testing.assert_array_equal(a.paired_sample_count[0, 2], [2, 2, 1, 1, 1])
        self.assertAlmostEqual(a.rms_baseline[0, 2, 0], np.sqrt(2.5))
        self.assertAlmostEqual(a.nrms[0, 2, 0], 200/3)
        self.assertTrue(np.isnan(a.signed_difference[0, 2, 3]))

    def test_nonfinite_inputs_excluded_without_invalid_arithmetic(self):
        from reservoir_time_machine.attributes.difference import calculate_attributes, calculate_sample_attributes
        b = np.full((1, 1, 3), np.inf)
        with np.errstate(all='raise'):
            for result in [calculate_attributes(b, b), calculate_sample_attributes(b, b, 1)]:
                self.assertTrue(np.isnan(result.nrms).all())
                self.assertTrue((result.paired_sample_count == 0).all())

    def test_all_attribute_routes_and_hand_window_reference(self):
        import tempfile
        from reservoir_time_machine.attributes.difference import calculate_attributes, calculate_sample_attributes
        from reservoir_time_machine.attributes.chunked import calculate_attributes_chunked
        rng = np.random.default_rng(6)
        b = rng.normal(size=(3, 4, 11)).astype(np.float32)
        m = (b * 1.1).copy(); b[0, 0] = np.nan; m[1, 2, 5] = np.inf
        fields = ['signed_difference', 'absolute_difference', 'normalized_difference',
                  'rms_baseline', 'rms_monitor', 'nrms', 'paired_sample_count']
        for w in [1, 3, 4, 9, 15]:
            full = calculate_attributes(b, m, window_samples=w)
            with tempfile.TemporaryDirectory() as tmp:
                chunk = calculate_attributes_chunked(b, m, tmp, inline_chunk=2,
                                                     crossline_chunk=3, window_samples=w)
                for f in fields:
                    np.testing.assert_allclose(getattr(chunk, f), getattr(full, f),
                                               rtol=2e-6, atol=1e-6, equal_nan=True)
            for sample in range(b.shape[2]):
                one = calculate_sample_attributes(b, m, sample, window_samples=w)
                for f in fields:
                    np.testing.assert_allclose(getattr(one, f), getattr(full, f)[:, :, sample],
                                               rtol=2e-6, atol=1e-6, equal_nan=True)
                lo, hi = max(0, sample-w//2), min(11, sample+(w+1)//2)
                x, y = b[1, 2, lo:hi].astype(float), m[1, 2, lo:hi].astype(float)
                valid = np.isfinite(x) & np.isfinite(y)
                self.assertEqual(one.paired_sample_count[1, 2], valid.sum())
                if valid.any():
                    rb = np.sqrt(np.mean(x[valid]**2)); rm = np.sqrt(np.mean(y[valid]**2))
                    self.assertAlmostEqual(one.rms_baseline[1, 2], rb)
                    self.assertAlmostEqual(one.rms_monitor[1, 2], rm)
                    expected = 200*np.sqrt(np.mean((y[valid]-x[valid])**2))/(rb+rm)
                    self.assertAlmostEqual(one.nrms[1, 2], expected)
        const = calculate_attributes(np.ones((1, 1, 5)), np.full((1, 1, 5), 2.0))
        np.testing.assert_allclose(const.signed_difference, 1)
        np.testing.assert_allclose(const.absolute_difference, 1)
        np.testing.assert_allclose(const.normalized_difference, 1/(1.5+1e-8))
        np.testing.assert_allclose(const.nrms, 200/3)

    def test_invalid_parameters_rejected_before_chunk_output(self):
        import tempfile
        from pathlib import Path
        from reservoir_time_machine.attributes.difference import calculate_attributes, calculate_sample_attributes
        from reservoir_time_machine.attributes.chunked import calculate_attributes_chunked
        b = np.ones((1, 1, 5))
        for kw in [{'epsilon': np.nan}, {'epsilon': np.inf}, {'epsilon': 0}, {'epsilon': 'bad'},
                   {'window_samples': 2.5}, {'window_samples': True}, {'window_samples': 0}]:
            with self.assertRaises(ValueError): calculate_attributes(b, b, **kw)
            with self.assertRaises(ValueError): calculate_sample_attributes(b, b, 2, **kw)
            with tempfile.TemporaryDirectory() as tmp:
                target = Path(tmp) / 'not-created'
                with self.assertRaises(ValueError): calculate_attributes_chunked(b, b, target, **kw)
                self.assertFalse(target.exists())
        with self.assertRaises(ValueError): calculate_sample_attributes(b, b, 1.5)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError): calculate_attributes_chunked(b, b, tmp, output_dtype='int32')


class ReadinessTests(unittest.TestCase):
    def test_unknown_evidence_only_permits_preliminary_screening(self):
        from reservoir_time_machine.models import SurveyMetadata, SeismicVolume
        from reservoir_time_machine.data.readiness import assess_capabilities
        values = np.zeros((2, 2, 3)); mask = np.zeros((2, 2), bool)
        meta = {'paired_grid_available': True, 'trace_key_pairing_verified': True,
                'inline_ids': [10, 11], 'crossline_ids': [20, 21]}
        volumes = [SeismicVolume(values, SurveyMetadata(n, n, values.shape,
                   sample_axis_domain='unknown', source_metadata=meta), mask) for n in ['b', 'm']]
        status = assess_capabilities(*volumes)
        self.assertTrue(status['trace_key_preliminary_screening']['available'])
        self.assertEqual(sum(v['available'] for v in status.values()), 1)
        for name, capability in status.items():
            if name != 'trace_key_preliminary_screening': self.assertTrue(capability['missing_requirements'])
        # A bare verified flag without its evidence reference must not enable readiness.
        meta['physical_4d_registration_verified'] = True
        self.assertFalse(assess_capabilities(*volumes)['physical_4d_registration']['available'])
        meta['trace_key_pairing_verified'] = False
        self.assertFalse(assess_capabilities(*volumes)['trace_key_preliminary_screening']['available'])

    def test_readiness_requires_reference_and_consistent_units(self):
        from reservoir_time_machine.models import SurveyMetadata, SeismicVolume
        from reservoir_time_machine.data.readiness import assess_capabilities
        values = np.zeros((2, 2, 3)); mask = np.zeros((2, 2), bool)
        meta = {'paired_grid_available': True, 'trace_key_pairing_verified': True,
                'inline_ids': [10, 11], 'crossline_ids': [20, 21],
                'sample_origin': 0, 'sample_origin_units': 's',
                'physical_4d_registration_verified': True,
                'registration_verification_reference': 'fixture-registration-review'}
        volumes = [SeismicVolume(values, SurveyMetadata(n, n, values.shape,
                   sample_axis_domain='two_way_time', sample_units='s', source_metadata=meta), mask)
                   for n in ['b', 'm']]
        evidence = {'spatial_verified': True, 'coordinate_reference': 'fixture-local',
                    'coordinate_units': 'm', 'spatial_verification_reference': 'fixture-navigation-review',
                    'well_depth_time_verified': True, 'well_depth_time_reference': 'fixture-checkshot-review',
                    'model_provenance_verified': True, 'model_input_policy_verified': True,
                    'model_validation_reference': 'fixture-validation', 'calibration_verified': True,
                    'calibration_method': 'fixture-method', 'calibration_reference': 'fixture-held-out-review'}
        self.assertTrue(all(c['available'] for c in assess_capabilities(*volumes, evidence=evidence).values()))
        evidence.pop('calibration_reference')
        self.assertFalse(assess_capabilities(*volumes, evidence=evidence)['probability_calibration']['available'])
        meta['sample_origin_units'] = 'unknown'
        self.assertFalse(assess_capabilities(*volumes)['physical_4d_registration']['available'])

    def test_cache_identity_tracks_masks_archives_sources_and_metadata(self):
        import tempfile
        from pathlib import Path
        from reservoir_time_machine.data_ingest.cache_identity import CACHE_FILES, paired_cache_identity
        from reservoir_time_machine.data_audit.audit import BASELINE, MONITOR
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); cache = root / 'cache'; cache.mkdir()
            for name in CACHE_FILES: (cache/name).write_bytes(b'initial')
            for name in [BASELINE, MONITOR]:
                path = root/name; path.parent.mkdir(exist_ok=True); path.write_bytes(b'source')
            identity = paired_cache_identity(root, cache)
            self.assertEqual(identity, paired_cache_identity(root, cache))
            for path in [*(cache/name for name in CACHE_FILES), root/BASELINE, root/MONITOR]:
                before = paired_cache_identity(root, cache)
                path.write_bytes(path.read_bytes()+b'x')
                self.assertNotEqual(before, paired_cache_identity(root, cache))

    def test_cache_metadata_rejects_duplicate_and_overlapping_keys(self):
        from reservoir_time_machine.data_ingest.cache_identity import validate_cache_index_metadata
        meta = {'shape': [2, 2, 3], 'inline_ids': [10, 11], 'crossline_ids': [20, 21],
                'paired_trace_count': 3}
        missing = np.array([[False, False], [False, True]])
        validate_cache_index_metadata(meta, missing, np.array([[11, 21]], dtype=np.int32))
        for keys in [np.array([[10, 20]]), np.array([[12, 21], [12, 21]]), np.array([[0, 20]])]:
            with self.assertRaises(ValueError): validate_cache_index_metadata(meta, missing, keys)
        meta['inline_ids'] = [10, 10]
        with self.assertRaises(ValueError): validate_cache_index_metadata(meta, missing, np.empty((0, 2)))


class WellIngestionTests(unittest.TestCase):
    def test_workbook_row_three_blank_duplicate_and_extra_columns_preserved(self):
        import tempfile
        from pathlib import Path
        try:
            from openpyxl import Workbook
        except ImportError:
            self.skipTest("openpyxl unavailable")
        from reservoir_time_machine.data_ingest.well_information import inspect_well_information, load_well_information
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'well.xlsx'; w = Workbook(); sheet = w.active
            sheet.append([]); sheet.append([])
            sheet.append(['Well Name', 'X', 'X', None]); sheet.append([])
            sheet.append(['Well-A', 1, 2, 'extra']); sheet.append(['Well-B', None, 4, None])
            w.save(path); w.close()
            report = inspect_well_information(path)['sheets'][0]
            self.assertEqual(report['header_row'], 3); self.assertEqual(report['data_rows'], 2)
            rows = load_well_information(path, column_mapping={'well_id': 'Well Name'})
            self.assertEqual(rows[0]['source_row'], 5)
            self.assertEqual(rows[0]['source_values']['X__column_2'], 1)
            self.assertEqual(rows[0]['source_values']['X__column_3'], 2)
            self.assertEqual(rows[0]['source_values']['source__column_4'], 'extra')
            self.assertIsNone(rows[1]['source_values']['X__column_2'])
            with self.assertRaises(ValueError): load_well_information(path, column_mapping={'x': 'absent'})

    def test_marker_first_record_whitespace_and_unknown_field_alignment(self):
        import tempfile
        from pathlib import Path
        from reservoir_time_machine.data_ingest.well_tables import read_well_table
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'markers.txt'
            path.write_text('#NAME\n#MDKB (m)\n#TVDSS (m)\nSeabed\t\t69\nTop_A 100 90\n')
            table = read_well_table(path)
            self.assertEqual(len(table.rows), 2)
            self.assertEqual(table.rows[0][0], 'Seabed')
            self.assertEqual(table.rows[1], ('Top_A', '100', '90'))
            self.assertEqual(table.raw_data_lines[0], 'Seabed\t\t69')
            self.assertEqual(table.declared_columns, ('NAME', 'MDKB', 'TVDSS'))
            self.assertEqual(table.row_width_mismatch_count, 1)
            self.assertEqual(table.columns, ('NAME', 'unmapped_column_2', 'unmapped_column_3'))
            self.assertEqual(table.units, (None, None, None))
            path.write_text('#NAME\n#TVDSS (m)\nI-1 100\nI-2 200\n')
            table = read_well_table(path)
            self.assertEqual(table.columns, ('NAME', 'TVDSS'))
            self.assertEqual(table.units, (None, 'm'))
            self.assertEqual(len(table.rows), 2)

    def test_audit_reuses_corrected_marker_reader_without_raw_values(self):
        import tempfile
        from pathlib import Path
        from reservoir_time_machine.data_audit.audit import _inspect_text_table
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'marker.txt'; path.write_text('#NAME\n#TVDSS (m)\nI-1 100\nI-2 200\n')
            report = _inspect_text_table(path)
            self.assertEqual(report['data_row_count'], 2)
            self.assertFalse(report['sample_values_included'])
            self.assertNotIn('100', str(report))

    def test_generic_delimiters_missing_values_and_surplus_fields_preserved(self):
        import tempfile
        from pathlib import Path
        from reservoir_time_machine.data_ingest.well_tables import read_well_table
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'generic.tsv'
            path.write_text('field_a\tfield_b\nalpha\t\nomega\t3\textra\n')
            table = read_well_table(path)
            self.assertEqual(table.rows[0], ('alpha', '', ''))
            self.assertEqual(table.rows[1], ('omega', '3', 'extra'))
            path.write_text('Top_A 100\nTop_B 200\n')
            self.assertEqual(len(read_well_table(path).rows), 2)


if __name__ == '__main__':
    unittest.main()
