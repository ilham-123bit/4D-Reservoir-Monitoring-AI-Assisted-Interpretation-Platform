"""Bounded read-only Phase 1 checks. No outputs or cache volumes are created.

Run with HPC Python, PYTHONPATH=src and explicit RTM_CHALLENGE3_ROOT/RTM_CACHE_DIR.
Only one sample plane and its attribute window are analyzed; source arrays stay read-only.
"""
import json
import os
from pathlib import Path
import time
import resource
import numpy as np

from reservoir_time_machine.attributes.difference import calculate_sample_attributes, calculate_attributes
from reservoir_time_machine.data.readiness import assess_capabilities
from reservoir_time_machine.data_ingest.cache_identity import paired_cache_identity, CACHE_FILES
from reservoir_time_machine.data_ingest.segy_loader import load_challenge3_paired_volumes
from reservoir_time_machine.data_ingest.well_information import inspect_well_information, load_well_information
from reservoir_time_machine.data_ingest.well_tables import read_well_table
from reservoir_time_machine.data_ingest.las_loader import inspect_las
from reservoir_time_machine.detection.statistical import statistical_screen
from reservoir_time_machine.detection.clustering import connected_zones
from reservoir_time_machine.visualization.slices import display_mask, display_slice, display_region, region_representative


def main():
    root, cache = Path(os.environ['RTM_CHALLENGE3_ROOT']), Path(os.environ['RTM_CACHE_DIR'])
    identity_before = paired_cache_identity(root, cache)
    cache_stat = {name: ((cache/name).stat().st_size, (cache/name).stat().st_mtime_ns,
                         (cache/name).stat().st_ctime_ns) for name in CACHE_FILES}
    start = time.perf_counter()
    b, m = load_challenge3_paired_volumes(root, output_dir=cache)
    attrs = calculate_sample_attributes(b.data, m.data, 650)
    detection = statistical_screen(attrs.absolute_difference)
    labels, zones = connected_zones(detection.mask, detection.score, min_size=8)
    seconds = time.perf_counter()-start
    for zone in zones:
        location = region_representative(labels, zone, detection.score)
        assert labels[location] == zone.label and np.isfinite(attrs.signed_difference[location])
    for zone in sorted(zones, key=lambda z: z.voxel_count, reverse=True)[:10]:
        assert display_region(labels, zone.label, 700).any()
    assert np.isnan(attrs.rms_baseline[b.missing_trace_mask]).all()
    assert (attrs.paired_sample_count[b.missing_trace_mask] == 0).all()
    assert not detection.mask[b.missing_trace_mask].any()
    coarse = display_mask(detection.mask, 700)
    stride_lost_bins = int(np.count_nonzero(coarse & ~display_slice(detection.mask, 700)))
    # Actual-data numerical parity on a bounded 64x80x9 tile, entirely in memory.
    tile_b = b.data[:64, :80, 646:655]
    tile_m = m.data[:64, :80, 646:655]
    tile_full = calculate_attributes(tile_b, tile_m)
    for field in ('signed_difference', 'absolute_difference', 'normalized_difference',
                  'rms_baseline', 'rms_monitor', 'nrms', 'paired_sample_count'):
        np.testing.assert_allclose(getattr(tile_full, field)[:, :, 4],
                                   getattr(attrs, field)[:64, :80],
                                   rtol=2e-6, atol=1e-6, equal_nan=True)
    capability = assess_capabilities(b, m)
    assert capability['trace_key_preliminary_screening']['available']
    assert sum(c['available'] for c in capability.values()) == 1
    workbook = root/'well_data/Challenge3_WellList_Information.xlsx'
    structure = inspect_well_information(workbook, header_row=3, max_column=13)
    records = load_well_information(workbook, column_mapping={'well_id': 'Well Name'},
                                    header_row=3, max_column=13)
    assert len(records) == 8 and all(r['well_id'] for r in records)
    assert structure['sheets'][0]['data_rows'] == 8
    tables = {}
    for path in sorted((root/'well_data').glob('*/*.txt')):
        table = read_well_table(path)
        source_lines = [line for line in path.read_text().splitlines()
                        if line.strip() and not line.lstrip().startswith(('#', '!', ';'))]
        assert len(table.rows) == len(source_lines)
        assert len(table.raw_data_lines) == len(source_lines)
        assert all(len(row) == len(table.columns) for row in table.rows)
        tables[path.name] = {'records': len(table.rows), 'variable_width_rows': table.row_width_mismatch_count,
                             'header_origin': table.header_origin}
    logs = {}
    for path in sorted((root/'well_data/las').glob('*.las')):
        log = inspect_las(path)
        assert all(0 <= c['null_count'] <= c['sample_count'] for c in log['curves'])
        assert all(len(values) == log['sample_count'] for values in log['data'].values())
        logs[path.name] = {'samples': log['sample_count'], 'curves': len(log['curves'])}
    assert len(logs) == 11
    assert identity_before == paired_cache_identity(root, cache)
    assert cache_stat == {name: ((cache/name).stat().st_size, (cache/name).stat().st_mtime_ns,
                                (cache/name).stat().st_ctime_ns) for name in CACHE_FILES}
    print(json.dumps({'actual_sample': 650, 'paired_cells': int(np.isfinite(attrs.signed_difference).sum()),
        'flagged_cells': int(detection.mask.sum()), 'regions': len(zones),
        'all_representatives_valid': True, 'bounded_real_tile_attribute_parity': True, 'additional_flagged_display_bins_preserved': stride_lost_bins,
        'support_range': [int(attrs.paired_sample_count.min()), int(attrs.paired_sample_count.max())],
        'cache_identity_and_stats_unchanged': True,
        'sample_attributes_detection_loader_seconds': seconds,
        'process_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'capabilities': capability, 'workbook_records': len(records), 'tables': tables, 'logs': logs}, indent=2))


if __name__ == '__main__':
    main()
