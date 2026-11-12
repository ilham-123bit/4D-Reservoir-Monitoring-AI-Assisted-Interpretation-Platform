"""Seven-page AppTest smoke checks on mathematical arrays, without network calls.

These are not browser, full-volume, or geological tests. Optional dashboard packages
must be available. Dataset loading and cache identity are replaced in this process.
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np


class DashboardRenderingTests(unittest.TestCase):
    def test_seven_pages_and_region_navigation(self):
        try:
            import plotly
            import streamlit as st
            from streamlit.testing.v1 import AppTest
        except ImportError:
            self.skipTest('Streamlit unavailable')
        from reservoir_time_machine.models import SurveyMetadata, SeismicVolume
        rng = np.random.default_rng(31)
        b = rng.normal(size=(16, 18, 20)).astype(np.float32)
        m = b.copy(); m[4:8, 5:9, 9] += 8; m[12:15, 13:16, 9] += 10
        missing = np.zeros(b.shape[:2], bool); missing[0, 0] = True
        b[0, 0] = np.nan; m[0, 0] = np.nan
        meta = {'paired_grid_available': True, 'trace_key_pairing_verified': True,
                'physical_4d_registration_verified': False, 'inline_ids': list(range(16)),
                'crossline_ids': list(range(18)), 'baseline_trace_count': 287,
                'paired_trace_count': 287, 'monitor_only_trace_count': 0}
        volumes = tuple(SeismicVolume(v, SurveyMetadata(n, n, v.shape,
                       sample_axis_domain='unknown', source_metadata=meta), missing)
                       for n, v in [('b', b), ('m', m)])
        charts = []
        from streamlit.delta_generator import DeltaGenerator
        original_chart = DeltaGenerator.plotly_chart
        def capture(self, fig, *args, **kwargs):
            charts.append(fig)
            return original_chart(self, fig, *args, **kwargs)
        st.cache_resource.clear()
        with tempfile.TemporaryDirectory() as tmp, patch.dict('os.environ', {
            'RTM_CHALLENGE3_ROOT': tmp, 'RTM_CACHE_DIR': tmp}), \
            patch('reservoir_time_machine.data_ingest.manifest.challenge3_availability',
                  return_value={'available': True, 'well_inventory_available': False}), \
            patch('reservoir_time_machine.data_ingest.cache_identity.paired_cache_identity', return_value='fixture-v1'), \
            patch('reservoir_time_machine.data_ingest.segy_loader.load_challenge3_paired_volumes', return_value=volumes), \
            patch.object(DeltaGenerator, 'plotly_chart', capture), \
            patch('reservoir_time_machine.llm.gemma_client.explain_evidence', side_effect=AssertionError('network path prohibited')):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py'), default_timeout=30).run()
            self.assertEqual(list(app.exception), [])
            overview_text = '\n'.join(v.value for v in app.markdown)
            self.assertLess(overview_text.index('Survey overview & QC'), overview_text.index('Scientific readiness'))
            self.assertIn('Physical 4D registration · N/A', overview_text)
            self.assertNotIn('cause N/A', overview_text)
            self.assertNotIn('Registration N/A', overview_text)
            warning = 'Trace-key pairing is available on the Baseline grid. Physical 4D registration is not yet verified. Interpret paired attributes as preliminary screening only.'
            self.assertIn(warning, [x.value for x in app.warning])
            for page in ['3D explorer', '4D comparison', 'Change detection', 'Survey transition',
                         'Well inventory', 'Evidence explanation']:
                charts.clear()
                app.sidebar.radio[0].set_value(page).run()
                self.assertEqual([e.message for e in app.exception], [], page)
                self.assertIn(warning, [x.value for x in app.warning], page)
                if page == '4D comparison':
                    # Preserve the reviewed compact UI: controls on the page,
                    # no duplicate sidebar widgets/header, collapsed evidence.
                    self.assertEqual(len(app.sidebar.slider), 0)
                    self.assertEqual(len(app.sidebar.selectbox), 0)
                    self.assertEqual([x.label for x in app.slider],
                                     ['Sample Index', 'Inline index', 'Crossline index'])
                    panel = next(x for x in app.expander if x.label == 'Evidence · Seismic / Well / ML / AI')
                    self.assertFalse(panel.proto.expanded)
                    text = '\n'.join(x.value for x in app.markdown)
                    for label in ['Seismic evidence', 'Well evidence', 'ML evidence', 'AI evidence']:
                        self.assertIn(label, text)
                    self.assertNotIn('<div class="rtm-title">Reservoir Time Machine</div>', text)
                    self.assertIn('4D comparison · Reservoir Time Machine', text)
                    # Change controls, inspect all three orientations, and check
                    # selected-point metrics remain tied to the same source cell.
                    next(x for x in app.slider if x.label == 'Sample Index').set_value(0)
                    next(x for x in app.slider if x.label == 'Inline index').set_value(5)
                    next(x for x in app.slider if x.label == 'Crossline index').set_value(6).run()
                    from reservoir_time_machine.attributes.difference import calculate_sample_attributes
                    selected = calculate_sample_attributes(b, m, 0)
                    for orientation in ['Sample plane', 'Inline section', 'Crossline section']:
                        charts.clear()
                        next(x for x in app.radio if x.label == 'Synchronized section').set_value(orientation).run()
                        self.assertEqual([e.message for e in app.exception], [], orientation)
                        self.assertIn(warning, [x.value for x in app.warning])
                        metrics = {x.label: x.value for x in app.metric}
                        self.assertEqual(metrics['Baseline amplitude'], f'{b[5,6,0]:.5g}')
                        self.assertEqual(metrics['Windowed NRMS'], f'{selected.nrms[5,6]:.2f}%')
                        self.assertTrue(any('Window support at selected cell: 5 finite paired samples' in x.value
                                            for x in app.caption))
                        baseline_chart = next(fig for fig in charts if fig.layout.title.text == 'BASELINE')
                        np.testing.assert_array_equal(baseline_chart.data[0].x, baseline_chart.data[1].x)
                        np.testing.assert_array_equal(baseline_chart.data[0].y, baseline_chart.data[1].y)
                    next(x for x in app.slider if x.label == 'Sample Index').set_value(9).run()
                if page == 'Change detection':
                    next(v for v in app.checkbox if v.label=='Run expert full-sample screening').check().run()
                    selector = next(x for x in app.selectbox if x.label == 'Inspect connected region')
                    selector.select_index(1).run()
                    self.assertEqual([e.message for e in app.exception], [])
                    # Evidence layers share source-index coordinates, not display indices.
                    evidence = [fig for fig in charts if 'Baseline evidence' in (fig.layout.title.text or '')][-1]
                    np.testing.assert_array_equal(evidence.data[0].x, evidence.data[1].x)
                    np.testing.assert_array_equal(evidence.data[0].y, evidence.data[1].y)
        st.cache_resource.clear()


if __name__ == '__main__':
    unittest.main()
