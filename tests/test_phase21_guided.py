"""Report-first, automatic member focus, source wells and endpoint animation tests."""
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from test_phase2 import fixture
from reservoir_time_machine.analysis.preliminary import run_preliminary
from reservoir_time_machine.analysis.guided_report import automatic_location,page_report
from reservoir_time_machine.attributes.difference import calculate_sample_attributes
from reservoir_time_machine.detection.statistical import statistical_screen
from reservoir_time_machine.detection.clustering import connected_zones
from reservoir_time_machine.visualization.finding_animation import finding_animation
from reservoir_time_machine.ai.grounding import finding_packet


class GuidedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b,cls.m,cls.missing=fixture()
        cls.result=run_preliminary(cls.b,cls.m,cls.missing,dataset_id='guided-fixture',sample=10,bounds=(0,64,0,64))
        cls.finding=cls.result['findings'][0]
        cls.wells={'well_count':1,'logs':[{'source':'synthetic.las','curves':[{'mnemonic':'GR','unit':'API'}]}],
            'tables':[{'source':'markers.txt','records':3}],'missing_association_evidence':['shared CRS','time-depth tie'],'association_available':False,'errors':[]}

    def test_automatic_location_is_valid_member_and_empty_is_explicit(self):
        attrs=calculate_sample_attributes(self.b,self.m,10)
        screen=statistical_screen(attrs.absolute_difference,threshold_sigma=3.5)
        labels,zones=connected_zones(screen.mask,screen.score,min_size=8)
        focus=automatic_location(labels,zones,screen.score)
        self.assertEqual(labels[tuple(focus['location'])],focus['label'])
        self.assertIsNone(automatic_location(labels,[],screen.score))

    def test_shared_reports_and_typed_packet_include_wells_without_implied_tie(self):
        for page in ('Overview and QC','3D explorer','4D comparison','Change detection','Survey transition','Well inventory','Evidence explanation'):
            report=page_report(self.result,self.finding,self.wells,page,{'sample':10})
            self.assertEqual(report['finding_id'],self.finding['id'])
            self.assertFalse(report['wells']['association_available'])
            self.assertIn('GR [API]',report['wells']['curve_types'])
            self.assertFalse(report['interpretation']['provenance']['external_transmission'])
        packet=finding_packet(self.result,self.finding,well_context=self.wells)
        fact=next(f for f in packet.facts if f['name']=='available_well_context')
        self.assertFalse(fact['value']['association_available'])
        self.assertLess(len(packet.to_json().encode()),24000)

    def test_animation_endpoints_coordinates_missing_and_component_membership(self):
        f=self.finding;r=self.result;i,x,s=f['representative']
        fig=finding_animation(self.b,self.m,self.missing,r,f)
        self.assertEqual([v.name for v in fig.frames],['Baseline','Monitor'])
        self.assertEqual(len(fig.frames[0].traces),3)
        np.testing.assert_allclose(fig.frames[0].data[0].z,self.b[:,:,s],equal_nan=True)
        np.testing.assert_allclose(fig.frames[1].data[0].z,np.where(np.isfinite(self.b[:,:,s]),self.m[:,:,s],np.nan),equal_nan=True)
        self.assertTrue(np.isnan(np.asarray(fig.data[0].z)[0]).all())
        np.testing.assert_array_equal(fig.data[0].x,np.arange(64))
        np.testing.assert_array_equal(fig.data[-1].z==1,r['labels']==f['label'])
        for frame in fig.frames:
            self.assertIn('inline '+str(i),frame.layout.title.text)
            self.assertIn('crossline '+str(x),frame.layout.title.text)
        bad={**f,'representative':[0,0,s]}
        with self.assertRaises(ValueError):finding_animation(self.b,self.m,self.missing,r,bad)

    def test_bad_marker_is_reported_without_losing_other_sources(self):
        from reservoir_time_machine.data_ingest.well_evidence import well_evidence_summary
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'well_data/markers').mkdir(parents=True);(root/'well_data/markers/bad.txt').write_text('bad')
            with patch('reservoir_time_machine.data_ingest.well_evidence.read_well_table',side_effect=ValueError('bad')):
                report=well_evidence_summary(root)
            self.assertEqual(report['errors'],[{'source':'bad.txt','type':'ValueError'}])
            self.assertFalse(report['association_available'])

    def test_dashboard_opens_report_and_focused_animation_without_manual_search(self):
        import streamlit as st
        from streamlit.testing.v1 import AppTest
        from reservoir_time_machine.models import SurveyMetadata,SeismicVolume
        meta={'paired_grid_available':True,'trace_key_pairing_verified':True,'inline_ids':list(range(64)),'crossline_ids':list(range(64)),
            'paired_trace_count':4032,'baseline_trace_count':4032,'monitor_only_trace_count':0}
        volumes=tuple(SeismicVolume(v,SurveyMetadata(n,n,v.shape,sample_axis_domain='unknown',source_metadata=meta),self.missing)
                      for n,v in [('b',self.b),('m',self.m)])
        figures=[]
        from streamlit.delta_generator import DeltaGenerator
        original=DeltaGenerator.plotly_chart
        original_module=st.plotly_chart
        def capture_module(fig,*args,**kwargs):figures.append(fig);return original_module(fig,*args,**kwargs)
        def capture(obj,fig,*args,**kwargs):figures.append(fig);return original(obj,fig,*args,**kwargs)
        st.cache_resource.clear();st.cache_data.clear()
        with tempfile.TemporaryDirectory() as tmp,patch.dict('os.environ',{'RTM_CHALLENGE3_ROOT':tmp,'RTM_CACHE_DIR':tmp,'RTM_ML_CACHE_DIR':'','RTM_LLM_BASE_URL':'','RTM_LLM_ALLOW_CHALLENGE3':'0','RTM_LLM_DATA_POLICY_CONFIRMED':'0','RTM_FINDINGS_REGISTRY':tmp+'/registry/registry.json'}), \
            patch('reservoir_time_machine.data_ingest.manifest.challenge3_availability',return_value={'available':True,'well_inventory_available':True}), \
            patch('reservoir_time_machine.data_ingest.cache_identity.paired_cache_identity',return_value='guided-ui'), \
            patch('reservoir_time_machine.data_ingest.segy_loader.load_challenge3_paired_volumes',return_value=volumes), \
            patch('reservoir_time_machine.data_ingest.well_evidence.well_evidence_summary',return_value=self.wells), \
            patch('reservoir_time_machine.llm.gemma_client._client',side_effect=AssertionError('External request forbidden')) as client, \
            patch.object(DeltaGenerator,'plotly_chart',capture),patch.object(st,'plotly_chart',capture_module):
            from reservoir_time_machine.analysis.registry import save_registry
            seed=run_preliminary(self.b,self.m,self.missing,dataset_id='guided-ui',sample=10,bounds=(0,64,0,64))
            save_registry([seed],tmp+'/registry',dataset_id='guided-ui',shape=list(self.b.shape),
                plan=[{'sample':10,'bounds':[0,64,0,64]}],wells=self.wells)
            app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py'),default_timeout=30).run()
            self.assertFalse(list(app.exception))
            focus=[app.session_state[k] for k in ('shared_inline','shared_crossline','shared_sample')]
            for page in ('Overview and QC','3D explorer','4D comparison','Change detection','Survey transition','Well inventory','Evidence explanation'):
                figures.clear();app.sidebar.radio[0].set_value(page).run()
                self.assertFalse(list(app.exception),page)
                self.assertTrue(any('Automated evidence report' in v.value for v in app.markdown),page)
                self.assertTrue(any('1 workbook records' in v.value for v in app.markdown),page)
                if page=='Survey transition':
                    self.assertTrue(any(len(v.frames)==2 for v in figures),str([(v.layout.title.text,len(v.frames)) for v in figures])+str([v.value for v in app.info]))
                    fig=next(v for v in figures if len(v.frames)==2)
                    for trace in fig.data:
                        if trace.name=='Finding representative':self.assertEqual(len(trace.x),1)
                    self.assertEqual([v.name for v in fig.frames],['Baseline','Monitor'])
            client.assert_not_called()
        st.cache_resource.clear();st.cache_data.clear()


if __name__=='__main__':unittest.main()
