"""Registry, coverage, local rank, completed-only loading and evidence report fixtures."""
import copy,json,tempfile
from pathlib import Path
import unittest
import numpy as np
from test_phase2 import fixture
from reservoir_time_machine.analysis.preliminary import run_preliminary
from reservoir_time_machine.analysis.registry import save_registry,load_registry,regular_plan,validate_plan,sha
from reservoir_time_machine.analysis.technical_report import storyboard,report_markdown


class Phase3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        b,m,mask=fixture();cls.shape=list(b.shape)
        cls.result=run_preliminary(b,m,mask,dataset_id='registry-test',sample=10,bounds=(0,64,0,64))
        cls.wells={'well_count':0,'logs':[],'tables':[],'missing_association_evidence':['CRS','time-depth tie'],'errors':[]}
        cls.plan=[{'sample':10,'bounds':[0,64,0,64]},{'sample':3,'bounds':[0,64,0,64]}]

    def make(self,root):return save_registry([self.result],root,dataset_id='registry-test',shape=self.shape,plan=self.plan,wells=self.wells)

    def test_completed_partial_registry_and_reproducible_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=self.make(tmp);r=load_registry(p,'registry-test',self.shape);c=r['body']['coverage']
            self.assertEqual((c['completed_tiles'],c['pending_tiles']),(1,1))
            self.assertEqual(c['analyzed_center_cells'],4096)
            self.assertAlmostEqual(c['center_fraction'],1/21)
            self.assertFalse(r['body']['ranking']['survey_comparable'])
            self.assertEqual(r['entries'][0]['finding']['representative'],self.result['findings'][0]['representative'])
            self.assertEqual(r['id'],load_registry(self.make(tmp),'registry-test',self.shape)['id'])

    def test_checksum_stale_dataset_and_artifact_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=self.make(tmp)
            with self.assertRaises(ValueError):load_registry(p,'other',self.shape)
            body=json.loads(p.read_text());body['body']['coverage']['completed_tiles']=12;p.write_text(json.dumps(body))
            with self.assertRaises(ValueError):load_registry(p,'registry-test',self.shape)
            p=self.make(tmp)
            ledger=json.loads(p.read_text())['body']['local_interpretation_records']
            record=Path(tmp)/'interpretations'/(next(iter(ledger.values()))+'.json')
            original=record.read_bytes();record.write_bytes(b'corrupt')
            with self.assertRaises(ValueError):load_registry(p,'registry-test',self.shape)
            record.write_bytes(original)
            maps=Path(tmp)/'artifacts'/self.result['run_id']/'maps.npz'
            with maps.open('ab') as stream:stream.write(b'corrupt')
            with self.assertRaises(ValueError):load_registry(p,'registry-test',self.shape)

    def test_plan_nonoverlap_and_honest_unanalysed_extent(self):
        plan=regular_plan([1029,1248,1301]);self.assertEqual(len(plan),27)
        validate_plan(plan,[1029,1248,1301])
        with self.assertRaises(ValueError):validate_plan([self.plan[0],self.plan[0]],self.shape)
        with self.assertRaises(ValueError):validate_plan([{'sample':100,'bounds':[0,64,0,64]}],self.shape)
        with self.assertRaises(ValueError):validate_plan([{'sample':10,'bounds':[0,65,0,64]}],[128,128,21])

    def test_report_and_eight_scenes_use_same_ids_numbers_and_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=load_registry(self.make(tmp),'registry-test',self.shape);e=r['entries'][0]
            scenes=storyboard(r,e);self.assertEqual(len(scenes),8)
            self.assertIn(e['id'],scenes[0][1]);self.assertIn(str(e['finding']['representative'][0]),scenes[1][1])
            text=report_markdown(r)
            self.assertIn(e['id'],text);self.assertIn('No survey-wide rank',text)
            self.assertIn('No spatial/depth/time tie established',text)
            self.assertIn('deterministic local',text)
            self.assertIn(str(e['finding']['metrics']['cells']),text)

    def test_empty_completed_plan_and_refusal_to_write_seismic_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=save_registry([],tmp,dataset_id='registry-test',shape=self.shape,plan=self.plan,wells=self.wells)
            r=load_registry(p,'registry-test',self.shape);self.assertEqual(r['entries'],[])
            self.assertEqual(r['body']['coverage']['pending_tiles'],2)
            root=Path(tmp)/'input';root.mkdir();(root/'baseline.npy').write_bytes(b'protected');(root/'monitor.npy').write_bytes(b'protected')
            with self.assertRaises(ValueError):save_registry([],root/'child',dataset_id='registry-test',shape=self.shape,plan=self.plan,wells=self.wells)



class PresentationWorkflowTests(unittest.TestCase):
    def test_timed_next_finding_synchronizes_registry_sections_report_without_fitting(self):
        import time
        import streamlit as st
        from streamlit.testing.v1 import AppTest
        from unittest.mock import patch
        from reservoir_time_machine.models import SurveyMetadata,SeismicVolume
        from reservoir_time_machine.data_ingest.well_evidence import well_evidence_summary
        b,m,mask=fixture();b=np.concatenate((b,b));m=np.concatenate((m,m));mask=np.concatenate((mask,mask))
        results=[run_preliminary(b,m,mask,dataset_id='carousel',sample=10,bounds=v) for v in ((0,64,0,64),(64,128,0,64))]
        self.assertTrue(all(r['findings'] for r in results))
        meta={'paired_grid_available':True,'trace_key_pairing_verified':True,'inline_ids':list(range(128)),
            'crossline_ids':list(range(64)),'paired_trace_count':8064,'baseline_trace_count':8064,'monitor_only_trace_count':0}
        volumes=tuple(SeismicVolume(v,SurveyMetadata(n,n,v.shape,sample_axis_domain='unknown',source_metadata=meta),mask) for n,v in [('b',b),('m',m)])
        st.cache_resource.clear();st.cache_data.clear()
        with tempfile.TemporaryDirectory() as tmp:
            path=save_registry(results,tmp+'/completed',dataset_id='carousel',shape=list(b.shape),
                plan=[{'sample':10,'bounds':list(r['bounds'])} for r in results],wells=well_evidence_summary(tmp))
            from reservoir_time_machine.analysis.registry import load_registry
            from reservoir_time_machine.visualization.finding_animation import finding_animation
            from reservoir_time_machine.visualization.presentation_cache import presentation_identity,save_presentation
            registry=load_registry(path,'carousel',list(b.shape));presentation_root=tmp+'/presentations'
            for entry in registry['entries']:
                identity=presentation_identity(registry['id'],entry['run_id'],entry['finding']['id'],'carousel')
                save_presentation(finding_animation(b,m,mask,entry['result'],entry['finding']),presentation_root,identity)
            with patch.dict('os.environ',{'RTM_CHALLENGE3_ROOT':tmp,'RTM_CACHE_DIR':tmp,'RTM_FINDINGS_REGISTRY':str(path),'RTM_PRESENTATION_CACHE':presentation_root,'RTM_LLM_BASE_URL':'','RTM_LLM_ALLOW_CHALLENGE3':'0','RTM_LLM_DATA_POLICY_CONFIRMED':'0'}), \
                patch('reservoir_time_machine.data_ingest.manifest.challenge3_availability',return_value={'available':True,'well_inventory_available':False}), \
                patch('reservoir_time_machine.data_ingest.cache_identity.paired_cache_identity',return_value='carousel'), \
                patch('reservoir_time_machine.data_ingest.segy_loader.load_challenge3_paired_volumes',return_value=volumes), \
                patch('reservoir_time_machine.analysis.preliminary.run_preliminary',side_effect=AssertionError('Fitting is forbidden on page load')) as fit, \
                patch('reservoir_time_machine.preprocessing.qc.quality_report',side_effect=AssertionError('No full-volume scan on startup')) as qc, \
                patch('reservoir_time_machine.llm.gemma_client._client',side_effect=AssertionError('No external request')) as client:
                app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py'),default_timeout=30).run()
                self.assertFalse(list(app.exception))
                app.sidebar.radio[0].set_value('Survey transition').run()
                old=app.session_state['registry_selection']
                next(v for v in app.button if v.label=='Next scene').click().run()
                self.assertEqual(app.session_state['presentation_scene'],1)
                app.session_state['presentation_scene']=7;app.session_state['presentation_playing']=True
                app.session_state['presentation_last_tick']=time.monotonic()-30;app.run()
                self.assertFalse(list(app.exception),str([v.message for v in app.exception]))
                self.assertNotEqual(app.session_state['registry_selection'],old)
                expected=next(f for r in results for f in r['findings'] if f['id']==app.session_state['registry_selection'])
                self.assertEqual([app.session_state[k] for k in ('shared_inline','shared_crossline','shared_sample')],expected['representative'])
                next(v for v in app.button if v.label=='Pause presentation').click().run()
                self.assertFalse(app.session_state['presentation_playing'])
                fit.assert_not_called();qc.assert_not_called();client.assert_not_called()
        st.cache_resource.clear();st.cache_data.clear()

if __name__=='__main__':unittest.main()
