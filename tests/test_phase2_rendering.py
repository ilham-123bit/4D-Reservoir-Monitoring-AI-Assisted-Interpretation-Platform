"""Existing-page integration, stable finding links and mocked-only Gemma interaction."""
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np


class PreliminaryDashboardTests(unittest.TestCase):
    def test_shared_finding_cached_fit_and_explicit_mocked_transmission(self):
        import streamlit as st
        from streamlit.testing.v1 import AppTest
        from reservoir_time_machine.models import SurveyMetadata,SeismicVolume
        from reservoir_time_machine.analysis.preliminary import run_preliminary
        from reservoir_time_machine.ai.grounding import FindingEvidence,deterministic_response
        rng=np.random.default_rng(6);b=rng.normal(size=(64,64,21)).astype('float32')
        m=(b+rng.normal(scale=.03,size=b.shape)).astype('float32');m[28:42,28:42,6:15]*=2
        missing=np.zeros((64,64),bool);missing[0,:]=True;b[missing]=np.nan;m[missing]=np.nan
        meta={'paired_grid_available':True,'trace_key_pairing_verified':True,'inline_ids':list(range(64)),
              'crossline_ids':list(range(64)),'paired_trace_count':4032,'baseline_trace_count':4032,'monitor_only_trace_count':0}
        volumes=tuple(SeismicVolume(v,SurveyMetadata(n,n,v.shape,sample_axis_domain='unknown',source_metadata=meta),missing)
                      for n,v in [('b',b),('m',m)])
        requests=[]
        def create(**kwargs):
            body=json.loads(kwargs['messages'][-1]['content']);requests.append(body)
            packet=FindingEvidence(body['finding_id'],body['run_id'],tuple(body['facts']),tuple(body['limitations']))
            response=json.dumps(deterministic_response(packet))
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=response))])
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        st.cache_resource.clear();st.cache_data.clear()
        with tempfile.TemporaryDirectory() as tmp, patch.dict('os.environ',{'RTM_CHALLENGE3_ROOT':tmp,'RTM_CACHE_DIR':tmp,'RTM_ML_CACHE_DIR':'','RTM_LLM_BASE_URL':'http://mock.invalid','RTM_LLM_DATA_POLICY_CONFIRMED':'1','RTM_LLM_ALLOW_CHALLENGE3':'1','RTM_LLM_DATA_POLICY_REFERENCE':'mock-only','RTM_FINDINGS_REGISTRY':tmp+'/registry/registry.json'}), \
            patch('reservoir_time_machine.data_ingest.manifest.challenge3_availability',return_value={'available':True,'well_inventory_available':False}), \
            patch('reservoir_time_machine.data_ingest.cache_identity.paired_cache_identity',return_value='phase2-ui-fixture'), \
            patch('reservoir_time_machine.data_ingest.segy_loader.load_challenge3_paired_volumes',return_value=volumes), \
            patch('reservoir_time_machine.analysis.preliminary.run_preliminary',wraps=run_preliminary) as analysis, \
            patch('reservoir_time_machine.llm.gemma_client._client',return_value=client):
            from reservoir_time_machine.analysis.registry import save_registry
            from reservoir_time_machine.data_ingest.well_evidence import well_evidence_summary
            seed=run_preliminary(b,m,missing,dataset_id='phase2-ui-fixture',sample=10,bounds=(0,64,0,64))
            save_registry([seed],tmp+'/registry',dataset_id='phase2-ui-fixture',shape=list(b.shape),
                plan=[{'sample':10,'bounds':[0,64,0,64]}],wells=well_evidence_summary(tmp))
            app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py'),default_timeout=30).run()
            app.sidebar.radio[0].set_value('Change detection').run()
            self.assertFalse(list(app.exception));self.assertEqual(requests,[])
            selector=next(x for x in app.selectbox if x.label=='Preliminary finding')
            selector.select_index(0).run()
            next(x for x in app.button if x.label=='Inspect finding in linked sections').click().run()
            saved=app.session_state['selected_preliminary_finding'];finding_id=saved['id'];location=saved['representative']
            app.sidebar.radio[0].set_value('4D comparison').run()
            self.assertFalse(list(app.exception))
            self.assertTrue(any(finding_id in x.value for x in app.caption))
            self.assertEqual([app.session_state[k] for k in ['shared_inline','shared_crossline','shared_sample']],location)
            self.assertEqual(next(x for x in app.metric if x.label=='Baseline amplitude').value,f'{b[tuple(location)]:.5g}')
            app.sidebar.radio[0].set_value('Evidence explanation').run()
            self.assertFalse(list(app.exception));self.assertEqual(requests,[])
            self.assertEqual(app.session_state['selected_preliminary_finding']['id'],finding_id)
            authorize=next(x for x in app.checkbox if x.key=='phase2_transmission')
            authorize.check().run();self.assertEqual(requests,[])
            next(x for x in app.button if x.label=='Request grounded Gemma explanation').click().run()
            self.assertFalse(list(app.exception));self.assertEqual(len(requests),1)
            self.assertEqual(requests[0]['finding_id'],finding_id)
            app.run();self.assertEqual(len(requests),1)
            next(x for x in app.button if x.label=='Request grounded Gemma explanation').click().run()
            self.assertEqual(len(requests),1)  # explicit repeat also reuses validated evidence/model cache
            with patch.dict('os.environ',{'RTM_LLM_MODEL':'mock-other-model'}):
                app.run()
                self.assertTrue(any('deterministic local summary' in x.value for x in app.info))
                next(x for x in app.button if x.label=='Request grounded Gemma explanation').click().run()
                self.assertEqual(len(requests),2)
            with patch.dict('os.environ',{'RTM_LLM_DATA_POLICY_CONFIRMED':'0','RTM_LLM_ALLOW_CHALLENGE3':'0'}):
                app.run()
                self.assertTrue(next(x for x in app.checkbox if x.key=='phase2_transmission').disabled)
                self.assertTrue(next(x for x in app.button if x.label=='Request grounded Gemma explanation').disabled)
                self.assertEqual(len(requests),2)
            self.assertFalse(list(app.exception))
            self.assertEqual(analysis.call_count,0)  # completed registry loading must never fit on reruns
        st.cache_resource.clear();st.cache_data.clear()


if __name__=='__main__':unittest.main()
