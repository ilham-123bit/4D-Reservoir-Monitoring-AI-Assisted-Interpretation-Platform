import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from reservoir_time_machine.analysis.full_field_registry import (
    load_full_field_registry, publish_full_field_registry,
)
from reservoir_time_machine.processing.full_field import _canonical, _hash_file


class FullFieldRegistryTests(unittest.TestCase):
    def _fixture(self, root):
        shape=(4,5,6); source='a'*64; dataset='b'*64
        attr_dir=root/'attributes'; model_dir=root/'models'; attr_dir.mkdir(); model_dir.mkdir()
        outputs={}
        for name in ('signed_difference','absolute_difference','normalized_difference','rms_baseline',
                     'rms_monitor','nrms','paired_sample_count'):
            dtype=np.uint8 if name=='paired_sample_count' else np.float32
            a=np.zeros(shape,dtype=dtype); path=attr_dir/f'{name}.npy'; np.save(path,a)
            outputs[name]={'file':path.name,'size_bytes':path.stat().st_size,'sha256':_hash_file(path),
                           'shape':list(shape),'dtype':str(dtype)}
        attr_body={'schema':'rtm-full-field-attributes-1','status':'complete',
            'configuration_sha256':source,'configuration':{'shape':list(shape),
            'source_domain_shape':list(shape),'source_bounds':[0,4,0,5,0,6]},'outputs':outputs}
        (attr_dir/'complete.json').write_text(json.dumps({'body':attr_body,'sha256':hashlib.sha256(_canonical(attr_body)).hexdigest()}))
        model_outputs={}
        for name in ('mahalanobis-score.npy','mahalanobis-flag.npy'):
            a=np.zeros(shape,dtype=np.float32 if 'score' in name else np.uint8);np.save(model_dir/name,a)
            model_outputs[name]={'file':name,'size_bytes':(model_dir/name).stat().st_size,
                'sha256':_hash_file(model_dir/name),'shape':list(shape),'dtype':str(a.dtype)}
        model_config={'source_identity':source,'models':['mahalanobis']}
        model_config_sha=hashlib.sha256(_canonical(model_config)).hexdigest()
        (model_dir/'manifest.json').write_text(json.dumps({'configuration':model_config,
            'configuration_sha256':model_config_sha}))
        model_body={'schema':'rtm-full-field-model-scores-1','status':'complete',
            'configuration_sha256':model_config_sha,'outputs':model_outputs}
        (model_dir/'complete.json').write_text(json.dumps(model_body))
        labels=np.zeros(shape,dtype=np.int32); labels[1,2,3]=1
        labels_path=root/'component_labels.npy';np.save(labels_path,labels)
        member=np.ones((1,1,1),dtype=bool)
        finding={'id':'field-test-0001','component_label':1,'source_identity':source,
            'array_location':[1,2,3], 'segy_location':{'inline_id':101,'crossline_id':202,'sample_index':3},
            'extent_in_array_indices':[[1,1],[2,2],[3,3]],'component_voxels':1,
            'member_mask_sha256':hashlib.sha256(member.tobytes()).hexdigest(),
            'measurements':{'mean_absolute_difference':0.1,'mean_signed_difference':0.1,
                'mean_nrms_percent':2.0,'minimum_paired_sample_support':9},
            'investigation_priority_index':0.7,
            'detectors':{'mahalanobis':{'score_is_probability':False}},'edge_metrics':{'status':'computed'},
            'edge_artifact_risk':0.1}
        findings={'schema':'rtm-full-field-findings-1','status':'complete_for_supplied_full_volume',
            'source_identity':source,'array_shape':list(shape),'processed_voxels':int(np.prod(shape)),
            'valid_voxels':int(np.prod(shape))-1,'coverage_fraction':(int(np.prod(shape))-1)/int(np.prod(shape)),
            'finding_count':1,'findings':[finding],'model_names':['mahalanobis'],
            'ranking_scope':'full-volume screening; priority is not probability',
            'limitations':['registration unverified']}
        findings_path=root/'findings.json';findings_path.write_text(json.dumps(findings))
        return shape,source,dataset,attr_dir,model_dir,labels_path,findings_path,findings

    def test_publish_loads_full_volume_and_preserves_finding_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);shape,source,dataset,ad,md,labels,fp,findings=self._fixture(root)
            published=publish_full_field_registry(findings_path=fp,attributes_directory=ad,
                model_directory=md,labels_path=labels,output_root=root/'registry',dataset_identity=dataset,
                shape=shape,inline_ids=[100,101,102,103],crossline_ids=[200,201,202,203,204],
                well_context={'association_available':False,'well_count':8})
            loaded=load_full_field_registry(published['pointer_path'],dataset_identity=dataset,shape=shape,
                inline_ids=[100,101,102,103],crossline_ids=[200,201,202,203,204])
            self.assertEqual(loaded['entries'][0]['id'],'field-test-0001')
            self.assertEqual(loaded['entries'][0]['finding']['array_location'],[1,2,3])
            self.assertEqual(loaded['entries'][0]['interpretation']['source'],'deterministic_local_full_field_interpretation')
            self.assertEqual(len(loaded['entries'][0]['interpretation']['actions']),3)

    def test_rejects_partial_findings_and_wrong_header_mapping(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);shape,source,dataset,ad,md,labels,fp,findings=self._fixture(root)
            findings['processed_voxels']=1
            fp.write_text(json.dumps(findings))
            with self.assertRaises(ValueError):
                publish_full_field_registry(findings_path=fp,attributes_directory=ad,model_directory=md,
                    labels_path=labels,output_root=root/'registry',dataset_identity=dataset,shape=shape,
                    inline_ids=[100,101,102,103],crossline_ids=[200,201,202,203,204],well_context={})

    def test_loader_rejects_changed_dataset_or_axes(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);shape,source,dataset,ad,md,labels,fp,findings=self._fixture(root)
            result=publish_full_field_registry(findings_path=fp,attributes_directory=ad,model_directory=md,
                labels_path=labels,output_root=root/'registry',dataset_identity=dataset,shape=shape,
                inline_ids=[100,101,102,103],crossline_ids=[200,201,202,203,204],well_context={})
            with self.assertRaises(ValueError):
                load_full_field_registry(result['pointer_path'],dataset_identity='c'*64,shape=shape,
                    inline_ids=[100,101,102,103],crossline_ids=[200,201,202,203,204])
            with self.assertRaises(ValueError):
                load_full_field_registry(result['pointer_path'],dataset_identity=dataset,shape=shape,
                    inline_ids=[100,101,102,103],crossline_ids=[200,201,203,204,205])

    def test_failed_publication_preserves_last_valid_pointer(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);shape,source,dataset,ad,md,labels,fp,findings=self._fixture(root)
            output=root/'registry'
            result=publish_full_field_registry(findings_path=fp,attributes_directory=ad,
                model_directory=md,labels_path=labels,output_root=output,dataset_identity=dataset,
                shape=shape,inline_ids=[100,101,102,103],crossline_ids=[200,201,202,203,204],well_context={})
            pointer=Path(result['pointer_path']);before=pointer.read_bytes()
            findings['processed_voxels']=1;fp.write_text(json.dumps(findings))
            with self.assertRaises(ValueError):
                publish_full_field_registry(findings_path=fp,attributes_directory=ad,
                    model_directory=md,labels_path=labels,output_root=output,dataset_identity=dataset,
                    shape=shape,inline_ids=[100,101,102,103],crossline_ids=[200,201,202,203,204],well_context={})
            self.assertEqual(pointer.read_bytes(),before)

    def test_streamlit_pages_share_selected_full_field_finding_without_fitting(self):
        import streamlit as st
        from streamlit.testing.v1 import AppTest
        from unittest.mock import patch
        from reservoir_time_machine.models import SeismicVolume,SurveyMetadata
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);shape,source,dataset,ad,md,labels,fp,findings=self._fixture(root)
            published=publish_full_field_registry(findings_path=fp,attributes_directory=ad,
                model_directory=md,labels_path=labels,output_root=root/'registry',dataset_identity=dataset,
                shape=shape,inline_ids=[100,101,102,103],crossline_ids=[200,201,202,203,204],well_context={'association_available':False})
            b=np.ones(shape,dtype=np.float32);m=b.copy();m[1,2,3]=1.1;mask=np.zeros(shape[:2],dtype=bool)
            meta={'paired_grid_available':True,'trace_key_pairing_verified':True,
                'inline_ids':[100,101,102,103],'crossline_ids':[200,201,202,203,204],
                'paired_trace_count':20,'baseline_trace_count':20,'monitor_only_trace_count':0}
            volumes=tuple(SeismicVolume(a,SurveyMetadata(n,n,shape,sample_axis_domain='unknown',source_metadata=meta),mask)
                          for n,a in [('base',b),('monitor',m)])
            st.cache_resource.clear();st.cache_data.clear()
            with patch.dict('os.environ',{'RTM_CHALLENGE3_ROOT':temp,'RTM_CACHE_DIR':temp,
                    'RTM_FINDINGS_REGISTRY':str(root/'legacy-registry.json'),
                    'RTM_FULL_FIELD_REGISTRY':published['pointer_path'],'RTM_LLM_BASE_URL':'',
                    'RTM_LLM_ALLOW_CHALLENGE3':'0','RTM_LLM_DATA_POLICY_CONFIRMED':'0'}), \
                 patch('reservoir_time_machine.data_ingest.manifest.challenge3_availability',return_value={'available':True,'well_inventory_available':False}), \
                 patch('reservoir_time_machine.data_ingest.cache_identity.paired_cache_identity',return_value=dataset), \
                 patch('reservoir_time_machine.data_ingest.segy_loader.load_challenge3_paired_volumes',return_value=volumes), \
                 patch('reservoir_time_machine.analysis.preliminary.run_preliminary',side_effect=AssertionError('No page-load fitting')):
                app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py'),default_timeout=30).run()
                self.assertFalse(list(app.exception),str([x.message for x in app.exception]))
                location=[app.session_state[k] for k in ('shared_inline','shared_crossline','shared_sample')]
                self.assertEqual(location,[1,2,3])
                self.assertTrue(any('field-test-0001' in str(x.value) for x in app.markdown))
                for page in ['3D explorer','4D comparison','Change detection','Survey transition','Well inventory','Evidence explanation']:
                    app.sidebar.radio[0].set_value(page).run()
                    self.assertFalse(list(app.exception),page+': '+str([x.message for x in app.exception]))
                    self.assertEqual([app.session_state[k] for k in ('shared_inline','shared_crossline','shared_sample')],location)
                    self.assertTrue(any('field-test-0001' in str(x.value) for x in app.markdown),page)
            st.cache_resource.clear();st.cache_data.clear()


if __name__=='__main__': unittest.main()
