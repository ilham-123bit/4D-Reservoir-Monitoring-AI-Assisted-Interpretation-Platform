"""Bounded ML, provenance, grounding and no-network regressions (unittest compatible)."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from reservoir_time_machine.analysis.features import extract_features, FEATURE_NAMES, robust_scaling
from reservoir_time_machine.analysis.preliminary import run_preliminary, PreliminaryConfig
from reservoir_time_machine.analysis.artifacts import load_artifact
from reservoir_time_machine.ai.grounding import finding_packet, deterministic_response, validate_response, interpret, GemmaFindingProvider
from reservoir_time_machine.detection.contracts import DetectionRequest, DetectionOutput
from reservoir_time_machine.ai.detectors import BayesianChangeDetector, GaussianChangeAssumptions
from reservoir_time_machine.ai.evidence_fusion import summarize_detection_agreement


def fixture():
    rng = np.random.default_rng(6)
    b = rng.normal(size=(64,64,21)).astype('float32')
    m = (b+rng.normal(scale=.03,size=b.shape)).astype('float32')
    m[28:42,28:42,6:15] *= 2
    missing = np.zeros((64,64),bool); missing[0,:] = True
    b[missing] = np.nan; m[missing] = np.nan
    b[1,1,7:14] = np.nan
    return b,m,missing


class FeatureTests(unittest.TestCase):
    def test_definitions_order_and_boundary_support(self):
        b = np.ones((16,16,12));m = 2*b;missing = np.zeros((16,16),bool)
        missing[2,3] = True
        f = extract_features(b,m,missing,sample=0,bounds=(0,16,0,16))
        self.assertEqual(FEATURE_NAMES,('normalized_difference','nrms_fraction','log_rms_ratio','relative_absolute_difference'))
        np.testing.assert_allclose(f.features[0,0],[1/(1.5+1e-8),1/3,np.log((2+1e-8)/(1+1e-8)),1/(3+1e-8)])
        scaled=extract_features(2*b,2*m,missing,sample=0,bounds=(0,16,0,16))
        self.assertNotEqual(f.content_sha256,scaled.content_sha256)
        self.assertEqual(f.available_window,5)
        self.assertEqual(f.attributes.paired_sample_count[0,0],5)
        self.assertTrue(np.isnan(f.features[2,3]).all());self.assertFalse(f.valid[2,3])

    def test_scaling_reference_only_and_invalid_support_exclusion(self):
        b,m,missing = fixture()
        f = extract_features(b,m,missing,sample=10,bounds=(0,64,0,64))
        self.assertFalse(f.valid[1,1]);self.assertFalse(f.valid[0].any())
        reference = np.array([[1,10],[2,20],[3,30]],float)
        center,scale,_ = robust_scaling(reference)
        np.testing.assert_array_equal(center,[2,20]);self.assertTrue((scale>0).all())
        with self.assertRaises(ValueError): extract_features(b,m,missing,sample=10,bounds=(0,128,0,128))
        for kwargs in [{'max_reference':10000},{'trees':200},{'quantile':np.nan}]:
            with self.assertRaises(ValueError): PreliminaryConfig(**kwargs)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b,cls.m,cls.missing = fixture()
        cls.result = run_preliminary(cls.b,cls.m,cls.missing,dataset_id='math-fixture',sample=10,bounds=(0,64,0,64))

    def test_eligible_models_finiteness_masks_and_reference_separation(self):
        r = self.result
        for name in ['mahalanobis','pca','isolation_forest','gmm']:
            self.assertEqual(r['model_status'][name]['status'],'PRELIMINARY OPERATIONAL')
            self.assertTrue(np.isfinite(r['scores'][name][r['valid']]).all())
            self.assertTrue(np.isnan(r['scores'][name][~r['valid']]).all())
            self.assertFalse(r['masks'][name][~r['valid']].any())
        self.assertFalse((r['fit_mask']&r['threshold_mask']).any())
        self.assertFalse((r['fit_mask']&r['inspection']).any())
        self.assertFalse((r['threshold_mask']&r['inspection']).any())
        self.assertFalse(r['fit_mask'][~r['valid']].any())
        self.assertEqual(r['provenance']['feature_order'],list(FEATURE_NAMES))
        self.assertFalse(r['provenance']['calibration_verified'])
        self.assertFalse(r['score_is_probability'])

    def test_neural_and_bayesian_not_silently_enabled(self):
        self.assertEqual(self.result['model_status']['siamese']['status'],'IMPLEMENTED BUT NOT TRAINED')
        self.assertEqual(self.result['model_status']['conv_autoencoder']['status'],'IMPLEMENTED BUT NOT TRAINED')
        self.assertEqual(self.result['model_status']['bayesian']['status'],'NOT CONFIGURED')
        self.assertNotIn('siamese',self.result['scores'])
        request = DetectionRequest({'x':np.array([[[2.,0.,np.nan]]])})
        output = BayesianChangeDetector('x',GaussianChangeAssumptions(prior_change_probability=.5)).detect(request)
        self.assertAlmostEqual(float(output.score[0,0,0]),2)
        self.assertTrue(np.isnan(output.score[0,0,2]))
        self.assertFalse(output.mask[0,0,2])

    def test_spatial_finding_identity_membership_and_reproducibility(self):
        first = self.result
        second = run_preliminary(self.b,self.m,self.missing,dataset_id='math-fixture',sample=10,bounds=(0,64,0,64))
        self.assertEqual(first['run_id'],second['run_id'])
        self.assertEqual([f['id'] for f in first['findings']],[f['id'] for f in second['findings']])
        self.assertTrue(first['findings'])
        for finding in first['findings']:
            i,x,s = finding['representative']
            self.assertEqual(first['labels'][i,x],finding['label'])
            self.assertEqual(s,10)
            self.assertEqual(finding['metrics']['cells'],int((first['labels']==finding['label']).sum()))
            self.assertEqual(finding['metrics']['baseline_amplitude'],float(self.b[i,x,s]))
            self.assertIn('unavailable',finding['well_association'])
            terms=finding['priority_terms']
            self.assertAlmostEqual(finding['priority'],.5*terms['statistical_fraction']
                +.3*terms['mean_ml_inspection_fraction']*terms['ml_inspection_coverage']
                +.2*terms['relative_grid_extent'])
        for name in first['scores']: np.testing.assert_allclose(first['scores'][name],second['scores'][name],equal_nan=True)
        changed = run_preliminary(self.b,self.m,self.missing,dataset_id='different',sample=10,bounds=(0,64,0,64))
        self.assertNotEqual(first['run_id'],changed['run_id'])

    def test_artifact_cache_identity_roundtrip_and_corruption_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            one = run_preliminary(self.b,self.m,self.missing,dataset_id='cache-fixture',sample=10,bounds=(0,64,0,64),artifact_root=tmp)
            with patch('reservoir_time_machine.analysis.preliminary.MahalanobisDetector',side_effect=AssertionError('cache must bypass fitting')):
                two = run_preliminary(self.b,self.m,self.missing,dataset_id='cache-fixture',sample=10,bounds=(0,64,0,64),artifact_root=tmp)
            self.assertTrue(two['artifact_cache_hit'])
            self.assertEqual(one['findings'],two['findings'])
            np.testing.assert_array_equal(one['labels'],two['labels'])
            target = Path(tmp)/one['run_id']
            with self.assertRaises(ValueError): load_artifact(target,'wrong-run')
            with (target/'maps.npz').open('ab') as stream: stream.write(b'corruption')
            with self.assertRaises(ValueError): load_artifact(target,one['run_id'])

    def test_artifact_refuses_input_cache_and_corrupt_metadata(self):
        from reservoir_time_machine.analysis.artifacts import save_artifact
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'baseline.npy').write_bytes(b'input');(root/'monitor.npy').write_bytes(b'input')
            with self.assertRaises(ValueError):save_artifact(self.result,root/'derivatives')
        with tempfile.TemporaryDirectory() as tmp:
            target=save_artifact(self.result,tmp)
            envelope=json.loads((target/'complete.json').read_text())
            envelope['metadata']['findings'][0]['metrics']['cells']+=1
            (target/'complete.json').write_text(json.dumps(envelope))
            with self.assertRaises(ValueError):load_artifact(target,self.result['run_id'])

    def test_untrained_neural_architecture_tensor_contract_only(self):
        import torch
        from reservoir_time_machine.ai.torch_models import build_siamese_3d,build_conv_autoencoder_3d,ConvAutoencoder3DConfig
        torch.set_num_threads(1)
        source=torch.ones((1,1,16,16,16))
        with torch.inference_mode():
            pair=build_siamese_3d(channels=(4,8)).eval()(source,source)
            reconstruction=build_conv_autoencoder_3d(ConvAutoencoder3DConfig(channels=(4,8))).eval()(source)
        self.assertEqual(tuple(pair.shape),tuple(source.shape));self.assertTrue(torch.isfinite(pair).all())
        self.assertEqual(tuple(reconstruction.shape),tuple(source.shape));self.assertTrue(torch.isfinite(reconstruction).all())
        # These random tensors are discarded; pipeline outputs never include these networks.
        self.assertNotIn('siamese',self.result['scores'])

    def test_no_candidates_or_reference_is_explicit(self):
        b = np.ones((8,8,12));r = run_preliminary(b,b,np.zeros((8,8),bool),dataset_id='flat',sample=5,bounds=(0,8,0,8))
        self.assertEqual(r['findings'],[])
        self.assertEqual(r['model_status']['mahalanobis']['status'],'NOT CONFIGURED')


class GroundingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        b,m,missing = fixture()
        cls.result = run_preliminary(b,m,missing,dataset_id='grounding-fixture',sample=10,bounds=(0,64,0,64))
        cls.packet = finding_packet(cls.result,cls.result['findings'][0])

    def test_local_grounding_and_no_unauthorized_calls(self):
        def forbidden(packet): raise AssertionError('Unauthorized provider called')
        result = interpret(self.packet,provider=forbidden,authorized=False)
        self.assertFalse(result['provenance']['external_transmission'])
        self.assertIn('not authorized',result['provenance']['unavailable_reason'])
        ids = {f['id'] for f in self.packet.facts}
        self.assertTrue(all(set(c['evidence_ids'])<=ids for c in result['claims']))
        self.assertTrue(any(c['category']=='hypothesis' for c in result['claims']))

    def test_rejects_claims_numbers_unknown_refs_and_semantically_wrong_refs(self):
        valid = deterministic_response(self.packet)
        variants=[]
        item=copy.deepcopy(valid);item['claims'][0]['text']='Confirmed fluid migration';variants.append(item)
        item=copy.deepcopy(valid);item['claims'][0]['code']='bypassed_oil_confirmed';variants.append(item)
        item=copy.deepcopy(valid);item['measurements'][0]['value']+=123;variants.append(item)
        item=copy.deepcopy(valid);item['claims'][0]['evidence_ids']=['invented'];variants.append(item)
        item=copy.deepcopy(valid);item['claims'][0]['evidence_ids']=[self.packet.facts[-1]['id']];variants.append(item)
        item=copy.deepcopy(valid);item['finding_id']='wrong';variants.append(item)
        for item in variants:
            with self.assertRaises(ValueError):validate_response(self.packet,item)
        self.assertEqual(interpret(self.packet,provider=lambda p:variants[0],authorized=True)['provenance']['source'],'deterministic_fallback')

    def test_mocked_gemma_end_to_end_and_legacy_import_repair(self):
        from reservoir_time_machine.settings import LLMSettings
        from reservoir_time_machine.ai.interfaces import EvidencePacket,GemmaEvidenceProvider
        response = json.dumps(deterministic_response(self.packet))
        calls=[]
        def create(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=response))])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with patch('reservoir_time_machine.llm.gemma_client._client',return_value=client):
            result=interpret(self.packet,provider=GemmaFindingProvider(LLMSettings(base_url='http://mock.invalid',policy_confirmed=True,allow_challenge3=True,policy_reference='mock-only')),authorized=True)
        self.assertEqual(result['provenance']['source'],'validated_provider_selection')
        self.assertEqual(calls[0]['response_format']['type'],'json_schema')
        with patch('reservoir_time_machine.llm.gemma_client.explain_evidence',return_value='mocked'):
            self.assertEqual(GemmaEvidenceProvider().explain(EvidencePacket('public-test',data_classification='public')),'mocked')

    def test_unavailable_provider_falls_back_without_secret_error_text(self):
        def failing(packet):raise RuntimeError('SECRET-token')
        result=interpret(self.packet,provider=failing,authorized=True)
        self.assertEqual(result['provenance']['error_type'],'RuntimeError')
        self.assertNotIn('SECRET',json.dumps(result))


class AgreementTests(unittest.TestCase):
    def test_joint_validity_and_incompatible_provenance(self):
        shape=(1,1,3)
        def out(name,score,run='same'):
            a=np.array(score).reshape(shape)
            return DetectionOutput(name,'1',a,np.ones(shape,bool),'strength','uncalibrated',{'dataset_id':'a','run_id':run},(),{})
        a=out('a',[1,np.nan,2]);b=out('b',[np.nan,1,2])
        summary=summarize_detection_agreement([a,b])['pairwise_mask_overlap']['a__b']
        self.assertEqual(summary['joint_valid_count'],1);self.assertEqual(summary['intersection_count'],1)
        with self.assertRaises(ValueError):summarize_detection_agreement([a,out('b',[1,1,1],run='different')])


if __name__=='__main__':unittest.main()
