"""Phase 2.1: synthetic-only transport, grounding, consent, cache and failure tests."""
import copy
import json
import tempfile
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from reservoir_time_machine.settings import LLMSettings
from reservoir_time_machine.ai.synthetic_evidence import synthetic_cases
from reservoir_time_machine.ai.grounding import (FindingEvidence, GemmaFindingProvider, interpret,
    deterministic_response, validate_response, required_codes, response_schema,
    explanation_identity, transmission_allowed)
from reservoir_time_machine.llm.gemma_client import explain_evidence, request_slot, ServiceBusy, test_connection as check_connection


class GemmaSettingsTests(unittest.TestCase):
    def test_environment_and_default_policy_disabled(self):
        with patch.dict('os.environ',{},clear=True):
            settings=LLMSettings.from_environment()
        self.assertFalse(settings.real_data_enabled)
        self.assertIsNone(settings.base_url)
        self.assertEqual(settings.model,'google/gemma-4-26B-A4B-it')
        with patch.dict('os.environ',{'RTM_LLM_BASE_URL':'http://mock.invalid/v1','RTM_LLM_TIMEOUT':'12',
              'RTM_LLM_MAX_TOKENS':'512','RTM_LLM_DATA_POLICY_CONFIRMED':'0','RTM_LLM_ALLOW_CHALLENGE3':'0'}):
            settings=LLMSettings.from_environment()
        self.assertEqual(settings.timeout_seconds,12)
        self.assertEqual(settings.max_tokens,512)
        self.assertFalse(settings.real_data_enabled)

    def test_invalid_settings_fail_closed(self):
        for kwargs in ({'timeout_seconds':float('nan')},{'timeout_seconds':121},{'max_tokens':4097},
                       {'max_tokens':True},{'policy_confirmed':'false'},{'timeout_seconds':True},{'structured_mode':'arbitrary'},
                       {'base_url':'http://user:secret@mock.invalid'},{'base_url':'file:///tmp/api'}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):LLMSettings(**kwargs)
        with patch.dict('os.environ',{'RTM_LLM_ALLOW_CHALLENGE3':'perhaps'}):
            with self.assertRaises(ValueError):LLMSettings.from_environment()

    def test_policy_requires_all_three_conditions(self):
        base=LLMSettings(base_url='http://mock.invalid')
        self.assertFalse(replace(base,allow_challenge3=True).real_data_enabled)
        self.assertFalse(replace(base,allow_challenge3=True,policy_confirmed=True).real_data_enabled)
        self.assertTrue(replace(base,allow_challenge3=True,policy_confirmed=True,policy_reference='mock authorization').real_data_enabled)

    def test_identity_changes_with_evidence_model_endpoint_and_validator_configuration(self):
        packet=synthetic_cases()[0];settings=LLMSettings(base_url='http://mock.invalid')
        identity=explanation_identity(packet,settings)
        for changed in (replace(settings,model='other'),replace(settings,base_url='http://other.invalid'),
                        replace(settings,max_tokens=512),replace(settings,structured_mode='json_object')):
            self.assertNotEqual(identity,explanation_identity(packet,changed))
        self.assertEqual(identity,explanation_identity(packet,replace(settings,api_key='SECRET')))
        self.assertNotIn('SECRET',settings.configuration_id())
        self.assertNotEqual(identity,explanation_identity(synthetic_cases()[1],settings))


class GemmaGroundingTests(unittest.TestCase):
    def test_four_controlled_cases_have_all_required_limits_and_exact_measurements(self):
        for packet in synthetic_cases():
            with self.subTest(packet=packet.finding_id):
                self.assertLess(len(packet.to_json().encode()),24000)
                response=validate_response(packet,deterministic_response(packet))
                self.assertEqual({c['code'] for c in response['claims']},set(required_codes(packet)))
                schema=response_schema(packet)
                self.assertEqual(schema['properties']['finding_id']['const'],packet.finding_id)
                explanation=interpret(packet)
                self.assertFalse(explanation['provenance']['external_transmission'])
                self.assertEqual(explanation['provenance']['source'],'deterministic_local')
                self.assertIn('validator_version',explanation['provenance'])
        self.assertIn('weak_support',required_codes(synthetic_cases()[1]))
        self.assertIn('conflicting_detectors',required_codes(synthetic_cases()[2]))

    def test_reject_fabricated_measurement_reference_claim_duplicates_and_wrong_semantics(self):
        packet=synthetic_cases()[2];valid=deterministic_response(packet);bad=[]
        value=copy.deepcopy(valid);value['measurements'][0]['value']+=1;bad.append(value)
        value=copy.deepcopy(valid);value['claims'][0]['evidence_ids']=['invented'];bad.append(value)
        value=copy.deepcopy(valid);value['claims'][0]['code']='confirmed_fluid_migration';bad.append(value)
        value=copy.deepcopy(valid);value['claims'][0]['text']='This proves bypassed oil';bad.append(value)
        value=copy.deepcopy(valid);value['claims'].append(value['claims'][0]);bad.append(value)
        value=copy.deepcopy(valid);value['claims'][0]['evidence_ids']=[packet.facts[-1]['id']];bad.append(value)
        value=copy.deepcopy(valid);value['claims']=[c for c in value['claims'] if c['code']!='conflicting_detectors'];bad.append(value)
        for response in bad:
            with self.subTest(response=response),self.assertRaises(ValueError):validate_response(packet,response)
            result=interpret(packet,provider=lambda p:response,authorized=True)
            self.assertEqual(result['provenance']['source'],'deterministic_fallback')
            self.assertNotIn('This proves bypassed oil',json.dumps(result))

    def test_unique_fact_ids_required(self):
        packet=synthetic_cases()[0]
        with self.assertRaises(ValueError):replace(packet,facts=packet.facts+(packet.facts[0],))

    def test_real_or_unclassified_packets_cannot_call_client_without_policy(self):
        packet=replace(synthetic_cases()[0],data_classification='unknown')
        settings=LLMSettings(base_url='http://mock.invalid')
        with patch('reservoir_time_machine.llm.gemma_client._client',side_effect=AssertionError('No transmission')) as client:
            result=interpret(packet,provider=GemmaFindingProvider(settings),authorized=True)
            self.assertEqual(result['provenance']['source'],'deterministic_local')
            self.assertFalse(result['provenance']['provider_invoked']);client.assert_not_called()
            with self.assertRaises(PermissionError):GemmaFindingProvider(settings)(packet)
            with self.assertRaises(PermissionError):explain_evidence({'data_classification':'unknown'},settings)
        self.assertFalse(transmission_allowed(synthetic_cases()[0],settings,authorized=False))
        self.assertFalse(transmission_allowed(replace(synthetic_cases()[0],finding_id='real-finding'),settings,authorized=True))

    def test_mocked_end_to_end_four_cases_have_gemma_provenance_and_strict_schema(self):
        settings=LLMSettings(base_url='http://mock.invalid/v1')
        for packet in synthetic_cases():
            calls=[]
            def create(**kwargs):
                calls.append(kwargs)
                return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',
                    message=SimpleNamespace(content=json.dumps(deterministic_response(packet))))])
            client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
            with patch('reservoir_time_machine.llm.gemma_client._client',return_value=client):
                result=interpret(packet,provider=GemmaFindingProvider(settings),authorized=True)
            self.assertEqual(len(calls),1);self.assertEqual(calls[0]['max_tokens'],2048)
            self.assertEqual(calls[0]['temperature'],0)
            self.assertEqual(calls[0]['response_format']['type'],'json_schema')
            self.assertEqual(result['provenance']['source'],'validated_provider_selection')
            self.assertEqual(result['provenance']['model'],settings.model)
            self.assertEqual(len(result['provenance']['response_sha256']),64)


class GemmaTransportTests(unittest.TestCase):
    def test_client_has_timeout_and_zero_retries(self):
        from reservoir_time_machine.llm.gemma_client import _client
        with patch('openai.OpenAI') as factory:
            _client(LLMSettings(base_url='http://mock.invalid',timeout_seconds=12))
        self.assertEqual(factory.call_args.kwargs['timeout'],12)
        self.assertEqual(factory.call_args.kwargs['max_retries'],0)

    def test_concurrent_request_is_rejected_before_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings=LLMSettings(base_url='http://mock.invalid',lock_directory=tmp)
            with request_slot(settings),patch('reservoir_time_machine.llm.gemma_client._client') as client:
                with self.assertRaises(ServiceBusy):GemmaFindingProvider(settings)(synthetic_cases()[0])
                client.assert_not_called()
            with request_slot(settings):pass  # lock released, including on failure

    def test_timeout_connection_and_invalid_json_return_sanitized_fallback(self):
        packet=synthetic_cases()[0];settings=LLMSettings(base_url='http://mock.invalid')
        for error in (TimeoutError('SECRET'),ConnectionError('SECRET')):
            with patch('reservoir_time_machine.llm.gemma_client._client',side_effect=error) as client:
                result=interpret(packet,provider=GemmaFindingProvider(settings),authorized=True)
            self.assertEqual(result['provenance']['source'],'deterministic_fallback')
            self.assertEqual(client.call_count,1);self.assertNotIn('SECRET',json.dumps(result))
        result=interpret(packet,provider=lambda p:'not json SECRET',authorized=True)
        self.assertEqual(result['provenance']['source'],'deterministic_fallback')
        self.assertNotIn('SECRET',json.dumps(result))
        # CLI must not report a safe fallback as a successful live validation.
        import importlib.util
        from pathlib import Path
        spec=importlib.util.spec_from_file_location('synthetic_cli',Path(__file__).resolve().parents[1]/'scripts/validate_gemma_synthetic.py')
        cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
        with tempfile.TemporaryDirectory() as tmp,patch.dict('os.environ',{
            'RTM_LLM_BASE_URL':'http://mock.invalid','RTM_LLM_DATA_POLICY_CONFIRMED':'0','RTM_LLM_ALLOW_CHALLENGE3':'0'}), \
            patch('sys.argv',['validate_gemma_synthetic.py','--live','--output',tmp+'/results']), \
            patch('reservoir_time_machine.llm.gemma_client._client',side_effect=RuntimeError('SECRET')) as client:
            with self.assertRaises(SystemExit) as stopped:cli.main()
            self.assertEqual(stopped.exception.code,2)
            summary=json.loads(Path(tmp+'/results/summary.json').read_text())
            self.assertEqual(summary['requests_attempted'],1)
            self.assertEqual(client.call_count,1)
            self.assertNotIn('SECRET',json.dumps(summary))

    def test_truncation_empty_oversize_and_json_object_mode(self):
        settings=LLMSettings(base_url='http://mock.invalid',structured_mode='json_object')
        for content,reason in (('{}','length'),('', 'stop'),('x'*24001,'stop')):
            client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs:
                SimpleNamespace(choices=[SimpleNamespace(finish_reason=reason,message=SimpleNamespace(content=content))]))))
            with patch('reservoir_time_machine.llm.gemma_client._client',return_value=client):
                with self.assertRaises(ValueError):GemmaFindingProvider(settings)(synthetic_cases()[0])
        with patch('reservoir_time_machine.llm.gemma_client._client') as client:
            with self.assertRaises(ValueError):explain_evidence({'x':'x'*24001},settings)
            client.assert_not_called()
        mismatched=SimpleNamespace(model='wrong-model',choices=[SimpleNamespace(message=SimpleNamespace(content='{}'))])
        with patch('reservoir_time_machine.llm.gemma_client._client',return_value=SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs:mismatched)))):
            with self.assertRaises(ValueError):GemmaFindingProvider(settings)(synthetic_cases()[0])
        calls=[]
        def create(**kwargs):
            calls.append(kwargs);return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{}'))])
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with patch('reservoir_time_machine.llm.gemma_client._client',return_value=client):
            GemmaFindingProvider(settings)(synthetic_cases()[0])
        self.assertEqual(calls[0]['response_format'],{'type':'json_object'})

    def test_model_list_is_explicit_sanitized_and_not_generation_validation(self):
        settings=LLMSettings(base_url='http://mock.invalid')
        client=SimpleNamespace(models=SimpleNamespace(list=lambda:SimpleNamespace(data=[SimpleNamespace(id=settings.model)])))
        with patch('reservoir_time_machine.llm.gemma_client._client',return_value=client):
            result=check_connection(settings)
        self.assertTrue(result['configured_model_listed'])
        with patch('reservoir_time_machine.llm.gemma_client._client',side_effect=RuntimeError('SECRET')):
            result=check_connection(settings)
        self.assertNotIn('SECRET',json.dumps(result));self.assertFalse(result['connected'])


if __name__=='__main__':unittest.main()
