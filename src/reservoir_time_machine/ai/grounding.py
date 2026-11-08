"""Constrained interpretation: provider selects reviewed statements, never invents prose/facts.

Open-ended LLM prose cannot be certified by a keyword filter. This initial pathway
accepts only statement codes, evidence IDs and exact copied measurements, then renders
local text. External selection adds no independent scientific verification.
"""
import json
import math
import hashlib
import numpy as np
from datetime import datetime, timezone
from dataclasses import dataclass

STATEMENTS = {
    'change_observed':('calculated_attribute','Paired seismic attributes show a response difference in this bounded candidate.'),
    'screening_interest':('detection','This preliminary candidate requires checking its threshold, spatial extent and valid support; screening does not establish cause.'),
    'dependent_methods':('interpretation','Detector agreement is descriptive; shared attributes make methods dependent.'),
    'alternatives_required':('hypothesis','Acquisition, processing, scaling, noise and registration effects remain alternative explanations.'),
    'reservoir_mechanism_unresolved':('hypothesis','A reservoir-related response is a hypothesis requiring verified registration and independent geological or production evidence.'),
    'recovery_not_established':('interpretation','This finding does not establish missed recovery. It can prioritize evidence collection before evaluating a potential recovery investigation target.'),
    'verify_repeatability':('recommendation','Review acquisition/processing history and verify physical registration and repeatability first.'),
    'weak_support':('limitation','Paired-window or inspection support is limited; spatial coherence alone cannot establish confidence.'),
    'conflicting_detectors':('interpretation','Statistical and ML evidence are inconsistent; review reference populations, thresholds and repeatability before interpreting a reservoir mechanism.'),
    'obtain_independent_context':('recommendation','Obtain a verified well tie, stratigraphic context and production/pressure evidence before evaluating reservoir mechanisms or recovery implications.'),
}

REQUIRED_FACT = {
    'change_observed':'mean_absolute_difference', 'screening_interest':'candidate_membership',
    'dependent_methods':'method_dependence', 'alternatives_required':'alternative_causes',
    'reservoir_mechanism_unresolved':'physical_registration', 'recovery_not_established':'well_association',
    'weak_support':'support_quality', 'conflicting_detectors':'method_agreement',
    'verify_repeatability':'physical_registration', 'obtain_independent_context':'well_association',
}

VALIDATOR_VERSION = "grounding-2.1-1"


@dataclass(frozen=True)
class FindingEvidence:
    finding_id: str
    run_id: str
    facts: tuple
    limitations: tuple
    data_classification: str = 'unknown'

    def __post_init__(self):
        if self.data_classification not in {'unknown','restricted','confidential','public','synthetic'}:
            raise ValueError('Unknown finding data classification')
        ids = [f.get('id') for f in self.facts]
        if not self.finding_id or not self.run_id or not ids or len(ids)!=len(set(ids)) or any(not isinstance(i,str) or not i for i in ids):
            raise ValueError('Unique, nonempty evidence IDs and finding/run IDs required')
        if any(not all(key in fact for key in ('id','name','category','value','source')) for fact in self.facts):
            raise ValueError('Incomplete typed evidence fact')

    def to_dict(self):
        return {'schema_version':'finding-evidence-1','finding_id':self.finding_id,'run_id':self.run_id,
                'data_classification':self.data_classification,'facts':list(self.facts),
                'required_codes':required_codes(self),
                'fact_type_for_code':REQUIRED_FACT,
                'response_schema':response_schema(self),
                'limitations':list(self.limitations),
                'required_response':{'finding_id':self.finding_id,
                    'claims':[{'code':'one allowed code','evidence_ids':['fact id']}],
                    'measurements':[{'evidence_id':'numeric fact id','value':'exact supplied numeric value'}]},
                'allowed_statements':{key:{'category':v[0],'text':v[1]} for key,v in STATEMENTS.items()}}

    def to_json(self):
        payload = json.dumps(self.to_dict(),allow_nan=False,sort_keys=True)
        if len(payload.encode())>24000: raise ValueError('Finding evidence exceeds bounded payload')
        return payload


def finding_packet(result, finding, *, well_context=None):
    if finding['run_id']!=result['run_id'] or finding not in result['findings']:
        raise ValueError('Finding does not belong to this analysis run')
    facts = []
    for name,value in finding['metrics'].items():
        facts.append({'id':finding['id']+':'+name,'category':'observation' if name in {'baseline_amplitude','monitor_amplitude'} else 'calculated_attribute',
                      'name':name,'value':value,'source':result['run_id']+':component:'+str(finding['label'])})
    for name,evidence in finding['method_evidence'].items():
        facts.append({'id':finding['id']+':method:'+name,'category':'statistical_detection' if name=='statistical' else 'ml_detection',
                      'name':name,'value':evidence['flagged_inspection_cells'],'inspection_support':evidence['inspection_support'],
                      'source':result['run_id']+':'+name,'calibration':'uncalibrated'})
    members=(result['labels']==finding['label']) & result['inspection']
    for name,score in result['scores'].items():
        supported=members & np.isfinite(score)
        if supported.any():
            facts.append({'id':finding['id']+':inspection_score:'+name,'category':'statistical_detection' if name=='statistical' else 'ml_detection',
                'name':name+'_mean_inspection_score','value':float(np.mean(score[supported],dtype=np.float64)),
                'source':result['run_id']+':held_out_inspection','calibration':'uncalibrated local score; not probability'})
    fractions = [v['fraction'] for v in finding['method_evidence'].values() if v['fraction'] is not None]
    disagreement = 'conflicting' if any(v>0 for v in fractions) and any(v==0 for v in fractions) else 'unassessed_or_consistent'
    limited = finding['priority_terms']['ml_inspection_coverage'] < .5 or finding['metrics']['minimum_paired_support'] < result['quality']['available_window_samples']
    facts.extend([
        {'id':finding['id']+':support_quality','category':'limitation','name':'support_quality','value':'limited' if limited else 'adequate','source':result['run_id']+':support_policy'},
        {'id':finding['id']+':agreement','category':'detection','name':'method_agreement','value':disagreement,'source':result['run_id']+':inspection_masks'},
        {'id':finding['id']+':dataset','category':'provenance','name':'dataset_identity','value':result['provenance']['dataset_id'],'source':result['run_id']},
        {'id':finding['id']+':analysis','category':'provenance','name':'analysis_provenance',
         'value':{k:result['provenance'][k] for k in ('configuration','software','feature_order','reference_policy','calibration_verified')},'source':result['run_id']},
        {'id':finding['id']+':quality','category':'limitation','name':'support_context','value':result['quality'],'source':result['run_id']+':validity_masks'},
        {'id':finding['id']+':detectors','category':'provenance','name':'detector_context',
         'value':{name:{k:v[k] for k in ('status','threshold','score_meaning','calibration_status') if k in v} for name,v in result['model_status'].items()},'source':result['run_id']},
        {'id':finding['id']+':membership','category':'calculated_attribute','name':'spatial_coherence',
         'value':{'membership_sha256':finding['membership_sha256'],'touches_tile_boundary':finding['touches_tile_boundary'],
                  'meaning':'connected cells in a bounded sample plane; grid indices only'},'source':result['run_id']+':components'},
        {'id':finding['id']+':extent','category':'calculated_attribute','name':'grid_extent','value':finding['bounds'],'source':result['run_id']+':component_membership'},
        {'id':finding['id']+':indices','category':'observation','name':'source_indices','value':finding['representative'],'source':result['run_id']+':component:'+str(finding['label'])},
        {'id':finding['id']+':candidate','category':'detection','name':'candidate_membership','value':finding['metrics']['cells'],'source':result['run_id']+':component:'+str(finding['label'])},
        {'id':finding['id']+':dependence','category':'limitation','name':'method_dependence','value':'shared correlated features; not independent confirmation','source':'feature construction'},
        {'id':finding['id']+':alternatives','category':'hypothesis','name':'alternative_causes','value':'acquisition/processing/scaling/noise/registration','source':'unresolved survey comparability'},
        {'id':finding['id']+':registration','category':'limitation','name':'physical_registration','value':'unverified','source':'dataset capability assessment'},
        {'id':finding['id']+':well','category':'limitation','name':'well_association','value':finding['well_association'],'source':'well eligibility gate'},
    ])
    if well_context is not None:
        facts.append({'id':finding['id']+':well_sources','category':'source_context','name':'available_well_context',
            'value':{'workbook_records':well_context['well_count'],'las_sources':len(well_context['logs']),
                'table_sources':len(well_context['tables']),'association_available':False,
                'missing_association_evidence':well_context['missing_association_evidence'],
                'well_inventory_sha256':hashlib.sha256(json.dumps(well_context,sort_keys=True,allow_nan=False).encode()).hexdigest(),
                'curve_types_and_units':sorted({(c['mnemonic']+' ['+c['unit']+']') for log in well_context['logs'] for c in log['curves']})},
            'source':'local well-source inventory; no spatial or depth/time association'})
    return FindingEvidence(finding['id'],result['run_id'],tuple(facts),tuple(finding['limitations']))


def validate_response(packet, response):
    if isinstance(response,str):
        if len(response.encode())>24000: raise ValueError('Provider response too large')
        response = json.loads(response)
    if not isinstance(response,dict) or set(response)!={'finding_id','claims','measurements'}:
        raise ValueError('Unsupported interpretation fields or free-text claims')
    if response['finding_id']!=packet.finding_id: raise ValueError('Finding ID mismatch')
    facts = {f['id']:f for f in packet.facts}
    if not isinstance(response['claims'],list) or not 1 <= len(response['claims']) <= 16:
        raise ValueError('Bounded claims required')
    mandatory = set(required_codes(packet))
    seen = set()
    for claim in response['claims']:
        if not isinstance(claim,dict) or set(claim)!={'code','evidence_ids'} or claim['code'] not in STATEMENTS:
            raise ValueError('Unsupported claim or reservoir assertion')
        ids = claim['evidence_ids']
        if not isinstance(ids,list) or not ids or any(i not in facts for i in ids):
            raise ValueError('Unknown or missing evidence reference')
        if not any(facts[i]['name']==REQUIRED_FACT[claim['code']] for i in ids):
            raise ValueError('Evidence reference does not support this statement type')
        if claim['code'] in seen or claim['code'] not in mandatory:
            raise ValueError('Duplicate or evidence-inapplicable statement')
        seen.add(claim['code'])
    if mandatory != seen: raise ValueError('Scientific limitations and follow-up must be explicit')
    if not isinstance(response['measurements'],list) or len(response['measurements'])>32:
        raise ValueError('Bounded measurements required')
    for item in response['measurements']:
        if not isinstance(item,dict) or set(item)!={'evidence_id','value'} or item['evidence_id'] not in facts:
            raise ValueError('Unknown measurement reference')
        value,expected = item['value'],facts[item['evidence_id']]['value']
        if (isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value)
                or isinstance(expected,bool) or not isinstance(expected,(int,float)) or value!=expected):
            raise ValueError('Unsupported measurement')
    return response


def deterministic_response(packet):
    references = {f['name']:f['id'] for f in packet.facts}
    return {'finding_id':packet.finding_id,
            'claims':[{'code':code,'evidence_ids':[references[REQUIRED_FACT[code]]]} for code in required_codes(packet)],
            'measurements':[{'evidence_id':f['id'],'value':f['value']} for name in
                            ('mean_nrms_percent','mean_signed_difference','mean_absolute_difference','minimum_paired_support')
                            for f in packet.facts if f['name']==name]}


def interpret(packet, *, provider=None, authorized=False):
    packet.to_json()
    provenance = {'source':'deterministic_local','external_transmission':False,'validation':'constrained_codes_and_exact_evidence',
                  'finding_id':packet.finding_id,'run_id':packet.run_id,'provider_invoked':False,
                  'evidence_sha256':hashlib.sha256(packet.to_json().encode()).hexdigest(),
                  'validator_version':VALIDATOR_VERSION,'cache_hit':False}
    response = deterministic_response(packet)
    if provider is not None and hasattr(provider,'settings') and not transmission_allowed(packet,provider.settings,authorized=authorized):
        provenance['unavailable_reason'] = 'Real evidence transmission disabled: policy confirmation and authorization required'
        authorized = False
    if provider is not None and authorized:
        try:
            response = validate_response(packet,provider(packet))
            provenance.update(source='validated_provider_selection',external_transmission=True,provider_invoked=True,
                              response_sha256=hashlib.sha256(json.dumps(response,sort_keys=True).encode()).hexdigest(),
                              generated_utc=datetime.now(timezone.utc).isoformat())
            if hasattr(provider,'settings'):
                provenance.update(model=provider.settings.model,model_configuration_id=provider.settings.configuration_id())
        except Exception as exc:
            provenance.update(source='deterministic_fallback',external_transmission=None,provider_invoked=True,error_type=type(exc).__name__)
    elif provider is not None:
        provenance.setdefault('unavailable_reason','External provider not authorized; no call made')
    response = validate_response(packet,response)
    return {'provenance':provenance,'claims':[{'category':STATEMENTS[c['code']][0],
             'text':STATEMENTS[c['code']][1],'evidence_ids':c['evidence_ids']} for c in response['claims']],
             'measurements':response['measurements'],'limitations':list(packet.limitations)}


def required_codes(packet):
    values = {f['name']:f['value'] for f in packet.facts}
    codes = [code for code in STATEMENTS if code not in {'weak_support','conflicting_detectors'}]
    if values.get('support_quality')=='limited': codes.append('weak_support')
    if values.get('method_agreement')=='conflicting': codes.append('conflicting_detectors')
    return codes


def response_schema(packet):
    # Dynamic IDs and exact allowed numeric pairs constrain syntax. The local
    # semantic validator still checks which fact type supports each statement.
    numeric = [f for f in packet.facts if isinstance(f['value'],(int,float)) and not isinstance(f['value'],bool)]
    measurement = {'oneOf':[{'type':'object','additionalProperties':False,
        'required':['evidence_id','value'],'properties':{'evidence_id':{'const':f['id']},'value':{'const':f['value']}}}
        for f in numeric]} if numeric else {'type':'object','additionalProperties':False}
    codes = required_codes(packet)
    return {'type':'object','additionalProperties':False,'required':['finding_id','claims','measurements'],
        'properties':{'finding_id':{'const':packet.finding_id},
            'claims':{'type':'array','minItems':len(codes),'maxItems':len(codes),
                'items':{'type':'object','additionalProperties':False,'required':['code','evidence_ids'],
                    'properties':{'code':{'enum':codes},'evidence_ids':{'type':'array','minItems':1,'maxItems':3,
                        'items':{'enum':[f['id'] for f in packet.facts]}}}}},
            'measurements':{'type':'array','minItems':1 if numeric else 0,'maxItems':4,'items':measurement}}}


def explanation_identity(packet, settings):
    return hashlib.sha256((VALIDATOR_VERSION+packet.to_json()+settings.configuration_id()).encode()).hexdigest()


def transmission_allowed(packet, settings, authorized=False):
    synthetic = packet.data_classification=='synthetic' and packet.finding_id.startswith('synthetic-') and packet.run_id.startswith('synthetic-')
    return bool(authorized and (synthetic or settings.real_data_enabled))


class GemmaFindingProvider:
    """Consent plus deployment policy required; synthetic fixtures are explicitly identified."""
    def __init__(self, settings): self.settings = settings

    def __call__(self, packet):
        if not transmission_allowed(packet,self.settings,authorized=True):
            raise PermissionError('Real evidence transmission disabled pending policy and explicit authorization')
        from ..llm.gemma_client import explain_evidence
        return explain_evidence(packet.to_dict(),self.settings,structured=True,response_schema=response_schema(packet))
