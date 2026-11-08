"""Live development test: synthetic constants only. Never opens Challenge 3 files.

Explicit --live and output directory required for network access. Four cases maximum,
sequential requests, no retries. No endpoint/key is stored in the output logs.
"""
import argparse
import json
from pathlib import Path
import time
from reservoir_time_machine.ai.synthetic_evidence import synthetic_cases
from reservoir_time_machine.ai.grounding import GemmaFindingProvider,interpret,validate_response,deterministic_response
from reservoir_time_machine.settings import LLMSettings


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--case',choices=[p.finding_id for p in synthetic_cases()])
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    settings=LLMSettings.from_environment()
    packets=[p for p in synthetic_cases() if args.case is None or p.finding_id==args.case]
    results=[]
    for packet in packets:
        captured={}
        def provider(p):
            captured['raw']=GemmaFindingProvider(settings)(p)
            return captured['raw']
        start=time.perf_counter()
        result=interpret(packet,provider=provider if args.live else None,authorized=args.live)
        # Offline malicious response injection validates rejection/fallback without another query.
        fabricated=deterministic_response(packet);fabricated['measurements'][0]['value']+=12345
        try:
            validate_response(packet,fabricated)
            raise AssertionError('Fabricated measurement accepted')
        except ValueError: pass
        prohibited=deterministic_response(packet);prohibited['claims'][0]['code']='confirmed_fluid_migration'
        fallback=interpret(packet,provider=lambda p:prohibited,authorized=True)
        assert fallback['provenance']['source']=='deterministic_fallback'
        record={'case':packet.finding_id,'model':settings.model,'configuration_id':settings.configuration_id(),
            'live_requested':args.live,'seconds':time.perf_counter()-start,'explanation':result,
            'fabricated_measurement_rejected':True,'prohibited_claim_replaced_with_fallback':True,
            'evidence':packet.to_dict(),'raw_synthetic_response':captured.get('raw')}
        (output/(packet.finding_id+'.json')).write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
        results.append({'case':packet.finding_id,'source':result['provenance']['source'],
                        'error_type':result['provenance'].get('error_type'),'seconds':record['seconds']})
        print(json.dumps(results[-1]),flush=True)
        if args.live and result['provenance'].get('error_type') in {'APIConnectionError','APITimeoutError','ServiceBusy','RuntimeError'}:
            break  # do not issue more requests when transport/dependency is unavailable
    (output/'summary.json').write_text(json.dumps({'cases':results,'synthetic_only':True,
        'requests_attempted':len(results) if args.live else 0,'api_success_is_not_scientific_validation':True},indent=2)+'\n')

    if args.live and (len(results)!=len(packets) or any(r['source']!='validated_provider_selection' for r in results)):
        raise SystemExit(2)  # fallback is safe, but does not constitute live-provider validation


if __name__=='__main__':main()
