"""Bounded actual-data Phase 2 validation; no training, external requests or volume writes.

Optional RTM_ML_CACHE_DIR must be an explicitly authorized separate derived-output root.
Three predeclared sample-650 tiles are exploratory checks, not independent geological validation.
"""
import json
import os
from pathlib import Path
import resource
import time
import numpy as np
from reservoir_time_machine.data_ingest.segy_loader import load_challenge3_paired_volumes
from reservoir_time_machine.data_ingest.cache_identity import paired_cache_identity
from reservoir_time_machine.data_ingest.well_evidence import well_evidence_summary
from reservoir_time_machine.analysis.preliminary import run_preliminary
from reservoir_time_machine.ai.grounding import finding_packet, interpret


def main():
    root,cache=Path(os.environ['RTM_CHALLENGE3_ROOT']),Path(os.environ['RTM_CACHE_DIR'])
    before=paired_cache_identity(root,cache)
    b,m=load_challenge3_paired_volumes(root,output_dir=cache)
    assert b.metadata.source_metadata['trace_key_pairing_verified'] is True
    missing=np.asarray(b.missing_trace_mask)|np.asarray(m.missing_trace_mask)
    checks=[];start=time.perf_counter()
    tiles=[(512,576,576,640),(224,288,1184,1248),(560,624,1184,1248)]
    for bounds in tiles:
        result=run_preliminary(b.data,m.data,missing,dataset_id=before,sample=650,bounds=bounds,
                               artifact_root=os.environ.get('RTM_ML_CACHE_DIR') or None)
        for name in ['mahalanobis','pca','isolation_forest','gmm']:
            assert result['model_status'][name]['status']=='PRELIMINARY OPERATIONAL',result['model_status']
            assert np.isfinite(result['scores'][name][result['valid']]).all()
            assert not result['masks'][name][~result['valid']].any()
        summaries=[]
        for finding in result['findings']:
            i,x,s=finding['representative'];i0,_,x0,_=bounds
            assert result['labels'][i-i0,x-x0]==finding['label'] and s==650
            packet=finding_packet(result,finding);explanation=interpret(packet)
            assert not explanation['provenance']['external_transmission']
            summaries.append({k:finding[k] for k in ['id','representative','bounds','metrics','priority','priority_terms','touches_tile_boundary','method_evidence']})
        checks.append({'bounds':bounds,'run_id':result['run_id'],'valid_cells':int(result['valid'].sum()),
                       'model_status':result['model_status'],'reference_rows':result['provenance']['fit_rows'],
                       'threshold_rows':result['provenance']['threshold_rows'],'inspection_cells':result['provenance']['inspection_cells'],
                       'runtime_seconds':result['runtime_seconds'],'artifact_cache_hit':result['artifact_cache_hit'],
                       'finding_count':len(summaries),'top_findings':summaries[:3], 'agreement':result['agreement']})
    wells=well_evidence_summary(root)
    assert wells['well_count']==8 and len(wells['logs'])==11 and not wells['association_available']
    assert before==paired_cache_identity(root,cache)
    print(json.dumps({'scope':'three bounded sample-650 tiles; two selected from earlier statistical screening (selection bias)',
        'tiles':checks,'well_context':{'wells':wells['well_count'],'las_sources':len(wells['logs']),
        'table_sources':len(wells['tables']),'association_available':wells['association_available'],'errors':wells['errors']},
        'cache_identity_unchanged':True,'external_requests':0,'runtime_seconds':time.perf_counter()-start,
        'process_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'scientific_limit':'No geological/production/recovery validation; unregistered preliminary screening only.'},indent=2))


if __name__=='__main__':main()
