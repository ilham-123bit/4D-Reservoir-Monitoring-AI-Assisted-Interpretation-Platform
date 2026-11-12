#!/usr/bin/env python3
"""Validate completed HPC products and atomically publish the dashboard registry."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from reservoir_time_machine.analysis.full_field_registry import publish_full_field_registry
from reservoir_time_machine.data_ingest.cache_identity import paired_cache_identity, paired_cache_provenance
from reservoir_time_machine.data_ingest.well_evidence import well_evidence_summary
from reservoir_time_machine.processing.full_field import _canonical
import hashlib

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--challenge-root',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--attributes',type=Path,required=True)
    p.add_argument('--models',type=Path,required=True)
    p.add_argument('--findings',type=Path,required=True)
    p.add_argument('--component-labels',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    if not a.challenge_root.is_dir() or not a.cache.is_dir():p.error('Challenge data/cache paths must exist')
    root=Path(__import__('os').environ.get('RTM_HPC_OUTPUT_ROOT','')).resolve()
    output=a.output.resolve()
    if not root.is_dir() or (output!=root and root not in output.parents):p.error('Output must remain under approved RTM_HPC_OUTPUT_ROOT')
    import json
    metadata=json.loads((a.cache/'metadata.json').read_text(encoding='utf-8'))
    dataset_id=paired_cache_identity(a.challenge_root,a.cache)
    cache_id,cache_details=paired_cache_provenance(a.cache)
    attributes=json.loads((a.attributes/'complete.json').read_text(encoding='utf-8'))['body']
    shape=tuple(metadata['shape'])
    source_details={'schema':'rtm-full-field-input-identity-2','paired_cache_sha256':cache_id,
                    'paired_cache':cache_details,'source_bounds':[0,shape[0],0,shape[1],0,shape[2]],
                    'source_domain_shape':list(shape)}
    expected_source=hashlib.sha256(json.dumps(source_details,sort_keys=True,separators=(',',':'),
                                               allow_nan=False).encode()).hexdigest()
    if (attributes.get('configuration',{}).get('source_identity') != expected_source
            or attributes.get('configuration',{}).get('source_identity_details') != source_details):
        p.error('Attribute source identity does not match this paired cache and full-volume domain')
    result=publish_full_field_registry(findings_path=a.findings,attributes_directory=a.attributes,
        model_directory=a.models,labels_path=a.component_labels,output_root=output,
        dataset_identity=dataset_id,shape=metadata['shape'],inline_ids=metadata['inline_ids'],
        crossline_ids=metadata['crossline_ids'],well_context=well_evidence_summary(a.challenge_root))
    print(json.dumps(result,indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())
