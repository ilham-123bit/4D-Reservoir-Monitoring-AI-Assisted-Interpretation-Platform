"""Export completed findings locally; no model fitting and no external requests."""
import argparse,os,json
from pathlib import Path
import numpy as np
from reservoir_time_machine.analysis.registry import load_registry
from reservoir_time_machine.data_ingest.cache_identity import paired_cache_identity
from reservoir_time_machine.data_ingest.segy_loader import load_challenge3_paired_volumes
from reservoir_time_machine.visualization.presentation_export import export_presentation


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--registry',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(os.environ['RTM_CHALLENGE3_ROOT']).resolve();cache=Path(os.environ['RTM_CACHE_DIR']).resolve();out=Path(a.output).resolve()
    if out.exists() or any(out==v or v in out.parents for v in (root,cache)):raise ValueError('Use a fresh export outside protected data/cache')
    identity=paired_cache_identity(root,cache);b,m=load_challenge3_paired_volumes(root,output_dir=cache)
    r=load_registry(a.registry,identity,list(b.data.shape));out.parent.mkdir(parents=True,exist_ok=True)
    export_presentation(r,b.data,m.data,np.asarray(b.missing_trace_mask)|np.asarray(m.missing_trace_mask),out)
    assert paired_cache_identity(root,cache)==identity
    print(json.dumps({'presentation':str(out),'findings':min(8,len(r['entries'])),'registry':r['id'],'external_requests':0}))

if __name__=='__main__':main()
