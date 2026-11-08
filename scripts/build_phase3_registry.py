"""Explicit bounded/scheduled discovery job. Never called by dashboard startup."""
import argparse,os,json,time,fcntl,resource
from pathlib import Path
from reservoir_time_machine.analysis.registry import regular_plan,save_registry,load_registry,validate_plan
from reservoir_time_machine.analysis.artifacts import load_artifact
from reservoir_time_machine.analysis.preliminary import run_preliminary
from reservoir_time_machine.analysis.technical_report import report_markdown
from reservoir_time_machine.data_ingest.cache_identity import paired_cache_identity
from reservoir_time_machine.data_ingest.segy_loader import load_challenge3_paired_volumes
from reservoir_time_machine.data_ingest.well_evidence import well_evidence_summary
import numpy as np


def main():
    job_start=time.perf_counter()
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--execute',action='store_true')
    p.add_argument('--smoke',action='store_true',help='Explicit bounded login-node fixture/real-data smoke: at most ONE tile')
    p.add_argument('--max-tiles',type=int,default=1);p.add_argument('--seconds',type=int,default=120)
    p.add_argument('--plan',help='JSON list of up to 64 nonoverlapping bounded tiles; extends a completed plan without removing existing scope')
    p.add_argument('--import-artifact',action='append',default=[]);args=p.parse_args()
    if not 1<=args.max_tiles<=4 or not 1<=args.seconds<=600:raise ValueError('Job budget is 1–4 tiles and 1–600 seconds')
    if args.execute and not (os.environ.get('SLURM_JOB_ID') or os.environ.get('PBS_JOBID') or (args.smoke and args.max_tiles==1)):
        raise PermissionError('Execution requires a batch allocation or explicit one-tile --smoke')
    root=Path(os.environ['RTM_CHALLENGE3_ROOT']).resolve();cache=Path(os.environ['RTM_CACHE_DIR']).resolve();out=Path(args.output).resolve()
    if '.reservoir_time_machine_cache' in out.parts or any(out==v or v in out.parents for v in (root,cache)):raise ValueError('Output cannot overlap original data/cache')
    identity=paired_cache_identity(root,cache)  # fail before loader if existing cache is incomplete
    b,m=load_challenge3_paired_volumes(root,output_dir=cache);shape=list(b.data.shape);plan=regular_plan(shape)
    out.mkdir(parents=True,exist_ok=True)
    with (out/'.analysis.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        results=[]
        if (out/'registry.json').exists():
            previous=load_registry(out/'registry.json',identity,shape);plan=previous['body']['plan'];results=previous['results']
        if args.plan:
            declared=Path(args.plan)
            if declared.stat().st_size>64000:raise ValueError('Plan file too large')
            additions=json.loads(declared.read_text());validate_plan(additions,shape)
            if not results and not (out/'registry.json').exists():plan=[]
            for item in additions:
                item.setdefault('selection','explicit user-declared sparse scope; representativeness unverified')
                if not any(v['sample']==item['sample'] and v['bounds']==item['bounds'] for v in plan):plan.append(item)
            validate_plan(plan,shape)
        for source in args.import_artifact:
            source=Path(source).resolve();r=load_artifact(source,source.name)
            if r['provenance']['dataset_id']!=identity:raise ValueError('Imported result belongs to another dataset')
            item={'sample':r['sample'],'bounds':list(r['bounds']),'selection':'opportunistic previously screened seed; selection bias, not representative survey sampling'}
            if not any(v['sample']==item['sample'] and v['bounds']==item['bounds'] for v in plan):plan.insert(0,item)
            else:
                for v in plan:
                    if v['sample']==item['sample'] and v['bounds']==item['bounds']:v['selection']=item['selection']
            if not any(v['run_id']==r['run_id'] for v in results):results.append(r)
        from reservoir_time_machine.data.readiness import assess_capabilities
        meta=b.metadata.source_metadata
        context={'capabilities':assess_capabilities(b,m),'sample_domain':b.metadata.sample_axis_domain,
            'sample_origin':meta.get('sample_origin'),'sample_interval':b.metadata.sample_interval,
            'sample_interval_units':b.metadata.sample_interval_units,
            'baseline_trace_count':meta.get('baseline_trace_count'),'paired_trace_count':meta.get('paired_trace_count'),
            'monitor_only_trace_count':meta.get('monitor_only_trace_count'),'missing_grid_cell_count':meta.get('missing_grid_cell_count'),
            'coverage_basis':'Baseline trace-key paired grid; retained Monitor-only traces are not projected into this analysis.'}
        wells=well_evidence_summary(root);path=save_registry(results,out,dataset_id=identity,shape=shape,plan=plan,wells=wells,scientific_context=context)
        start=time.monotonic();done=0
        if args.execute:
            missing=np.asarray(b.missing_trace_mask)|np.asarray(m.missing_trace_mask)
            for item in plan:
                if any(r['sample']==item['sample'] and list(r['bounds'])==item['bounds'] for r in results):continue
                if done>=args.max_tiles or time.monotonic()-start>=args.seconds:break
                r=run_preliminary(b.data,m.data,missing,dataset_id=identity,sample=item['sample'],bounds=tuple(item['bounds']),artifact_root=out/'artifacts')
                results.append(r);done+=1
                path=save_registry(results,out,dataset_id=identity,shape=shape,plan=plan,wells=wells,scientific_context=context)
        registry=load_registry(path,identity,shape);(out/'report.md').write_text(report_markdown(registry))
        assert paired_cache_identity(root,cache)==identity
        print(json.dumps({'registry':str(path),'identity':registry['id'],'coverage':registry['body']['coverage'],
            'findings':len(registry['entries']),'new_tiles_fitted':done,'external_requests':0,'runtime_seconds':time.perf_counter()-job_start,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},indent=2))

if __name__=='__main__':main()
