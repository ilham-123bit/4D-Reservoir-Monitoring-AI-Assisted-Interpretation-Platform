#!/usr/bin/env python3
"""Compare real-cache chunk profiles on one allocated HPC node; never submits jobs."""
import argparse,hashlib,json,os,resource,time
from pathlib import Path
import numpy as np
from reservoir_time_machine.processing.full_field import run_full_field_attributes,FIELDS
from reservoir_time_machine.data_ingest.cache_identity import paired_cache_provenance

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bounds',type=int,nargs=6,action='append',required=True,metavar=('I0','I1','X0','X1','S0','S1'))
    p.add_argument('--chunks',type=int,nargs=3,action='append',default=[])
    p.add_argument('--min-free-gib',type=float,default=1.0)
    a=p.parse_args(argv)
    if not os.environ.get('SLURM_JOB_ID'):p.error('Run only within an approved Slurm allocation')
    cache=a.cache.resolve();out=a.output.resolve()
    if not cache.is_dir() or not out.is_dir() or out==cache or cache in out.parents or out in cache.parents:p.error('Cache/output roots invalid or overlapping')
    profiles=a.chunks or [(16,32,128),(32,32,256),(64,64,256)]
    for prof in profiles:
        if len(prof)!=3 or min(prof)<1:p.error('Chunk dimensions must be positive')
    arrays={n:np.load(cache/f'{n}.npy',mmap_mode='r',allow_pickle=False) for n in ('baseline','monitor','missing_trace_mask')}
    shape=tuple(arrays['baseline'].shape);cache_id,details=paired_cache_provenance(cache)
    job=os.environ['SLURM_JOB_ID'];report={'schema':'rtm-chunk-benchmark-1','job_id':job,
        'cache_identity':cache_id,'cache_identity_scope':details['identity_scope'],'source_shape':list(shape),
        'profiles':[],'numerical_parity':{},'peak_rss_kib_process_highwater':None,
        'note':'Wall/CPU and output throughput are measured on this node. Input bytes are a lower-bound estimate; RSS is process high-water across profiles.'}
    all_outputs={}
    for roi_num,bounds in enumerate(a.bounds):
        i0,i1,x0,x1,s0,s1=bounds
        if not(0<=i0<i1<=shape[0] and 0<=x0<x1<=shape[1] and 0<=s0<s1<=shape[2]):p.error('ROI bounds outside cache')
        local_shape=(i1-i0,x1-x0,s1-s0)
        if local_shape[0]<96 or local_shape[1]<96 or local_shape[2]<128:p.error('Benchmark ROIs must be representative (>=96×96×128)')
        sl=(slice(i0,i1),slice(x0,x1),slice(s0,s1)); mask_sl=sl[:2]
        before,after=arrays['baseline'][sl],arrays['monitor'][sl];missing=arrays['missing_trace_mask'][mask_sl]
        roi_key=f'roi{roi_num}-i{i0}-{i1}-x{x0}-{x1}-s{s0}-{s1}'
        all_outputs[roi_key]={}
        for chunks in profiles:
            name='chunk-'+'x'.join(map(str,chunks));identity=hashlib.sha256(f'{cache_id}:{bounds}:{chunks}:9:0.8'.encode()).hexdigest()
            destination=out/'chunk-benchmarks'/job/roi_key/name
            t0=time.perf_counter();c0=time.process_time()
            result=run_full_field_attributes(before,after,missing,destination,source_identity=identity,
                chunk_shape=tuple(chunks),window_samples=9,min_support_fraction=.8,
                source_bounds=tuple(bounds),source_domain_shape=shape,protected_cache=cache,
                min_free_bytes=int(a.min_free_gib*1024**3))
            wall=time.perf_counter()-t0;cpu=time.process_time()-c0
            if result.get('status')!='complete':raise RuntimeError(f'incomplete benchmark product: {destination}')
            run=Path(result['run_directory']);maps={f:np.load(run/f'{f}.npy',mmap_mode='r',allow_pickle=False) for f in FIELDS}
            if name in all_outputs[roi_key]:raise RuntimeError('Duplicate benchmark profile')
            all_outputs[roi_key][name]=maps
            journal=run/'chunks.jsonl';nchunks=result['total_chunks'];output_bytes=result['output_bytes']
            input_lower=int((before.size+after.size)*4+missing.size)
            report['profiles'].append({'roi':roi_key,'bounds':bounds,'shape':local_shape,'chunk_shape':chunks,
                'chunks':nchunks,'wall_seconds':wall,'process_cpu_seconds':cpu,
                'allocated_cpu_utilization_percent':100*cpu/max(wall*int(os.environ.get('SLURM_CPUS_PER_TASK','1')),1e-9),
                'processed_voxels_per_second':int(np.prod(local_shape)/max(wall,1e-9)),
                'output_bytes':output_bytes,'journal_bytes':journal.stat().st_size,
                'output_bytes_per_wall_second':output_bytes/max(wall,1e-9),
                'minimum_input_bytes_read_estimate':input_lower,'cache_hit':result.get('cache_hit',False),
                'chunk_sync_count':nchunks,'map_flush_calls_estimate':nchunks*len(FIELDS),
                'peak_rss_kib_process_highwater':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
        baseline_maps=all_outputs[roi_key]['chunk-16x32x128']
        for profile in profiles[1:]:
            key='chunk-'+'x'.join(map(str,profile));
            parity={f:bool(np.array_equal(baseline_maps[f],all_outputs[roi_key][key][f],equal_nan=True)) for f in FIELDS}
            report['numerical_parity'][roi_key+':'+key]=parity
            if not all(parity.values()):raise RuntimeError(f'Numerical mismatch: {roi_key} {key}: {parity}')
    report['peak_rss_kib_process_highwater']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report_path=out/'chunk-benchmarks'/job/'benchmark-report.json';report_path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    report_path.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':'complete','report':str(report_path),'profiles':report['profiles'],
        'numerical_parity':report['numerical_parity'],'peak_rss_kib_process_highwater':report['peak_rss_kib_process_highwater']},indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())
