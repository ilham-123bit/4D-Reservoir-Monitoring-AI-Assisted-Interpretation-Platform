"""One sample-plane discovery and one bounded ML tile; local outputs only."""
import json,os,time,resource
from pathlib import Path
import numpy as np
from reservoir_time_machine.data_ingest.segy_loader import load_challenge3_paired_volumes
from reservoir_time_machine.data_ingest.cache_identity import paired_cache_identity
from reservoir_time_machine.data_ingest.well_evidence import well_evidence_summary
from reservoir_time_machine.attributes.difference import calculate_sample_attributes
from reservoir_time_machine.detection.statistical import statistical_screen
from reservoir_time_machine.detection.clustering import connected_zones
from reservoir_time_machine.analysis.guided_report import automatic_location,page_report
from reservoir_time_machine.analysis.preliminary import run_preliminary
from reservoir_time_machine.visualization.finding_animation import finding_animation


def main():
    output=Path(os.environ['RTM_GUIDED_VALIDATION_OUT']).resolve()
    root=Path(os.environ['RTM_CHALLENGE3_ROOT']);cache=Path(os.environ['RTM_CACHE_DIR'])
    if any(output==p.resolve() or p.resolve() in output.parents for p in (root,cache)):
        raise ValueError('Validation output must be outside original data and paired cache')
    output.mkdir(exist_ok=False)
    identity=paired_cache_identity(root,cache)
    start=time.perf_counter();b,m=load_challenge3_paired_volumes(root,output_dir=cache)
    missing=np.asarray(b.missing_trace_mask)|np.asarray(m.missing_trace_mask);sample=b.data.shape[2]//2
    attrs=calculate_sample_attributes(b.data,m.data,sample);screen=statistical_screen(attrs.absolute_difference,threshold_sigma=3.5)
    labels,zones=connected_zones(screen.mask,screen.score,min_size=8);focus=automatic_location(labels,zones,screen.score)
    assert focus and labels[tuple(focus['location'])]==focus['label']
    i,x=focus['location'];i0,x0=(i//64)*64,(x//64)*64;bounds=(i0,min(i0+64,b.data.shape[0]),x0,min(x0+64,b.data.shape[1]))
    result=run_preliminary(b.data,m.data,missing,dataset_id=identity,sample=sample,bounds=bounds,artifact_root=output/'bounded-artifacts')
    wells=well_evidence_summary(root);finding=result['findings'][0] if result['findings'] else None
    report=page_report(result,finding,wells,'Survey transition',{'sample':sample,'statistical_focus':focus,'ml_bounds':bounds})
    if finding:
        fig=finding_animation(b.data,m.data,missing,result,finding)
        assert [f.name for f in fig.frames]==['Baseline','Monitor']
        fig.write_html(output/'presentation.html',include_plotlyjs=True,auto_play=False)
    assert paired_cache_identity(root,cache)==identity
    record={'sample':sample,'regions':len(zones),'focus':focus,'tile':bounds,'findings':len(result['findings']),
        'report':report,'cache_identity_unchanged':True,'external_requests':0,
        'runtime_seconds':time.perf_counter()-start,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'presentation_saved':bool(finding)}
    (output/'validation.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in record.items() if k!='report'},indent=2))

if __name__=='__main__':main()
