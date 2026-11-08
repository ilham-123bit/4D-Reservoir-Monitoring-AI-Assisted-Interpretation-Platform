"""Versioned completed-artifact registry. Dashboard loading never fits a model."""
from pathlib import Path
import hashlib,json,os,re,zipfile
import numpy as np
from datetime import datetime,timezone
from .artifacts import load_artifact,save_artifact

VERSION='registry-3.1'

def canonical(value):return json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':')).encode()
def sha(value):return hashlib.sha256(canonical(value)).hexdigest()


def validate_plan(plan,shape):
    if len(shape)!=3 or any(type(v)!=int or v<1 for v in shape) or not 1<=len(plan)<=64:
        raise ValueError('Positive 3D shape and 1–64 bounded planned tiles required')
    seen=[]
    for item in plan:
        s=item['sample'];i0,i1,x0,x1=item['bounds']
        if (type(s)!=int or not 0<=s<shape[2] or any(type(v)!=int for v in item['bounds'])
            or not 0<=i0<i1<=shape[0] or not 0<=x0<x1<=shape[1] or i1-i0>64 or x1-x0>64):
            raise ValueError('Plan exceeds bounded tile/grid limits')
        for prev in seen:
            a,b,c,d=prev['bounds']
            if s==prev['sample'] and max(a,i0)<min(b,i1) and max(c,x0)<min(d,x1):
                raise ValueError('Overlapping center-grid coverage cannot be double-counted')
        seen.append(item)


def regular_plan(shape):
    """27 predeclared sparse tiles, 3 sample positions × 3×3 spatial anchors."""
    plan=[]
    for s in sorted({int((shape[2]-1)*f) for f in (.25,.5,.75)}):
        for i in sorted({min((int((shape[0]-1)*f)//64)*64,max(0,shape[0]-64)) for f in (1/6,.5,5/6)}):
            for x in sorted({min((int((shape[1]-1)*f)//64)*64,max(0,shape[1]-64)) for f in (1/6,.5,5/6)}):
                item={'sample':s,'bounds':[i,min(i+64,shape[0]),x,min(x+64,shape[1])],'selection':'predeclared sparse spatial/sample anchors'}
                if not any(v['sample']==s and max(v['bounds'][0],i)<min(v['bounds'][1],item['bounds'][1])
                    and max(v['bounds'][2],x)<min(v['bounds'][3],item['bounds'][3]) for v in plan):plan.append(item)
    validate_plan(plan,shape);return plan


def save_registry(results,root,*,dataset_id,shape,plan,wells,scientific_context=None):
    root=Path(root).resolve();validate_plan(plan,list(shape))
    if '.reservoir_time_machine_cache' in root.parts:raise ValueError('Protected cache directory cannot hold derived registry outputs')
    if any((p/'baseline.npy').exists() and (p/'monitor.npy').exists() for p in (root,*root.parents)):
        raise ValueError('Registry must remain outside original paired cache')
    root.mkdir(parents=True,exist_ok=True)
    if any((root/n).is_symlink() for n in ('artifacts','interpretations','snapshots','registry.json')):raise ValueError('Redirected registry output')
    runs=[];seen=set()
    for result in results:
        if result['provenance']['dataset_id']!=dataset_id:raise ValueError('Mixed dataset identities')
        match=next((p for p in plan if p['sample']==result['sample'] and p['bounds']==list(result['bounds'])),None)
        if match is None or result['run_id'] in seen:raise ValueError('Unplanned or duplicate completed tile')
        seen.add(result['run_id']);target=root/'artifacts'/result['run_id']
        if not target.exists():save_artifact(result,root/'artifacts')
        checked=load_artifact(target,result['run_id'])
        runs.append({'run_id':result['run_id'],'sample':result['sample'],'bounds':list(result['bounds']),
            'eligible_cells':int(checked['valid'].sum()),'finding_ids':[f['id'] for f in checked['findings']],
            'selection':match.get('selection','unrecorded')})
    from ..ai.grounding import finding_packet,interpret
    ledger={};records=root/'interpretations';records.mkdir(exist_ok=True)
    for run in runs:
        checked=load_artifact(root/'artifacts'/run['run_id'],run['run_id'])
        for finding in checked['findings']:
            packet=finding_packet(checked,finding,well_context=wells)
            explanation=interpret(packet)
            record={'finding_id':finding['id'],'evidence_sha256':explanation['provenance']['evidence_sha256'],
                'packet':packet.to_dict(),'interpretation':explanation}
            payload=canonical(record);checksum=hashlib.sha256(payload).hexdigest()
            file=records/(checksum+'.json')
            if not file.exists():
                with file.open('xb') as stream:stream.write(payload)
            ledger[finding['id']]=checksum
    completed=sum((r['bounds'][1]-r['bounds'][0])*(r['bounds'][3]-r['bounds'][2]) for r in runs)
    total=int(shape[0])*int(shape[1])*int(shape[2])
    body={'version':VERSION,'dataset_id':dataset_id,'shape':list(shape),'plan':plan,'runs':runs,'wells':wells,
        'local_interpretation_records':ledger,'scientific_context':scientific_context or {'status':'Not supplied with this fixture/run'},
        'coverage':{'planned_tiles':len(plan),'completed_tiles':len(runs),'pending_tiles':len(plan)-len(runs),
            'analyzed_center_cells':completed,'total_center_cells':total,'center_fraction':completed/total,
            'eligible_center_cells':sum(r['eligible_cells'] for r in runs),
            'meaning':'Sample-center grid cells only; 9-sample windows overlap. No physical reservoir coverage is inferred.'},
        'ranking':{'survey_comparable':False,'scope':'Within each completed tile only. Cross-tile scores/references are not calibrated; sequence follows plan, not global priority.'},
        'scientific_limits':['Physical 4D registration unverified','Sample domain/origin unresolved','No verified well association','No calibrated reservoir probability or recovery validation']}
    rid=sha(body);envelope={'registry_id':rid,'body':body,'sha256':rid,'created_utc':datetime.now(timezone.utc).isoformat()}
    if len(canonical(envelope))>2_000_000:raise ValueError('Registry index exceeds bounded manifest budget; use separate run registries')
    snapshots=root/'snapshots';snapshots.mkdir(exist_ok=True)
    snapshot=snapshots/(rid+'.json')
    if not snapshot.exists():snapshot.write_bytes(canonical(envelope))
    temporary=root/('registry.'+str(os.getpid())+'.tmp');temporary.write_bytes(canonical(envelope));temporary.replace(root/'registry.json')
    return root/'registry.json'


def load_registry(path,dataset_id,shape):
    path=Path(path).resolve()
    if path.stat().st_size>2_000_000:raise ValueError('Registry manifest too large')
    envelope=json.loads(path.read_text());body=envelope['body']
    if envelope['sha256']!=sha(body) or envelope['registry_id']!=sha(body) or body['version']!=VERSION:
        raise ValueError('Registry checksum/version mismatch')
    if body['dataset_id']!=dataset_id or body['shape']!=list(shape):raise ValueError('Stale registry dataset/grid')
    validate_plan(body['plan'],body['shape']);results=[];entries=[];seen=set()
    for run in body['runs']:
        rid=run['run_id']
        if not re.fullmatch('[a-f0-9]{64}',rid) or rid in seen:raise ValueError('Invalid/duplicate run identity')
        seen.add(rid)
        from .preliminary import VERSION as PIPELINE_VERSION,FEATURE_VERSION
        target=path.parent/'artifacts'/rid
        if target.resolve()!=target:raise ValueError('Redirected artifact path')
        try:
            with zipfile.ZipFile(target/'maps.npz') as archive:
                if len(archive.infolist())>32 or sum(v.file_size for v in archive.infolist())>8_000_000:
                    raise ValueError('Artifact exceeds bounded decompression budget')
        except zipfile.BadZipFile as exc:raise ValueError('Invalid artifact archive') from exc
        result=load_artifact(target,rid)
        if result['valid'].shape!=(run['bounds'][1]-run['bounds'][0],run['bounds'][3]-run['bounds'][2]):
            raise ValueError('Artifact shape does not match analyzed bounds')
        for finding in result['findings']:
            members=result['labels']==finding['label']
            if (int(members.sum())!=finding['metrics']['cells'] or
                hashlib.sha256(np.flatnonzero(members).astype('<i8').tobytes()).hexdigest()!=finding['membership_sha256']):
                raise ValueError('Finding component membership/provenance mismatch')
        if result['provenance']['software']['pipeline']!=PIPELINE_VERSION or result['provenance']['software']['features']!=FEATURE_VERSION:
            raise ValueError('Registry pipeline/feature version is stale')
        if result['provenance']['dataset_id']!=dataset_id or result['sample']!=run['sample'] or list(result['bounds'])!=run['bounds']:
            raise ValueError('Registry artifact source mismatch')
        if [f['id'] for f in result['findings']]!=run['finding_ids']:raise ValueError('Finding ledger mismatch')
        results.append(result)
        for rank,f in enumerate(result['findings'],1):
            digest=body['local_interpretation_records'][f['id']]
            if not re.fullmatch('[a-f0-9]{64}',digest):raise ValueError('Invalid interpretation record identity')
            record_path=path.parent/'interpretations'/(digest+'.json')
            if record_path.stat().st_size>64000:raise ValueError('Interpretation record exceeds budget')
            payload=record_path.read_bytes()
            if hashlib.sha256(payload).hexdigest()!=digest:raise ValueError('Interpretation ledger checksum mismatch')
            record=json.loads(payload)
            from ..ai.grounding import finding_packet,interpret
            packet=finding_packet(result,f,well_context=body['wells'])
            expected=interpret(packet)
            if record['finding_id']!=f['id'] or record['evidence_sha256']!=expected['provenance']['evidence_sha256'] or record['interpretation']!=expected or record['packet']!=packet.to_dict():
                raise ValueError('Interpretation ledger lost evidence correspondence')
            entries.append({'id':f['id'],'local_rank':rank,'run_id':rid,'result':result,'finding':f,'interpretation':record['interpretation'],'evidence_packet':record['packet']})
    if set(body['local_interpretation_records'])!={e['id'] for e in entries}:raise ValueError('Interpretation index differs from findings')
    if len({e['id'] for e in entries})!=len(entries):raise ValueError('Duplicate finding identity')
    # Recompute coverage instead of trusting serialized summary/rank declarations.
    center=sum((r['bounds'][1]-r['bounds'][0])*(r['bounds'][3]-r['bounds'][2]) for r in body['runs'])
    if (body['ranking']['survey_comparable'] is not False or center!=body['coverage']['analyzed_center_cells']
        or body['coverage']['total_center_cells']!=int(shape[0])*int(shape[1])*int(shape[2])
        or body['coverage']['center_fraction']!=center/(int(shape[0])*int(shape[1])*int(shape[2]))
        or body['coverage']['planned_tiles']!=len(body['plan'])
        or body['coverage']['eligible_center_cells']!=sum(int(r['valid'].sum()) for r in results)
        or len(results)!=body['coverage']['completed_tiles'] or len(body['plan'])-len(results)!=body['coverage']['pending_tiles']):
        raise ValueError('Unsupported ranking or inconsistent coverage')
    for run in body['runs']:
        if not any(p['sample']==run['sample'] and p['bounds']==run['bounds'] for p in body['plan']):raise ValueError('Unplanned completed run')
    return {'id':envelope['registry_id'],'body':body,'results':results,'entries':entries,'path':str(path)}
