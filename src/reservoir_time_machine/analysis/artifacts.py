"""Explicit-output cache of bounded derived results; never write inside paired caches."""
import hashlib
import json
from pathlib import Path
import numpy as np


def _canonical(value): return json.dumps(value,allow_nan=False,sort_keys=True,separators=(',',':')).encode()


def save_artifact(result, output_root):
    root = Path(output_root).expanduser().resolve()
    if any((parent/'baseline.npy').exists() and (parent/'monitor.npy').exists() for parent in (root,*root.parents)):
        raise ValueError('Artifact root must be outside the paired seismic cache')
    target = root/result['run_id']; target.mkdir(parents=True,exist_ok=False)
    arrays = {name:result[name] for name in ('valid','support','inspection','fit_mask','threshold_mask','labels')}
    for group in ('scores','masks'):
        for name,array in result[group].items(): arrays[group+'__'+name] = array
    path = target/'maps.npz'
    with path.open('xb') as stream: np.savez_compressed(stream,**arrays)
    summary = {k:v for k,v in result.items() if k not in {'scores','masks',*arrays}}
    summary['maps_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    summary['map_shapes'] = {k:list(v.shape) for k,v in arrays.items()}
    envelope = {'metadata':summary,'metadata_sha256':hashlib.sha256(_canonical(summary)).hexdigest()}
    with (target/'complete.json').open('x') as stream: json.dump(envelope,stream,allow_nan=False,sort_keys=True)
    return target


def load_artifact(path, expected_run_id):
    path = Path(path)
    try:
        envelope = json.loads((path/'complete.json').read_text()); summary = envelope['metadata']
        if hashlib.sha256(_canonical(summary)).hexdigest()!=envelope['metadata_sha256']:
            raise ValueError('Artifact metadata checksum mismatch')
        identity_keys = ('dataset_id','sample','bounds','configuration','software','feature_order','tile_content_sha256')
        identity = {key:summary['provenance'][key] for key in identity_keys}
        expected_identity = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        if summary['run_id']!=expected_run_id or expected_identity!=expected_run_id:
            raise ValueError('Artifact identity mismatch')
        maps = path/'maps.npz'
        if hashlib.sha256(maps.read_bytes()).hexdigest()!=summary['maps_sha256']:
            raise ValueError('Artifact map checksum mismatch')
        with np.load(maps,allow_pickle=False) as archive: arrays = {key:archive[key] for key in archive.files}
        if {k:list(v.shape) for k,v in arrays.items()}!=summary['map_shapes']:
            raise ValueError('Artifact map shape mismatch')
        result = {k:v for k,v in summary.items() if k not in {'maps_sha256','map_shapes'}}
        result['scores'],result['masks'] = {},{}
        for key,array in arrays.items():
            if '__' in key:
                group,name=key.split('__',1);result[group][name]=array
            else:result[key]=array
        valid=result['valid'];shape=valid.shape
        if valid.dtype!=np.bool_ or valid.ndim!=2 or valid.size>16384:
            raise ValueError('Invalid cached spatial validity grid')
        if any(a.shape!=shape for a in arrays.values()): raise ValueError('Inconsistent cached grids')
        if set(result['scores'])!=set(result['masks']): raise ValueError('Score/mask mismatch')
        for name,score in result['scores'].items():
            mask=result['masks'][name]
            if mask.dtype!=np.bool_ or not np.isfinite(score[valid]).all() or mask[~valid].any():
                raise ValueError('Invalid cached scores or detection support')
        if np.any(result['fit_mask']&result['threshold_mask']) or np.any(result['fit_mask']&result['inspection']):
            raise ValueError('Cached reference populations overlap')
        for finding in result['findings']:
            i,x,s=finding['representative'];i0,_,x0,_=result['bounds']
            if (s!=result['sample'] or not 0<=i-i0<shape[0] or not 0<=x-x0<shape[1]
                    or result['labels'][i-i0,x-x0]!=finding['label']):
                raise ValueError('Cached finding lost spatial correspondence')
        return result
    except (KeyError,TypeError,IndexError) as exc:
        raise ValueError('Malformed artifact; refusing reuse') from exc
