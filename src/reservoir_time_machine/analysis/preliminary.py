"""Bounded, reproducible classical screening on an explicitly unverified paired grid."""
from dataclasses import dataclass, asdict
import hashlib
import importlib.metadata
import json
import time
import platform
import numpy as np
from scipy.ndimage import binary_dilation
from ..ai.detectors import MahalanobisDetector, LinearAutoencoderDetector, IsolationForestDetector, GaussianMixtureDetector
from ..ai.evidence_fusion import summarize_detection_agreement
from ..detection.contracts import DetectionRequest, DetectionOutput
from ..detection.statistical import statistical_screen
from ..detection.clustering import connected_zones
from ..visualization.slices import region_representative
from .features import extract_features, robust_scaling, FEATURE_NAMES, VERSION as FEATURE_VERSION

VERSION = 'bounded-preliminary-4'
LIMITATIONS = (
    'Trace-key pairing only; physical 4D registration and sample domain/origin are unverified.',
    'Reference blocks are unlabeled survey samples and may contain actual change; no unchanged population is established.',
    'Threshold quantiles are exploratory reference cutoffs, not false-alarm rates or probabilities.',
    'Detectors share correlated seismic attributes; agreement is not independent geological confirmation.',
    'Results cover one bounded sample tile, not the full survey or a 3D reservoir body.',
    'No verified well tie, production response, fluid movement or recovery opportunity is established.',
)


@dataclass(frozen=True)
class PreliminaryConfig:
    window: int = 9
    min_support_fraction: float = .8
    max_reference: int = 2048
    max_threshold_rows: int = 1024
    min_reference: int = 32
    quantile: float = .98
    seed: int = 2026
    min_region: int = 8
    mad_multiplier: float = 3.5
    trees: int = 40

    def __post_init__(self):
        for name in ('window', 'max_reference', 'max_threshold_rows', 'min_reference', 'seed', 'min_region', 'trees'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name=='seed' else 1):
                raise ValueError(f'{name} must be an integer within budget')
        if (self.max_reference > 4096 or self.max_threshold_rows > 2048 or self.min_reference < 16
                or self.trees > 100 or self.trees < 10 or self.window > 31
                or self.min_reference > min(self.max_reference, self.max_threshold_rows)):
            raise ValueError('Reference/tree/window budget exceeded or too few reference rows')
        if (not np.isfinite([self.min_support_fraction,self.quantile,self.mad_multiplier]).all()
                or not 0 < self.min_support_fraction <= 1 or not .9 <= self.quantile < 1
                or self.mad_multiplier <= 0):
            raise ValueError('Invalid support/quantile/MAD settings')


def versions():
    result = {'pipeline': VERSION, 'features': FEATURE_VERSION, 'python':platform.python_version()}
    for name in ('numpy','scipy','scikit-learn'):
        try: result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: result[name] = 'unavailable'
    return result


def _sample_indices(indices, limit, rng):
    return np.sort(rng.choice(indices, size=min(len(indices),limit), replace=False))


def run_preliminary(baseline, monitor, missing, *, dataset_id, sample, bounds, config=None, artifact_root=None):
    config = config or PreliminaryConfig()
    start = time.perf_counter()
    tile = extract_features(baseline, monitor, missing, sample=sample, bounds=bounds,
                            window=config.window, min_support_fraction=config.min_support_fraction)
    shape = (*tile.valid.shape, 1)
    model_parameters = {'mahalanobis':{'regularization':1e-3,'robust_winsor_z':4.5},
                        'pca':{'latent_dimensions':2,'threshold_quantile':.99},
                        'isolation_forest':{'n_estimators':config.trees,'random_state':config.seed,'contamination':.01},
                        'gmm':{'components':2,'random_state':config.seed,'threshold_quantile':.99}}
    configuration = {**asdict(config), 'detector_parameters':model_parameters,
                     'priority_policy':'statistical_0.5_plus_coverage_adjusted_ml_0.3_plus_extent_0.2_v1'}
    identity = {'dataset_id': dataset_id, 'sample': sample, 'bounds': list(bounds),
                'configuration': configuration, 'software': versions(), 'feature_order': list(FEATURE_NAMES),
                'tile_content_sha256': tile.content_sha256}
    run_id = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    if artifact_root is not None:
        from pathlib import Path
        from .artifacts import load_artifact
        cached = Path(artifact_root).expanduser()/run_id
        if cached.exists():
            result = load_artifact(cached,run_id)
            result['artifact_cache_hit'] = True
            return result
    valid_flat = tile.valid.ravel(); blocks = tile.blocks.ravel()
    rng = np.random.default_rng(config.seed)
    fit_idx = _sample_indices(np.flatnonzero(valid_flat & (blocks < 2)), config.max_reference,rng)
    cutoff_idx = _sample_indices(np.flatnonzero(valid_flat & (blocks == 2)), config.max_threshold_rows,rng)
    inspection = tile.valid & (tile.blocks == 3)
    scores, masks, statuses, fitted_outputs = {}, {}, {}, []
    provenance = {**identity, 'run_id':run_id,
                  'reference_policy':'unlabeled 8x8 spatial blocks; groups 0/1 fit, 2 threshold, 3 inspection',
                  'fit_indices_sha256':hashlib.sha256(fit_idx.tobytes()).hexdigest(),
                  'threshold_indices_sha256':hashlib.sha256(cutoff_idx.tobytes()).hexdigest(),
                  'fit_rows':len(fit_idx), 'threshold_rows':len(cutoff_idx),
                  'inspection_cells':int(inspection.sum()), 'calibration_verified':False}
    factories = {
        'mahalanobis':lambda:MahalanobisDetector(**model_parameters['mahalanobis']),
        'pca':lambda:LinearAutoencoderDetector(**model_parameters['pca']),
        'isolation_forest':lambda:IsolationForestDetector(**model_parameters['isolation_forest']),
        'gmm':lambda:GaussianMixtureDetector(**model_parameters['gmm']),
    }
    statuses.update({
        'siamese':{'status':'IMPLEMENTED BUT NOT TRAINED', 'reason':'No verified trained checkpoint/label provenance configured; never initialized for predictions.'},
        'conv_autoencoder':{'status':'IMPLEMENTED BUT NOT TRAINED', 'reason':'No verified checkpoint or justified reference training run configured.'},
        'bayesian':{'status':'NOT CONFIGURED', 'reason':'Gaussian likelihood implementation exists; no justified change distribution or prior. Numerical scenario tests only.'},
    })
    matrix = tile.features.reshape(-1,len(FEATURE_NAMES))
    eligible = len(fit_idx) >= config.min_reference and len(cutoff_idx) >= config.min_reference and inspection.any()
    if eligible:
        center, scale, varying = robust_scaling(matrix[fit_idx])
        eligible = np.count_nonzero(varying) >= 3
        provenance['scaling'] = {'center':center.tolist(),'scale':scale.tolist(),
                                 'method':'fit-only median/MAD with fit-standard-deviation fallback',
                                 'varying_features':varying.tolist()}
    for name, factory in factories.items():
        if not eligible:
            statuses[name] = {'status':'NOT CONFIGURED','reason':'Insufficient spatially separated valid reference/threshold/inspection support or feature variance.'}
            continue
        model_start = time.perf_counter()
        try:
            # Reference/scoring budgets are bounded; threadpoolctl comes with sklearn.
            try:
                from threadpoolctl import threadpool_limits
                limit_context = threadpool_limits(limits=1)
            except ImportError:
                from contextlib import nullcontext
                limit_context = nullcontext()
            with limit_context:
                model = factory().fit((matrix[fit_idx]-center)/scale,feature_names=FEATURE_NAMES)
                if name=='gmm' and not model.estimator.converged_:
                    raise ValueError('Gaussian mixture did not converge')
                standardized = (tile.features-center)/scale
                request = DetectionRequest({n:standardized[...,j,None] for j,n in enumerate(FEATURE_NAMES)},
                                           input_provenance=provenance, processing_configuration=configuration)
                output = model.detect(request)
            score = output.score[...,0].copy()
            if not np.isfinite(score[tile.valid]).all():
                raise ValueError('Non-finite scores at valid source cells')
            cutoff = float(np.quantile(score.ravel()[cutoff_idx],config.quantile))
            mask = tile.valid & (score > cutoff)
            scores[name], masks[name] = score, mask
            statuses[name] = {'status':'PRELIMINARY OPERATIONAL', 'score_meaning':output.score_meaning,
                'threshold':cutoff,'threshold_quantile':config.quantile,'score_is_probability':False,
                'calibration_status':'uncalibrated_unlabeled_reference_cutoff',
                'runtime_seconds':time.perf_counter()-model_start,'validated_reservoir_inference':False,
                'parameters':model_parameters[name], 'internal_mask_ignored':True,
                'fitting_diagnostics':output.summary()['diagnostics']}
            fitted_outputs.append(DetectionOutput(name,VERSION,score[...,None],mask[...,None],output.score_meaning,
                'held_out_unlabeled_reference_quantile_uncalibrated',provenance,LIMITATIONS,configuration))
        except (ImportError, RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            statuses[name] = {'status':'DEPENDENCY UNAVAILABLE' if isinstance(exc,ImportError) else 'INTEGRATED BUT FAILING',
                              'reason':type(exc).__name__+': '+str(exc)}
    strength = np.where(tile.valid,tile.attributes.absolute_difference,np.nan)
    statistical = statistical_screen(strength,threshold_sigma=config.mad_multiplier)
    scores['statistical'], masks['statistical'] = strength, statistical.mask
    vote = np.zeros(tile.valid.shape,np.uint8)
    for mask in masks.values(): vote += mask.astype(np.uint8)
    # ML-only discoveries require two methods on inspection blocks. Statistical
    # candidates remain full-resolution within the tile; fitting cells cannot add ML corroboration.
    candidate = statistical.mask | (inspection & (vote >= 2))
    labels, zones = connected_zones(candidate,strength,min_size=config.min_region)
    findings = []
    invalid_neighbors = binary_dilation(~tile.valid) & tile.valid
    i0,i1,x0,x1 = bounds
    for zone in zones:
        membership = labels==zone.label
        rep = region_representative(labels,zone,strength)
        inspected_members = membership & inspection
        stats = {}
        for name in masks:
            supported = inspected_members & np.isfinite(scores[name])
            stats[name] = {'inspection_support':int(supported.sum()),
                           'flagged_inspection_cells':int((masks[name]&supported).sum()),
                           'fraction':float(masks[name][supported].mean()) if supported.any() else None}
        ml_fraction = [v['fraction'] for n,v in stats.items() if n!='statistical' and v['fraction'] is not None]
        statistical_fraction = float(statistical.mask[membership].mean())
        # Transparent 0..1 priority, never a calibrated probability/confidence.
        extent = min(1.0,np.log1p(zone.voxel_count)/np.log1p(tile.valid.sum()))
        agreement = float(np.mean(ml_fraction)) if ml_fraction else 0.0
        inspection_coverage = float(inspected_members.sum()/zone.voxel_count)
        priority = .5*statistical_fraction+.3*agreement*inspection_coverage+.2*extent
        member_hash = hashlib.sha256(np.flatnonzero(membership).astype('<i8').tobytes()).hexdigest()
        finding_id = 'finding-'+hashlib.sha256((run_id+member_hash).encode()).hexdigest()[:20]
        metrics = {'baseline_amplitude':float(baseline[rep[0]+i0,rep[1]+x0,sample]),
                   'monitor_amplitude':float(monitor[rep[0]+i0,rep[1]+x0,sample]),
                   'cells':zone.voxel_count,'mean_absolute_difference':float(np.mean(strength[membership])),
                   'mean_signed_difference':float(np.mean(tile.attributes.signed_difference[membership])),
                   'mean_nrms_percent':float(np.mean(tile.attributes.nrms[membership])),
                   'mean_rms_baseline':float(np.mean(tile.attributes.rms_baseline[membership])),
                   'mean_rms_monitor':float(np.mean(tile.attributes.rms_monitor[membership])),
                   'minimum_paired_support':int(tile.attributes.paired_sample_count[membership].min()),
                   'members_adjacent_to_invalid_cells':int((membership & invalid_neighbors).sum())}
        findings.append({'id':finding_id,'run_id':run_id,'label':zone.label,'sample':sample,
            'representative':[rep[0]+i0,rep[1]+x0,sample],
            'bounds':[[z[0]+offset,z[1]+offset] for z,offset in zip(zone.bounding_box,(i0,x0))],
            'membership_sha256':member_hash,'metrics':metrics,'method_evidence':stats,
            'priority':priority,'priority_terms':{'statistical_fraction':statistical_fraction,
                'mean_ml_inspection_fraction':agreement,'ml_inspection_coverage':inspection_coverage,
                'coverage_adjusted_ml_agreement':agreement*inspection_coverage,
                'relative_grid_extent':float(extent),'weights':[.5,.3,.2]},
            'touches_tile_boundary':any(lo==0 or hi==size-1 for (lo,hi),size in zip(zone.bounding_box,tile.valid.shape)),
            'limitations':list(LIMITATIONS),'well_association':'unavailable; shared CRS/datum/time-depth not verified'})
    findings.sort(key=lambda f:(-f['priority'],f['id']))
    agreement = summarize_detection_agreement(fitted_outputs) if len(fitted_outputs)>=2 else {}
    result = {'schema_version':'1','run_id':run_id,'provenance':provenance,'bounds':list(bounds),'sample':sample,
        'valid':tile.valid,'support':tile.attributes.paired_sample_count,'inspection':inspection,
        'fit_mask':np.isin(np.arange(tile.valid.size).reshape(tile.valid.shape),fit_idx),
        'threshold_mask':np.isin(np.arange(tile.valid.size).reshape(tile.valid.shape),cutoff_idx),
        'scores':scores,'masks':masks,'labels':labels,'findings':findings,'model_status':statuses,
        'agreement':agreement,'quality':{'eligible_cells':int(tile.valid.sum()),
            'tile_cells':int(tile.valid.size),'missing_traces':int(np.asarray(missing[i0:i1,x0:x1]).sum()),
            'insufficient_window_support':int((tile.attributes.paired_sample_count < np.ceil(tile.available_window*config.min_support_fraction)).sum()),
            'available_window_samples':tile.available_window},'limitations':list(LIMITATIONS),'score_is_probability':False,
        'runtime_seconds':time.perf_counter()-start, 'artifact_cache_hit':False}
    if artifact_root is not None:
        from .artifacts import save_artifact
        save_artifact(result,artifact_root)
    return result
