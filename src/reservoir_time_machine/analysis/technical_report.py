"""Evidence-led report and presentation scene records; no external requests."""
from .guided_report import page_report


def storyboard(registry,entry):
    f=entry['finding'];m=f['metrics'];i,x,s=f['representative'];w=registry['body']['wells'];c=registry['body']['coverage']
    return [
      ('Finding introduction',f"{f['id']} · local rank {entry['local_rank']} in tile {entry['result']['bounds']}. Investigation priority {f['priority']:.4f}, not probability. {c['completed_tiles']}/{c['planned_tiles']} planned tiles completed."),
      ('Spatial localization',f"Inline {i}, crossline {x}, sample {s}; region bounds {f['bounds']}, {m['cells']} connected cells. Coordinates are source indices; this is a 2D component, not a volumetric geological body."),
      ('Baseline versus Monitor',f"Representative amplitudes: Baseline {m['baseline_amplitude']:.5g}, Monitor {m['monitor_amplitude']:.5g}. Fixed endpoint scales preserve amplitude comparison; no acquired intermediate vintage exists."),
      ('Difference and anomaly evidence',f"Mean signed change {m['mean_signed_difference']:.5g}; mean absolute change {m['mean_absolute_difference']:.5g}; mean NRMS {m['mean_nrms_percent']:.2f}%. Minimum paired support {m['minimum_paired_support']}. ML/statistical counts are shown in the evidence report; correlated methods are not independent confirmation."),
      ('Cross-sectional investigation',f"Both orthogonal sections intersect inline {i}, crossline {x}, sample {s}. Context outside the classified sample has not been assigned a 3D anomaly label. Gray cells have invalid/missing paired measurements."),
      ('Well and geological context',f"{w['well_count']} workbook records, {len(w['logs'])} LAS sources and {len(w['tables'])} supporting tables. Association unavailable: "+'; '.join(w['missing_association_evidence'])+'.'),
      ('Grounded interpretation',' '.join(c['text'] for c in entry['interpretation']['claims'] if c['category'] in {'hypothesis','interpretation','limitation'})),
      ('Engineering recommendation',' '.join(c['text'] for c in entry['interpretation']['claims'] if c['category']=='recommendation'))]


def report_markdown(registry):
    b=registry['body'];c=b['coverage'];w=b['wells']
    lines=['# Reservoir Time Machine — technical interpretation report',
      '\n## Executive interpretation',f"Completed {c['completed_tiles']} of {c['planned_tiles']} planned tiles; {len(registry['entries'])} preliminary findings. No survey-wide rank or recovery conclusion is established.",
      '\n## Dataset and scientific readiness',f"Dataset identity `{b['dataset_id']}`; registry `{registry['id']}`; grid {b['shape']}. "+'; '.join(b['scientific_limits'])+'.',
      '\n## Analysis coverage and methodology',f"Analyzed {c['analyzed_center_cells']} of {c['total_center_cells']} sample-center cells ({100*c['center_fraction']:.6f}%); {c['eligible_center_cells']} eligible centers. {c['meaning']}",
      b['ranking']['scope'],
      'Scientific readiness/source context: '+str(b.get('scientific_context',{})),
      'Completed group selection: '+str([{'sample':v['sample'],'bounds':v['bounds'],'selection':v['selection']} for v in b['runs']]),
      'Features: normalized difference, NRMS fraction, log RMS ratio and relative absolute difference. Robust reference scaling; disjoint fitting, cutoff and inspection blocks. Thresholds are uncalibrated local reference quantiles.',
      '\n## Quality and repeatability','Missing/non-finite pairs and insufficient support are excluded. No full-volume QC report or verified physical registration is implied by bounded outputs.',
      '\n## Findings and seismic/ML evidence']
    for entry in registry['entries']:
        f=entry['finding'];r=entry['result'];report=page_report(r,f,w,'Evidence explanation',b['coverage'],shared_interpretation=entry['interpretation'])
        lines += [f"\n### {f['id']} — local rank {entry['local_rank']}",f"Run `{entry['run_id']}`; center {f['representative']}, bounds {f['bounds']}; tile-edge truncation {f['touches_tile_boundary']}.",
            'Calculated metrics: '+', '.join(f"{key}={value}" for key,value in f['metrics'].items())+'.',
            'Held-out inspection detection: '+str(f['method_evidence']),
            'Model statuses: '+str({k:v['status'] for k,v in r['model_status'].items()}),
            'Uncalibrated held-out mean scores (different model-specific scales, not probabilities): '+str({v['name']:v['value'] for v in entry['evidence_packet']['facts'] if v['name'].endswith('_mean_inspection_score')}),
            f"Interpretation source: deterministic local; evidence SHA256 `{report['interpretation']['provenance']['evidence_sha256']}`."]
        lines += [f"{v['category']}: {v['text']} Evidence: {', '.join(v['evidence_ids'])}" for v in report['interpretation']['claims']]
    lines += ['\n## Well evidence and association',f"{w['well_count']} workbook records, {len(w['logs'])} LAS and {len(w['tables'])} supporting tables. No spatial/depth/time tie established.",
        'Missing: '+'; '.join(w['missing_association_evidence']),
        'Source well groups: '+str(w.get('by_well',{})),
        '\n## Geological alternatives and recovery investigation','All candidates remain screening observations. Acquisition/processing/scaling/noise/registration may explain differences. No bypassed oil, connectivity or recovery opportunity is proven.',
        '\n## Recommended geoscience and engineering work','Verify registration and repeatability; review source quality and detector disagreement; establish CRS/datum/time-depth ties; obtain independent geology and production evidence; extend documented coverage using scheduled jobs.',
        '\n## Provenance and unresolved limitations',f"Version {b['version']}; dataset/run/feature/software identities are preserved in the registry and completed artifacts. Large-file source SHA256 and independent geological validation remain unavailable. No actual evidence was externally transmitted to produce this report."]
    return '\n\n'.join(lines)+'\n'
