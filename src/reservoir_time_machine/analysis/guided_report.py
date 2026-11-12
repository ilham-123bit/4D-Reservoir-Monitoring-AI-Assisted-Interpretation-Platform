"""Deterministic page reports from the SAME bounded finding and well inventory.

No external calls or well-seismic associations. Plane discovery is screening, not a
full-volume search; the rank is a transparent inspection heuristic, never probability.
"""
import math
from ..visualization.slices import region_representative
from ..ai.grounding import finding_packet, interpret


def automatic_location(labels, zones, score):
    if not zones:
        return None
    zone = max(zones, key=lambda z:(z.max_strength*math.sqrt(z.voxel_count),-z.label))
    return {'location':list(region_representative(labels,zone,score)), 'label':zone.label,
            'cells':zone.voxel_count,'ranking':'maximum absolute difference × sqrt(component cells)',
            'rank_value':float(zone.max_strength*math.sqrt(zone.voxel_count))}


def page_report(result, finding, wells, page, discovery, *, shared_interpretation=None):
    log_sources=wells.get('logs',[])
    curves=sorted({c['mnemonic']+(' ['+c['unit']+']' if c['unit'] else ' [unit unavailable]')
                   for log in log_sources for c in log['curves']})
    table_counts={}
    table_qc={}
    for table in wells.get('tables',[]):
        category=table.get('category','unknown')
        table_counts[category]=table_counts.get(category,0)+int(table.get('records',0))
        status=table.get('quality',{}).get('status','not assessed')
        table_qc.setdefault(category,{})[status]=table_qc.setdefault(category,{}).get(status,0)+1
    report={'page':page,'scope':discovery,'finding_id':None,
        'wells':{'well_records':wells['well_count'],'las_sources':len(log_sources),
                 'table_sources':len(wells.get('tables',[])),'curve_types':curves,
                 'source_well_groups':sorted(wells.get('by_well',{})),
                 'table_record_counts_by_category':table_counts,'table_qc_status_counts':table_qc,
                 'evidence_classification':wells.get('evidence_classification',{}),
                 'association_available':False,'missing':wells['missing_association_evidence'],
                 'errors':wells.get('errors',[])},
        'source':'deterministic local analysis report; no external transmission',
        'limitation':'This finding is classified within one completed tile; registry coverage lists other completed and pending tiles. No full-volume reservoir assessment is implied.'}
    report['page_assessment']={
        'Overview and QC':f"Selected tile {result['bounds']} has {int(result['valid'].sum())} eligible source cells. Main report findings remain locally ranked; data comparability must be established before reservoir interpretation.",
        '3D explorer':'Orthogonal sections locate the selected member. A 2D component is not proof of a 3D geological body; continuity away from this sample has not been classified.',
        '4D comparison':'The numerical evidence below compares the same candidate using paired support. NRMS measures dissimilarity, not fluid movement or calibrated confidence.',
        'Change detection':'Candidate priority uses statistical strength, ML inspection coverage and grid extent. Review zero/positive detector disagreement rather than counting correlated votes as confirmation.',
        'Survey transition':'Presentation is automatically framed on the report finding. Sample, inline and crossline endpoints and signed differences share the same representative; no synthetic time history is inferred.',
        'Well inventory':f"{len(curves)} curve types/units are available as source context. No log interval or well location is assigned to this seismic candidate until the association gate is satisfied.",
        'Evidence explanation':'Interpretation uses this report finding, typed calculated evidence and available well-source context. Unsupported numbers, references and geological assertions are rejected.'
    }.get(page,'Review the report scope and evidence before interpretation.')
    if 'completed_tiles' in discovery:
        report['page_assessment'] += f" Completed registry coverage: {discovery['completed_tiles']}/{discovery['planned_tiles']} tiles; {discovery['eligible_center_cells']} eligible sample centers."
    if finding:
        report.update(finding_id=finding['id'],location=finding['representative'],metrics=finding['metrics'],
            methods=finding['method_evidence'],bounds=finding['bounds'],priority=finding['priority'],
            interpretation=shared_interpretation or interpret(finding_packet(result,finding,well_context=wells)),
            discipline_actions=[
                {'discipline':'Geophysics','action':'Verify event alignment, repeatability, and edge sensitivity.',
                 'evidence_ids':[finding['id']+':registration',finding['id']+':quality'],
                 'uncertainty':'Physical 4D registration remains unverified.'},
                {'discipline':'Petrophysics','action':'Review source logs and markers after a valid well tie is established.',
                 'evidence_ids':[finding['id']+':well_sources',finding['id']+':well'],
                 'uncertainty':'No well interval is associated with this seismic finding.'},
                {'discipline':'Reservoir engineering','action':'Gather pressure and production history before recovery screening.',
                 'evidence_ids':[finding['id']+':well_sources',finding['id']+':registration'],
                 'uncertainty':'Pressure, production, and injection time series are unavailable here.'},
            ])
    return report
