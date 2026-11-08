"""Source-level well evidence; no seismic tie is constructed from inventory alone."""
from pathlib import Path
from .las_loader import inspect_las
from .well_tables import read_well_table
from .well_information import load_well_information


def well_evidence_summary(root):
    root = Path(root)
    source_identity=[{'path':str(p.relative_to(root)),'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in sorted((root/'well_data').rglob('*')) if p.is_file()]
    report = {'source_identity':source_identity,'association_available':False,
              'missing_association_evidence':['verified shared CRS/coordinate transform',
                   'verified sample domain/origin','well datum and trajectory QC','reviewed seismic time-depth tie'],
              'logs':[], 'tables':[], 'well_count':0,'errors':[], 'records':[], 'by_well':{}}
    workbook = root/'well_data/Challenge3_WellList_Information.xlsx'
    if workbook.exists():
        try:
            records = load_well_information(workbook,column_mapping={'well_id':'Well Name'},header_row=3,max_column=13)
            report['well_count'] = len(records)
            report['records']=[{'well_id':r['well_id'],'source':'Challenge3_WellList_Information.xlsx','source_row':r['source_row'],'source_sheet':r['source_sheet'],'source_header_row':r['source_header_row'],
                'declared_values':{k:str(v) if v is not None else None for k,v in r['source_values'].items()}} for r in records]
            report['coordinate_fields_present'] = all('X' in r['source_values'] and 'Y' in r['source_values'] for r in records)
        except (ImportError,ValueError,OSError,RuntimeError) as exc: report['errors'].append({'source':workbook.name,'type':type(exc).__name__})
    for path in sorted((root/'well_data/las').glob('*.las')):
        try:
            log = inspect_las(path)
            report['logs'].append({'source':path.name,'provenance':log['provenance'],
                'well_id':log['well_id'],'samples':log['sample_count'],'depth_unit':log['depth_unit'],
                'depth_start':log['depth_start'],'depth_stop':log['depth_stop'],
                'curves':[{k:c[k] for k in ('mnemonic','unit','description','sample_count','null_count')} for c in log['curves']],
                'note':'Curve provenance must be reviewed; petrophysical/interpreted curves are not direct fluid-change observations.'})
        except (ImportError,ValueError,OSError,RuntimeError) as exc: report['errors'].append({'source':path.name,'type':type(exc).__name__})
    for path in sorted((root/'well_data').glob('*/*.txt')):
        try:
            table = read_well_table(path)
            report['tables'].append({'category':path.parent.name,'source':path.name,'records':len(table.rows),
                'declared_columns':list(table.declared_columns),'declared_units':list(table.declared_units),
                'ambiguous_rows':table.row_width_mismatch_count})
        except (OSError,UnicodeError,ValueError) as exc:
            report['errors'].append({'source':path.name,'type':type(exc).__name__})

    for record in report['records']:
        key=str(record['well_id']).strip().casefold() if record['well_id'] is not None and str(record['well_id']).strip() else 'unassigned:workbook:'+str(record['source_sheet'])+':'+str(record['source_row'])
        group=report['by_well'].setdefault(key,{'source_identifiers':[],'workbook_records':[],'logs':[],
            'identity_rule':'exact case-insensitive source ID only; not a verified well-seismic tie'})
        group['source_identifiers'].append(record['well_id']);group['workbook_records'].append(record)
    for log in report['logs']:
        key=log['well_id'].strip().casefold() or 'unassigned:'+log['source']
        group=report['by_well'].setdefault(key,{'source_identifiers':[],'workbook_records':[],'logs':[],
            'identity_rule':'exact case-insensitive source ID only; not a verified well-seismic tie'})
        group['source_identifiers'].append(log['well_id']);group['logs'].append(log)
    return report
