"""Workbook tables with explicit or detected header rows and original row identity."""
from pathlib import Path
from typing import Any


def _table(sheet, *, header_row=None, max_column=None, last_data_row=None):
    if header_row is not None and (not isinstance(header_row, int) or header_row < 1):
        raise ValueError('header_row must be a positive row number')
    options = {'values_only': True}
    if max_column is not None:
        if not isinstance(max_column, int) or max_column < 1:
            raise ValueError('max_column must be positive')
        options['max_col'] = max_column
    if last_data_row is not None:
        if not isinstance(last_data_row, int) or last_data_row < (header_row or 1):
            raise ValueError('last_data_row precedes header')
        options['max_row'] = last_data_row
    iterator = enumerate(sheet.iter_rows(**options), 1)
    headers = None
    for number, row in iterator:
        nonempty = [value for value in row if value is not None and str(value).strip()]
        detected = len(nonempty) >= 2 and all(isinstance(v, str) for v in nonempty)
        if number == header_row or (header_row is None and detected):
            stop = max((i+1 for i, v in enumerate(row) if v is not None and str(v).strip()), default=0)
            headers = [str(v).strip() if v is not None else '' for v in row[:stop]]
            actual_header_row = number
            break
        if header_row is None and number >= 20:
            raise ValueError('No unambiguous header in first 20 rows; specify header_row')
    if not headers:
        raise ValueError('Worksheet has no usable table header')
    rows = []
    for number, row in iterator:
        if any(value is not None and str(value).strip() for value in row):
            stop = max(i+1 for i, v in enumerate(row) if v is not None)
            rows.append((number, tuple(row[:max(stop, len(headers))])))
    width = max([len(headers), *(len(row) for _, row in rows)])
    original_headers = headers + [''] * (width-len(headers))
    # Make blank/duplicate keys explicit so no source column can overwrite another.
    counts = {h: original_headers.count(h) for h in original_headers}
    columns = [h if h and counts[h] == 1 else f'{h or "source"}__column_{i+1}'
               for i, h in enumerate(original_headers)]
    if len(set(columns)) != len(columns):
        raise ValueError('Generated column names collide; supply an explicit source layout')
    return actual_header_row, columns, original_headers, rows


def inspect_well_information(path: str | Path, *, header_row=None, max_column=None,
                             last_data_row=None) -> dict[str, Any]:
    try:
        from openpyxl import load_workbook
    except ImportError as exc:
        raise RuntimeError('XLSX reading requires openpyxl') from exc
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        report = []
        for sheet in workbook.worksheets:
            number, columns, original, rows = _table(sheet, header_row=header_row,
                max_column=max_column, last_data_row=last_data_row)
            report.append({'sheet': sheet.title, 'columns': columns, 'original_headings': original,
                           'header_row': number, 'data_rows': len(rows), 'mapping_required': True})
        return {'sheets': report, 'source_values_preserved_by_loader': True,
                'note': 'Explicit column mapping required; CRS, units and datums are not inferred.'}
    finally:
        workbook.close()


def load_well_information(path: str | Path, *, column_mapping: dict[str, str],
                          sheet_name: str | None = None, header_row=None,
                          max_column=None, last_data_row=None) -> list[dict[str, Any]]:
    try:
        from openpyxl import load_workbook
    except ImportError as exc:
        raise RuntimeError('XLSX reading requires openpyxl') from exc
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        if sheet_name is not None and sheet_name not in workbook.sheetnames:
            raise ValueError(f'Worksheet {sheet_name!r} does not exist')
        sheet = workbook[sheet_name] if sheet_name else workbook.active
        number, columns, original, rows = _table(sheet, header_row=header_row,
            max_column=max_column, last_data_row=last_data_row)
        missing = set(column_mapping.values()) - set(columns)
        if missing:
            raise ValueError(f'Mapped spreadsheet columns do not exist: {sorted(missing)}')
        results = []
        for source_row, row in rows:
            values = {key: row[i] if i < len(row) else None for i, key in enumerate(columns)}
            mapped = {target: values[source] for target, source in column_mapping.items()}
            mapped.update(source_values=values, source_sheet=sheet.title, source_row=source_row,
                          source_file=Path(path).name, source_header_row=number,
                          source_headings=tuple(original))
            results.append(mapped)
        return results
    finally:
        workbook.close()
