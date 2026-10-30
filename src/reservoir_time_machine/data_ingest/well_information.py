"""Well-list spreadsheet loading with source columns preserved and no guessed mappings."""

from pathlib import Path
from typing import Any


def inspect_well_information(path: str | Path) -> dict[str, Any]:
    try:
        from openpyxl import load_workbook
    except ImportError as exc:
        raise RuntimeError("XLSX reading requires the competition-data extra: pip install .[competition-data]") from exc
    workbook = load_workbook(path, read_only=True, data_only=True)
    report = []
    for sheet in workbook.worksheets:
        iterator = sheet.iter_rows(values_only=True)
        headers = next(iterator, ())
        columns = [str(value).strip() if value is not None else "" for value in headers]
        count = sum(1 for row in iterator if any(value is not None for value in row))
        report.append({"sheet": sheet.title, "columns": columns, "data_rows": count,
                       "mapping_required": True})
    workbook.close()
    return {"sheets": report, "source_values_preserved_by_loader": False,
            "note": "Column mapping must be supplied from the inspected source headings; coordinates and datums are not inferred."}


def load_well_information(path: str | Path, *, column_mapping: dict[str, str],
                          sheet_name: str | None = None) -> list[dict[str, Any]]:
    """Load workbook rows with original source fields plus explicitly mapped attributes.

    `column_mapping` maps standardized keys (e.g. well_id, x, y, crs, datum) to exact
    spreadsheet headings. Missing mappings remain None and source values remain intact.
    """
    try:
        from openpyxl import load_workbook
    except ImportError as exc:
        raise RuntimeError("XLSX reading requires openpyxl") from exc
    workbook = load_workbook(path, read_only=True, data_only=True)
    if sheet_name is not None and sheet_name not in workbook.sheetnames:
        workbook.close()
        raise ValueError(f"Worksheet {sheet_name!r} does not exist")
    sheet = workbook[sheet_name] if sheet_name else workbook.active
    iterator = sheet.iter_rows(values_only=True)
    headers = [str(value).strip() if value is not None else "" for value in next(iterator, ())]
    indices = {heading: i for i, heading in enumerate(headers) if heading}
    missing_columns = [source for source in column_mapping.values() if source not in indices]
    if missing_columns:
        workbook.close()
        raise ValueError(f"Mapped spreadsheet columns do not exist: {missing_columns}")
    results = []
    for row in iterator:
        if not any(value is not None for value in row):
            continue
        source_values = {heading: row[i] if i < len(row) else None for heading, i in indices.items()}
        mapped = {target: source_values.get(source) for target, source in column_mapping.items()}
        mapped["source_values"] = source_values
        mapped["source_sheet"] = sheet.title
        results.append(mapped)
    workbook.close()
    return results
