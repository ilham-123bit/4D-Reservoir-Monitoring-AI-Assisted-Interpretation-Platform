"""Loss-preserving well tables; source metadata is not a seismic association."""
from dataclasses import dataclass
from pathlib import Path
import csv
import re


@dataclass(frozen=True)
class WellTable:
    source_file: str
    comment_lines: tuple[str, ...]
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    raw_data_lines: tuple[str, ...] = ()
    source_line_numbers: tuple[int, ...] = ()
    units: tuple[str | None, ...] = ()
    row_width_mismatch_count: int = 0
    header_origin: str = 'none'
    declared_columns: tuple[str, ...] = ()
    declared_units: tuple[str | None, ...] = ()


def _number(token):
    try:
        float(token)
        return True
    except ValueError:
        return token.casefold() in {'na', 'null', 'none', '-', ''}


def read_well_table(path: str | Path, *, has_header: bool | None = None) -> WellTable:
    """Preserve raw rows; pad missing trailing fields and never truncate surplus ones.

    Challenge well comments explicitly declare NAME/MDKB/TVDSS etc. Whitespace
    is collapsed only for those documented whitespace tables. Generic TSV/CSV
    keeps empty fields. Ambiguous generic text requires an explicit header policy.
    Units come only from source comment declarations and can remain unknown.
    """
    path = Path(path)
    comments, content, numbers = [], [], []
    for number, line in enumerate(path.read_text(errors='replace').splitlines(), 1):
        if not line.strip():
            continue
        if line.lstrip().startswith(('#', '!', ';')):
            comments.append(line.strip())
        else:
            content.append(line)
            numbers.append(number)
    declarations = []
    for line in comments:
        match = re.fullmatch(r'[#!;]\s*(NAME|MDKB|TVDSS|TVD|TWT|INCL|AZIM)\s*(?:\(([^)]+)\))?\s*', line, re.I)
        if match:
            declarations.append((match[1], match[2]))
    if not content:
        return WellTable(path.name, tuple(comments), (), ())
    declared = bool(declarations)
    delimiter = ',' if ',' in content[0] else ('\t' if '\t' in content[0] else None)

    def split(line):
        if declared or delimiter is None:
            return tuple(line.split())
        return tuple(value.strip() for value in next(csv.reader([line], delimiter=delimiter)))

    first = split(content[0])
    if has_header is None:
        if declared:
            header = False
        elif all(_number(value) for value in first):
            header = False
        elif any(_number(value) for value in first):
            # A named record with numeric fields is data, not a heading.
            header = False
        else:
            header = True  # retain historical generic all-text header support
    else:
        header = has_header
    raw = content[1:] if header else content
    line_numbers = numbers[1:] if header else numbers
    parsed = tuple(split(line) for line in raw)
    source_columns = tuple(name for name, _ in declarations) if declared and not header else (first if header else ())
    width = max([len(source_columns), *(len(row) for row in parsed)])
    columns = source_columns + tuple(f'column_{i+1}' for i in range(len(source_columns), width))
    units = tuple(unit for _, unit in declarations) if declared and not header else ()
    units += (None,) * (width-len(units))
    expected_width = len(source_columns) or len(first)
    mismatches = sum(len(row) != expected_width for row in parsed)
    declared_columns = tuple(name for name, _ in declarations)
    declared_units = tuple(unit for _, unit in declarations)
    if declared and mismatches:
        # Do not guess which declared depth field a short row omitted. Preserve
        # positional fields, but mark every numeric column's semantics unresolved.
        columns = tuple('NAME' if i == 0 and declarations[0][0].upper() == 'NAME'
                        else f'unmapped_column_{i+1}' for i in range(width))
        units = (None,) * width
    rows = tuple(row + ('',) * (width-len(row)) for row in parsed)
    return WellTable(path.name, tuple(comments), columns, rows, tuple(raw),
                     tuple(line_numbers), units, mismatches,
                     'source_comments' if declared and not header else ('data_header' if header else 'generated'),
                     declared_columns, declared_units)
