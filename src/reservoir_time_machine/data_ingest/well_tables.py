"""Loss-preserving reader for checkshot, deviation, and marker text tables."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WellTable:
    source_file: str
    comment_lines: tuple[str, ...]
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


def read_well_table(path: str | Path) -> WellTable:
    """Read delimiter-separated or whitespace data without assigning unit semantics."""
    path = Path(path)
    comments, content = [], []
    for line in path.read_text(errors="replace").splitlines():
        text = line.strip()
        if not text:
            continue
        if text.startswith(("#", "!", ";")):
            comments.append(text)
        else:
            content.append(text)
    if not content:
        return WellTable(path.name, tuple(comments), (), ())
    delimiter = "\t" if "\t" in content[0] else ("," if "," in content[0] else None)

    def split(line: str) -> tuple[str, ...]:
        return tuple(part.strip() for part in (line.split(delimiter) if delimiter else line.split()))

    first = split(content[0])
    header = any(any(character.isalpha() for character in part) for part in first)
    columns = first if header else tuple(f"column_{i+1}" for i in range(len(first)))
    data_lines = content[1:] if header else content
    rows = tuple(split(line) for line in data_lines)
    return WellTable(path.name, tuple(comments), columns, rows)
