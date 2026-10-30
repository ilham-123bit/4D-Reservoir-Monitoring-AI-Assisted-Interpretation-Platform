"""Well-data value objects that preserve source provenance and unknown references."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class WellRecord:
    original_identifier: str | None
    x: float | None = None
    y: float | None = None
    coordinate_units: str | None = None
    coordinate_reference: str | None = None
    datum: str | None = None
    source_file: str | None = None
    source_sheet: str | None = None
    source_fields: dict[str, Any] = field(default_factory=dict)
    missing_metadata: tuple[str, ...] = ()


@dataclass(frozen=True)
class WellLogCurve:
    mnemonic: str
    unit: str
    description: str
    values: np.ndarray


@dataclass(frozen=True)
class WellLogSet:
    original_identifier: str | None
    source_file: str
    provenance: str
    depth_mnemonic: str | None
    depth_unit: str | None
    curves: tuple[WellLogCurve, ...]
    source_metadata: dict[str, Any] = field(default_factory=dict)


def well_record_from_source(row: dict[str, Any], *, source_file: str | None = None) -> WellRecord:
    """Convert an explicitly mapped workbook row; never infer units, datum, or CRS."""
    missing = tuple(key for key in ("x", "y", "coordinate_units", "coordinate_reference", "datum")
                    if row.get(key) in (None, ""))
    identifier = row.get("well_id")
    return WellRecord(str(identifier) if identifier not in (None, "") else None,
                      _optional_float(row.get("x")), _optional_float(row.get("y")),
                      row.get("coordinate_units"), row.get("coordinate_reference"), row.get("datum"),
                      source_file, row.get("source_sheet"), dict(row.get("source_values", {})), missing)


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)
