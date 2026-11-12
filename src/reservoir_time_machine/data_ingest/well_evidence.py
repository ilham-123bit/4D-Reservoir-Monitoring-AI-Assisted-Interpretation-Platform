"""Well-source evidence grouped by source identifier, without fabricating a seismic tie."""

from pathlib import Path
import re

import numpy as np

from .las_loader import inspect_las
from .well_information import load_well_information
from .well_tables import read_well_table


_DERIVED_CURVE_NAMES = {
    "PHIE", "PHIT", "PERM", "PERMH", "PERM_TRFM", "SWE", "SWT", "NPHIHC",
    "VCLB", "VCLD", "VCLW", "VDOLO", "VGAS", "VLIME", "VOIL", "VSAND",
    "VSHALE", "VSILT", "VWATER", "COALF", "FCOAL",
}


def _well_key(identifier):
    """Normalize case only; preserve meaningful source identifier digits."""
    text = str(identifier or "").strip().casefold()
    match = re.match(r"^(well[-_ ]?\d+)", text)
    return match.group(1).replace("_", "-").replace(" ", "-") if match else ""


def _new_group():
    return {"source_identifiers": [], "workbook_records": [], "logs": [], "tables": [],
            "spatial_association_available": False, "depth_time_association_available": False,
            "identity_rule": "case-insensitive source well ID; filename grouping is source-level only"}


def _table_numeric_qc(category, table):
    """Check declared table structure; this is not a seismic tie or interpretation."""
    expected = {"checkshots": ("TVDSS", "TWT"),
                "deviations": ("MDKB", "TVD", "INCL", "AZIM")}.get(category)
    result = {"status": "source_records_only", "numeric_rows": 0}
    if table.row_width_mismatch_count:
        return {**result, "status": "unresolved_variable_width_rows",
                "reason": "row widths differ; original fields are preserved without positional repair"}
    if expected is None:
        return result
    if tuple(str(x).upper() for x in table.declared_columns) != expected:
        return {**result, "status": "declared_columns_unexpected",
                "reason": "source-declared columns differ from expected fields"}
    try:
        values = np.asarray([[float(value) for value in row[:len(expected)]] for row in table.rows], dtype=float)
    except (TypeError, ValueError):
        return {**result, "status": "non_numeric_or_missing_source_values",
                "reason": "a declared numeric source field is missing or non-numeric"}
    if values.ndim != 2 or values.shape[0] == 0 or not np.isfinite(values).all():
        return {**result, "status": "non_finite_source_values", "numeric_rows": int(values.shape[0]) if values.ndim == 2 else 0}
    units = tuple((value or "").strip().casefold() for value in table.declared_units)
    if category == "checkshots":
        depth_monotonic = bool(np.all(np.diff(values[:, 0]) >= 0))
        time_monotonic = bool(np.all(np.diff(values[:, 1]) > 0))
        units_valid = units == ("m", "ms")
        return {"status": "structurally_valid_declared_checkshot_pairs" if units_valid and depth_monotonic and time_monotonic else "checkshot_qc_warning",
                "numeric_rows": int(values.shape[0]), "declared_units": list(table.declared_units),
                "depth_non_decreasing": depth_monotonic, "twt_strictly_increasing": time_monotonic,
                "reason": "source pairs only; sample origin, datum relation, and physical 4D registration remain unverified"}
    md_unit_valid = bool(units and units[0] == "m")
    md_monotonic = bool(np.all(np.diff(values[:, 0]) > 0))
    tvd_monotonic = bool(np.all(np.diff(values[:, 1]) >= 0))
    return {"status": "depth_survey_structurally_valid_orientation_units_missing" if md_unit_valid and md_monotonic and tvd_monotonic else "deviation_qc_warning",
            "numeric_rows": int(values.shape[0]), "declared_units": list(table.declared_units),
            "md_increasing": md_monotonic, "tvd_non_decreasing": tvd_monotonic,
            "reason": "inclination/azimuth units and reference are unavailable; no XY trajectory is computed"}


def well_evidence_summary(root):
    """Read source well evidence, group it by source ID, and retain association gates.

    Workbook coordinates, LAS curves and text-table records remain source context.
    They do not become a finding-to-well relationship unless independent CRS,
    datum, trajectory and seismic sample ties are available.
    """
    root = Path(root)
    well_root = root / "well_data"
    source_identity = [{"path": str(p.relative_to(root)), "size": p.stat().st_size,
                        "mtime_ns": p.stat().st_mtime_ns}
                       for p in sorted(well_root.rglob("*")) if p.is_file()]
    report = {
        "source_identity": source_identity,
        "association_available": False,
        "missing_association_evidence": [
            "well X/Y units, CRS, and datum",
            "authoritative SEG-Y coordinate CRS and header-to-map transform",
            "well trajectory orientation reference and units",
            "verified well datum and depth reference",
            "verified seismic sample origin and domain",
            "reviewed well-to-seismic time-depth tie",
        ],
        "evidence_classification": {
            "workbook": "source metadata; X/Y coordinate references unresolved",
            "standard_las": "source log curves; raw-versus-processed status not independently established",
            "composite_interpreted_las": "composite/interpreted source curves",
            "derived_curve_names": sorted(_DERIVED_CURVE_NAMES),
            "checkshots": "source-declared depth/time pairs; not tied to seismic samples",
            "deviations": "source-declared depth/inclination/azimuth records; no XY trajectory derived",
            "markers": "interpreted source markers; seismic horizon association unverified",
        },
        "logs": [], "tables": [], "well_count": 0, "errors": [], "records": [], "by_well": {},
    }
    workbook = well_root / "Challenge3_WellList_Information.xlsx"
    if workbook.exists():
        try:
            records = load_well_information(workbook, column_mapping={"well_id": "Well Name"},
                                            header_row=3, max_column=13)
            report["well_count"] = len(records)
            for item in records:
                values = item["source_values"]
                record = {"well_id": item["well_id"], "source": workbook.name,
                    "source_row": item["source_row"], "source_sheet": item["source_sheet"],
                    "source_header_row": item["source_header_row"],
                    "declared_values": {k: str(v) if v is not None else None for k, v in values.items()},
                    "coordinate_evidence": {"x": values.get("X"), "y": values.get("Y"),
                        "units": None, "crs": None, "datum": None,
                        "status": "source X/Y present; coordinate units, CRS, and datum unavailable"}}
                report["records"].append(record)
                key = _well_key(item["well_id"]) or f"unassigned:workbook:{item['source_row']}"
                group = report["by_well"].setdefault(key, _new_group())
                group["source_identifiers"].append(item["well_id"])
                group["workbook_records"].append(record)
        except (ImportError, ValueError, OSError, RuntimeError) as exc:
            report["errors"].append({"source": workbook.name, "type": type(exc).__name__})

    for path in sorted((well_root / "las").glob("*.las")):
        try:
            las = inspect_las(path)
            curves = []
            for curve in las["curves"]:
                values = np.asarray(las["data"].get(curve["mnemonic"], ()), dtype=float)
                finite = values[np.isfinite(values)]
                derived = curve["mnemonic"].upper().split(":")[0] in _DERIVED_CURVE_NAMES
                curves.append({**{key: curve[key] for key in ("mnemonic", "unit", "description", "sample_count", "null_count")},
                    "finite_count": int(finite.size),
                    "finite_min": float(finite.min()) if finite.size else None,
                    "finite_median": float(np.median(finite)) if finite.size else None,
                    "finite_max": float(finite.max()) if finite.size else None,
                    "evidence_kind": "derived_or_interpreted_petrophysical_curve" if derived else "source_log_response_semantics_unverified",
                    "raw_measurement_status": "not established from mnemonic alone"})
            log = {"source": path.name, "provenance": las["provenance"], "well_id": las["well_id"],
                "samples": las["sample_count"], "depth_mnemonic": las["depth_mnemonic"],
                "depth_unit": las["depth_unit"], "depth_start": las["depth_start"],
                "depth_stop": las["depth_stop"],
                "depth_reference_status": "LAS axis unit is source-declared; MD/TVD/datum relation unresolved",
                "curves": curves,
                "note": "Petrophysical property curves are static source context, not temporal fluid-change observations."}
            report["logs"].append(log)
            key = _well_key(las["well_id"]) or f"unassigned:las:{path.name}"
            group = report["by_well"].setdefault(key, _new_group())
            group["source_identifiers"].append(las["well_id"])
            group["logs"].append(log)
        except (ImportError, ValueError, OSError, RuntimeError) as exc:
            report["errors"].append({"source": path.name, "type": type(exc).__name__})

    for path in sorted(well_root.glob("*/*.txt")):
        try:
            table = read_well_table(path)
            match = re.match(r"(?i)^(well[-_ ]?\d+)", path.stem)
            file_well = match.group(1).replace("_", "-").replace(" ", "-") if match else None
            item = {"category": path.parent.name, "source": path.name, "well_id_from_filename": file_well,
                "well_key": _well_key(file_well), "records": len(table.rows),
                "declared_columns": list(table.declared_columns), "columns": list(table.columns),
                "declared_units": list(table.declared_units), "units": list(table.units),
                "ambiguous_rows": table.row_width_mismatch_count, "header_origin": table.header_origin,
                "source_line_numbers": list(table.source_line_numbers),
                "raw_records": [list(row) for row in table.rows],
                "raw_data_lines": list(table.raw_data_lines),
                "quality": _table_numeric_qc(path.parent.name, table),
                "evidence_kind": ("declared_checkshot_pairs" if path.parent.name == "checkshots" else
                    "source_deviation_survey" if path.parent.name == "deviations" else "interpreted_formation_markers"),
                "association_scope": "filename groups source records by well only; no seismic position or horizon tie"}
            report["tables"].append(item)
            key = item["well_key"] or f"unassigned:table:{path.name}"
            group = report["by_well"].setdefault(key, _new_group())
            group["source_identifiers"].append(file_well)
            group["tables"].append(item)
        except (OSError, UnicodeError, ValueError) as exc:
            report["errors"].append({"source": path.name, "type": type(exc).__name__})

    report["association_assessment"] = {
        "status": "unavailable", "spatial_well_to_seismic": False,
        "vertical_well_to_sample": False,
        "grouped_source_well_count": sum(bool(v["workbook_records"] or v["logs"] or v["tables"])
                                          for v in report["by_well"].values()),
        "reason": "workbook X/Y lacks declared units/CRS/datum; SEG-Y map CRS and sample origin unresolved; deviation orientation reference incomplete",
        "well_data_role": "contextual log, marker, checkshot, and deviation evidence; no finding-specific association",
    }
    return report
