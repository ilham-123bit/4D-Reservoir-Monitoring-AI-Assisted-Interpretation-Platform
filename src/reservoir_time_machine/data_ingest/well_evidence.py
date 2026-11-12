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


def marker_checkshot_correlations(group):
    """Interpolate marker TVDSS against the same well's declared checkshot pairs.

    This is a source-table relationship only. It does not tie a well marker to
    a SEG-Y sample, and it refuses extrapolation or positional row repair.
    """
    checkshots = [item for item in group.get("tables", []) if item.get("category") == "checkshots"]
    markers = [item for item in group.get("tables", []) if item.get("category") == "markers"]
    valid_checkshots = [item for item in checkshots
        if item.get("quality", {}).get("status") == "structurally_valid_declared_checkshot_pairs"
        and item.get("declared_columns") == ["TVDSS", "TWT"]
        and item.get("declared_units") == ["m", "ms"]]
    if not valid_checkshots:
        return {"status": "unavailable", "reason": "No structurally valid same-well TVDSS/TWT checkshot table with declared m/ms units.", "markers": []}
    table = valid_checkshots[0]
    depths = np.asarray([float(row[0]) for row in table.get("raw_records", [])], dtype=float)
    times = np.asarray([float(row[1]) for row in table.get("raw_records", [])], dtype=float)
    order = np.argsort(depths)
    depths, times = depths[order], times[order]
    if len(depths) < 2 or not np.isfinite(depths).all() or not np.isfinite(times).all() or np.any(np.diff(depths) <= 0):
        return {"status": "unavailable", "reason": "Checkshot depth support is not finite and strictly increasing after sorting.", "markers": []}
    marker_table = next((item for item in markers
        if item.get("ambiguous_rows", 0) == 0
        and "TVDSS" in item.get("declared_columns", [])
        and item.get("declared_units", [])[item["declared_columns"].index("TVDSS")] == "m"), None)
    if marker_table is None:
        return {"status": "unavailable", "reason": "No unambiguous marker table with source-declared TVDSS in metres; MDKB values are not substituted.", "markers": []}
    columns = marker_table["declared_columns"]
    name_index = next((i for i, name in enumerate(columns) if name.upper() in {"NAME", "MARKER"}), None)
    depth_index = columns.index("TVDSS")
    if name_index is None:
        return {"status": "unavailable", "reason": "Marker table has no declared marker-name column.", "markers": []}
    output = []
    for row_number, row in enumerate(marker_table.get("raw_records", []), start=1):
        try:
            depth = float(row[depth_index])
        except (IndexError, TypeError, ValueError):
            continue
        if not np.isfinite(depth) or depth < depths[0] or depth > depths[-1]:
            continue
        output.append({"marker": str(row[name_index]), "source_row_index": row_number,
            "tvdss_m": depth, "interpolated_source_checkshot_twt_ms": float(np.interp(depth, depths, times)),
            "method": "linear interpolation within this well's source-declared TVDSS/TWT checkshot range"})
    return {"status": "source_marker_to_checkshot_interpolation_only" if output else "no_markers_within_checkshot_coverage",
        "checkshot_source": table.get("source"), "marker_source": marker_table.get("source"),
        "checkshot_tvdss_range_m": [float(depths[0]), float(depths[-1])],
        "checkshot_twt_range_ms": [float(np.min(times)), float(np.max(times))],
        "markers": output,
        "limitation": "Not a seismic tie: SEG-Y time domain/datum and well-to-grid spatial association remain unresolved."}


def marker_deviation_correlations(group):
    """Map marker MDKB to source-survey TVD only where both axes declare metres."""
    deviations = [item for item in group.get("tables", []) if item.get("category") == "deviations"
        and item.get("quality", {}).get("status", "").startswith("depth_survey_structurally_valid")
        and item.get("declared_columns") == ["MDKB", "TVD", "INCL", "AZIM"]
        and item.get("declared_units", [])[:2] == ["m", "m"]]
    markers = [item for item in group.get("tables", []) if item.get("category") == "markers"
        and item.get("ambiguous_rows", 0) == 0 and "MDKB" in item.get("declared_columns", [])
        and len(item.get("declared_units", [])) > item["declared_columns"].index("MDKB")
        and item["declared_units"][item["declared_columns"].index("MDKB")] == "m"]
    if not deviations or not markers:
        return {"status": "unavailable", "reason": "No same-well, source-declared metre MDKB deviation and unambiguous metre MDKB marker pair.", "markers": []}
    deviation = deviations[0]
    md_i, tvd_i = deviation["declared_columns"].index("MDKB"), deviation["declared_columns"].index("TVD")
    values = np.asarray([[float(row[md_i]), float(row[tvd_i])] for row in deviation.get("raw_records", [])], dtype=float)
    order = np.argsort(values[:, 0])
    values = values[order]
    if len(values) < 2 or not np.isfinite(values).all() or np.any(np.diff(values[:, 0]) <= 0) or np.any(np.diff(values[:, 1]) < 0):
        return {"status": "unavailable", "reason": "Deviation MDKB/TVD pairs are not finite and monotonic.", "markers": []}
    marker_table = markers[0]
    columns = marker_table["declared_columns"]
    name_i = next((i for i, name in enumerate(columns) if name.upper() in {"NAME", "MARKER"}), None)
    md_marker_i = columns.index("MDKB")
    if name_i is None:
        return {"status": "unavailable", "reason": "Marker table has no marker name column.", "markers": []}
    mapped = []
    for row_number, row in enumerate(marker_table.get("raw_records", []), start=1):
        try:
            md = float(row[md_marker_i])
        except (IndexError, TypeError, ValueError):
            continue
        if np.isfinite(md) and values[0, 0] <= md <= values[-1, 0]:
            mapped.append({"marker": str(row[name_i]), "source_row_index": row_number,
                "mdkb_m": md, "interpolated_source_deviation_tvd_m": float(np.interp(md, values[:, 0], values[:, 1])),
                "method": "linear interpolation within source MDKB/TVD deviation coverage"})
    return {"status": "source_marker_to_deviation_interpolation_only" if mapped else "no_markers_within_deviation_coverage",
        "deviation_source": deviation.get("source"), "marker_source": marker_table.get("source"),
        "deviation_mdkb_range_m": [float(values[0, 0]), float(values[-1, 0])],
        "markers": mapped,
        "limitation": "TVD reference is not reconciled to checkshot TVDSS; this does not establish TWT or seismic-grid position."}


def build_well_evidence_matrix(report):
    """Summarize actual source evidence for each well without assigning ties."""
    family_rules = {
        "gamma_ray": {"GR", "GR_P", "GAMMA", "CGR"},
        "resistivity": {"RT", "RT_P", "RXO", "RXO_P", "ILD", "LLD"},
        "sonic": {"DTC", "DTC_P", "DTCOM", "DTS", "DTS_P"},
        "density": {"DEN", "DENS_P", "DENB", "RHOB"},
        "neutron": {"NEUT_P", "NPHI", "PHIN"},
        "porosity_interpreted": {"PHIE", "PHIT"},
        "permeability_interpreted": {"PERM", "PERMH", "PERM_TRFM"},
        "saturation_or_volume_interpreted": {"SWE", "SWT", "VOIL", "VGAS", "VWATER"},
    }
    matrix = []
    for well_id, group in sorted(report.get("by_well", {}).items()):
        workbook = group.get("workbook_records", [])
        coordinate = workbook[0].get("coordinate_evidence", {}) if workbook else {}
        curves = {}
        log_summary = []
        for log in group.get("logs", []):
            log_summary.append({"source": log.get("source"), "provenance": log.get("provenance"),
                "samples": log.get("samples"), "depth_axis": log.get("depth_mnemonic"),
                "depth_unit": log.get("depth_unit"), "depth_range": [log.get("depth_start"), log.get("depth_stop")]})
            for curve in log.get("curves", []):
                mnemonic = curve.get("mnemonic", "").upper().split(":")[0]
                for family, names in family_rules.items():
                    if mnemonic in names:
                        curves.setdefault(family, set()).add(mnemonic)
        tables = group.get("tables", [])
        category = {name: [item for item in tables if item.get("category") == name] for name in ("markers", "checkshots", "deviations")}
        marker_records = []
        for item in category["markers"]:
            columns = item.get("declared_columns", [])
            name_index = next((i for i, value in enumerate(columns) if value.upper() in {"NAME", "MARKER"}), None)
            depth_columns = [name for name in ("MDKB", "TVDSS", "TVD") if name in columns]
            names = [str(row[name_index]) for row in item.get("raw_records", []) if name_index is not None and name_index < len(row)]
            marker_records.append({"source": item.get("source"), "count": item.get("records"),
                "declared_depth_columns": depth_columns, "names": names,
                "quality": item.get("quality", {}).get("status"),
                "twt_association": "checkshot_interpolated only when a same-well TVDSS value and valid checkshot support exist"})
        checkshot_records = [{"source": item.get("source"), "count": item.get("records"),
            "units": item.get("declared_units"), "quality": item.get("quality", {}).get("status")} for item in category["checkshots"]]
        deviation_records = [{"source": item.get("source"), "count": item.get("records"),
            "units": item.get("declared_units"), "quality": item.get("quality", {}).get("status")} for item in category["deviations"]]
        matrix.append({"well_id": well_id,
            "workbook_xy": [coordinate.get("x"), coordinate.get("y")] if coordinate else None,
            "xy_reference": "units, CRS, and datum not declared; not compared to SEG-Y trace coordinates",
            "las_sources": log_summary,
            "curve_families": {key: sorted(value) for key, value in curves.items()},
            "markers": marker_records, "checkshots": checkshot_records, "deviations": deviation_records,
            "marker_checkshot_relationship": marker_checkshot_correlations(group),
            "marker_deviation_relationship": marker_deviation_correlations(group),
            "seismic_association": "unverified"})
    return matrix


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
                low = float(finite.min()) if finite.size else None
                high = float(finite.max()) if finite.size else None
                unit = str(curve.get("unit") or "").strip().casefold().replace(" ", "")
                quality_flags = []
                if unit in {"frac", "fraction", "v/v", "vv"} and finite.size and (low < 0 or high > 1):
                    quality_flags.append("declared_fraction_outside_0_to_1_review_scaling_or_sentinels")
                if derived and finite.size and max(abs(low), abs(high)) >= 100:
                    quality_flags.append("extreme_finite_interpreted_value_review_source_encoding_or_sentinels")
                curves.append({**{key: curve[key] for key in ("mnemonic", "unit", "description", "sample_count", "null_count")},
                    "finite_count": int(finite.size),
                    "finite_min": low,
                    "finite_median": float(np.median(finite)) if finite.size else None,
                    "finite_max": high,
                    "value_quality_flags": quality_flags,
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
    report["well_evidence_matrix"] = build_well_evidence_matrix(report)
    return report
