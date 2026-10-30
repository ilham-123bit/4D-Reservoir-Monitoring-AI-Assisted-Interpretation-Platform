"""LAS curve reader preserving original versus composite/interpreted provenance."""

from pathlib import Path
from typing import Any

import numpy as np
from .well_models import WellLogCurve, WellLogSet


def inspect_las(path: str | Path) -> dict[str, Any]:
    try:
        import lasio
    except ImportError as exc:
        raise RuntimeError("LAS reading requires the competition-data extra: pip install .[competition-data]") from exc
    path = Path(path)
    las = lasio.read(str(path), mnemonic_case="preserve")
    depth = np.asarray(las.index, dtype=float)
    curves = []
    null_marker = las.well.NULL.value
    for curve in las.curves:
        values = np.asarray(las[curve.mnemonic], dtype=float)
        null_count = int((~np.isfinite(values)).sum())
        if null_marker is not None:
            null_count += int(np.count_nonzero(values == float(null_marker)))
        curves.append({"mnemonic": curve.mnemonic, "unit": curve.unit,
                       "description": curve.descr, "sample_count": int(values.size), "null_count": null_count})
    return {"source_file": path.name, "well_id": str(las.well.WELL.value or ""),
            "provenance": "composite_interpreted" if "Composite_Interp" in path.name else "standard",
            "las_version": str(las.version.VERS.value), "depth_mnemonic": las.curves[0].mnemonic if las.curves else None,
            "depth_unit": las.curves[0].unit if las.curves else None,
            "depth_start": float(depth[0]) if depth.size else None,
            "depth_stop": float(depth[-1]) if depth.size else None,
            "sample_count": int(depth.size), "curves": curves,
            "data": {curve["mnemonic"]: np.asarray(las[curve["mnemonic"]], dtype=float).copy() for curve in curves}}


def load_las(path: str | Path) -> WellLogSet:
    """Load one LAS file as a provenance-preserving well log object."""
    report = inspect_las(path)
    curves = tuple(WellLogCurve(item["mnemonic"], item["unit"], item["description"], report["data"][item["mnemonic"]])
                   for item in report["curves"])
    metadata = {key: report[key] for key in ("las_version", "depth_start", "depth_stop", "sample_count")}
    return WellLogSet(report["well_id"] or None, report["source_file"], report["provenance"],
                      report["depth_mnemonic"], report["depth_unit"], curves, metadata)
