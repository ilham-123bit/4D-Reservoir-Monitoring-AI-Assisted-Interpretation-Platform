"""Data availability checks that never substitute or fabricate missing inputs."""

from pathlib import Path

from ..data_audit.audit import BASELINE, MONITOR, WELLS


def challenge3_availability(root: str | Path) -> dict[str, object]:
    root = Path(root).expanduser()
    expected = {"baseline": BASELINE, "monitor": MONITOR, "well_inventory": WELLS}
    status = {name: (root / relative).is_file() for name, relative in expected.items()}
    missing = [expected[key] for key, present in status.items() if not present]
    return {"root": str(root), "available": status["baseline"] and status["monitor"],
            "well_inventory_available": status["well_inventory"], "expected_files": expected,
            "status": status, "missing": missing}
