"""Decision-support records built only from the validated Stage 3 package.

The queue ranks follow-up work, not hydrocarbon locations. It intentionally does
not calculate a recovery probability or turn statistical screening into geology.
"""
from __future__ import annotations


def investigation_records(registry: dict) -> list[dict]:
    """Return QC actions first, followed by historical seismic-screen records.

    The Stage 3 queue order is preserved verbatim. Historical Stage 2 rank is
    never silently replaced by the action order, and neither is called a
    recovery-target ranking.
    """
    findings = {item["id"]: item for item in registry.get("historical_findings", [])}
    actions_by_id = {
        item["finding"]["id"]: item
        for item in registry.get("qc_follow_up_actions", [])
        if isinstance(item.get("finding"), dict) and item["finding"].get("id") in findings
    }
    ordered_ids = [item["finding"]["id"] for item in registry.get("qc_follow_up_actions", [])
                   if item.get("finding", {}).get("id") in findings]
    ordered_ids.extend(item["id"] for item in registry.get("historical_findings", [])
                       if item["id"] not in set(ordered_ids))

    result = []
    for index, finding_id in enumerate(ordered_ids):
        finding = findings[finding_id]
        action = actions_by_id.get(finding_id)
        result.append({
            "id": finding_id,
            "kind": "qc_follow_up" if action else "historical_screening_record",
            "queue_rank": index + 1 if action else None,
            "historical_stage2_rank": finding.get("rank"),
            "finding": finding,
            "queue_basis": action.get("basis") if action else None,
            "observation": finding.get("finding_specific_observation", finding.get("interpretation", "")),
            "observation_confidence": finding.get("observation_confidence", "Not assessed"),
            "geological_confidence": finding.get("geological_interpretation_confidence", "Unresolved"),
            "alternatives": list(finding.get("plausible_alternatives", [])),
            "actions": {
                "geophysics": finding.get("geophysics_next_action", "No geophysics action is recorded."),
                "petrophysics": finding.get("petrophysics_next_action", "No verified well tie is available."),
                "reservoir_engineering": finding.get("reservoir_engineering_next_action", "No reservoir action is established."),
            },
            "recovery_opportunity": "unverified",
            "recovery_chance_of_success": "Not yet quantifiable — reservoir validation required.",
        })
    return result


def default_investigation_id(records: list[dict]) -> str | None:
    """Choose first validated QC action; otherwise first historical record."""
    if not records:
        return None
    first_action = next((item for item in records if item["kind"] == "qc_follow_up"), None)
    return (first_action or records[0])["id"]


def select_investigation(registry: dict, finding_id: str) -> dict:
    """Resolve one registry ID or fail rather than fabricating a location."""
    match = next((item for item in investigation_records(registry) if item["id"] == finding_id), None)
    if match is None:
        raise KeyError(f"finding ID is not present in the validated Stage 3 registry: {finding_id}")
    return match
