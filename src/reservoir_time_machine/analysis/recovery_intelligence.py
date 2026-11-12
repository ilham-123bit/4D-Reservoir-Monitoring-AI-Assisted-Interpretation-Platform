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


def reservoir_evolution_readiness(well_context: dict, stage3_assessment: dict) -> dict:
    """Gate production-response and low-response interpretations on required data.

    This does not label a seismic anomaly as depletion or an unswept reservoir.
    It returns explicit eligibility and the data actions needed to make those
    interpretations testable for the current evidence package.
    """
    association = well_context.get("association_assessment", {})
    tie = stage3_assessment.get("well_and_sample_tie", {})
    provenance = stage3_assessment.get("provenance", {})
    sample_domain_verified = provenance.get("sample_domain") not in {None, "unknown", "unverified"}
    sample_origin_verified = provenance.get("sample_origin") is not None
    registration_verified = provenance.get("physical_4d_registration") == "verified"
    spatial_tie = bool(association.get("spatial_well_to_seismic"))
    vertical_tie = bool(association.get("vertical_well_to_sample")) and sample_domain_verified and sample_origin_verified
    counts = {category: sum(len([table for table in item.get("tables", []) if table.get("category") == category])
                            for item in well_context.get("by_well", {}).values())
              for category in ("markers", "checkshots", "deviations")}
    limitations = [
        "SEG-Y trace SourceX/SourceY values are zero and CoordinateUnits is unset in the inspected source headers; the well-list X/Y CRS, units, and datum are not declared.",
        "SEG-Y headers report a 2,000 microsecond sample interval and zero recording delay, while the textual header reports 2 ms sampling and 2,600 ms record length; the files do not explicitly identify the sample domain as TWT or provide a verified geological datum tie.",
        "The supplied checkshots provide well-local source-declared TVDSS/TWT pairs, but they do not establish a well-to-seismic grid relationship.",
        "Physical Baseline/Monitor 4D registration remains unverified.",
        "The enumerated challenge inputs contain no production, injection, or pressure-history files; static log curves are not production-history evidence.",
    ]
    prerequisites = [
        {"priority": 1, "action": "Obtain an authoritative survey coordinate reference and a trace-coordinate or navigation transform, plus well coordinate CRS/units/datum and deviation reference details.", "unlocks": "Well-to-grid spatial association and candidate/well proximity tests."},
        {"priority": 2, "action": "Obtain the processor/survey time-domain and datum statement confirming the SEG-Y sample origin/domain, and validate it against source checkshots or a synthetic tie.", "unlocks": "Reservoir marker-to-sample mapping and interval-restricted seismic analysis."},
        {"priority": 3, "action": "Validate local 1995/2006 registration, wavelet and amplitude repeatability in the target interval.", "unlocks": "Separating repeatable production-related response from acquisition/processing and registration effects."},
        {"priority": 4, "action": "Provide dated production, injection, completion/intervention and pressure records for the survey interval.", "unlocks": "Testing depletion, sweep, injector influence and limited-production-response hypotheses."},
        {"priority": 5, "action": "Provide reviewed reservoir horizons/fault interpretations and provenance for processed petrophysical curves and fluid contacts.", "unlocks": "Compartment connectivity, reservoir-quality and remaining-hydrocarbon investigation assessment."},
    ]
    discipline_actions = [
        {"discipline": "Geophysics", "action": "Confirm SEG-Y time-domain/datum and spatial navigation metadata; then validate local 1995/2006 registration, wavelet and amplitude repeatability before attributing change to production."},
        {"discipline": "Geology / geoscience", "action": "Provide reviewed reservoir horizons and fault/compartment interpretation tied to the survey grid; correlate named well markers only after spatial and time references are established."},
        {"discipline": "Petrophysics", "action": "Verify LAS curve provenance, units, depth reference and interpretation workflow; treat PHIE, PERM, SWE, VOIL and VGAS as interpreted curves until independently validated."},
        {"discipline": "Reservoir engineering", "action": "Supply time-aligned production, injection and pressure history to test depletion, sweep and injector-response hypotheses."},
        {"discipline": "Production engineering", "action": "Provide well-by-well rate, completion/intervention, shut-in and injection records spanning the 1995 and 2006 surveys."},
        {"discipline": "Drilling / well planning", "action": "No location, well type or drilling depth is supported until a reservoir interval, structure, well tie and development case are validated."},
    ]
    statuses = {
        "production_response": "eligible_for_reservoir_test" if spatial_tie and vertical_tie and registration_verified else "withheld_insufficient_tie_or_registration",
        "limited_response_or_poor_sweep": "eligible_for_comparative_screen" if spatial_tie and vertical_tie and registration_verified else "withheld_insufficient_reservoir_mask_and_production_context",
        "fluid_or_pressure_mechanism": "not_assessed_without_independent_well_or_production_constraints",
        "development_opportunity_count": int(stage3_assessment.get("candidate_interpretation", {}).get("qualified_recovery_opportunity_count", 0)),
    }
    return {"schema": "rtm-reservoir-evolution-readiness-1", "status": "reservoir_inference_withheld_pending_evidence",
        "statuses": statuses, "well_source_counts": {"wells": len(well_context.get("well_evidence_matrix", [])), **counts},
        "limitations": limitations, "next_data_actions": prerequisites,
        "discipline_actions": discipline_actions,
        "interpretation_rule": "Low seismic response is not evidence of remaining hydrocarbons by itself; compare only inside a verified reservoir interval with adequate detectability, 4D QC, connectivity, and production context.",
        "well_sample_tie_record": tie}
