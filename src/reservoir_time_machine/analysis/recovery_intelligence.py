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
        interpretation_record = _finding_interpretation_record(
            finding, assessment_id=registry.get("id", "unknown"), action=action
        )
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
            "interpretation_record": interpretation_record,
        })
    return result


def _finding_interpretation_record(finding: dict, *, assessment_id: str,
                                   action: dict | None) -> dict:
    """Expose one source-traceable, non-probabilistic Stage 3 evidence record."""
    flags = finding.get("detector_threshold_flagged_voxels",
                        finding.get("detector_flagged_voxels", {}))
    detector_flags = ({str(name): int(value) for name, value in flags.items()}
                      if isinstance(flags, dict) else {})
    location = finding.get("array_location", [None, None, None])
    extent = finding.get("extent")
    alternatives = list(finding.get("plausible_alternatives", []))
    repeatability = finding.get("local_repeatability", {})
    return {
        "schema": "rtm-investigation-interpretation-1",
        "id": finding["id"],
        "source_assessment_id": assessment_id,
        "source_record_rank": finding.get("rank"),
        "priority_basis": "Stage 3 QC follow-up action" if action else "historical Stage 2 screening order",
        "location": {
            "inline_id": finding.get("inline_id"),
            "crossline_id": finding.get("crossline_id"),
            "sample_index": finding.get("sample_index", location[2] if len(location) == 3 else None),
            "array_location": list(location),
            "extent_array_bounds_inclusive": extent,
            "extent_semantics": "component bounding extent; exact membership is in the protected label map",
        },
        "observed": {
            "signed_monitor_minus_baseline_mean_source_units": finding.get("signed_difference_mean"),
            "absolute_difference_mean_source_units": finding.get("absolute_difference_mean"),
            "nrms_mean_percent": finding.get("nrms_mean_percent"),
            "baseline_rms_mean_source_units": finding.get("rms_baseline_mean"),
            "monitor_rms_mean_source_units": finding.get("rms_monitor_mean"),
            "monitor_to_baseline_rms_ratio": finding.get("monitor_to_baseline_rms_ratio"),
            "normalized_difference_mean": finding.get("normalized_difference_mean"),
            "component_voxels": finding.get("component_voxels"),
            "source_amplitude_units": "unspecified",
            "sample_domain": "array sample index; absolute time/depth origin unverified",
        },
        "model_evidence": {
            "mad_only": bool(finding.get("mad_only", False)),
            "mad_member_fraction": finding.get("data_support", {}).get("MAD_member_fraction"),
            "detector_threshold_flagged_voxels": detector_flags,
            "detector_score_means_nonprobabilistic": finding.get("detector_score_means_nonprobabilistic", {}),
        },
        "data_quality": {
            "minimum_paired_sample_support": finding.get("min_paired_support"),
            "edge_risk": finding.get("edge_risk", finding.get("edge_artifact_risk")),
            "edge_qc": finding.get("edge_qc", {}),
            "local_repeatability": repeatability,
            "physical_4d_registration": "unverified",
            "trace_pairing": "exact Baseline-grid trace-key pairing; not physical registration",
        },
        "conditional_interpretation": finding.get(
            "finding_specific_observation",
            "Observed component-scale paired-array difference; geological cause is unresolved.",
        ),
        "alternatives": alternatives or [
            "Registration, phase, wavelet, amplitude-scaling, acquisition, processing, or noise differences remain possible alternatives."
        ],
        "confidence": {
            "seismic_observation": finding.get("observation_confidence", "Not assessed"),
            "geological_interpretation": finding.get("geological_interpretation_confidence", "Unresolved"),
        },
        "reservoir_relevance": "Unverified; no validated reservoir interval or well-to-seismic association is available.",
        "recovery_status": "Investigation record only; no recovery opportunity or drilling target is established.",
        "next_tests": {
            "geophysics": finding.get("geophysics_next_action", "Validate registration and repeatability."),
            "petrophysics": finding.get("petrophysics_next_action", "Verify well and depth/time references before correlation."),
            "reservoir_engineering": finding.get("reservoir_engineering_next_action", "Obtain interval-linked production and pressure evidence before recovery interpretation."),
        },
    }


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
