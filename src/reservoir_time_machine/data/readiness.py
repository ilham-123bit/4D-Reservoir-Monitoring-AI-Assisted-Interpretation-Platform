"""Conservative capability assessments from explicit, provenance-linked declarations.

This assesses supplied evidence, not independent geological truth. Missing evidence
never enables a capability. Structural cache identity is not source authentication.
"""
from dataclasses import dataclass, asdict
from typing import Mapping, Any
import math
import numpy as np


@dataclass(frozen=True)
class Capability:
    available: bool
    missing_requirements: tuple[str, ...]
    evidence_references: tuple[str, ...] = ()


def assess_capabilities(baseline, monitor, *, evidence: Mapping[str, Any] | None = None):
    """Do not infer model, calibration, map or well readiness from array shape."""
    a, b = baseline.metadata.source_metadata, monitor.metadata.source_metadata
    declared = dict(evidence or {})
    result = {}

    def assess(name, requirements, references=()):
        missing = tuple(label for label, satisfied in requirements if not satisfied)
        result[name] = Capability(not missing, missing, tuple(r for r in references if r))
        return not missing

    def known(value):
        return isinstance(value, str) and value.strip().casefold() not in {
            '', 'unknown', 'unverified', 'not supplied', 'not calculated', 'not established'}

    def finite_origin(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

    axes_match = all(
        a.get(name, a.get(name + '_selected')) is not None
        and b.get(name, b.get(name + '_selected')) is not None
        and tuple(a.get(name, a.get(name + '_selected'))) == tuple(b.get(name, b.get(name + '_selected')))
        and len(a.get(name, a.get(name + '_selected'))) == baseline.data.shape[axis]
        for axis, name in enumerate(('inline_ids', 'crossline_ids')))
    masks_match = (baseline.missing_trace_mask is not None and monitor.missing_trace_mask is not None
                   and np.array_equal(baseline.missing_trace_mask, monitor.missing_trace_mask))
    paired = assess('trace_key_preliminary_screening', [
        ('matching canonical grid', baseline.data.shape == monitor.data.shape),
        ('matching explicit trace axes', axes_match),
        ('matching missing-trace coverage', masks_match),
        ('Baseline trace-key verification', a.get('trace_key_pairing_verified') is True),
        ('Monitor trace-key verification', b.get('trace_key_pairing_verified') is True),
        ('paired grid available', a.get('paired_grid_available') is True and b.get('paired_grid_available') is True),
        ('explicit missing-trace masks', baseline.missing_trace_mask is not None and monitor.missing_trace_mask is not None),
    ])
    domain_known = (baseline.metadata.sample_axis_domain in {'two_way_time', 'one_way_time', 'depth'}
                    and baseline.metadata.sample_axis_domain == monitor.metadata.sample_axis_domain)
    axis_known = (domain_known and known(baseline.metadata.sample_units)
                  and baseline.metadata.sample_units == monitor.metadata.sample_units
                  and a.get('sample_origin_units') == baseline.metadata.sample_units
                  and known(baseline.metadata.sample_interval_units) and finite_origin(a.get('sample_origin'))
                  and a.get('sample_origin') == b.get('sample_origin')
                  and known(a.get('sample_origin_units'))
                  and a.get('sample_origin_units') == b.get('sample_origin_units')
                  and baseline.metadata.sample_interval == monitor.metadata.sample_interval
                  and baseline.metadata.sample_interval_units == monitor.metadata.sample_interval_units)
    registration_ref = b.get('registration_verification_reference')
    physical = assess('physical_4d_registration', [
        ('trace-key preliminary screening', paired),
        ('verified sample domain/origin/interval', axis_known),
        ('physical registration verification', b.get('physical_4d_registration_verified') is True),
        ('registration verification reference', known(registration_ref)),
    ], (registration_ref,))
    spatial_ref = declared.get('spatial_verification_reference')
    spatial = assess('spatial_coordinate_association', [
        ('verified coordinate association', declared.get('spatial_verified') is True),
        ('coordinate reference', known(declared.get('coordinate_reference'))),
        ('coordinate units', known(declared.get('coordinate_units'))),
        ('spatial verification reference', known(spatial_ref)),
    ], (spatial_ref,))
    well_ref = declared.get('well_depth_time_reference')
    assess('well_depth_time_association', [
        ('spatial coordinate association', spatial),
        ('verified sample domain/origin/interval', axis_known),
        ('verified well datum and depth/time mapping', declared.get('well_depth_time_verified') is True),
        ('well depth/time verification reference', known(well_ref)),
    ], (well_ref,))
    model_ref = declared.get('model_validation_reference')
    model = assess('model_inference', [
        ('physical 4D registration', physical),
        ('verified model fit/checkpoint provenance', declared.get('model_provenance_verified') is True),
        ('validated feature and missing-data policy', declared.get('model_input_policy_verified') is True),
        ('model validation reference', known(model_ref)),
    ], (model_ref,))
    calibration_ref = declared.get('calibration_reference')
    assess('probability_calibration', [
        ('eligible model inference', model),
        ('independent calibration verified', declared.get('calibration_verified') is True),
        ('calibration method', known(declared.get('calibration_method'))),
        ('independent calibration reference', known(calibration_ref)),
    ], (calibration_ref,))
    return {name: asdict(value) for name, value in result.items()}
