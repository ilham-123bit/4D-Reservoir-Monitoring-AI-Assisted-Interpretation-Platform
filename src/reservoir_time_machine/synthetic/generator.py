"""Reproducible 3D reflection-volume generator; manipulations are illustrative only."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from ..config import SyntheticConfig
from ..models import GroundTruth, SeismicVolume, SurveyMetadata, SyntheticScenario
from .wavelets import ricker


def _reflectivity(config: SyntheticConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ni, nc, ns = config.dimensions
    il, xl = np.meshgrid(np.arange(ni), np.arange(nc), indexing="ij")
    cube = np.zeros(config.dimensions, dtype=np.float32)
    fault = il > ni * config.fault_position_fraction + 0.035 * (xl - nc / 2)
    horizons: list[np.ndarray] = []
    rng = np.random.default_rng(config.seed + 101)
    centers = np.linspace(8, ns - 12, config.reflector_count)
    for index, center in enumerate(centers):
        dip = (il - ni / 2) * (0.07 + 0.015 * (index % 3))
        fold = 1.8 * np.sin(xl / nc * 2 * np.pi + index * 0.3) + 1.2 * np.sin(il / ni * np.pi)
        fault_throw = fault * (1.5 + 0.15 * index)
        horizon = center + dip + fold + fault_throw
        horizons.append(horizon)
        polarity = -1.0 if index % 2 else 1.0
        lateral = 0.65 + 0.25 * np.sin(il / ni * 2 * np.pi + index) * np.cos(xl / nc * np.pi)
        lateral *= rng.uniform(0.8, 1.2)
        samples = np.arange(ns, dtype=float)[None, None, :] - horizon[:, :, None]
        cube += (polarity * lateral[:, :, None] * ricker(samples, config.peak_frequency_hz + 2 * (index % 4), config.sample_interval)).astype(np.float32)
    # Attenuate events on one side to make the synthetic fault visibly discontinuous.
    cube[fault, :] *= 0.72
    return cube, np.stack(horizons), fault


def generate_scenario(config: SyntheticConfig | None = None) -> SyntheticScenario:
    """Build a paired baseline/monitor reflection cube with separate truth labels."""
    config = config or SyntheticConfig()
    rng = np.random.default_rng(config.seed)
    base, horizons, fault = _reflectivity(config)
    ni, nc, ns = config.dimensions
    il, xl = np.meshgrid(np.arange(ni), np.arange(nc), indexing="ij")
    # Modest acquisition footprint plus spatially correlated noise.
    footprint = (1 + config.footprint_amplitude * np.sin(2 * np.pi * il / max(ni, 1)))[:, :, None]
    trace_gain = rng.normal(1.0, config.trace_gain_std, (ni, nc, 1)).astype(np.float32)
    baseline = base * footprint * trace_gain
    baseline += rng.normal(0, config.noise_std, config.dimensions).astype(np.float32)
    if config.correlated_noise_std:
        noise = gaussian_filter(rng.normal(size=config.dimensions), sigma=(1.2, 1.2, 0.35))
        noise /= max(float(noise.std()), 1e-8)
        baseline += (config.correlated_noise_std * noise).astype(np.float32)
    monitor = baseline.copy()
    # A smaller independent vintage noise component models imperfect repeatability.
    monitor += rng.normal(0, config.noise_std * config.monitor_noise_fraction, config.dimensions).astype(np.float32)

    di = (il - ni * config.response_center_fraction[0]) / max(ni * config.response_radii_fraction[0], 1)
    dx = (xl - nc * config.response_center_fraction[1]) / max(nc * config.response_radii_fraction[1], 1)
    spatial_ellipse = di**2 + dx**2 <= 1
    sample_axis = np.arange(ns)[None, None, :]
    target_horizon = horizons[len(horizons) // 2]
    vertical_gate = np.abs(sample_axis - target_horizon[:, :, None]) <= 6
    region = spatial_ellipse[:, :, None] & vertical_gate
    amp_truth = np.zeros(config.dimensions, dtype=bool)
    shift_truth = np.zeros(config.dimensions, dtype=bool)
    front_truth = np.zeros(config.dimensions, dtype=bool)
    unchanged_truth = np.zeros(config.dimensions, dtype=bool)
    artifact_truth = np.zeros(config.dimensions, dtype=bool)

    if config.scenario == "amplitude_increase":
        monitor[region] *= 1.0 + config.change_strength
        amp_truth = region.copy()
    elif config.scenario == "amplitude_decrease":
        monitor[region] *= max(0.0, 1.0 - config.change_strength)
        amp_truth = region.copy()
    elif config.scenario == "weak_amplitude":
        monitor[region] *= 1.0 + min(config.change_strength, 0.08)
        amp_truth = region.copy()
    elif config.scenario == "time_shift":
        local_shift = spatial_ellipse.astype(float) * config.maximum_time_shift_samples
        coords = np.indices(config.dimensions, dtype=float)
        coords[2] -= local_shift[:, :, None]
        monitor = map_coordinates(monitor, coords, order=1, mode="nearest").astype(np.float32)
        shift_truth = np.broadcast_to(spatial_ellipse[:, :, None], config.dimensions).copy()
    elif config.scenario == "fluid_front":
        # A synthetic lateral front; this is geometric signal manipulation, not flow simulation.
        front = (xl > nc * 0.2) & (xl < nc * 0.8) & (il > ni * 0.34) & (il < ni * 0.70)
        front_region = front[:, :, None] & vertical_gate
        progression = np.clip((xl - nc * 0.2) / max(nc * 0.6, 1), 0, 1)[:, :, None]
        monitor[front_region] *= (1 + config.change_strength * np.broadcast_to(progression, config.dimensions)[front_region])
        front_truth = front_region
        amp_truth = front_region.copy()
    elif config.scenario == "acquisition_artifact":
        stripe = np.zeros((ni, nc), dtype=bool)
        stripe[:, nc // 2:nc // 2 + max(1, nc // 12)] = True
        artifact_truth = np.broadcast_to(stripe[:, :, None], config.dimensions).copy()
        monitor += artifact_truth * (config.change_strength * np.sin(sample_axis * 0.35)).astype(np.float32)
    elif config.scenario == "noise_only":
        monitor += rng.normal(0, config.noise_std * 1.5, config.dimensions).astype(np.float32)
    if config.scenario == "unchanged":
        unchanged_truth = np.ones(config.dimensions, dtype=bool)
    else:
        injected = amp_truth | shift_truth | front_truth | artifact_truth
        unchanged_truth = ~injected

    if config.degraded_trace_fraction:
        degraded = rng.random((ni, nc)) < config.degraded_trace_fraction
        monitor[degraded, :] *= 0.25
        degraded_truth = np.broadcast_to(degraded[:, :, None], config.dimensions)
        artifact_truth |= degraded_truth
        unchanged_truth &= ~degraded_truth

    missing = np.zeros((ni, nc), dtype=bool)
    if config.missing_trace_fraction:
        missing = rng.random((ni, nc)) < config.missing_trace_fraction
        baseline[missing, :] = np.nan
        monitor[missing, :] = np.nan

    params = asdict(config)
    metadata_base = SurveyMetadata("MERIDIAN-SURVEY-A-YEAR-1", "Survey A — Year 1", config.dimensions,
                                   sample_interval=config.sample_interval, random_seed=config.seed,
                                   generation_parameters=params, synthetic=True)
    metadata_monitor = SurveyMetadata("MERIDIAN-SURVEY-B-YEAR-11", "Survey B — Year 11", config.dimensions,
                                      sample_interval=config.sample_interval, random_seed=config.seed,
                                      generation_parameters=params, synthetic=True)
    fault_truth = np.broadcast_to(fault[:, :, None], config.dimensions).copy()
    truth = GroundTruth(amp_truth, shift_truth, front_truth, unchanged_truth, artifact_truth,
                        fault_truth, config.scenario)
    return SyntheticScenario(SeismicVolume(baseline, metadata_base, missing),
                             SeismicVolume(monitor, metadata_monitor, missing), truth, params)
