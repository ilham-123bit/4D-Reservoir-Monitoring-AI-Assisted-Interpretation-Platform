"""Repeatable multi-seed synthetic detector comparison."""

from dataclasses import asdict
from time import perf_counter

from scipy.ndimage import shift as shift_volume

from ..config import SyntheticConfig
from ..detection.statistical import difference_detector, nrms_detector, statistical_detector, spatial_detector
from ..models import SYNTHETIC_LABEL
from .metrics import evaluate_detection
from ..synthetic import generate_scenario

DETECTORS = {
    "difference": difference_detector,
    "nrms": nrms_detector,
    "robust_statistical": statistical_detector,
    "spatial_coherent": spatial_detector,
}


def run_benchmark(*, seeds=(3, 7, 19), strengths=(0.2, 0.45), noise_levels=(0.02, 0.05),
                  registration_errors=(0, 1),
                  scenarios=("amplitude_increase", "amplitude_decrease", "weak_amplitude", "time_shift", "fluid_front",
                             "unchanged", "acquisition_artifact", "noise_only"),
                  dimensions=(24, 24, 64)) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scenario_name in scenarios:
        for seed in seeds:
            for strength in strengths:
                for noise in noise_levels:
                    scenario = generate_scenario(SyntheticConfig(dimensions=dimensions, seed=seed,
                        scenario=scenario_name, change_strength=strength, noise_std=noise))
                    truth = (scenario.ground_truth.amplitude_change | scenario.ground_truth.time_shift |
                             scenario.ground_truth.fluid_front)
                    for reg_error in registration_errors:
                        monitor = (shift_volume(scenario.monitor.data, (0, 0, reg_error), order=0, mode="nearest")
                                   if reg_error else scenario.monitor.data)
                        for detector_name, detector in DETECTORS.items():
                            start = perf_counter()
                            result = detector(scenario.baseline.data, monitor)
                            elapsed = perf_counter() - start
                            metrics = evaluate_detection(result.mask, truth, runtime_seconds=elapsed)
                            rows.append({"scenario": scenario_name, "seed": seed, "detector": detector_name,
                                "strength": strength, "noise_std": noise,
                                "registration_error_samples": reg_error,
                                **asdict(metrics), "synthetic": True, "label": SYNTHETIC_LABEL})
    return rows
