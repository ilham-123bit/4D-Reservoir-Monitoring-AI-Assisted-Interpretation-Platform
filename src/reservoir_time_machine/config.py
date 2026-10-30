"""Validated, field-independent synthetic generation settings."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SyntheticConfig:
    dimensions: tuple[int, int, int] = (48, 48, 96)
    seed: int = 7
    sample_interval: float = 0.002
    reflector_count: int = 8
    peak_frequency_hz: float = 28.0
    fault_position_fraction: float = 0.56
    response_center_fraction: tuple[float, float] = (0.58, 0.53)
    response_radii_fraction: tuple[float, float] = (0.16, 0.20)
    footprint_amplitude: float = 0.025
    noise_std: float = 0.035
    correlated_noise_std: float = 0.012
    trace_gain_std: float = 0.025
    monitor_noise_fraction: float = 0.35
    missing_trace_fraction: float = 0.0
    degraded_trace_fraction: float = 0.0
    scenario: str = "amplitude_increase"
    change_strength: float = 0.35
    maximum_time_shift_samples: float = 2.0

    def __post_init__(self) -> None:
        if len(self.dimensions) != 3 or any(not isinstance(d, int) or d < 16 for d in self.dimensions):
            raise ValueError("dimensions must contain three values, each at least 16")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.reflector_count < 3:
            raise ValueError("reflector_count must be at least 3")
        levels = (self.peak_frequency_hz, self.fault_position_fraction, self.footprint_amplitude,
                  self.noise_std, self.correlated_noise_std, self.trace_gain_std, self.monitor_noise_fraction,
                  self.change_strength, self.maximum_time_shift_samples, self.sample_interval,
                  *self.response_center_fraction, *self.response_radii_fraction)
        if not all(math.isfinite(value) for value in levels):
            raise ValueError("generation parameters must be finite")
        if self.noise_std < 0 or self.correlated_noise_std < 0 or self.trace_gain_std < 0:
            raise ValueError("noise levels must be non-negative")
        if self.change_strength < 0 or self.maximum_time_shift_samples < 0 or self.sample_interval <= 0:
            raise ValueError("change strength/lag must be non-negative and sample interval positive")
        if self.peak_frequency_hz <= 0 or not 0 < self.fault_position_fraction < 1:
            raise ValueError("peak frequency must be positive and fault position must be inside the grid")
        if self.peak_frequency_hz + 6 >= 1.0 / (2.0 * self.sample_interval):
            raise ValueError("reflector peak frequencies must be below Nyquist")
        if self.monitor_noise_fraction < 0 or self.footprint_amplitude < 0:
            raise ValueError("monitor noise fraction and footprint amplitude must be non-negative")
        if any(not 0 < value < 1 for value in self.response_center_fraction) or any(value <= 0 for value in self.response_radii_fraction):
            raise ValueError("response center must lie within normalized grid and radii must be positive")
        if not 0 <= self.missing_trace_fraction < 1:
            raise ValueError("missing_trace_fraction must be in [0, 1)")
        if not 0 <= self.degraded_trace_fraction < 1:
            raise ValueError("degraded_trace_fraction must be in [0, 1)")
        if self.scenario not in {
            "amplitude_increase", "amplitude_decrease", "weak_amplitude", "time_shift",
            "fluid_front", "unchanged", "acquisition_artifact", "noise_only",
        }:
            raise ValueError("unknown synthetic scenario")
