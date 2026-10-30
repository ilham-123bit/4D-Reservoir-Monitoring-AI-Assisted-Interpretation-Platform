"""Generate reproducible synthetic 3D paired-vintage data."""

import argparse
import json
from pathlib import Path

import numpy as np

from .config import SyntheticConfig
from .synthetic import generate_scenario


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic Meridian Field 3D seismic volumes.")
    parser.add_argument("--output", type=Path, default=Path("data/synthetic/meridian_synthetic.npz"))
    parser.add_argument("--dimensions", type=int, nargs=3, metavar=("INLINE", "CROSSLINE", "SAMPLE"), default=(48, 48, 96))
    parser.add_argument("--size", type=int, help="Compatibility shorthand: set inline and crossline; sample count stays 96.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--scenario", choices=("amplitude_increase", "amplitude_decrease", "weak_amplitude", "time_shift", "fluid_front", "unchanged", "acquisition_artifact", "noise_only"), default="amplitude_increase")
    args = parser.parse_args()
    dims = (args.size, args.size, args.dimensions[2]) if args.size else tuple(args.dimensions)
    scenario = generate_scenario(SyntheticConfig(dimensions=dims, seed=args.seed, scenario=args.scenario))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, baseline=scenario.baseline.data, monitor=scenario.monitor.data,
                        missing_trace_mask=scenario.baseline.missing_trace_mask,
                        metadata=json.dumps({"baseline": scenario.baseline.metadata.__dict__,
                                             "monitor": scenario.monitor.metadata.__dict__,
                                             "parameters": scenario.parameters, "label": scenario.label}))
    # Labels are kept in a separate sidecar and are never part of detector inputs.
    truth_path = args.output.with_name(args.output.stem + ".ground_truth.npz")
    np.savez_compressed(truth_path, amplitude_change=scenario.ground_truth.amplitude_change,
                        time_shift=scenario.ground_truth.time_shift, fluid_front=scenario.ground_truth.fluid_front,
                        unchanged=scenario.ground_truth.unchanged,
                        acquisition_artifact=scenario.ground_truth.acquisition_artifact,
                        structural_fault=scenario.ground_truth.structural_fault,
                        scenario_label=scenario.ground_truth.scenario_label,
                        synthetic_label=scenario.label)
    print(f"{scenario.label}")
    print(f"Wrote {scenario.baseline.data.shape} (inline, crossline, sample) volumes: {args.output}")
    print(f"Ground-truth labels: {truth_path}")


if __name__ == "__main__":
    main()
