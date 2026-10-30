"""Command-line synthetic detector benchmark runner."""

import argparse
from pathlib import Path

import pandas as pd

from .benchmark import run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run synthetic multi-condition detector benchmarks.")
    parser.add_argument("--output", type=Path, default=Path("data/synthetic/benchmark_results.csv"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[3, 7, 19])
    parser.add_argument("--dimensions", type=int, nargs=3, default=(16, 16, 32), metavar=("INLINE", "CROSSLINE", "SAMPLE"))
    args = parser.parse_args()
    rows = run_benchmark(seeds=tuple(args.seeds), dimensions=tuple(args.dimensions))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print("SYNTHETIC — FOR DEVELOPMENT AND DEMONSTRATION ONLY")
    print(f"Wrote {len(rows)} detector evaluations to {args.output}")


if __name__ == "__main__":
    main()
