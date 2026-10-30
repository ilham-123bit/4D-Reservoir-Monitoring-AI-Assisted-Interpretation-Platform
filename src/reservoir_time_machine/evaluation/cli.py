"""Deprecated CLI retained for import compatibility; synthetic output is disabled."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Legacy synthetic benchmark command (disabled).")
    parser.parse_args()
    raise SystemExit("Synthetic benchmark generation is not an active workflow. Run the Challenge 3 data audit instead.")
