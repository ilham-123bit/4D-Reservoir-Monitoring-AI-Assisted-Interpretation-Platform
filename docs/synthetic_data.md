# Legacy synthetic prototype modules

The earlier development phase included a fictional seismic-like volume generator, demo dashboard, and benchmark tests. Those files are retained for source-history and implementation reference in the legacy configuration/model definitions, `src/reservoir_time_machine/synthetic/`, `src/reservoir_time_machine/evaluation/benchmark.py`, `legacy/`, and `tests/legacy_synthetic/`.

They are not part of the active application workflow, are not used as a fallback when competition files are missing, and are excluded from the default test collection. Do not use their generated data or metrics as Challenge 3 evidence, geological validation, or competition results. The current public application consumes only configured source files and otherwise displays a data-unavailable message.
