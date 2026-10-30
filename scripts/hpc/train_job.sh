#!/usr/bin/env bash
set -euo pipefail

: "${RTM_CODE_ROOT:?Set RTM_CODE_ROOT to the project code directory available on HPC}"
: "${RTM_HPC_OUTPUT_ROOT:?Set RTM_HPC_OUTPUT_ROOT to authorized derived-output storage}"
: "${RTM_EXPERIMENT_CONFIG:?Set RTM_EXPERIMENT_CONFIG to the HPC-private JSON configuration}"
: "${RTM_CHALLENGE3_ROOT:?Set RTM_CHALLENGE3_ROOT to the authorized input directory}"

export PYTHONPATH="${RTM_CODE_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

python -m reservoir_time_machine.hpc.environment --strict \
  --output "${RTM_HPC_OUTPUT_ROOT}/environment-${SLURM_JOB_ID:-unknown}.json"
python -m reservoir_time_machine.hpc.training --config "${RTM_EXPERIMENT_CONFIG}"
