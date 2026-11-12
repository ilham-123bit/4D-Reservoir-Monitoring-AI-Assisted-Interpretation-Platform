#!/usr/bin/env bash
set -euo pipefail
: "${RTM_CODE_ROOT:?Set RTM_CODE_ROOT to reviewed source}"
: "${RTM_HPC_OUTPUT_ROOT:?Set approved private output root}"
: "${RTM_CHALLENGE3_ROOT:?Set authorized input root}"
: "${RTM_INFERENCE_CONFIG:?Set private inference config with verified checkpoint and adapter}"
: "${SLURM_JOB_ID:?Run only inside an authorized Slurm allocation}"
[[ -d "$RTM_CODE_ROOT" && -d "$RTM_HPC_OUTPUT_ROOT" && -d "$RTM_CHALLENGE3_ROOT" && -f "$RTM_INFERENCE_CONFIG" ]] || {
  echo "Configured source/data/output/config paths must exist." >&2; exit 2;
}
export PYTHONPATH="${RTM_CODE_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
exec python -m reservoir_time_machine.hpc.inference --config "$RTM_INFERENCE_CONFIG"
