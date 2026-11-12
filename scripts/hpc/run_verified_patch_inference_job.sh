#!/usr/bin/env bash
set -euo pipefail
: "${RTM_CODE_ROOT:?Set RTM_CODE_ROOT to reviewed source}"
: "${RTM_PAIRED_CACHE_DIR:?Set existing paired cache}"
: "${RTM_HPC_OUTPUT_ROOT:?Set private derived-output root}"
: "${RTM_PATCH_INFERENCE_CONFIG:?Set private checkpoint/patch config}"
: "${SLURM_JOB_ID:?Run only inside an authorized Slurm GPU allocation}"
[[ -d "$RTM_CODE_ROOT" && -d "$RTM_PAIRED_CACHE_DIR" && -d "$RTM_HPC_OUTPUT_ROOT" && -f "$RTM_PATCH_INFERENCE_CONFIG" ]] || {
  echo "Configured source/cache/output/config paths must exist." >&2; exit 2;
}
export PYTHONPATH="${RTM_CODE_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
exec python "${RTM_CODE_ROOT}/scripts/hpc/run_verified_patch_inference.py" --config "$RTM_PATCH_INFERENCE_CONFIG"
