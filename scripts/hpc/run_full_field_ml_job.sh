#!/usr/bin/env bash
set -euo pipefail
: "${RTM_CODE_ROOT:?Set RTM_CODE_ROOT to reviewed source}"
: "${RTM_PAIRED_CACHE_DIR:?Set RTM_PAIRED_CACHE_DIR to paired cache}"
: "${RTM_HPC_OUTPUT_ROOT:?Set RTM_HPC_OUTPUT_ROOT to approved private output}"
: "${RTM_FIELD_ML_CONFIG:?Set RTM_FIELD_ML_CONFIG to private population config}"
: "${RTM_OPERATOR_SUBMISSION_APPROVED:?Set YES for this operator-authorized run}"
: "${SLURM_JOB_ID:?Run only inside Slurm}"
[[ "$RTM_OPERATOR_SUBMISSION_APPROVED" == "YES" ]] || {
  echo "Explicit operator authorization was not propagated to the Slurm job." >&2; exit 2;
}
[[ -d "$RTM_CODE_ROOT" && -d "$RTM_PAIRED_CACHE_DIR" && -d "$RTM_HPC_OUTPUT_ROOT" && -f "$RTM_FIELD_ML_CONFIG" ]] || {
  echo "Configured source/cache/output/config paths must exist." >&2; exit 2;
}
export PYTHONPATH="${RTM_CODE_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
# Full-field scorer parallelizes disjoint chunks with bounded Python threads.
# Keep native BLAS/OpenMP pools single-threaded to avoid oversubscription.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
exec python "${RTM_CODE_ROOT}/scripts/hpc/run_full_field_ml.py" --config "$RTM_FIELD_ML_CONFIG"
