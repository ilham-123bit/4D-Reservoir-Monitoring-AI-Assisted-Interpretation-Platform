#!/usr/bin/env bash
# Slurm-only worker. It runs on a compute allocation, never on the login node.
main() {
  local code_root python_bin rc
  : "${SLURM_JOB_ID:?This worker must be started by Slurm}"
  : "${RTM_BOUNDED_OUTPUT_DIR:?Launcher must provide a unique output directory}"
  # Slurm stages batch scripts under a spool directory. The submitter passes the
  # stable shared-filesystem checkout explicitly; never infer it from BASH_SOURCE.
  : "${RTM_CODE_ROOT:?Set RTM_CODE_ROOT to the stable staged project checkout}"
  code_root="$(cd -- "$RTM_CODE_ROOT" && pwd -P)" || {
    printf 'ERROR: configured RTM_CODE_ROOT cannot be resolved: %s\n' "$RTM_CODE_ROOT" >&2
    return 2
  }
  if [[ ! -d "$code_root/src" || ! -f "$code_root/scripts/hpc/run_bounded_ml_pilot.py" ]]; then
    printf 'ERROR: staged project source or bounded-pilot entry point is unavailable: %s\n' "$code_root" >&2
    return 2
  fi
  python_bin="${RTM_PYTHON:-python3}"
  export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
  export PYTHONPATH="$code_root/src${PYTHONPATH:+:$PYTHONPATH}"
  "$python_bin" "$code_root/scripts/hpc/run_bounded_ml_pilot.py" \
    --config "${RTM_BOUNDED_PILOT_CONFIG:-$code_root/config/hpc/well2_bounded_pilot.json}" \
    --paired-cache "${RTM_PAIRED_CACHE_DIR:?Set RTM_PAIRED_CACHE_DIR to the authorized paired cache}" \
    --challenge3-root "${RTM_CHALLENGE3_ROOT:?Set RTM_CHALLENGE3_ROOT to the authorized challenge data root}" \
    --output-root "${RTM_HPC_OUTPUT_ROOT:?Set RTM_HPC_OUTPUT_ROOT to authorized private derived storage}" \
    --output-dir "$RTM_BOUNDED_OUTPUT_DIR" --experiment-id well2-coal-control-v1 \
    --reference-bounds 64 192 1088 1216 --calibration-bounds 869 997 64 192 \
    --sample-bounds 850 901 --scoring-bounds 600 728 131 259 850 901 \
    --max-samples 10000 --max-score-voxels 1000000 \
    --max-output-bytes 134217728 --max-memory-mib 2048 \
    --compare-chunks 16x16x64 32x32x128 64x64x128 --threads 4 --seed 2026
  rc=$?
  return "$rc"
}
main "$@"
