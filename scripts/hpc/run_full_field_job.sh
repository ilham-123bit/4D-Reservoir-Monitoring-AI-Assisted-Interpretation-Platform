#!/usr/bin/env bash
set -euo pipefail

: "${RTM_CODE_ROOT:?Set RTM_CODE_ROOT to this reviewed checkout}"
: "${RTM_PAIRED_CACHE_DIR:?Set RTM_PAIRED_CACHE_DIR to the existing paired NPY cache}"
: "${RTM_HPC_OUTPUT_ROOT:?Set RTM_HPC_OUTPUT_ROOT to approved private derived-output storage}"
: "${SLURM_JOB_ID:?Run only inside a Slurm allocation}"
[[ -d "$RTM_CODE_ROOT" && -d "$RTM_PAIRED_CACHE_DIR" && -d "$RTM_HPC_OUTPUT_ROOT" ]] || {
  echo "Code, cache, and output roots must already exist." >&2; exit 2;
}
export PYTHONPATH="${RTM_CODE_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
pilot_args=()
if [[ -n "${RTM_PILOT_BOUNDS:-}" ]]; then
  read -r -a pilot_bounds <<< "$RTM_PILOT_BOUNDS"
  [[ "${#pilot_bounds[@]}" -eq 6 ]] || { echo "RTM_PILOT_BOUNDS must contain six integer array bounds." >&2; exit 2; }
  for bound in "${pilot_bounds[@]}"; do
    [[ "$bound" =~ ^[0-9]+$ ]] || { echo "RTM_PILOT_BOUNDS must contain non-negative integers." >&2; exit 2; }
  done
  pilot_args=(--pilot-bounds "${pilot_bounds[@]}")
fi
exec python "${RTM_CODE_ROOT}/scripts/hpc/process_full_field.py" \
  --cache "$RTM_PAIRED_CACHE_DIR" --output "${RTM_HPC_OUTPUT_ROOT}/full-field-attributes" \
  --chunk "${RTM_INLINE_CHUNK:-16}" "${RTM_CROSSLINE_CHUNK:-32}" "${RTM_SAMPLE_CHUNK:-128}" \
  --window "${RTM_NRMS_WINDOW:-9}" --min-support "${RTM_MIN_SUPPORT_FRACTION:-0.8}" \
  --min-free-gib "${RTM_MIN_FREE_GIB:-10}" "${pilot_args[@]}"
