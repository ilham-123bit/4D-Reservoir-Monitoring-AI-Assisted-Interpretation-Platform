#!/usr/bin/env bash
set -euo pipefail
: "${RTM_SLURM_ACCOUNT:?Set approved Slurm account}"
: "${RTM_SLURM_PARTITION:?Set a currently available GPU/MIG partition}"
: "${RTM_SLURM_GPU_REQUEST:?Set site-approved GPU/MIG GRES request}"
: "${RTM_SLURM_CPUS:?Set approved CPU count}"
: "${RTM_SLURM_MEMORY:?Set approved memory}"
: "${RTM_SLURM_TIME:?Set approved walltime}"
: "${RTM_CODE_ROOT:?Set reviewed code checkout}"
: "${RTM_HPC_OUTPUT_ROOT:?Set approved private output root}"
: "${RTM_PAIRED_CACHE_DIR:?Set existing paired cache}"
: "${RTM_PATCH_INFERENCE_CONFIG:?Set private verified-checkpoint config}"
command -v sbatch >/dev/null && command -v squeue >/dev/null && command -v flock >/dev/null || {
  echo "Required Slurm commands unavailable." >&2; exit 2;
}
[[ -d "$RTM_CODE_ROOT" && -d "$RTM_HPC_OUTPUT_ROOT" && -d "$RTM_PAIRED_CACHE_DIR" && -f "$RTM_PATCH_INFERENCE_CONFIG" ]] || {
  echo "Configured source/cache/output/config paths must exist." >&2; exit 2;
}
exec 9>"${RTM_HPC_OUTPUT_ROOT}/.rtm-slurm-submit.lock"
flock -n 9 || { echo "Another job submission check is active." >&2; exit 2; }
states="$(squeue --account "$RTM_SLURM_ACCOUNT" --noheader --format='%T')"
running="$(printf '%s\n' "$states" | awk '$1 == "RUNNING" {n++} END {print n+0}')"
total="$(printf '%s\n' "$states" | awk 'NF {n++} END {print n+0}')"
if (( running >= 1 || total >= 2 )); then
  echo "Team GPU job limit reached; no job submitted." >&2; exit 2
fi
exec sbatch --account "$RTM_SLURM_ACCOUNT" --partition "$RTM_SLURM_PARTITION" \
  --gres "$RTM_SLURM_GPU_REQUEST" --cpus-per-task "$RTM_SLURM_CPUS" \
  --mem "$RTM_SLURM_MEMORY" --time "$RTM_SLURM_TIME" \
  --output "${RTM_HPC_OUTPUT_ROOT}/slurm-patch-inference-%j.log" --export=ALL \
  "$RTM_CODE_ROOT/scripts/hpc/run_verified_patch_inference_job.sh"
