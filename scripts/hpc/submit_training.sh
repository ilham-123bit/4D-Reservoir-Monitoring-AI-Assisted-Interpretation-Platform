#!/usr/bin/env bash
set -euo pipefail

# Fail closed: resource names/limits are supplied only after inspecting the actual site.
: "${RTM_SLURM_ACCOUNT:?Set the team account after checking the HPC manual}"
: "${RTM_SLURM_PARTITION:?Set the partition after inspecting site Slurm configuration}"
: "${RTM_SLURM_GPU_REQUEST:?Set the GPU/MIG GRES request confirmed by site documentation}"
: "${RTM_SLURM_CPUS:?Set an approved CPU request}"
: "${RTM_SLURM_MEMORY:?Set an approved memory request}"
: "${RTM_SLURM_TIME:?Set an approved walltime}"
: "${RTM_CODE_ROOT:?Set code path available on HPC}"
: "${RTM_HPC_OUTPUT_ROOT:?Set authorized output root}"
: "${RTM_EXPERIMENT_CONFIG:?Set HPC-private JSON config}"
: "${RTM_CHALLENGE3_ROOT:?Set authorized data root}"

if [[ "${RTM_GPU_KERNEL_ACTIVE:-0}" == "1" ]]; then
  echo "Stop the team's active GPU kernel before submitting a separate GPU job." >&2
  exit 2
fi
command -v sbatch >/dev/null || { echo "sbatch unavailable; run only in the authorized HPC environment." >&2; exit 2; }
command -v squeue >/dev/null || { echo "squeue unavailable; cannot check team allocation limits." >&2; exit 2; }
command -v flock >/dev/null || { echo "flock unavailable; cannot serialize the team quota check." >&2; exit 2; }
[[ -d "$RTM_HPC_OUTPUT_ROOT" ]] || { echo "RTM_HPC_OUTPUT_ROOT must be an existing team-shared directory." >&2; exit 2; }
exec 9>"${RTM_HPC_OUTPUT_ROOT}/.rtm-slurm-submit.lock"
flock -n 9 || { echo "Another team submission check is in progress." >&2; exit 2; }
states="$(squeue --account "$RTM_SLURM_ACCOUNT" --noheader --format='%T')"
running="$(printf '%s\n' "$states" | awk '$1 == "RUNNING" {n++} END {print n+0}')"
total="$(printf '%s\n' "$states" | awk 'NF {n++} END {print n+0}')"
if (( running >= 1 )); then
  echo "Account already has a running Slurm job; respecting one-running-job limit." >&2
  exit 2
fi
if (( total >= 2 )); then
  echo "Account already has two submitted jobs; respecting two-job limit." >&2
  exit 2
fi

script="${RTM_CODE_ROOT}/scripts/hpc/train_job.sh"
exec sbatch --account "$RTM_SLURM_ACCOUNT" --partition "$RTM_SLURM_PARTITION" \
  --gres "$RTM_SLURM_GPU_REQUEST" --cpus-per-task "$RTM_SLURM_CPUS" \
  --mem "$RTM_SLURM_MEMORY" --time "$RTM_SLURM_TIME" "$script"
