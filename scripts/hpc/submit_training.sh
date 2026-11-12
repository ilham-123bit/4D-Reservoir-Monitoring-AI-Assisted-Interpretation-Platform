#!/usr/bin/env bash
set -euo pipefail

# Fail closed: resource names/limits are supplied only after inspecting the actual site.
: "${RTM_SLURM_ACCOUNT:?Set the team account after checking the HPC manual}"
: "${RTM_SLURM_PARTITION:?Set the partition after inspecting site Slurm configuration}"
: "${RTM_SLURM_QOS:?Set the QoS confirmed for this account and partition}"
: "${RTM_SLURM_CONSTRAINT:?Set the approved GPU node constraint}"
: "${RTM_SLURM_MCS_LABEL:?Set the mandatory MCS label from site instructions}"
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
[[ -f "${RTM_CODE_ROOT}/scripts/hpc/train_job.sh" ]] || { echo "Training batch worker is missing from RTM_CODE_ROOT." >&2; exit 2; }
[[ -f "$RTM_EXPERIMENT_CONFIG" ]] || { echo "Configured training experiment file is unavailable." >&2; exit 2; }
mkdir -p "${RTM_HPC_OUTPUT_ROOT}/logs"
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
if ! submission="$(sbatch --parsable --account "$RTM_SLURM_ACCOUNT" \
  --partition "$RTM_SLURM_PARTITION" --qos "$RTM_SLURM_QOS" \
  --constraint "$RTM_SLURM_CONSTRAINT" --mcs-label "$RTM_SLURM_MCS_LABEL" \
  --nodes=1 --ntasks=1 --gres "$RTM_SLURM_GPU_REQUEST" \
  --cpus-per-task "$RTM_SLURM_CPUS" --mem "$RTM_SLURM_MEMORY" \
  --time "$RTM_SLURM_TIME" \
  --output "${RTM_HPC_OUTPUT_ROOT}/logs/%x-%j.out" \
  --error "${RTM_HPC_OUTPUT_ROOT}/logs/%x-%j.err" "$script" 2>&1)"; then
  printf 'sbatch failed; no training job was accepted.\n%s\n' "$submission" >&2
  exit 2
fi
job_id="${submission%%;*}"
printf 'Submitted GPU training job %s\n' "$job_id"
printf 'stdout: %s/logs/train_job-%s.out\n' "$RTM_HPC_OUTPUT_ROOT" "$job_id"
printf 'stderr: %s/logs/train_job-%s.err\n' "$RTM_HPC_OUTPUT_ROOT" "$job_id"
