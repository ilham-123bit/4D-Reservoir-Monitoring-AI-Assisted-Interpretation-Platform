#!/usr/bin/env bash
set -euo pipefail

# Resource values are mandatory after reviewing the live site preflight. No
# partition, memory, walltime, account, or output path is assumed by this script.
: "${RTM_SLURM_ACCOUNT:?Set after checking HPC policy}"
: "${RTM_SLURM_PARTITION:?Choose from current sinfo output}"
: "${RTM_SLURM_CPUS:?Set approved CPU count}"
: "${RTM_SLURM_MEMORY:?Set approved memory request}"
: "${RTM_SLURM_TIME:?Set approved walltime}"
: "${RTM_CODE_ROOT:?Set reviewed code checkout}"
: "${RTM_PAIRED_CACHE_DIR:?Set existing paired cache}"
: "${RTM_HPC_OUTPUT_ROOT:?Set approved private output root}"
[[ "$RTM_SLURM_CPUS" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid CPU count." >&2; exit 2; }
[[ -d "$RTM_CODE_ROOT" && -d "$RTM_PAIRED_CACHE_DIR" && -d "$RTM_HPC_OUTPUT_ROOT" ]] || {
  echo "Code/cache/output roots must already exist." >&2; exit 2;
}
command -v sbatch >/dev/null && command -v squeue >/dev/null && command -v flock >/dev/null || {
  echo "Slurm commands or flock unavailable." >&2; exit 2;
}
exec 9>"${RTM_HPC_OUTPUT_ROOT}/.rtm-cpu-submit.lock"
flock -n 9 || { echo "Another submission check is active." >&2; exit 2; }
states="$(squeue --account "$RTM_SLURM_ACCOUNT" --user "$USER" --noheader --format='%T')"
running="$(printf '%s\n' "$states" | awk '$1 == "RUNNING" {n++} END {print n+0}')"
pending="$(printf '%s\n' "$states" | awk 'NF {n++} END {print n+0}')"
if (( running >= 1 || pending >= 2 )); then
  echo "Team job limit reached; no job submitted." >&2; exit 2
fi
exec sbatch --qos=normal --nodes=1 --ntasks=1 --constraint=nodetype-h200.B-3072 --mcs-label=apgce --gres=gpu:nvidia_h200_nvl_2g.35gb:1 --account "$RTM_SLURM_ACCOUNT" --partition "$RTM_SLURM_PARTITION" \
  --cpus-per-task "$RTM_SLURM_CPUS" --mem "$RTM_SLURM_MEMORY" --time "$RTM_SLURM_TIME" \
  --output "${RTM_HPC_OUTPUT_ROOT}/slurm-full-field-%j.log" \
  --export=ALL "$RTM_CODE_ROOT/scripts/hpc/run_full_field_job.sh"
