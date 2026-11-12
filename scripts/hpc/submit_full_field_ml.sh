#!/usr/bin/env bash
set -euo pipefail
: "${RTM_SLURM_ACCOUNT:?Set approved Slurm account}"
: "${RTM_SLURM_PARTITION:?Choose from current sinfo output}"
: "${RTM_SLURM_CPUS:?Set approved CPU count}"
: "${RTM_SLURM_MEMORY:?Set approved memory request}"
: "${RTM_SLURM_TIME:?Set approved walltime}"
: "${RTM_SLURM_QOS:?Set the QoS approved for this account and partition}"
: "${RTM_SLURM_CONSTRAINT:?Set the verified node constraint}"
: "${RTM_SLURM_MCS_LABEL:?Set the verified MCS label}"
: "${RTM_CODE_ROOT:?Set reviewed code checkout}"
: "${RTM_PAIRED_CACHE_DIR:?Set existing paired cache}"
: "${RTM_HPC_OUTPUT_ROOT:?Set approved private output root}"
: "${RTM_FIELD_ML_CONFIG:?Set private reference/calibration configuration}"
: "${RTM_CHALLENGE3_ROOT:?Set authorized original seismic data root for path protection}"
: "${RTM_OPERATOR_SUBMISSION_APPROVED:?Set YES only for this authorized submission}"
[[ "$RTM_OPERATOR_SUBMISSION_APPROVED" == "YES" ]] || { echo "Explicit operator submission authorization is required." >&2; exit 2; }
[[ "$RTM_SLURM_CPUS" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid CPU count." >&2; exit 2; }
(( RTM_SLURM_CPUS <= 128 )) || { echo "CPU request exceeds the verified 128-CPU node capacity." >&2; exit 2; }
[[ "$RTM_SLURM_ACCOUNT" == "petronas" ]] || { echo "This team wrapper requires the verified petronas account." >&2; exit 2; }
[[ "$RTM_SLURM_PARTITION" == "petronas_apgce" ]] || { echo "This team wrapper requires the verified petronas_apgce partition." >&2; exit 2; }
[[ -d "$RTM_CODE_ROOT" && -d "$RTM_PAIRED_CACHE_DIR" && -d "$RTM_HPC_OUTPUT_ROOT" && -d "$RTM_CHALLENGE3_ROOT" && -f "$RTM_FIELD_ML_CONFIG" ]] || {
  echo "Configured source/cache/output/config paths must exist." >&2; exit 2;
}
command -v python >/dev/null || { echo "python unavailable for calibration review gate." >&2; exit 2; }
python "$RTM_CODE_ROOT/scripts/hpc/validate_full_field_ml_review.py" \
  --config "$RTM_FIELD_ML_CONFIG" --output-root "$RTM_HPC_OUTPUT_ROOT" \
  --paired-cache "$RTM_PAIRED_CACHE_DIR" --protected-data-root "$RTM_CHALLENGE3_ROOT"
command -v sbatch >/dev/null && command -v squeue >/dev/null && command -v scontrol >/dev/null \
  && command -v sacctmgr >/dev/null && command -v flock >/dev/null || {
  echo "Required Slurm commands or flock unavailable." >&2; exit 2;
}
partition_qos="$(scontrol show partition "$RTM_SLURM_PARTITION" | sed -n 's/.*AllowQos=\([^ ]*\).*/\1/p' | tail -n 1)"
if [[ -z "$partition_qos" ]]; then
  echo "Could not verify partition QoS policy; no job submitted." >&2; exit 2
fi
partition_record="$(scontrol show partition "$RTM_SLURM_PARTITION")"
partition_max_time="$(printf '%s\n' "$partition_record" | sed -n 's/.*MaxTime=\([^ ]*\).*/\1/p' | head -n 1)"
time_to_seconds() {
  local value="$1" days=0 hours=0 minutes=0 seconds=0
  if [[ "$value" == "UNLIMITED" || "$value" == "0" ]]; then printf '0\n'; return 0; fi
  if [[ "$value" =~ ^([0-9]+)-([0-9]{1,2}):([0-9]{2}):([0-9]{2})$ ]]; then
    days="${BASH_REMATCH[1]}"; hours="${BASH_REMATCH[2]}"; minutes="${BASH_REMATCH[3]}"; seconds="${BASH_REMATCH[4]}"
  elif [[ "$value" =~ ^([0-9]{1,2}):([0-9]{2}):([0-9]{2})$ ]]; then
    hours="${BASH_REMATCH[1]}"; minutes="${BASH_REMATCH[2]}"; seconds="${BASH_REMATCH[3]}"
  else return 1; fi
  printf '%s\n' "$((days*86400 + hours*3600 + minutes*60 + seconds))"
}
if [[ -z "$partition_max_time" ]]; then
  echo "Could not verify partition MaxTime; no job submitted." >&2; exit 2
fi
requested_seconds="$(time_to_seconds "$RTM_SLURM_TIME")" || { echo "Invalid RTM_SLURM_TIME format." >&2; exit 2; }
maximum_seconds="$(time_to_seconds "$partition_max_time")" || { echo "Unrecognized partition MaxTime: $partition_max_time; no job submitted." >&2; exit 2; }
if (( maximum_seconds > 0 && requested_seconds > maximum_seconds )); then
  echo "Requested walltime exceeds partition MaxTime=$partition_max_time; no job submitted." >&2; exit 2
fi
if [[ "$partition_qos" != "ALL" ]] && ! printf '%s\n' "$partition_qos" | tr ',' '\n' | grep -Fxq "$RTM_SLURM_QOS"; then
  echo "Requested QoS is not allowed by the partition; no job submitted." >&2; exit 2
fi
association_qos="$(sacctmgr show assoc where user="$USER" account="$RTM_SLURM_ACCOUNT" format=QOS -nP | tr ',;|' '\n\n\n' | sed 's/[[:space:]]//g' | awk 'NF')"
if ! printf '%s\n' "$association_qos" | grep -Fxq "$RTM_SLURM_QOS"; then
  echo "Requested QoS is not verified in the account association; no job submitted." >&2; exit 2
fi
exec 9>"${RTM_HPC_OUTPUT_ROOT}/.rtm-cpu-submit.lock"
flock -n 9 || { echo "Another submission check is active." >&2; exit 2; }
states="$(squeue --user "$USER" --account "$RTM_SLURM_ACCOUNT" --noheader --format='%T')"
running="$(printf '%s\n' "$states" | awk '$1 == "RUNNING" {n++} END {print n+0}')"
total="$(printf '%s\n' "$states" | awk 'NF {n++} END {print n+0}')"
if (( running >= 1 || total >= 2 )); then
  echo "Team job limit reached; no job submitted." >&2; exit 2
fi
duplicate="$(squeue --user "$USER" --account "$RTM_SLURM_ACCOUNT" --name rtm-stage2-ml --noheader --format='%T')"
if [[ -n "$duplicate" ]]; then
  echo "An active or pending Stage 2 ML job already exists; no job submitted." >&2; exit 2
fi
exec sbatch --account "$RTM_SLURM_ACCOUNT" --partition "$RTM_SLURM_PARTITION" \
  --qos "$RTM_SLURM_QOS" --constraint "$RTM_SLURM_CONSTRAINT" --mcs-label "$RTM_SLURM_MCS_LABEL" \
  --nodes 1 --ntasks 1 --cpus-per-task "$RTM_SLURM_CPUS" --mem "$RTM_SLURM_MEMORY" --time "$RTM_SLURM_TIME" \
  --job-name rtm-stage2-ml --output "${RTM_HPC_OUTPUT_ROOT}/slurm-field-ml-%j.out" \
  --error "${RTM_HPC_OUTPUT_ROOT}/slurm-field-ml-%j.err" --export=ALL \
  "$RTM_CODE_ROOT/scripts/hpc/run_full_field_ml_job.sh"
