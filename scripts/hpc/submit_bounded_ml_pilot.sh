#!/usr/bin/env bash
# Safe detached submission wrapper for the bounded Well-2 Stage 2 pilot.
# Invoke as: bash scripts/hpc/submit_bounded_ml_pilot.sh --dry-run|--submit

main() {
  local mode="${1:-}" rc output_dir validation_output partition_qos association_qos
  local states running total duplicate response job_id
  if [[ "$#" -ne 1 || ( "$mode" != "--dry-run" && "$mode" != "--submit" ) ]]; then
    printf 'ERROR: choose exactly one mode: --dry-run or --submit\n' >&2
    return 2
  fi

  local code_root
  code_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)" || {
    printf 'ERROR: cannot resolve the staged source directory.\n' >&2; return 2;
  }
  local python_bin="${RTM_PYTHON:-python3}"
  local config="${RTM_BOUNDED_PILOT_CONFIG:-$code_root/config/hpc/well2_bounded_pilot.json}"
  local cache="${RTM_PAIRED_CACHE_DIR:-}"
  local data_root="${RTM_CHALLENGE3_ROOT:-}"
  local output_root="${RTM_HPC_OUTPUT_ROOT:-}"
  local account="petronas" partition="petronas_apgce" qos="normal"
  local cpus=4 memory=32G walltime=02:00:00

  if [[ -z "$cache" || -z "$data_root" || -z "$output_root" ]]; then
    printf 'ERROR: set RTM_PAIRED_CACHE_DIR, RTM_CHALLENGE3_ROOT, and RTM_HPC_OUTPUT_ROOT to authorized HPC paths.\n' >&2
    return 2
  fi

  local path
  for path in "$config" "$cache" "$data_root" "$output_root" "$code_root/scripts/hpc/run_bounded_ml_pilot.py" "$code_root/scripts/hpc/validate_bounded_ml_pilot_config.py"; do
    if [[ ! -e "$path" ]]; then printf 'ERROR: required path does not exist: %s\n' "$path" >&2; return 2; fi
    case "$path" in *[[:space:],]*) printf 'ERROR: configured path contains characters unsafe for Slurm export: %s\n' "$path" >&2; return 2;; esac
  done
  if [[ ! -d "$cache" || ! -d "$data_root" || ! -d "$output_root" || ! -f "$config" ]]; then
    printf 'ERROR: config must be a file; cache, source-data and private output paths must be directories.\n' >&2
    return 2
  fi
  if [[ ! -d "$output_root/full-field-attributes/attributes-0a361aa415beca729b53" ]]; then
    printf 'ERROR: verified Stage 1 attribute directory is missing under the private output root.\n' >&2
    return 2
  fi
  if [[ ! -x "$(command -v "$python_bin" 2>/dev/null)" ]]; then
    printf 'ERROR: Python executable is unavailable: %s\n' "$python_bin" >&2; return 2
  fi
  if [[ "$mode" == "--submit" ]]; then
    for tool in sbatch squeue scontrol sacctmgr flock; do
      command -v "$tool" >/dev/null 2>&1 || { printf 'ERROR: required command is unavailable: %s\n' "$tool" >&2; return 2; }
    done
  fi

  validation_output="$("$python_bin" "$code_root/scripts/hpc/validate_bounded_ml_pilot_config.py" \
    --config "$config" --output-root "$output_root" 2>&1)"
  rc=$?
  if (( rc != 0 )); then
    printf 'ERROR: bounded-pilot configuration check failed (status %s):\n%s\n' "$rc" "$validation_output" >&2
    return 2
  fi
  printf 'Bounded-pilot configuration: %s\n' "$validation_output"

  local stamp
  stamp="$(date -u +%Y%m%dT%H%M%SZ)" || { printf 'ERROR: cannot create unique run timestamp.\n' >&2; return 2; }
  local suffix="${RTM_BOUNDED_RUN_SUFFIX:-${stamp}-$$}"
  if [[ ! "$suffix" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$ ]]; then
    printf 'ERROR: RTM_BOUNDED_RUN_SUFFIX must be a safe 1-80 character identifier.\n' >&2; return 2
  fi
  output_dir="$output_root/well2-coal-control-v1-${suffix}"
  if [[ -e "$output_dir" || -L "$output_dir" ]]; then
    printf 'ERROR: unique output directory collision: %s\n' "$output_dir" >&2; return 2
  fi
  case "$output_dir" in "$cache"/*|"$data_root"/*) printf 'ERROR: unsafe output path.\n' >&2; return 2;; esac
  case "$output_dir" in *[[:space:],]*) printf 'ERROR: output path contains characters unsafe for Slurm export.\n' >&2; return 2;; esac

  local pilot_args=(
    --config "$config" --paired-cache "$cache" --challenge3-root "$data_root"
    --output-root "$output_root" --output-dir "$output_dir"
    --experiment-id well2-coal-control-v1
    --reference-bounds 64 192 1088 1216
    --calibration-bounds 869 997 64 192
    --sample-bounds 850 901
    --scoring-bounds 600 728 131 259 850 901
    --max-samples 10000 --max-score-voxels 1000000
    --max-output-bytes 134217728 --max-memory-mib 2048
    --compare-chunks 16x16x64 32x32x128 64x64x128
    --threads 4 --seed 2026
  )
  local dry_output
  dry_output="$("$python_bin" "$code_root/scripts/hpc/run_bounded_ml_pilot.py" "${pilot_args[@]}" --dry-run 2>&1)"
  rc=$?
  if (( rc != 0 )); then
    printf 'ERROR: bounded-pilot dry-run validation failed (status %s):\n%s\n' "$rc" "$dry_output" >&2
    return 2
  fi
  printf 'Bounded pilot dry-run: validated geometry, budgets and existing artifacts.\n%s\n' "$dry_output"
  printf 'Experiment: well2-coal-control-v1\nOutput (new, private): %s\n' "$output_dir"
  printf 'Slurm: account=%s partition=%s qos=%s CPUs=%s memory=%s walltime=%s GPU=none\n' "$account" "$partition" "$qos" "$cpus" "$memory" "$walltime"
  printf 'Submission template: sbatch --parsable --account %s --partition %s --qos %s --constraint=nodetype-h200.B-3072 --mcs-label=apgce --nodes=1 --ntasks=1 --cpus-per-task %s --mem %s --time %s --output %s/slurm-well2-ml-%%j.out --error %s/slurm-well2-ml-%%j.err --export=ALL,RTM_BOUNDED_OUTPUT_DIR=<unique-private-run> %s/scripts/hpc/run_bounded_ml_pilot_job.sh\n' \
    "$account" "$partition" "$qos" "$cpus" "$memory" "$walltime" "$output_root" "$output_root" "$code_root"

  if [[ "$mode" == "--dry-run" ]]; then
    printf 'DRY RUN ONLY: no directory created, no model fitted, no job submitted.\n'
    return 0
  fi

  local output_stat
  output_stat="$(stat -c '%u %a' "$output_root" 2>/dev/null)" || { printf 'ERROR: cannot verify private output-root ownership/mode.\n' >&2; return 2; }
  local owner mode_bits
  read -r owner mode_bits <<< "$output_stat"
  if [[ "$owner" != "$(id -u)" ]] || (( (8#$mode_bits & 077) != 0 )); then
    printf 'ERROR: output root must be owned by the current user and inaccessible to group/others.\n' >&2; return 2
  fi
  if [[ -z "${SLURM_ACCOUNT:-}" && "$account" != petronas ]]; then
    printf 'ERROR: configured account does not match the verified account.\n' >&2; return 2
  fi
  partition_qos="$(scontrol show partition "$partition" 2>&1)"
  rc=$?
  if (( rc != 0 )) || [[ "$partition_qos" != *"State=UP"* ]] || \
     [[ "$partition_qos" != *"AllowQos=ALL"* && "$partition_qos" != *"AllowQos=${qos}"* && "$partition_qos" != *"AllowQos="*"${qos}"* ]]; then
    printf 'ERROR: could not verify QoS %s is allowed on partition %s: %s\n' "$qos" "$partition" "$partition_qos" >&2
    return 2
  fi
  association_qos="$(sacctmgr show assoc where user="${USER:?USER is not set}" account="$account" format=QOS -nP 2>&1)"
  rc=$?
  if (( rc != 0 )) || ! printf '%s\n' "$association_qos" | tr ',;|' '\n\n\n' | grep -Fxq "$qos"; then
    printf 'ERROR: QoS %s is not verified for account %s: %s\n' "$qos" "$account" "$association_qos" >&2
    return 2
  fi
  states="$(squeue --user "$USER" --account "$account" --noheader --format='%T' 2>&1)"
  rc=$?
  if (( rc != 0 )); then printf 'ERROR: squeue failed; no submission: %s\n' "$states" >&2; return 2; fi
  running="$(printf '%s\n' "$states" | awk '$1 == "RUNNING" {n++} END {print n+0}')"
  total="$(printf '%s\n' "$states" | awk 'NF {n++} END {print n+0}')"
  if (( running >= 1 || total >= 2 )); then
    printf 'ERROR: team Slurm limit reached (running=%s submitted=%s); no job submitted.\n' "$running" "$total" >&2
    return 2
  fi
  duplicate="$(squeue --user "$USER" --account "$account" --name rtm-well2-coal-control-v1 --noheader --format='%A %j %T' 2>&1)"
  rc=$?
  if (( rc != 0 )); then printf 'ERROR: duplicate-job check failed; no submission: %s\n' "$duplicate" >&2; return 2; fi
  if [[ -n "$duplicate" ]]; then printf 'ERROR: conflicting Well-2 pilot job exists; no submission: %s\n' "$duplicate" >&2; return 2; fi

  local lock_file="$output_root/.rtm-bounded-ml-submit.lock"
  exec 9>"$lock_file" || { printf 'ERROR: cannot open submission lock.\n' >&2; return 2; }
  flock -n 9 || { printf 'ERROR: another bounded-pilot submission check is active.\n' >&2; return 2; }
  printf 'Type exactly SUBMIT well2-coal-control-v1 to submit, or anything else to cancel: '
  IFS= read -r response || { printf '\nERROR: confirmation input was not received.\n' >&2; return 2; }
  if [[ "$response" != "SUBMIT well2-coal-control-v1" ]]; then printf 'Submission cancelled; no job submitted.\n'; return 1; fi

  local worker="$code_root/scripts/hpc/run_bounded_ml_pilot_job.sh"
  [[ -f "$worker" ]] || { printf 'ERROR: Slurm worker script is missing: %s\n' "$worker" >&2; return 2; }
  local submit_output
  submit_output="$(sbatch --parsable --account "$account" --partition "$partition" --qos "$qos" \
    --constraint=nodetype-h200.B-3072 --mcs-label=apgce --nodes=1 --ntasks=1 \
    --cpus-per-task "$cpus" --mem "$memory" --time "$walltime" --job-name rtm-well2-coal-control-v1 \
    --output "$output_root/slurm-well2-ml-%j.out" --error "$output_root/slurm-well2-ml-%j.err" \
    --export="ALL,RTM_CODE_ROOT=$code_root,RTM_BOUNDED_OUTPUT_DIR=$output_dir,RTM_PYTHON=$python_bin,RTM_BOUNDED_PILOT_CONFIG=$config,RTM_PAIRED_CACHE_DIR=$cache,RTM_CHALLENGE3_ROOT=$data_root,RTM_HPC_OUTPUT_ROOT=$output_root" \
    "$worker" 2>&1)"
  rc=$?
  if (( rc != 0 )); then printf 'ERROR: sbatch failed (status %s); interactive shell remains available:\n%s\n' "$rc" "$submit_output" >&2; return 2; fi
  job_id="${submit_output%%;*}"
  if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then printf 'ERROR: sbatch returned an unrecognized job ID: %s\n' "$submit_output" >&2; return 2; fi
  printf 'SUBMITTED Slurm job ID: %s\n' "$job_id"
  printf 'stdout: %s/slurm-well2-ml-%s.out\nstderr: %s/slurm-well2-ml-%s.err\n' "$output_root" "$job_id" "$output_root" "$job_id"
  return 0
}

main "$@"
