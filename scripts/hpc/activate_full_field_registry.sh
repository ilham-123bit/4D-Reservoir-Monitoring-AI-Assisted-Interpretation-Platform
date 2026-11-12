#!/usr/bin/env bash
# Publish the completed private full-field findings registry; NO ML rerun.
# Requires an authorized Slurm allocation on the APGCE HPC.
set -Eeuo pipefail

PROJECT=/data/epic20/APGCE/apgce_team_02/4D-Reservoir-Monitoring-AI-Assisted-Interpretation-Platform
PRIVATE=/data/epic20/APGCE/apgce_team_02/rtm-derived-private
INPUT=/data/epic20/APGCE/input_data/challenge3
RUN="$PRIVATE/full-field-ml-runs/stage2b-design-a-20261111"
CACHE="$PROJECT/.reservoir_time_machine_cache/challenge3_paired_20261102"
ATTR="$PRIVATE/full-field-attributes/attributes-0a361aa415beca729b53"
MODELS="$RUN/classical-models/models-a345c185619b3e148b1f"
FINDINGS="$RUN/full-field-findings/findings.json"
LABELS="$RUN/full-field-findings/components/component_labels.npy"
OUTPUT="$RUN/full-field-findings/registry"
PYTHON=/cm/shared/apps/python/python312/bin/python

git -C "$PROJECT" rev-parse --is-inside-work-tree >/dev/null
test "$(git -C "$PROJECT" branch --show-current)" = main
git -C "$PROJECT" diff --quiet
git -C "$PROJECT" diff --cached --quiet
test -d "$INPUT" && test -d "$CACHE"
for file in "$ATTR/complete.json" "$MODELS/complete.json" "$FINDINGS" "$LABELS"; do
  if [[ ! -r "$file" ]]; then
    printf 'Missing required completed artifact: %s\n' "$file" >&2
    exit 1
  fi
done
test -x "$PYTHON"
test -f "$PROJECT/scripts/hpc/publish_full_field_registry.py"

install -d -m 700 "$OUTPUT"
export RTM_CODE_ROOT="$PROJECT" RTM_CHALLENGE3_ROOT="$INPUT"
export RTM_PAIRED_CACHE_DIR="$CACHE" RTM_HPC_OUTPUT_ROOT="$PRIVATE"

JOB=$(mktemp "$HOME/rtm-activate-registry.XXXXXX.slurm")
trap 'rm -f "$JOB"' EXIT
cat > "$JOB" <<'SLURM'
#!/usr/bin/env bash
#SBATCH --job-name=rtm-registry
#SBATCH --account=petronas
#SBATCH --partition=petronas_apgce
#SBATCH --qos=normal
#SBATCH --mcs-label=apgce
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
set -Eeuo pipefail
PRIVATE=/data/epic20/APGCE/apgce_team_02/rtm-derived-private
RUN="$PRIVATE/full-field-ml-runs/stage2b-design-a-20261111"
export PYTHONPATH="$RTM_CODE_ROOT/src"
exec /cm/shared/apps/python/python312/bin/python "$RTM_CODE_ROOT/scripts/hpc/publish_full_field_registry.py" \
  --challenge-root "$RTM_CHALLENGE3_ROOT" \
  --cache "$RTM_PAIRED_CACHE_DIR" \
  --attributes "$PRIVATE/full-field-attributes/attributes-0a361aa415beca729b53" \
  --models "$RUN/classical-models/models-a345c185619b3e148b1f" \
  --findings "$RUN/full-field-findings/findings.json" \
  --component-labels "$RUN/full-field-findings/components/component_labels.npy" \
  --output "$RUN/full-field-findings/registry"
SLURM

echo 'Submitting validation/publication using existing completed products.'
sbatch --wait --export=ALL "$JOB"
POINTER="$OUTPUT/latest.json"
test -r "$POINTER"

"$PYTHON" - "$POINTER" <<'PY'
import json, os, sys, tempfile
from pathlib import Path
pointer = Path(sys.argv[1]).resolve(strict=True)
config = Path.home() / ".config/reservoir-time-machine/runtime.json"
data = json.loads(config.read_text()) if config.exists() else {}
if not isinstance(data, dict) or set(data) - {"findings_registry", "full_field_registry"}:
    raise SystemExit("Unexpected runtime config keys. No changes made.")
data["full_field_registry"] = str(pointer)
config.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
fd, temp = tempfile.mkstemp(prefix=".runtime-", dir=config.parent)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(data, stream, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temp, 0o600)
    os.replace(temp, config)
finally:
    Path(temp).unlink(missing_ok=True)
print("PRIVATE REGISTRY POINTER CONFIGURED:", pointer)
PY
echo "Registry available. Refresh the original Reservoir Time Machine desktop app."
