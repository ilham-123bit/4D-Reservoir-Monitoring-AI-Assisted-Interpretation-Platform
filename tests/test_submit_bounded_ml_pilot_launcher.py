"""Synthetic shell-level safety tests; these never invoke a real sbatch."""
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[1]
LAUNCHER = SOURCE / "scripts/hpc/submit_bounded_ml_pilot.sh"
REAL_PYTHON = os.environ.get("PYTHON", "python3")


def config_fixture():
    return json.loads((SOURCE / "config/hpc/well2_bounded_pilot.json").read_text(encoding="utf-8"))


class BoundedLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.config = self.root / "config.json"
        self.output = self.root / "private-output"
        self.output.mkdir(mode=0o700)
        attribute_dir = self.output / "full-field-attributes/attributes-0a361aa415beca729b53"
        attribute_dir.mkdir(parents=True)
        (attribute_dir / "complete.json").write_text("{}", encoding="utf-8")
        self.cache = self.root / "paired-cache"
        self.cache.mkdir(mode=0o700)
        self.data = self.root / "challenge3"
        self.data.mkdir(mode=0o700)
        self.write_config()
        self.write_command("python", f'''#!/usr/bin/env bash
case "$1" in
  */validate_bounded_ml_pilot_config.py) exec {REAL_PYTHON} "$@" ;;
  */run_bounded_ml_pilot.py)
    if [[ "${{RTM_TEST_PILOT_FAIL:-}}" == YES ]]; then echo 'synthetic pilot dry-run failure' >&2; exit 9; fi
    echo '{{"status":"dry_run_validated","models_fit":false}}'; exit 0 ;;
  *) echo "unexpected Python target: $1" >&2; exit 90 ;;
esac
''')
        self.write_command("scontrol", "#!/usr/bin/env bash\necho 'PartitionName=petronas_apgce State=UP AllowQos=ALL'\n")
        self.write_command("sacctmgr", "#!/usr/bin/env bash\necho normal\n")
        self.write_command("squeue", '''#!/usr/bin/env bash
if [[ "${RTM_TEST_CONFLICT:-}" == YES && "$*" == *"--name"* ]]; then echo '9001 rtm-well2-coal-control-v1 RUNNING'; fi
exit 0
''')
        self.write_command("sbatch", '''#!/usr/bin/env bash
if [[ -n "${RTM_TEST_SBATCH_CAPTURE:-}" ]]; then printf '%s\\n' "$*" > "$RTM_TEST_SBATCH_CAPTURE"; fi
if [[ "${RTM_TEST_SBATCH_FAIL:-}" == YES ]]; then echo 'synthetic scheduler rejection' >&2; exit 9; fi
echo 812345
''')
        self.env = os.environ.copy()
        self.env.update({
            "PATH": f"{self.bin}:{os.environ['PATH']}",
            "RTM_PYTHON": str(self.bin / "python"),
            "RTM_BOUNDED_PILOT_CONFIG": str(self.config),
            "RTM_PAIRED_CACHE_DIR": str(self.cache),
            "RTM_CHALLENGE3_ROOT": str(self.data),
            "RTM_HPC_OUTPUT_ROOT": str(self.output),
            "RTM_BOUNDED_RUN_SUFFIX": "test-run",
            "USER": "synthetic-user",
        })

    def write_config(self, config=None):
        self.config.write_text(json.dumps(config or config_fixture()), encoding="utf-8")

    def write_command(self, name, body):
        path = self.bin / name
        path.write_text(body, encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    def invoke(self, mode, input_text=None, **extra):
        env = self.env.copy()
        env.update(extra)
        return subprocess.run(["bash", str(LAUNCHER), mode], input=input_text, text=True,
                              capture_output=True, env=env, timeout=20)

    def test_synthetic_approved_dry_run_validates_without_submission_or_writes(self):
        result = self.invoke("--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("DRY RUN ONLY", result.stdout)
        self.assertIn('"models_fit":false', result.stdout.replace(" ", ""))
        self.assertFalse((self.output / "well2-coal-control-v1-test-run").exists())

    def test_scientific_config_invalidity_fails_safely_without_admin_forms(self):
        config = config_fixture()
        config["mandatory_edge_qc"] = False
        self.write_config(config)
        result = self.invoke("--dry-run")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("edge QC", result.stderr)

    def test_submit_confirmation_is_the_operator_authorization(self):
        result = self.invoke("--submit", "SUBMIT well2-coal-control-v1\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("SUBMITTED Slurm job ID: 812345", result.stdout)

    def test_incorrect_calibration_bounds_fail_closed(self):
        config = config_fixture()
        config["calibration_bounds"] = [896, 1024, 64, 192]
        self.write_config(config)
        result = self.invoke("--dry-run")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("bounds mismatch", result.stderr)

    def test_conflicting_job_prevents_sbatch(self):
        result = self.invoke("--submit", "SUBMIT well2-coal-control-v1\n", RTM_TEST_CONFLICT="YES")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("conflicting Well-2 pilot job", result.stderr)

    def test_sbatch_failure_returns_without_affecting_caller(self):
        result = self.invoke("--submit", "SUBMIT well2-coal-control-v1\n", RTM_TEST_SBATCH_FAIL="YES")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("sbatch failed", result.stderr)
        self.assertIn("interactive shell remains available", result.stderr)

    def test_output_collision_prevents_submission(self):
        (self.output / "well2-coal-control-v1-test-run").mkdir()
        result = self.invoke("--submit", "SUBMIT well2-coal-control-v1\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("output directory collision", result.stderr)

    def test_operator_declining_confirmation_does_not_submit(self):
        result = self.invoke("--submit", "no\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Submission cancelled", result.stdout)

    def test_successful_synthetic_submit_reports_job_id_without_running_worker(self):
        capture = self.root / "sbatch-args.txt"
        result = self.invoke("--submit", "SUBMIT well2-coal-control-v1\n",
                             RTM_TEST_SBATCH_CAPTURE=str(capture))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("SUBMITTED Slurm job ID: 812345", result.stdout)
        args = capture.read_text()
        for required in ("--account petronas", "--partition petronas_apgce", "--qos normal",
                         "--constraint=nodetype-h200.B-3072", "--mcs-label=apgce",
                         "--nodes=1", "--ntasks=1", "--cpus-per-task 4", "--mem 32G",
                         "--time 02:00:00", "slurm-well2-ml-%j.out", "slurm-well2-ml-%j.err"):
            self.assertIn(required, args)
        self.assertNotIn("--gres", args)


if __name__ == "__main__":
    unittest.main()
