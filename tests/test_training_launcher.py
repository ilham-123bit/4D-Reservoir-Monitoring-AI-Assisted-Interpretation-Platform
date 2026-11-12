"""Detached GPU launcher tests; fake Slurm commands only, never submit jobs."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class TrainingLauncherTests(unittest.TestCase):
    def _run(self, *, fail_sbatch=False):
        temp = tempfile.TemporaryDirectory()
        root = Path(temp.name)
        bindir = root / "bin"
        bindir.mkdir()
        code = root / "source"
        (code / "scripts/hpc").mkdir(parents=True)
        (code / "scripts/hpc/train_job.sh").write_text("#!/usr/bin/env bash\n")
        config = root / "pilot.json"
        config.write_text("{}\n")
        out = root / "output"
        out.mkdir()
        (bindir / "squeue").write_text("#!/usr/bin/env bash\nexit 0\n")
        (bindir / "sbatch").write_text("#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" > \"$RTM_TEST_SBATCH_ARGS\"\n"
            + ("echo simulated-rejection >&2; exit 42\n" if fail_sbatch else "echo 765432\n"))
        for name in ("squeue", "sbatch"):
            (bindir / name).chmod(0o755)
        env = os.environ.copy()
        env.update({"PATH": f"{bindir}:{env['PATH']}", "RTM_SLURM_ACCOUNT": "petronas",
            "RTM_SLURM_PARTITION": "petronas_apgce", "RTM_SLURM_QOS": "normal",
            "RTM_SLURM_CONSTRAINT": "nodetype-h200.B-3072", "RTM_SLURM_MCS_LABEL": "apgce",
            "RTM_SLURM_GPU_REQUEST": "gpu:nvidia_h200_nvl_2g.35gb:1", "RTM_SLURM_CPUS": "4",
            "RTM_SLURM_MEMORY": "32G", "RTM_SLURM_TIME": "02:00:00", "RTM_CODE_ROOT": str(code),
            "RTM_HPC_OUTPUT_ROOT": str(out), "RTM_EXPERIMENT_CONFIG": str(config),
            "RTM_CHALLENGE3_ROOT": str(root / "input"), "RTM_TEST_SBATCH_ARGS": str(root / "args.txt"),
            "RTM_GPU_KERNEL_ACTIVE": "0"})
        (root / "input").mkdir()
        result = subprocess.run(["bash", str(Path(__file__).resolve().parents[1] / "scripts/hpc/submit_training.sh")],
            env=env, text=True, capture_output=True, check=False)
        return temp, root, result

    def test_submits_detached_job_with_site_gpu_controls_and_separate_logs(self):
        temp, root, result = self._run()
        try:
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Submitted GPU training job 765432", result.stdout)
            self.assertIn("train_job-765432.out", result.stdout)
            args = (root / "args.txt").read_text()
            for value in ("--qos normal", "--constraint nodetype-h200.B-3072", "--mcs-label apgce",
                          "--nodes=1", "--ntasks=1", "--gres gpu:nvidia_h200_nvl_2g.35gb:1",
                          "--cpus-per-task 4", "--mem 32G", "--time 02:00:00", "%x-%j.out", "%x-%j.err"):
                self.assertIn(value, args)
        finally:
            temp.cleanup()

    def test_sbatch_failure_returns_nonzero_without_parent_shell_exit(self):
        temp, _, result = self._run(fail_sbatch=True)
        try:
            self.assertEqual(result.returncode, 2)
            self.assertIn("no training job was accepted", result.stderr)
            self.assertIn("simulated-rejection", result.stderr)
        finally:
            temp.cleanup()


if __name__ == "__main__":
    unittest.main()
