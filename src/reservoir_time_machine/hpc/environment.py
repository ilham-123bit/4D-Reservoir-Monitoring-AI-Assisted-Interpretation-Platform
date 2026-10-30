"""Read-only inventory of Python ML frameworks and visible GPU allocation."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any


def _version(name: str) -> dict[str, Any]:
    if importlib.util.find_spec(name) is None:
        return {"available": False}
    try:
        module = importlib.import_module(name)
        return {"available": True, "version": getattr(module, "__version__", "unknown")}
    except Exception as exc:
        return {"available": True, "import_error": f"{type(exc).__name__}: {exc}"}


def _redacted_executable() -> str:
    value = sys.executable
    home = str(Path.home())
    return value.replace(home, "$HOME") if home and home in value else Path(value).name


def _run(command: list[str]) -> dict[str, Any]:
    if not shutil.which(command[0]):
        return {"available": False, "reason": f"{command[0]} is not installed or not on PATH"}
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
        return {"available": True, "return_code": result.returncode,
                "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except Exception as exc:
        return {"available": True, "error": f"{type(exc).__name__}: {exc}"}


def collect_environment() -> dict[str, Any]:
    """Probe runtime details without changing modules, devices, or scheduler state."""
    distribution = {}
    if platform.system() == "Linux" and hasattr(platform, "freedesktop_os_release"):
        try:
            distribution = platform.freedesktop_os_release()
        except OSError:
            distribution = {"unavailable": True}
    report: dict[str, Any] = {
        "purpose": "read_only_hpc_environment_validation",
        "python": {"version": platform.python_version(), "executable": _redacted_executable(),
                   "expected_version": "3.12", "matches_expected": sys.version_info[:2] == (3, 12)},
        "platform": {"system": platform.system(), "release": platform.release(),
                     "machine": platform.machine(), "distribution": distribution,
                     "is_expected_rocky_linux": str(distribution.get("ID", "")).casefold() == "rocky"},
        "libraries": {name: _version(name) for name in (
            "numpy", "scipy", "sklearn", "segyio", "lasio", "openpyxl", "tensorboard", "plotly", "pyvista")},
        "slurm_context": {"inside_slurm_job": bool(os.environ.get("SLURM_JOB_ID")),
                          "gpu_visibility_variable_set": "CUDA_VISIBLE_DEVICES" in os.environ,
                          "allocated_cpu_threads": os.environ.get("SLURM_CPUS_PER_TASK"),
                          "memory_allocation_mb": os.environ.get("SLURM_MEM_PER_NODE")},
        "nvidia_smi": _run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version",
                            "--format=csv,noheader"]),
    }
    try:
        import torch
        torch_info: dict[str, Any] = {"available": True, "version": torch.__version__,
                                     "built_cuda_version": torch.version.cuda,
                                     "cudnn_version": torch.backends.cudnn.version(),
                                     "cuda_available": torch.cuda.is_available(),
                                     "visible_device_count": torch.cuda.device_count()}
        devices = []
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            free, total = torch.cuda.mem_get_info(index)
            devices.append({"visible_index": index, "name": properties.name,
                            "compute_capability": f"{properties.major}.{properties.minor}",
                            "reported_total_memory_bytes": int(properties.total_memory),
                            "free_memory_bytes_at_probe": int(free), "total_memory_bytes_at_probe": int(total),
                            "multiprocessor_count": int(properties.multi_processor_count)})
        torch_info["visible_devices"] = devices
        report["pytorch"] = torch_info
    except Exception as exc:
        report["pytorch"] = {"available": importlib.util.find_spec("torch") is not None,
                             "import_or_probe_error": f"{type(exc).__name__}: {exc}"}
    try:
        import tensorflow as tf
        tf_devices = []
        for index, device in enumerate(tf.config.list_physical_devices("GPU")):
            details = tf.config.experimental.get_device_details(device)
            try:
                memory = tf.config.experimental.get_memory_info(device.name)
            except Exception as exc:
                memory = {"unavailable": f"{type(exc).__name__}: {exc}"}
            tf_devices.append({"visible_index": index, "name": device.name, "details": details,
                               "memory_info": memory})
        report["tensorflow"] = {"available": True, "version": tf.__version__,
            "built_with_cuda": bool(tf.test.is_built_with_cuda()),
            "build_info": tf.sysconfig.get_build_info(),
            "visible_gpu_count": len(tf_devices), "visible_devices": tf_devices}
    except Exception as exc:
        report["tensorflow"] = {"available": importlib.util.find_spec("tensorflow") is not None,
                                "import_or_probe_error": f"{type(exc).__name__}: {exc}"}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="Optional JSON report path; existing files are never overwritten.")
    parser.add_argument("--strict", action="store_true", help="Require Python 3.12/Rocky/Linux, Slurm and CUDA-enabled PyTorch (training baseline).")
    parser.add_argument("--require-tensorflow-cuda", action="store_true", help="Also fail unless TensorFlow sees a CUDA GPU.")
    args = parser.parse_args()
    report = collect_environment()
    report["overall_status"] = "inspection_complete"
    rendered = json.dumps(report, indent=2, sort_keys=True, default=str)
    if args.output:
        target = Path(args.output).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(rendered + "\n")
    print(rendered)
    strict_ok = (report["python"]["matches_expected"]
                 and report["platform"]["system"] == "Linux"
                 and report["platform"]["is_expected_rocky_linux"]
                 and report["slurm_context"]["inside_slurm_job"]
                 and report["slurm_context"]["gpu_visibility_variable_set"]
                 and report.get("pytorch", {}).get("available")
                 and report.get("pytorch", {}).get("cuda_available")
                 and report.get("pytorch", {}).get("visible_device_count", 0) > 0)
    if args.require_tensorflow_cuda:
        strict_ok = (strict_ok and report.get("tensorflow", {}).get("available")
                     and report.get("tensorflow", {}).get("built_with_cuda")
                     and report.get("tensorflow", {}).get("visible_gpu_count", 0) > 0)
    if (args.strict or args.require_tensorflow_cuda) and not strict_ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
