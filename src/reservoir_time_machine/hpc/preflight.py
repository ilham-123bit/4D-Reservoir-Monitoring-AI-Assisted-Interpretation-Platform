"""Read-only, timeout-bounded HPC resource and storage preflight."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def _query(argv: list[str], timeout: float = 8.0) -> dict:
    try:
        result = subprocess.run(argv, check=False, capture_output=True, text=True, timeout=timeout)
        return {"available": True, "returncode": result.returncode,
                "stdout": result.stdout.strip(), "stderr": result.stderr.strip()[:500]}
    except FileNotFoundError:
        return {"available": False, "reason": "command unavailable"}
    except subprocess.TimeoutExpired:
        return {"available": False, "reason": f"timed out after {timeout:g}s"}


def collect_preflight(storage_paths=()) -> dict:
    """Collect scheduler, Python/PyTorch, and caller-selected storage facts only."""
    partitions = _query(["sinfo", "-h", "-o", "%P|%c|%m|%l|%G"], timeout=12)
    storage = {}
    for raw in storage_paths:
        path = Path(raw).expanduser().resolve()
        try:
            usage = shutil.disk_usage(path)
            storage[str(path)] = {"exists": path.exists(), "total_bytes": usage.total,
                                  "used_bytes": usage.used, "free_bytes": usage.free}
        except OSError as exc:
            storage[str(path)] = {"exists": path.exists(), "error": type(exc).__name__}
    quota = _query(["quota", "-s"], timeout=8)
    module_list = _query(["modulecmd", "python", "list"], timeout=8)
    try:
        import numpy
        numpy_info = {"available": True, "version": numpy.__version__}
    except Exception as exc:
        numpy_info = {"available": False, "reason": type(exc).__name__}
    try:
        import torch
        torch_info = {"available": True, "version": torch.__version__,
                      "cuda_available": bool(torch.cuda.is_available()),
                      "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0}
    except Exception as exc:
        torch_info = {"available": False, "reason": type(exc).__name__}
    return {"schema": "rtm-hpc-preflight-1", "timestamp_unix": int(time.time()),
            "python": {"version": sys.version.split()[0], "executable": sys.executable,
                       "slurm_job_id_present": bool(os.environ.get("SLURM_JOB_ID"))},
            "slurm_partitions": partitions, "quota": quota, "module_environment": module_list,
            "scientific_runtime": {"numpy": numpy_info, "pytorch": torch_info},
            "storage": storage,
            "note": "Read-only inspection only; no job was submitted. Verify policy and allocation before compute."}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage", action="append", default=[], help="existing directory to inspect (repeatable)")
    parser.add_argument("--output", type=Path, help="optional private output JSON path; parent must already exist")
    args = parser.parse_args(argv)
    report = collect_preflight(args.storage)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        target = args.output.expanduser().resolve()
        if not target.parent.is_dir() or target.exists():
            parser.error("--output parent must exist and target must not already exist")
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
