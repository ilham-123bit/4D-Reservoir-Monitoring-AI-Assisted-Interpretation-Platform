#!/usr/bin/env python3
"""Guarded removal of a completed disposable bounded Stage 2 ML benchmark.

Dry-run is the default. This tool never removes source data or full-field
products and deliberately refuses incomplete, resumable, running, malformed,
or non-private runs.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import time

PILOT_SCHEMA = "rtm-bounded-ml-pilot-1"
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
PRESERVE = ("pilot_metrics.json", "pilot_config_snapshot.json")
TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
                   "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE"}


def _no_symlinks(path: Path) -> bool:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        try:
            if stat.S_ISLNK(current.lstat().st_mode):
                return False
        except FileNotFoundError:
            return False
    return True


def _private_owned(path: Path) -> bool:
    st = path.stat()
    return st.st_uid == os.getuid() and stat.S_IMODE(st.st_mode) & 0o077 == 0


def _scheduler_state(job_id: str) -> str:
    """Require squeue to report no active job and sacct a terminal state."""
    try:
        queue = subprocess.run(["squeue", "-h", "-j", job_id, "-o", "%T"],
                               check=True, capture_output=True, text=True, timeout=15)
        if queue.stdout.strip():
            return "ACTIVE"
        accounting = subprocess.run(["sacct", "-n", "-X", "-j", job_id,
                                     "--format=State"], check=True,
                                    capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"could not verify Slurm state: {exc}") from exc
    states = [line.strip().split()[0].split("+")[0].upper()
              for line in accounting.stdout.splitlines() if line.strip()]
    terminal = [state for state in states if state in TERMINAL_STATES]
    if not terminal:
        return "UNKNOWN"
    # Allocation and batch steps can produce duplicate records. Any active or
    # nonterminal state remains a refusal; all observed states must be terminal.
    if any(state not in TERMINAL_STATES for state in states):
        return "UNKNOWN"
    return terminal[0]


def _walk_regular_files(root: Path):
    for current, dirs, files in os.walk(root, followlinks=False):
        base = Path(current)
        for name in dirs + files:
            path = base / name
            if path.is_symlink():
                raise ValueError(f"symlink in run tree is not permitted: {path}")
        for name in files:
            path = base / name
            if not path.is_file():
                raise ValueError(f"non-regular file in run tree: {path}")
            yield path


def inspect_run(output_root: Path, run_id: str, protected_paths=()):
    if not RUN_ID_RE.fullmatch(run_id) or run_id in {".", ".."}:
        raise ValueError("run ID must be a simple explicit directory name")
    root = output_root.expanduser().absolute()
    if not root.is_dir() or not _no_symlinks(root) or not _private_owned(root):
        raise ValueError("output root must be a real, user-owned private directory")
    run = root / run_id
    if not run.exists() or run.parent != root or not _no_symlinks(run):
        raise ValueError("run must be a direct, non-symlink child of output root")
    resolved = run.resolve(strict=True)
    if resolved.parent != root.resolve(strict=True) or not _private_owned(run):
        raise ValueError("run path is not canonical, direct-child, private and user-owned")
    for value in protected_paths:
        protected = Path(value).expanduser().resolve(strict=True)
        if resolved == protected or resolved in protected.parents or protected in resolved.parents:
            raise ValueError(f"run overlaps protected path: {protected}")
    if len(protected_paths) < 3:
        raise ValueError("supply original-data, paired-cache and verified-attributes protected paths")
    metrics_path = run / "pilot_metrics.json"
    config_path = run / "pilot_config_snapshot.json"
    if not metrics_path.is_file() or not config_path.is_file():
        raise ValueError("pilot metrics and configuration snapshot are required")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (metrics.get("schema") != PILOT_SCHEMA or metrics.get("status") != "complete"
            or metrics.get("output_class") != "disposable_stage2_benchmark"
            or metrics.get("resumable") is not False):
        raise ValueError("run is not a completed non-resumable disposable Stage 2 benchmark")
    job_id = metrics.get("slurm_job_id")
    if not isinstance(job_id, str) or not job_id.isdigit():
        raise ValueError("run lacks a verifiable Slurm job ID")
    if config.get("schema") != "rtm-bounded-ml-pilot-config-1":
        raise ValueError("configuration snapshot schema is incompatible")
    files = list(_walk_regular_files(run))
    relative = {p.relative_to(run).as_posix() for p in files}
    if not set(PRESERVE).issubset(relative):
        raise ValueError("required preservation records are missing")
    expected_dirs = {str(Path(item["run_directory"]).parts[0])
                     for item in metrics.get("chunk_benchmarks", [])
                     if isinstance(item, dict) and isinstance(item.get("run_directory"), str)}
    actual_chunk_dirs = {p.name for p in run.iterdir() if p.is_dir()}
    actual_top_files = {p.name for p in run.iterdir() if p.is_file()}
    if actual_chunk_dirs != expected_dirs or actual_top_files != set(PRESERVE):
        raise ValueError("run root does not match the declared completed chunk benchmark structure")
    model_outputs = {f"{model}-{kind}.npy" for model in
                    ("mahalanobis", "pca", "isolation_forest", "gmm")
                    for kind in ("score", "flag")}
    allowed_score_files = model_outputs | {"manifest.json", "chunks.jsonl", "complete.json", ".lock"}
    for rel in relative - set(PRESERVE):
        parts = Path(rel).parts
        if (len(parts) != 4 or parts[0] not in expected_dirs
                or parts[1] != "model-scores"
                or not re.fullmatch(r"models-[0-9a-f]{20}", parts[2])
                or parts[3] not in allowed_score_files):
            raise ValueError(f"unexpected output structure prevents cleanup: {rel}")
    for chunk_name in expected_dirs:
        chunk_path = run / chunk_name
        if {p.name for p in chunk_path.iterdir()} != {"model-scores"}:
            raise ValueError(f"unexpected chunk output structure: {chunk_name}")
    return {"run": resolved, "job_id": job_id, "metrics": metrics,
            "config": config, "files": files,
            "bytes": sum(p.stat().st_size for p in files),
            "preserve_files": [p for p in files if p.suffix.lower() in {".json", ".jsonl"}]}


def _capacity(path: Path):
    usage = shutil.disk_usage(path)
    return {"total_bytes": usage.total, "used_bytes": usage.used,
            "free_bytes": usage.free, "sampled_unix_time": int(time.time()),
            "note": "Filesystem free-space reporting may lag deletion and does not establish quota or retention policy."}


def _quota_status(path: Path):
    try:
        result = subprocess.run(["lfs", "quota", "-u", getpass.getuser(), str(path)],
                                check=True, capture_output=True, text=True, timeout=20)
        return {"available": True, "command": "lfs quota -u <current-user> <path>",
                "output": result.stdout.strip()[-4000:]}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "command": "lfs quota -u <current-user> <path>",
                "reason": str(exc)}


def cleanup(*, output_root: Path, run_id: str, protected_paths=(), confirm_id=None,
            storage_authorization_reference=None):
    info = inspect_run(output_root, run_id, protected_paths)
    state = _scheduler_state(info["job_id"])
    if state not in TERMINAL_STATES:
        raise RuntimeError(f"refusing cleanup because Slurm state is {state}")
    before = _capacity(info["run"].parent)
    quota_before = _quota_status(info["run"].parent)
    summary_root = info["run"].parent / "cleanup-records"
    record = summary_root / f"{run_id}-{info['metrics']['slurm_job_id']}"
    preview = {"run_id": run_id, "canonical_path": str(info["run"]),
               "slurm_job_id": info["job_id"], "slurm_state": state,
               "file_count": len(info["files"]), "bytes_to_remove": info["bytes"],
               "files": [str(p.relative_to(info["run"])) for p in info["files"]],
               "capacity_before": before, "preserved_record_directory": str(record),
               "quota_before": quota_before,
               "preserved_metadata_files": [str(p.relative_to(info["run"]))
                                            for p in info["preserve_files"]],
               "storage_authorization_reference": storage_authorization_reference,
               "dry_run": confirm_id is None}
    if confirm_id is None:
        return preview
    if confirm_id != run_id:
        raise ValueError("destructive confirmation must exactly match the run ID")
    if not quota_before["available"]:
        raise RuntimeError("quota status is unavailable; refusing destructive cleanup")
    if not isinstance(storage_authorization_reference, str) or not storage_authorization_reference.strip() or any(
            t in storage_authorization_reference.upper()
            for t in ("PLACEHOLDER", "REPLACE", "INSERT", "TODO", "ACTUAL_")):
        raise ValueError("actual storage authorization reference is required for deletion")
    storage_authorization_reference = storage_authorization_reference.strip()
    if (summary_root.is_symlink() or
            (summary_root.exists() and not _private_owned(summary_root))):
        raise PermissionError("cleanup records directory must not be a symlink and must be private")
    if record.exists() or record.is_symlink():
        raise FileExistsError("preservation record path already exists; refusing overwrite")
    summary_root.mkdir(mode=0o700, exist_ok=True)
    if not _private_owned(summary_root):
        raise PermissionError("cleanup records directory must be private and user-owned")
    record.mkdir(mode=0o700)
    try:
        for source in info["preserve_files"]:
            destination = record / source.relative_to(info["run"])
            destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        for source in info["preserve_files"]:
            destination = record / source.relative_to(info["run"])
            if source.read_bytes() != destination.read_bytes():
                raise OSError(f"preserved metadata does not match source: {source}")
        (record / "cleanup_record.json").write_text(
            json.dumps({**preview, "dry_run": False}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        os.chmod(record / "cleanup_record.json", 0o600)
        # Recheck path identity and scheduler state immediately before removal.
        current = inspect_run(output_root, run_id, protected_paths)
        if current["run"] != info["run"] or current["bytes"] != info["bytes"]:
            raise RuntimeError("run changed after preview; refusing deletion")
        state = _scheduler_state(info["job_id"])
        if state not in TERMINAL_STATES:
            raise RuntimeError(f"Slurm state changed to {state}; refusing deletion")
        shutil.rmtree(info["run"])
    except Exception:
        shutil.rmtree(record, ignore_errors=True)
        raise
    preview.update(dry_run=False, capacity_after=_capacity(summary_root),
                   quota_after=_quota_status(summary_root), removed=True,
                   capacity_reclamation_note="Lustre quota/free-space counters may update asynchronously; verify again after site accounting delay.")
    record_file = record / "cleanup_record.json"
    temporary = record / ".cleanup-record.tmp"
    temporary.write_text(json.dumps(preview, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, record_file)
    return preview


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--protected-path", type=Path, action="append", default=[])
    parser.add_argument("--confirm-id", help="must exactly equal --run-id to delete")
    parser.add_argument("--storage-authorization-reference")
    parser.add_argument("--dry-run", action="store_true", help="show preview; this is already the default")
    args = parser.parse_args(argv)
    try:
        result = cleanup(output_root=args.output_root, run_id=args.run_id,
                         protected_paths=args.protected_path,
                         confirm_id=args.confirm_id,
                         storage_authorization_reference=args.storage_authorization_reference)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
