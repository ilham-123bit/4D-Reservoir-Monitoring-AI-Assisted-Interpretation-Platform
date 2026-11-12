"""Checkpoint-verified, overlap-context, resumable neural volume inference."""

from __future__ import annotations

from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat

import numpy as np

from ..ai.torch_models import ConvAutoencoder3DConfig, build_conv_autoencoder_3d, build_siamese_3d
from ..processing.chunks import iter_chunks
from ..processing.full_field import _append, _atomic_json, _canonical, _hash_bytes, _hash_file, _read_journal


@dataclass(frozen=True)
class VerifiedTrainedModel:
    model: object
    model_kind: str
    checkpoint_sha256: str
    checkpoint_path: str
    epoch: int
    dataset_provenance: dict
    device: str


def load_verified_trained_model(checkpoint_path, completion_path, output_root, *, device="cuda:0"):
    """Load only the best checkpoint named and checksummed by a completed training run."""
    import torch
    root = Path(output_root).expanduser().resolve()
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    completion_path = Path(completion_path).expanduser().resolve()
    for path in (checkpoint_path, completion_path):
        if path != root and root not in path.parents:
            raise ValueError("checkpoint and training completion record must remain under private output root")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("status") != "complete" or completion.get("epochs", 0) < 1:
        raise ValueError("training completion record does not establish a completed trained checkpoint")
    digest = _hash_file(checkpoint_path)
    if completion.get("best_checkpoint_sha256") != digest:
        raise ValueError("best checkpoint hash differs from the completed training record")
    experiment_path = completion_path.with_name("experiment_manifest.json")
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    provenance = completion.get("dataset_provenance", {})
    if provenance != experiment.get("dataset_provenance"):
        raise ValueError("training provenance differs between experiment and completion records")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    kind = checkpoint.get("model_kind")
    config = checkpoint.get("model_configuration", {})
    if int(checkpoint.get("epoch", 0)) < 1 or checkpoint.get("dataset_provenance") != provenance:
        raise ValueError("checkpoint has no matching trained epoch/provenance")
    if kind == "siamese":
        model = build_siamese_3d(in_channels=int(config.get("in_channels", 1)),
                                 channels=tuple(config.get("channels", [8, 16, 32])))
    elif kind == "autoencoder":
        model = build_conv_autoencoder_3d(ConvAutoencoder3DConfig(
            in_channels=int(config.get("in_channels", 1)),
            channels=tuple(config.get("channels", [8, 16, 32, 64]))))
    else:
        raise ValueError("checkpoint model kind is unsupported")
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return VerifiedTrainedModel(model, kind, digest, str(checkpoint_path), int(checkpoint["epoch"]), provenance, str(device))


def _extract_padded(array, start, patch_shape):
    slices = tuple(slice(max(0, s), min(size, s + width))
                   for s, width, size in zip(start, patch_shape, array.shape))
    block = np.asarray(array[slices], dtype=np.float32)
    pads = tuple((max(0, -s), max(0, s + width - size))
                 for s, width, size in zip(start, patch_shape, array.shape))
    mode = "reflect" if all(axis > 1 for axis in block.shape) else "edge"
    if any(left or right for left, right in pads):
        block = np.pad(block, pads, mode=mode)
    return block, slices


def run_overlap_volume_inference(verified: VerifiedTrainedModel, baseline, monitor,
                                 missing_trace_mask, output_root, *, source_identity: str,
                                 patch_shape=(32, 32, 64), context_overlap=(8, 8, 16),
                                 min_support_fraction: float = 1.0, max_chunks: int | None = None,
                                 protected_cache=None) -> dict:
    """Run bounded patches; non-overlapping output cores use overlapping seismic context.

    Core outputs are assigned once, so a crash after a write but before journaling is
    idempotently repaired on resume. Patches containing missing or non-finite paired
    samples are excluded and recorded in the coverage map.
    """
    import torch
    before, after = np.asanyarray(baseline), np.asanyarray(monitor)
    missing = np.asanyarray(missing_trace_mask)
    if before.ndim != 3 or before.shape != after.shape or missing.shape != before.shape[:2]:
        raise ValueError("matching paired volumes and trace mask are required")
    patch = tuple(map(int, patch_shape)); halo = tuple(map(int, context_overlap))
    if len(patch) != 3 or len(halo) != 3 or any(p < 8 for p in patch) or any(h < 0 or 2*h >= p for p, h in zip(patch, halo)):
        raise ValueError("patch dimensions must be >=8 and overlap context must leave a positive core")
    core_shape = tuple(p - 2*h for p, h in zip(patch, halo))
    if any(s < p for s, p in zip(before.shape, patch)):
        raise ValueError("patch dimensions must fit the survey volume")
    if len(source_identity) != 64 or float(min_support_fraction) != 1.0:
        raise ValueError("source identity and validity threshold are required")
    if verified.epoch < 1 or len(verified.checkpoint_sha256) != 64 or verified.model_kind not in {"siamese", "autoencoder"}:
        raise ValueError("inference requires a checkpoint verified by load_verified_trained_model")
    root = Path(output_root).expanduser().resolve()
    if protected_cache is not None:
        cache = Path(protected_cache).expanduser().resolve()
        if root == cache or cache in root.parents or root in cache.parents:
            raise ValueError("neural output root must not overlap source cache")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if root.stat().st_uid != os.getuid() or stat.S_IMODE(root.stat().st_mode) & 0o077:
        raise PermissionError("neural outputs must be private and user-owned")
    required_bytes = int(np.prod(before.shape, dtype=np.int64)) * 5
    disk = os.statvfs(root)
    if disk.f_bavail * disk.f_frsize < required_bytes:
        raise OSError("insufficient free space for full-volume neural scores and coverage")
    config = {"schema": "rtm-neural-volume-inference-1", "model_kind": verified.model_kind,
              "checkpoint_sha256": verified.checkpoint_sha256, "epoch": verified.epoch,
              "dataset_provenance": verified.dataset_provenance, "source_identity": source_identity,
              "shape": list(before.shape), "patch_shape": list(patch), "context_overlap": list(halo),
              "core_shape": list(core_shape), "min_support_fraction": min_support_fraction,
              "score_meaning": ("raw Siamese change-head logit" if verified.model_kind == "siamese"
                                else "absolute difference in Baseline/Monitor autoencoder reconstruction residual")}
    config_sha = hashlib.sha256(_canonical(config)).hexdigest()
    run_dir = root / f"neural-inference-{config_sha[:20]}"
    run_dir.mkdir(mode=0o700, exist_ok=True)
    if run_dir.stat().st_uid != os.getuid() or stat.S_IMODE(run_dir.stat().st_mode) & 0o077:
        raise PermissionError("neural inference run directory must be private")
    score_path, coverage_path = run_dir / "score.npy", run_dir / "coverage.npy"
    manifest_path, journal_path, complete_path = run_dir / "manifest.json", run_dir / "chunks.jsonl", run_dir / "complete.json"
    with (run_dir / ".lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        events = _read_journal(journal_path)
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("configuration_sha256") != config_sha:
                raise ValueError("neural inference manifest belongs to another checkpoint or input")
        else:
            if events or complete_path.exists() or score_path.exists() or coverage_path.exists():
                raise ValueError("untracked neural outputs exist without a matching manifest")
            manifest = {"configuration": config, "configuration_sha256": config_sha, "status": "running"}
            _atomic_json(manifest_path, manifest)
            _append(journal_path, {"event": "start", "configuration_sha256": config_sha})
            events = _read_journal(journal_path)
        if not events or events[0].get("configuration_sha256") != config_sha:
            raise ValueError("neural inference journal has no matching start event")
        completed = {event["chunk_key"]: event for event in events if event.get("event") == "chunk_complete"}
        if score_path.exists() != coverage_path.exists():
            raise ValueError("only one neural output array exists; refusing unsafe resume")
        if not score_path.exists() and completed:
            raise FileNotFoundError("neural output arrays are missing although chunks were journaled")
        if not score_path.exists():
            score_map = np.lib.format.open_memmap(score_path, mode="w+", dtype=np.float32, shape=before.shape)
            coverage = np.lib.format.open_memmap(coverage_path, mode="w+", dtype=np.uint8, shape=before.shape)
            score_map[:] = np.nan
            coverage[:] = 0
            score_map.flush(); coverage.flush()
        else:
            score_map = np.load(score_path, mmap_mode="r+", allow_pickle=False)
            coverage = np.load(coverage_path, mmap_mode="r+", allow_pickle=False)
            if score_map.shape != before.shape or coverage.shape != before.shape:
                raise ValueError("neural output shape mismatch")
        total_chunks = int(np.prod([(size + width - 1) // width
                                    for size, width in zip(before.shape, core_shape)], dtype=np.int64))
        done = 0
        exclusion_counts = {"missing_trace_in_context": 0, "nonfinite_pair_in_context": 0}
        device = torch.device(verified.device)
        verified.model.to(device).eval()
        for chunk in iter_chunks(before.shape, core_shape):
            key = ",".join(str(part.start) for part in chunk.core)
            core = chunk.core
            starts = tuple(part.start - halo[axis] for axis, part in enumerate(core))
            base_patch, clipped = _extract_padded(before, starts, patch)
            monitor_patch, _ = _extract_padded(after, starts, patch)
            mask_patch, _ = _extract_padded(missing[..., None], starts[:2] + (0,), (patch[0], patch[1], 1))
            input_hash = _hash_bytes(base_patch, monitor_patch, mask_patch)
            previous = completed.get(key)
            if previous:
                if previous.get("input_sha256") != input_hash:
                    raise ValueError(f"neural inference input changed for output core {key}")
                if (_hash_bytes(np.asarray(score_map[core])) != previous.get("score_sha256")
                        or _hash_bytes(np.asarray(coverage[core])) != previous.get("coverage_sha256")):
                    raise ValueError(f"neural inference output changed for core {key}")
                continue
            if max_chunks is not None and done >= max_chunks:
                break
            has_missing = bool(np.asarray(mask_patch).any())
            has_nonfinite = not np.isfinite(base_patch).all() or not np.isfinite(monitor_patch).all()
            invalid = has_missing or has_nonfinite
            exclusion_reason = "missing_trace_in_context" if has_missing else (
                "nonfinite_pair_in_context" if has_nonfinite else None)
            core_shape_actual = tuple(part.stop-part.start for part in core)
            if invalid:
                values = np.full(core_shape_actual, np.nan, dtype=np.float32)
                valid_output = np.zeros(core_shape_actual, dtype=np.uint8)
                exclusion_counts[exclusion_reason] += 1
            else:
                base_tensor = torch.from_numpy(base_patch.copy())[None, None].to(device)
                monitor_tensor = torch.from_numpy(monitor_patch.copy())[None, None].to(device)
                with torch.inference_mode():
                    if verified.model_kind == "siamese":
                        full_score = verified.model(base_tensor, monitor_tensor)[0, 0]
                    else:
                        base_error = (verified.model(base_tensor) - base_tensor).square().mean(dim=1)[0]
                        monitor_error = (verified.model(monitor_tensor) - monitor_tensor).square().mean(dim=1)[0]
                        full_score = torch.abs(monitor_error - base_error)
                crop = tuple(slice(halo[axis], halo[axis] + core_shape_actual[axis]) for axis in range(3))
                values = full_score[crop].detach().float().cpu().numpy()
                values = np.asarray(values, dtype=np.float32)
                valid_output = np.isfinite(values).astype(np.uint8)
                values[valid_output == 0] = np.nan
            score_map[core] = values
            coverage[core] = valid_output
            score_map.flush(); coverage.flush()
            event = {"event": "chunk_complete", "chunk_key": key, "input_sha256": input_hash,
                     "score_sha256": _hash_bytes(values), "coverage_sha256": _hash_bytes(valid_output),
                     "valid_voxels": int(valid_output.sum()), "excluded_patch": bool(invalid),
                     "exclusion_reason": exclusion_reason}
            _append(journal_path, event)
            completed[key] = event
            done += 1
        if len(completed) != total_chunks:
            manifest.update(status="incomplete", completed_chunks=len(completed), total_chunks=total_chunks)
            _atomic_json(manifest_path, manifest)
            return {**manifest, "run_directory": str(run_dir)}
        for name, path in (("score", score_path), ("coverage", coverage_path)):
            if not path.is_file():
                raise FileNotFoundError(f"missing neural inference output: {name}")
        body = {"schema": config["schema"], "configuration_sha256": config_sha,
                "status": "complete", "total_cores": total_chunks, "completed_cores": len(completed),
                "valid_output_voxels": int(coverage.sum(dtype=np.uint64)),
                "excluded_cores": sum(bool(event.get("excluded_patch")) for event in completed.values()),
                "exclusion_reasons": {reason: sum(event.get("exclusion_reason") == reason
                                                    for event in completed.values())
                                      for reason in exclusion_counts},
                "score_meaning": config["score_meaning"], "score_is_probability": False,
                "checkpoint_sha256": verified.checkpoint_sha256,
                "dataset_provenance": verified.dataset_provenance,
                "outputs": {"score": {"file": score_path.name, "sha256": _hash_file(score_path),
                                       "size_bytes": score_path.stat().st_size},
                            "coverage": {"file": coverage_path.name, "sha256": _hash_file(coverage_path),
                                         "size_bytes": coverage_path.stat().st_size}},
                "limitations": ["Only patch cores whose full input context is finite and has no missing trace are scored.",
                                "Context overlap reduces artificial seams; scores remain model-specific and uncalibrated.",
                                "Synthetic training does not establish geological validity."]}
        if complete_path.exists():
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
            if complete.get("sha256") != hashlib.sha256(_canonical(complete.get("body", {}))).hexdigest():
                raise ValueError("neural completion manifest checksum mismatch")
            previous = complete["body"]
            if previous.get("configuration_sha256") != config_sha:
                raise ValueError("neural completion manifest belongs to another configuration")
            for item in previous["outputs"].values():
                if _hash_file(run_dir / item["file"]) != item["sha256"]:
                    raise ValueError("completed neural output checksum mismatch")
            return {**previous, "run_directory": str(run_dir), "cache_hit": True}
        _atomic_json(complete_path, {"body": body, "sha256": hashlib.sha256(_canonical(body)).hexdigest()})
        manifest.update(status="complete", completed_chunks=total_chunks, total_chunks=total_chunks)
        _atomic_json(manifest_path, manifest)
        return {**body, "run_directory": str(run_dir), "cache_hit": False}
