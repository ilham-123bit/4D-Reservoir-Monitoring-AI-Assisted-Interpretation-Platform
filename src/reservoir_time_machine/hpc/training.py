"""Epoch-checkpointed PyTorch training runner for an HPC-provided dataset adapter.

This runner never creates input data. It is fail-closed unless executed on Linux inside
a Slurm GPU allocation with authorized data/output roots and a team-owned adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import random
import re
import sys
import tempfile
from typing import Any

import numpy as np

from ..ai.torch_models import ConvAutoencoder3DConfig, build_conv_autoencoder_3d, build_siamese_3d
from ..hpc.environment import collect_environment
from ..processing.provenance import configuration_fingerprint, sha256_file


def _adapter(reference: str):
    module_name, separator, function_name = reference.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError("dataset adapter must be specified as python.module:function")
    return getattr(importlib.import_module(module_name), function_name)


def _safe_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", value):
        raise ValueError("experiment/run names may contain only letters, digits, dot, underscore, and dash")
    return value


def _contained_path(root: Path, value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    path = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if path != root and root not in path.parents:
        raise ValueError("checkpoint, log, and run outputs must remain under RTM_HPC_OUTPUT_ROOT")
    return path


def _sanitized_provenance(value: dict[str, Any]) -> dict[str, Any]:
    permitted = {"dataset_id", "contract_version", "adapter_version", "source_hashes",
                 "geometry_validation_id", "data_classification"}
    return {key: value[key] for key in permitted if key in value}


def _fingerprint_config(config: dict[str, Any]) -> str:
    # Resume location is operational state, not part of the model/training experiment.
    return configuration_fingerprint({key: value for key, value in config.items() if key != "resume_checkpoint"})


def _save_checkpoint(torch, path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)  # HPC Linux local/shared filesystem atomic rename
    finally:
        temporary.unlink(missing_ok=True)


def _batch_loss(model, batch: dict[str, Any], model_kind: str, torch, device):
    if model_kind == "siamese":
        required = {"baseline", "monitor", "target"}
        if not required.issubset(batch):
            raise ValueError("Siamese batches require baseline, monitor, and independently justified target labels")
        baseline = torch.as_tensor(batch["baseline"], dtype=torch.float32, device=device)
        monitor = torch.as_tensor(batch["monitor"], dtype=torch.float32, device=device)
        target = torch.as_tensor(batch["target"], dtype=torch.float32, device=device)
        if not torch.isfinite(baseline).all() or not torch.isfinite(monitor).all():
            raise ValueError("Siamese batch contains non-finite seismic; adapter must supply a declared validity policy")
        logits = model(baseline, monitor)
        if target.ndim == logits.ndim - 1:
            target = target.unsqueeze(1)
        if target.shape != logits.shape or not torch.isfinite(target).all():
            raise ValueError("Siamese target mask must be finite and match output logits")
        if ((target < 0) | (target > 1)).any():
            raise ValueError("Siamese training targets must be in [0,1]")
        loss_map = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
        valid_mask = batch.get("valid_mask")
        if valid_mask is not None:
            valid = torch.as_tensor(valid_mask, dtype=torch.bool, device=device)
            if valid.ndim == logits.ndim - 1:
                valid = valid.unsqueeze(1)
            if valid.shape != logits.shape or not valid.any():
                raise ValueError("valid_mask must match logits and include valid voxels")
            loss = loss_map[valid].mean()
        else:
            loss = loss_map.mean()
        return loss
    if "input" not in batch:
        raise ValueError("autoencoder batches require input")
    source = torch.as_tensor(batch["input"], dtype=torch.float32, device=device)
    valid = torch.isfinite(source)
    target = torch.nan_to_num(source)
    reconstruction = model(target)
    if reconstruction.shape != target.shape:
        raise ValueError("autoencoder reconstruction shape differs from input")
    squared_error = (reconstruction-target).square()
    explicit_mask = batch.get("valid_mask")
    if explicit_mask is not None:
        supplied = torch.as_tensor(explicit_mask, dtype=torch.bool, device=device)
        if supplied.shape != valid.shape:
            raise ValueError("autoencoder valid_mask must match input shape")
        valid &= supplied
    if not valid.any():
        raise ValueError("autoencoder batch has no valid input samples")
    return squared_error[valid].mean()


def train(config_path: str | Path) -> Path:
    if sys.platform != "linux" or sys.version_info[:2] != (3, 12) or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("training requires Python 3.12 inside the authorized Linux Slurm compute allocation")
    try:
        import torch
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise RuntimeError("HPC PyTorch and TensorBoard packages must be available in the supplied environment") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device is visible to this Slurm allocation; refusing CPU fallback")
    raw_data_root, raw_output_root = os.environ.get("RTM_CHALLENGE3_ROOT"), os.environ.get("RTM_HPC_OUTPUT_ROOT")
    if not raw_data_root or not raw_output_root:
        raise RuntimeError("set RTM_CHALLENGE3_ROOT and RTM_HPC_OUTPUT_ROOT to authorized HPC directories")
    data_root, output_root = Path(raw_data_root).expanduser().resolve(), Path(raw_output_root).expanduser().resolve()
    if not data_root.is_dir() or not output_root.is_dir():
        raise RuntimeError("set RTM_CHALLENGE3_ROOT and RTM_HPC_OUTPUT_ROOT to existing authorized HPC directories")
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if config.get("data", {}).get("adapter") in (None, "", "team_adapter.module:function"):
        raise ValueError("configure a trusted HPC-only data adapter before training")
    kind = config.get("model", {}).get("kind")
    if kind == "siamese":
        labels = config.get("training", {})
        if not labels.get("label_source") or not labels.get("independent_validation_source"):
            raise ValueError("Siamese training requires defensible training and independent validation label sources")
    elif kind == "autoencoder":
        if not config.get("training", {}).get("reference_population_id"):
            raise ValueError("autoencoder training requires a documented reference population identifier")
    else:
        raise ValueError("model.kind must be 'siamese' or 'autoencoder'")
    seed = int(config.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    deterministic = bool(config.get("deterministic", True))
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    device = torch.device("cuda:0")
    model_config = config["model"].get("parameters", {})
    if kind == "siamese":
        model = build_siamese_3d(in_channels=int(model_config.get("in_channels", 1)),
                                 channels=tuple(model_config.get("channels", [8, 16, 32])))
    else:
        model = build_conv_autoencoder_3d(ConvAutoencoder3DConfig(
            in_channels=int(model_config.get("in_channels", 1)),
            channels=tuple(model_config.get("channels", [8, 16, 32, 64]))))
    model = model.to(device)
    adapter = _adapter(config["data"]["adapter"])
    bundle = adapter(config.get("data", {}), challenge_root=data_root)
    if not isinstance(bundle, dict) or not all(key in bundle for key in ("train_loader", "validation_loader")):
        raise TypeError("dataset adapter must return train_loader and validation_loader")
    if not isinstance(bundle.get("provenance", {}), dict):
        raise TypeError("dataset adapter provenance must be a dictionary of approved manifest fields")
    train_loader, validation_loader = bundle["train_loader"], bundle["validation_loader"]
    learning_rate = float(config.get("training", {}).get("learning_rate", 1e-4))
    weight_decay = float(config.get("training", {}).get("weight_decay", 1e-5))
    epochs = int(config.get("training", {}).get("epochs", 10))
    if learning_rate <= 0 or epochs < 1:
        raise ValueError("learning_rate and epochs must be positive")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    experiment = _safe_name(config.get("experiment_name", "rtm"))
    run_name = _safe_name(config.get("run_name", "unconfigured"))
    run_dir = _contained_path(output_root, output_root / "experiments" / experiment / run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = _contained_path(output_root, run_dir / "checkpoints")
    log_dir = _contained_path(output_root, run_dir / "tensorboard")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    resume = config.get("resume_checkpoint")
    if resume:
        resume_path = _contained_path(output_root, resume)
        if not resume_path.is_file():
            raise FileNotFoundError("configured resume checkpoint does not exist under output root")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        expected_fingerprint = _fingerprint_config(config)
        if checkpoint.get("configuration_sha256") != expected_fingerprint:
            raise ValueError("resume checkpoint configuration fingerprint differs from current config")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"])
    else:
        if any(checkpoint_dir.iterdir()):
            raise FileExistsError("checkpoint directory is non-empty; configure resume_checkpoint or choose a new run_name")
        start_epoch = 0
    config_hash = _fingerprint_config(config)
    manifest_path = run_dir / "experiment_manifest.json"
    if manifest_path.exists() and not resume:
        raise FileExistsError("experiment manifest already exists; choose a new run_name")
    if not manifest_path.exists():
        environment = collect_environment()
        project_root = Path(__file__).resolve().parents[3]
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=project_root,
                                  capture_output=True, text=True, check=False)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=project_root,
                               capture_output=True, text=True, check=False)
        manifest = {"schema_version": "1.0", "experiment_name": experiment, "run_name": run_name,
                    "model_kind": kind, "model_version": "1.0", "configuration": config,
                    "configuration_sha256": config_hash, "seed": seed,
                    "deterministic_algorithms_requested": deterministic,
                    "dataset_provenance": _sanitized_provenance(bundle.get("provenance", {})),
                    "runtime": {"python": platform_version(), "torch": torch.__version__,
                                "cuda_build": torch.version.cuda,
                                "visible_device_name": torch.cuda.get_device_name(0),
                                "visible_device_properties": str(torch.cuda.get_device_properties(0)),
                                "code_revision": revision.stdout.strip() if revision.returncode == 0 else "unavailable",
                                "code_checkout_dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
                                "environment_report": environment},
                    "data_path_policy": "input files are read in place; only derived checkpoints/logs are written to output root"}
        with manifest_path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True, default=str)
            stream.write("\n")
    writer = SummaryWriter(log_dir=str(log_dir))
    best_validation = float("inf")
    if resume:
        best_validation = float(checkpoint.get("best_validation_loss", best_validation))
    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            training_losses = []
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                loss = _batch_loss(model, batch, kind, torch, device)
                loss.backward()
                optimizer.step()
                training_losses.append(float(loss.detach().cpu()))
            model.eval()
            validation_losses = []
            with torch.inference_mode():
                for batch in validation_loader:
                    validation_losses.append(float(_batch_loss(model, batch, kind, torch, device).cpu()))
            if not training_losses or not validation_losses:
                raise RuntimeError("dataset adapter returned an empty train or validation loader")
            train_loss = float(np.mean(training_losses))
            validation_loss = float(np.mean(validation_losses))
            writer.add_scalar("loss/train", train_loss, epoch + 1)
            writer.add_scalar("loss/validation", validation_loss, epoch + 1)
            writer.flush()
            payload = {"model_kind": kind, "model_state": model.state_dict(),
                       "optimizer_state": optimizer.state_dict(), "epoch": epoch + 1,
                       "best_validation_loss": min(best_validation, validation_loss),
                       "train_loss": train_loss, "validation_loss": validation_loss,
                       "configuration_sha256": config_hash, "model_configuration": model_config, "seed": seed,
                       "torch_version": torch.__version__, "cuda_build": torch.version.cuda}
            _save_checkpoint(torch, checkpoint_dir / "last.pt", payload)
            if validation_loss < best_validation:
                best_validation = validation_loss
                payload["best_validation_loss"] = best_validation
                _save_checkpoint(torch, checkpoint_dir / "best.pt", payload)
        completion = {"status": "complete", "epochs": epochs, "best_validation_loss": best_validation,
                      "last_checkpoint_sha256": sha256_file(checkpoint_dir / "last.pt"),
                      "best_checkpoint_sha256": sha256_file(checkpoint_dir / "best.pt"),
                      "score_status": "model output is not a calibrated probability without separate validation"}
        with (run_dir / "training_complete.json").open("x", encoding="utf-8") as stream:
            json.dump(completion, stream, indent=2, sort_keys=True)
            stream.write("\n")
    finally:
        writer.close()
    return run_dir


def platform_version() -> str:
    import platform
    return platform.python_version()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="HPC-private JSON experiment configuration")
    args = parser.parse_args()
    print(train(args.config))


if __name__ == "__main__":
    main()
