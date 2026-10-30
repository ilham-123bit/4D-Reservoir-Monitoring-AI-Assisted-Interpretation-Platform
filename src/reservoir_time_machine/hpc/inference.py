"""Tiled inference runner for an HPC-private adapter and saved Torch checkpoint."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np

from ..ai.torch_models import ConvAutoencoder3DConfig, build_conv_autoencoder_3d, build_siamese_3d
from ..processing.provenance import sha256_file


def _adapter(reference: str):
    module_name, separator, function_name = reference.partition(":")
    if not separator:
        raise ValueError("inference adapter must be python.module:function")
    return getattr(importlib.import_module(module_name), function_name)


def _safe_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", value):
        raise ValueError("tile and run identifiers must be safe file-name tokens")
    return value


def _under(root: Path, item: str | Path) -> Path:
    candidate = Path(item).expanduser()
    path = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if path != root and root not in path.parents:
        raise ValueError("checkpoint and inference outputs must remain under RTM_HPC_OUTPUT_ROOT")
    return path


def infer(config_path: str | Path) -> Path:
    if sys.platform != "linux" or sys.version_info[:2] != (3, 12) or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("inference requires Python 3.12 inside the authorized Linux Slurm GPU allocation")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is unavailable in the selected HPC environment") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device is visible; refusing CPU fallback")
    raw_root = os.environ.get("RTM_HPC_OUTPUT_ROOT")
    raw_data = os.environ.get("RTM_CHALLENGE3_ROOT")
    if not raw_root or not raw_data:
        raise RuntimeError("set authorized RTM_HPC_OUTPUT_ROOT and RTM_CHALLENGE3_ROOT")
    output_root, data_root = Path(raw_root).expanduser().resolve(), Path(raw_data).expanduser().resolve()
    if not output_root.is_dir() or not data_root.is_dir():
        raise RuntimeError("configured authorized HPC input/output directories must exist")
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    checkpoint_path = _under(output_root, config["checkpoint"])
    if not checkpoint_path.is_file():
        raise FileNotFoundError("checkpoint is missing under authorized output root")
    checkpoint = torch.load(checkpoint_path, map_location="cuda:0", weights_only=False)
    kind = checkpoint.get("model_kind")
    model_config = checkpoint.get("model_configuration", {})
    if kind == "siamese":
        model = build_siamese_3d(in_channels=int(model_config.get("in_channels", 1)),
                                 channels=tuple(model_config.get("channels", [8, 16, 32])))
    elif kind == "autoencoder":
        model = build_conv_autoencoder_3d(ConvAutoencoder3DConfig(
            in_channels=int(model_config.get("in_channels", 1)),
            channels=tuple(model_config.get("channels", [8, 16, 32, 64]))))
    else:
        raise ValueError("checkpoint contains an unsupported model_kind")
    model.load_state_dict(checkpoint["model_state"])
    model.to("cuda:0").eval()
    adapter_ref = config.get("data", {}).get("adapter")
    if not adapter_ref or adapter_ref == "team_adapter.module:function":
        raise ValueError("configure a trusted HPC-private inference tile adapter")
    tile_iterator = _adapter(adapter_ref)(config.get("data", {}), challenge_root=data_root)
    run_name = _safe_name(config.get("run_name", f"inference-epoch-{checkpoint.get('epoch', 0)}"))
    run_dir = _under(output_root, output_root / "inference" / run_name)
    run_dir.mkdir(parents=True, exist_ok=False)
    threshold = float(config.get("threshold", 0.0))
    records = []
    with torch.inference_mode():
        for batch in tile_iterator:
            tile_ids = batch.get("tile_ids")
            if tile_ids is None:
                raise ValueError("inference adapter batches must include one safe tile_id per patch")
            if kind == "siamese":
                before = torch.as_tensor(np.asarray(batch["baseline"]), dtype=torch.float32, device="cuda:0")
                after = torch.as_tensor(np.asarray(batch["monitor"]), dtype=torch.float32, device="cuda:0")
                if not torch.isfinite(before).all() or not torch.isfinite(after).all():
                    raise ValueError("inference patches must be finite; adapter must apply and report a validity policy")
                scores = model(before, after).squeeze(1)
                score_meaning = "raw Siamese change-head logit"
            else:
                source = torch.as_tensor(np.asarray(batch["input"]), dtype=torch.float32, device="cuda:0")
                valid = torch.isfinite(source)
                reconstruction = model(torch.nan_to_num(source))
                scores = (reconstruction-source.nan_to_num()).square().mean(dim=1)
                scores = scores.masked_fill(~valid.all(dim=1), float("nan"))
                score_meaning = "mean squared reconstruction residual over channels"
            score_array = scores.detach().cpu().numpy().astype(np.float32, copy=False)
            if len(tile_ids) != score_array.shape[0]:
                raise ValueError("tile_ids count must match the inference batch size")
            for tile_id, score in zip(tile_ids, score_array):
                safe_id = _safe_name(str(tile_id))
                result_path = run_dir / f"{safe_id}.npz"
                if result_path.exists():
                    raise FileExistsError(f"duplicate inference tile id: {safe_id}")
                mask = np.isfinite(score) & (score >= threshold)
                with result_path.open("xb") as stream:
                    np.savez(stream, score=score, mask=mask)
                records.append({"tile_id": safe_id, "file": result_path.name,
                                "shape": list(score.shape), "detection_count": int(mask.sum())})
    report = {"schema_version": "1.0", "model_kind": kind, "checkpoint_sha256": sha256_file(checkpoint_path),
              "checkpoint_epoch": checkpoint.get("epoch"), "threshold": threshold,
              "score_meaning": score_meaning if records else "no tiles supplied",
              "calibration_status": "uncalibrated", "score_is_probability": False,
              "tile_count": len(records), "tiles": records,
              "data_policy": "tiles read from authorized HPC inputs; derived scores written only beneath output root"}
    with (run_dir / "inference_manifest.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="HPC-private JSON inference configuration")
    args = parser.parse_args()
    print(infer(args.config))


if __name__ == "__main__":
    main()
