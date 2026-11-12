"""Spatially separated paired 3D patch adapter for controlled synthetic training.

The labels describe injected synthetic perturbations only. They are not Angsi
reservoir-change labels and cannot validate geological interpretations.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Sequence

import numpy as np


def _shape3(values: Sequence[int]) -> tuple[int, int, int]:
    shape = tuple(int(v) for v in values)
    if len(shape) != 3 or any(v < 8 for v in shape):
        raise ValueError("3D patch dimensions must each be at least 8 voxels")
    return shape


def _centers(baseline, missing, shape, split: str, count: int, seed: int):
    ni, nx, ns = baseline.shape
    pi, px, ps = shape
    if min(ni, nx, ns) < max(shape):
        raise ValueError("patch shape exceeds available seismic volume")
    # Three inline-only geographic partitions with a full patch-width guard band.
    # This prevents overlapping patches from crossing train/validation/test splits.
    b1, b2 = int(ni * 0.60), int(ni * 0.82)
    guard = pi
    bounds = {"train": (pi // 2, max(pi // 2, b1 - guard)),
              "validation": (min(ni - pi // 2, b1 + guard), max(b1 + guard, b2 - guard)),
              "test": (min(ni - pi // 2, b2 + guard), ni - pi // 2)}
    if split not in bounds:
        raise ValueError("split must be train, validation, or test")
    lo, hi = bounds[split]
    if hi <= lo or count < 1:
        raise ValueError("split has no eligible center range or count is invalid")
    rng = np.random.default_rng(seed)
    selected, seen = [], set()
    attempts = max(1000, count * 100)
    for _ in range(attempts):
        ci = int(rng.integers(lo, hi))
        cx = int(rng.integers(px // 2, nx - (px - px // 2) + 1))
        cs = int(rng.integers(ps // 2, ns - (ps - ps // 2) + 1))
        start = (ci - pi // 2, cx - px // 2, cs - ps // 2)
        key = start
        if key in seen:
            continue
        slices = tuple(slice(v, v + size) for v, size in zip(start, shape))
        if np.asarray(missing[slices[:2]]).any():
            continue
        patch = np.asarray(baseline[slices])
        if not np.isfinite(patch).all() or float(np.sqrt(np.mean(patch.astype(np.float64) ** 2))) <= 1e-8:
            continue
        seen.add(key)
        selected.append(start)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"only found {len(selected)} valid patches for {split}; requested {count}")
    return np.asarray(selected, dtype=np.int32)


class SyntheticPairedPatchDataset:
    """Lazy controlled pair generation from baseline seismic patches."""

    def __init__(self, baseline, starts, patch_shape, *, seed: int, change_fraction: float = 0.5):
        self.baseline = baseline
        self.starts = np.asarray(starts, dtype=np.int32)
        self.patch_shape = _shape3(patch_shape)
        self.seed = int(seed)
        if not 0 <= change_fraction <= 1:
            raise ValueError("change_fraction must be between 0 and 1")
        self.change_fraction = float(change_fraction)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        import torch
        start = self.starts[index]
        slices = tuple(slice(int(v), int(v) + size) for v, size in zip(start, self.patch_shape))
        base = np.asarray(self.baseline[slices], dtype=np.float32).copy()
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(index)]))
        scale = float(np.sqrt(np.mean(base.astype(np.float64) ** 2)))
        if scale <= 1e-8 or not np.isfinite(scale):
            raise ValueError("selected patch has invalid RMS")
        base /= scale
        monitor = base.copy()
        target = np.zeros(self.patch_shape, dtype=np.float32)
        changed = bool(rng.random() < self.change_fraction)
        if changed:
            axes = [np.arange(size, dtype=np.float32) for size in self.patch_shape]
            center = [float(rng.uniform(size * 0.3, size * 0.7)) for size in self.patch_shape]
            radii = [float(rng.uniform(max(1.5, size * 0.08), max(2.0, size * 0.22))) for size in self.patch_shape]
            grids = np.meshgrid(*axes, indexing="ij")
            radius2 = sum(((grid - c) / r) ** 2 for grid, c, r in zip(grids, center, radii))
            target = (radius2 <= 1.0).astype(np.float32)
            amp = float(rng.uniform(0.15, 0.55))
            carrier = np.sin(np.arange(self.patch_shape[2], dtype=np.float32) * rng.uniform(0.4, 1.2))
            monitor += (amp * target * carrier[None, None, :]).astype(np.float32)
        else:
            # Acquisition-like nuisance controls must not be labeled as localized
            # reservoir changes. These are intentionally simple synthetic cases.
            if rng.random() < 0.5:
                monitor *= np.float32(rng.uniform(0.98, 1.02))
            if rng.random() < 0.25:
                monitor = np.roll(monitor, int(rng.choice([-1, 1])), axis=2)
        return {"baseline": torch.from_numpy(base[None]), "monitor": torch.from_numpy(monitor[None]),
                "target": torch.from_numpy(target[None]),
                "valid_mask": torch.ones((1, *self.patch_shape), dtype=torch.bool),
                "source_index": torch.from_numpy(np.asarray(start, dtype=np.int64)),
                "synthetic_change": torch.tensor(changed)}


class SyntheticReconstructionPatchDataset:
    """Baseline-reference reconstruction pairs with controlled input corruption."""

    def __init__(self, baseline, starts, patch_shape, *, seed: int):
        self.baseline, self.starts = baseline, np.asarray(starts, dtype=np.int32)
        self.patch_shape, self.seed = _shape3(patch_shape), int(seed)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        import torch
        start = self.starts[index]
        slices = tuple(slice(int(v), int(v) + size) for v, size in zip(start, self.patch_shape))
        target = np.asarray(self.baseline[slices], dtype=np.float32).copy()
        rms = float(np.sqrt(np.mean(target.astype(np.float64) ** 2)))
        if rms <= 1e-8 or not np.isfinite(rms):
            raise ValueError("selected reference patch has invalid RMS")
        target /= rms
        source = target.copy()
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(index)]))
        source += rng.normal(0.0, float(rng.uniform(0.005, 0.03)), source.shape).astype(np.float32)
        if rng.random() < 0.5:
            # Random contiguous dropout is an input corruption only; the target
            # remains the uncorrupted baseline patch.
            i0 = int(rng.integers(0, self.patch_shape[0] // 2))
            source[i0:i0 + max(1, self.patch_shape[0] // 8)] = 0
        return {"input": torch.from_numpy(source[None]), "target": torch.from_numpy(target[None]),
                "valid_mask": torch.ones((1, *self.patch_shape), dtype=torch.bool),
                "source_index": torch.from_numpy(np.asarray(start, dtype=np.int64))}


def build_synthetic_training_bundle(data_config: dict, *, challenge_root: Path) -> dict:
    """Training adapter for the existing Slurm runner; outputs remain synthetic-only."""
    import torch
    from torch.utils.data import DataLoader
    cache_text = os.environ.get("RTM_CACHE_DIR")
    if not cache_text:
        raise RuntimeError("RTM_CACHE_DIR must identify the existing authorized paired cache")
    cache = Path(cache_text).expanduser().resolve()
    if not cache.is_dir():
        raise FileNotFoundError("configured paired cache directory is unavailable")
    baseline = np.load(cache / "baseline.npy", mmap_mode="r", allow_pickle=False)
    missing = np.load(cache / "missing_trace_mask.npy", mmap_mode="r", allow_pickle=False)
    shape = _shape3(data_config.get("patch_shape", (16, 16, 32)))
    count = int(data_config.get("samples_per_split", 256))
    batch = int(data_config.get("batch_size", 4))
    seed = int(data_config.get("seed", 2026))
    if count < 1 or batch < 1:
        raise ValueError("samples_per_split and batch_size must be positive")
    datasets = {}
    starts_by_split = {}
    for offset, split in enumerate(("train", "validation", "test")):
        starts = _centers(baseline, missing, shape, split, count, seed + offset * 1009)
        starts_by_split[split] = starts
        datasets[split] = SyntheticPairedPatchDataset(baseline, starts, shape, seed=seed + offset * 1009)
    digest = hashlib.sha256(b"".join(starts_by_split[name].tobytes() for name in starts_by_split)).hexdigest()
    generator = torch.Generator().manual_seed(seed)
    loaders = {name: DataLoader(dataset, batch_size=batch, shuffle=(name == "train"),
                                num_workers=0, generator=generator if name == "train" else None)
               for name, dataset in datasets.items()}
    return {"train_loader": loaders["train"], "validation_loader": loaders["validation"],
            "test_loader": loaders["test"],
            "provenance": {"dataset_id": "angsi-baseline-substrate-synthetic-pairs",
                           "contract_version": "synthetic-paired-patches-1",
                           "adapter_version": "spatial-split-1", "source_hashes": {},
                           "data_classification": "synthetic labels on real seismic texture; not geological ground truth",
                           "training_objective": "voxelwise supervised BCE on injected synthetic localized amplitude bodies",
                           "label_semantics": "1=injected synthetic perturbation; 0=unchanged or configured nuisance control",
                           "split_policy": "disjoint inline bands with one-patch-width guard bands",
                           "patch_shape": list(shape), "samples_per_split": count,
                           "split_starts_sha256": digest,
                           "challenge_root_used_for_data": False}}


def build_synthetic_autoencoder_bundle(data_config: dict, *, challenge_root: Path) -> dict:
    """Reference-only denoising autoencoder adapter; not a geological label source."""
    import torch
    from torch.utils.data import DataLoader
    cache_text = os.environ.get("RTM_CACHE_DIR")
    if not cache_text:
        raise RuntimeError("RTM_CACHE_DIR must identify the existing authorized paired cache")
    cache = Path(cache_text).expanduser().resolve()
    baseline = np.load(cache / "baseline.npy", mmap_mode="r", allow_pickle=False)
    missing = np.load(cache / "missing_trace_mask.npy", mmap_mode="r", allow_pickle=False)
    shape = _shape3(data_config.get("patch_shape", (16, 16, 32)))
    count, batch, seed = int(data_config.get("samples_per_split", 256)), int(data_config.get("batch_size", 4)), int(data_config.get("seed", 2026))
    if count < 1 or batch < 1:
        raise ValueError("samples_per_split and batch_size must be positive")
    datasets, starts_by_split = {}, {}
    for offset, split in enumerate(("train", "validation", "test")):
        starts = _centers(baseline, missing, shape, split, count, seed + offset * 1009)
        starts_by_split[split] = starts
        datasets[split] = SyntheticReconstructionPatchDataset(baseline, starts, shape, seed=seed + offset * 1009)
    digest = hashlib.sha256(b"".join(starts_by_split[name].tobytes() for name in starts_by_split)).hexdigest()
    generator = torch.Generator().manual_seed(seed)
    loaders = {name: DataLoader(dataset, batch_size=batch, shuffle=(name == "train"), num_workers=0,
                                generator=generator if name == "train" else None)
               for name, dataset in datasets.items()}
    return {"train_loader": loaders["train"], "validation_loader": loaders["validation"],
            "test_loader": loaders["test"],
            "provenance": {"dataset_id": "angsi-baseline-reference-synthetic-corruptions",
                           "contract_version": "synthetic-reconstruction-patches-1",
                           "adapter_version": "spatial-split-1", "source_hashes": {},
                           "data_classification": "real baseline texture with synthetic input corruption; reference population not geologically verified",
                           "training_objective": "self-supervised denoising reconstruction of baseline patches",
                           "label_semantics": "target is uncorrupted baseline reference patch; not a change label",
                           "split_policy": "disjoint inline bands with one-patch-width guard bands",
                           "patch_shape": list(shape), "samples_per_split": count,
                           "split_starts_sha256": digest,
                           "challenge_root_used_for_data": False}}
