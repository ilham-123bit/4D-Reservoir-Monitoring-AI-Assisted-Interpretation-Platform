from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from reservoir_time_machine.hpc.patch_data import (
    SyntheticPairedPatchDataset,
    SyntheticReconstructionPatchDataset,
    _centers,
)
from reservoir_time_machine.hpc.training import _batch_loss


class NeuralPatchAdapterTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(81)
        self.volume = rng.normal(size=(240, 40, 40)).astype(np.float32)
        self.missing = np.zeros((240, 40), dtype=bool)

    def test_spatial_train_validation_test_centers_are_disjoint_and_valid(self):
        patches = {name: _centers(self.volume, self.missing, (8, 8, 8), name, 12, 44)
                   for name in ("train", "validation", "test")}
        for first, second in (("train", "validation"), ("validation", "test"), ("train", "test")):
            a, b = patches[first], patches[second]
            self.assertTrue(a[:, 0].max() + 8 <= b[:, 0].min() or b[:, 0].max() + 8 <= a[:, 0].min())
        for starts in patches.values():
            self.assertTrue(np.isfinite(self.volume[starts[0, 0]:starts[0, 0]+8,
                                                       starts[0, 1]:starts[0, 1]+8,
                                                       starts[0, 2]:starts[0, 2]+8]).all())

    def test_siamese_dataset_outputs_only_controlled_synthetic_voxel_labels(self):
        starts = _centers(self.volume, self.missing, (8, 8, 8), "train", 32, 2026)
        dataset = SyntheticPairedPatchDataset(self.volume, starts, (8, 8, 8), seed=2026)
        results = [dataset[i] for i in range(len(dataset))]
        self.assertTrue(any(bool(item["synthetic_change"]) for item in results))
        self.assertTrue(any(not bool(item["synthetic_change"]) for item in results))
        for item in results:
            self.assertEqual(tuple(item["baseline"].shape), (1, 8, 8, 8))
            self.assertEqual(tuple(item["target"].shape), (1, 8, 8, 8))
            self.assertTrue(torch.isfinite(item["baseline"]).all())
            if not bool(item["synthetic_change"]):
                self.assertEqual(float(item["target"].sum()), 0.0)

    def test_autoencoder_loss_uses_clean_target_not_corrupted_input(self):
        source = torch.tensor([[[[[2.0, 4.0]]]]])
        target = torch.tensor([[[[[1.0, 1.0]]]]])
        model = torch.nn.Identity()
        loss = _batch_loss(model, {"input": source, "target": target}, "autoencoder", torch, "cpu")
        self.assertEqual(float(loss), 5.0)

    def test_deliberately_trained_siamese_fixture_checkpoint_round_trip(self):
        from reservoir_time_machine.ai.torch_models import build_siamese_3d
        from reservoir_time_machine.hpc.training import _save_checkpoint
        model = build_siamese_3d(in_channels=1, channels=(2, 4))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        batch = next(iter(torch.utils.data.DataLoader(
            SyntheticPairedPatchDataset(self.volume,
                _centers(self.volume, self.missing, (8, 8, 8), "train", 4, 23),
                (8, 8, 8), seed=23), batch_size=2)))
        optimizer.zero_grad(set_to_none=True)
        loss = _batch_loss(model, batch, "siamese", torch, "cpu")
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            before = model(batch["baseline"], batch["monitor"]).clone()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "trained-test-only.pt"
            _save_checkpoint(torch, path, {"model_state": model.state_dict(), "epoch": 1,
                                           "training_scope": "synthetic fixture only"})
            restored = build_siamese_3d(in_channels=1, channels=(2, 4))
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            restored.load_state_dict(checkpoint["model_state"])
            restored.eval()
            with torch.inference_mode():
                after = restored(batch["baseline"], batch["monitor"])
        torch.testing.assert_close(before, after)
        self.assertGreater(int(checkpoint["epoch"]), 0)

    def test_cache_adapter_reads_only_existing_mmaps_and_returns_test_split(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            np.save(root / "baseline.npy", self.volume)
            np.save(root / "missing_trace_mask.npy", self.missing)
            old = os.environ.get("RTM_CACHE_DIR")
            os.environ["RTM_CACHE_DIR"] = str(root)
            try:
                from reservoir_time_machine.hpc.patch_data import build_synthetic_training_bundle
                bundle = build_synthetic_training_bundle(
                    {"patch_shape": [8, 8, 8], "samples_per_split": 4, "batch_size": 2, "seed": 4},
                    challenge_root=root)
                self.assertIn("test_loader", bundle)
                self.assertEqual(bundle["provenance"]["data_classification"].split(";")[0],
                                 "synthetic labels on real seismic texture")
                batch = next(iter(bundle["validation_loader"]))
                self.assertEqual(tuple(batch["baseline"].shape), (2, 1, 8, 8, 8))
            finally:
                if old is None:
                    os.environ.pop("RTM_CACHE_DIR", None)
                else:
                    os.environ["RTM_CACHE_DIR"] = old


if __name__ == "__main__":
    unittest.main()
