from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from reservoir_time_machine.ai.torch_models import build_siamese_3d
from reservoir_time_machine.hpc.patch_data import SyntheticPairedPatchDataset
from reservoir_time_machine.hpc.patch_data import _centers
from reservoir_time_machine.hpc.patch_inference import load_verified_trained_model, run_overlap_volume_inference
from reservoir_time_machine.hpc.training import _save_checkpoint


class PatchInferenceTests(unittest.TestCase):
    def _trained_checkpoint(self, root):
        provenance = {"dataset_id": "controlled-synthetic-test", "data_classification": "synthetic only"}
        model = build_siamese_3d(in_channels=1, channels=(2, 4))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        rng = np.random.default_rng(41)
        base = rng.normal(size=(240, 16, 16)).astype(np.float32)
        starts = _centers(base, np.zeros((240, 16), bool), (8, 8, 8), "train", 2, 4)
        # Controlled synthetic patches exercise one actual optimization step.
        batch = next(iter(torch.utils.data.DataLoader(
            SyntheticPairedPatchDataset(base, starts, (8, 8, 8), seed=5), batch_size=2)))
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch["baseline"], batch["monitor"])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch["target"])
        loss.backward(); optimizer.step()
        model_dir = root / "trained-test"
        model_dir.mkdir(mode=0o700)
        checkpoint_path = model_dir / "best.pt"
        _save_checkpoint(torch, checkpoint_path, {"model_kind": "siamese", "epoch": 1,
            "model_configuration": {"in_channels": 1, "channels": [2, 4]},
            "model_state": model.state_dict(), "dataset_provenance": provenance})
        digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        (model_dir / "experiment_manifest.json").write_text(json.dumps({"dataset_provenance": provenance}))
        completion = model_dir / "training_complete.json"
        completion.write_text(json.dumps({"status": "complete", "epochs": 1,
                                          "best_checkpoint_sha256": digest,
                                          "dataset_provenance": provenance}))
        return checkpoint_path, completion

    def test_trained_checkpoint_validation_overlap_core_resume_and_coverage(self):
        rng = np.random.default_rng(9)
        baseline = rng.normal(size=(10, 10, 10)).astype(np.float32)
        monitor = baseline.copy()
        monitor[4:7, 4:7, 4:7] += 0.5
        missing = np.zeros((10, 10), dtype=bool)
        missing[4, 4] = True
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint, completion = self._trained_checkpoint(root)
            verified = load_verified_trained_model(checkpoint, completion, root, device="cpu")
            result_root = root / "inference"
            partial = run_overlap_volume_inference(verified, baseline, monitor, missing, result_root,
                source_identity=hashlib.sha256(b"paired-synthetic").hexdigest(),
                patch_shape=(8, 8, 8), context_overlap=(2, 2, 2), max_chunks=1)
            self.assertEqual(partial["status"], "incomplete")
            result = run_overlap_volume_inference(verified, baseline, monitor, missing, result_root,
                source_identity=hashlib.sha256(b"paired-synthetic").hexdigest(),
                patch_shape=(8, 8, 8), context_overlap=(2, 2, 2))
            self.assertEqual(result["status"], "complete")
            score = np.load(Path(result["run_directory"]) / "score.npy", mmap_mode="r")
            coverage = np.load(Path(result["run_directory"]) / "coverage.npy", mmap_mode="r")
            self.assertEqual(score.shape, baseline.shape)
            self.assertTrue(np.isnan(score[coverage == 0]).all())
            self.assertTrue(np.isfinite(score[coverage == 1]).all())
            cached = run_overlap_volume_inference(verified, baseline, monitor, missing, result_root,
                source_identity=hashlib.sha256(b"paired-synthetic").hexdigest(),
                patch_shape=(8, 8, 8), context_overlap=(2, 2, 2))
            self.assertTrue(cached["cache_hit"])

    def test_checkpoint_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint, completion = self._trained_checkpoint(root)
            completion.write_text(json.dumps({"status": "complete", "epochs": 1,
                "best_checkpoint_sha256": "0" * 64,
                "dataset_provenance": {"dataset_id": "controlled-synthetic-test", "data_classification": "synthetic only"}}))
            with self.assertRaisesRegex(ValueError, "hash differs"):
                load_verified_trained_model(checkpoint, completion, root, device="cpu")


if __name__ == "__main__":
    unittest.main()
