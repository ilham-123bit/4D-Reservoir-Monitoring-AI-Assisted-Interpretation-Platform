import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/hpc/run_full_field_ml.py"
SPEC = importlib.util.spec_from_file_location("full_field_ml_entrypoint", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class _CalibrationReached(RuntimeError):
    pass


class FullFieldEntrypointCalibrationTests(unittest.TestCase):
    def test_publication_validates_streamed_candidate_catalogues(self):
        from reservoir_time_machine.analysis.full_field_findings import _write_streamed_envelope
        from reservoir_time_machine.processing.full_field import _atomic_json

        source_identity = hashlib.sha256(b"streamed-catalogue-source").hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staging = root / "staging"
            staging.mkdir()
            candidate_body = {"schema": "rtm-full-field-candidate-catalogue-1",
                "status": "complete_for_supplied_full_volume", "source_identity": source_identity,
                "candidate_count": 1, "candidates": None}
            qc_body = {"schema": "rtm-full-field-qc-catalogue-1",
                "status": "complete_for_supplied_full_volume", "source_identity": source_identity,
                "quarantined_count": 0, "items": None}
            candidate_sha = _write_streamed_envelope(staging / "candidate_catalogue.json",
                candidate_body, "candidates", iter([{"id": "candidate-1"}]))
            qc_sha = _write_streamed_envelope(staging / "qc_catalogue.json", qc_body, "items", iter([]))
            _atomic_json(staging / "findings.json", {"status": "complete_for_supplied_full_volume",
                "source_identity": source_identity, "additional_candidate_catalogue": {
                    "file": "candidate_catalogue.json", "sha256": candidate_sha},
                "quarantined_qc_catalogue": {"file": "qc_catalogue.json", "sha256": qc_sha}})
            package = MODULE._publish_findings_package(staging, root / "published", source_identity)
            self.assertEqual(package["source_identity"], source_identity)
            self.assertTrue((root / "published" / "candidate_catalogue.json").is_file())

    def test_exploratory_calibration_design_reaches_entrypoint_validator(self):
        """Exercise main through the previously failing buffer-argument evaluation."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_root = root / "private-output"
            output_root.mkdir(mode=0o700)
            attributes = output_root / "attributes"
            attributes.mkdir(mode=0o700)
            analysis = output_root / "analysis"
            cache = root / "paired-cache"
            cache.mkdir(mode=0o700)
            npy_mask = cache / "missing_trace_mask.npy"
            np.save(npy_mask, np.zeros((2, 2), dtype=bool), allow_pickle=False)

            config = {
                "schema": "rtm-full-field-ml-config-1",
                "status": "UNLABELED_EXPLORATORY_SCREENING",
                "execution_scope": "operator_authorized_unlabeled_full_field_screening",
                "attribute_run_directory": "attributes",
                "analysis_output_directory": "analysis",
                "reference_bounds": [0, 1, 0, 1],
                "calibration_bounds": [1, 2, 1, 2],
                "calibration_design": {
                    "minimum_buffer_grid_cells": 32,
                    "minimum_occupied_fraction": 0.95,
                },
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            shape = [2, 2, 2]
            cache_details = {"fixture": "entrypoint-regression"}
            cache_identity = "paired-cache-fixture-identity"
            source_details = {
                "schema": "rtm-full-field-input-identity-2",
                "paired_cache_sha256": cache_identity,
                "paired_cache": cache_details,
                "source_bounds": [0, 2, 0, 2, 0, 2],
                "source_domain_shape": shape,
            }
            source_identity = hashlib.sha256(json.dumps(
                source_details, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()).hexdigest()
            body = {
                "status": "complete",
                "configuration": {
                    "shape": shape,
                    "source_bounds": source_details["source_bounds"],
                    "source_domain_shape": shape,
                    "source_identity_details": source_details,
                    "source_identity": source_identity,
                    "window_samples": 9,
                },
                "outputs": {},
            }
            (attributes / "complete.json").write_text(json.dumps({
                "body": body,
                "sha256": hashlib.sha256(MODULE._canonical(body)).hexdigest(),
            }), encoding="utf-8")

            received = {}

            def stop_at_calibration(reference, calibration, **kwargs):
                received.update(kwargs)
                raise _CalibrationReached("entrypoint reached calibration validator")

            environment = {
                "SLURM_JOB_ID": "synthetic-regression-job",
                "RTM_HPC_OUTPUT_ROOT": str(output_root),
                "RTM_PAIRED_CACHE_DIR": str(cache),
                "RTM_OPERATOR_SUBMISSION_APPROVED": "YES",
            }
            with patch.dict(os.environ, environment, clear=False), \
                    patch.object(MODULE, "paired_cache_provenance",
                                 return_value=(cache_identity, cache_details)), \
                    patch.object(MODULE, "validate_spatial_calibration_design",
                                 side_effect=stop_at_calibration):
                with self.assertRaisesRegex(_CalibrationReached, "reached calibration validator"):
                    MODULE.main(["--config", str(config_path)])

            self.assertEqual(received["buffer_grid_cells"], 32)
            self.assertEqual(received["minimum_occupied_fraction"], 0.95)
            self.assertEqual(received["review"], config["calibration_design"])

    def test_completed_model_score_run_is_reused_only_when_fit_identity_matches(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "models-fixture"
            run.mkdir()
            shape = (2, 2, 2)
            source_identity = "a" * 64
            review = {"execution_mode": "unlabeled_exploratory_screening",
                      "minimum_buffer_grid_cells": 32}
            names = ["gmm", "isolation_forest", "mahalanobis", "pca"]
            fit = {
                "source_identity": source_identity,
                "sample_bounds": [0, 2], "window_samples": 9, "random_state": 2026,
                "reference": {"bounds": [0, 1, 0, 1], "id": "ref"},
                "calibration": {"bounds": [1, 2, 1, 2], "id": "cal"},
                "spatial_design": {"review": review},
                "models": {name: {"threshold_quantile": 0.99} for name in names},
            }
            run_config = {"source_identity": source_identity, "shape": list(shape),
                "models": names, "min_support_fraction": 0.8, "window_samples": 9,
                "chunk_shape": [64, 64, 256], "fit_record": fit}
            config_sha = hashlib.sha256(MODULE._canonical(run_config)).hexdigest()
            manifest = {"configuration_sha256": config_sha, "configuration": run_config}
            outputs = {}
            for name in names:
                for kind in ("score", "flag"):
                    filename = f"{name}-{kind}.npy"
                    path = run / filename
                    np.save(path, np.zeros(shape, dtype=np.float32 if kind == "score" else np.uint8),
                            allow_pickle=False)
                    outputs[filename] = {"file": filename, "shape": list(shape),
                        "size_bytes": path.stat().st_size,
                        "sha256": MODULE._hash_file(path)}
            complete = {"schema": "rtm-full-field-model-scores-1", "status": "complete",
                "completed_chunks": 1, "total_chunks": 1,
                "configuration_sha256": config_sha, "fit_record": fit, "outputs": outputs}
            (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (run / "complete.json").write_text(json.dumps(complete), encoding="utf-8")
            config = {"reference_population_id": "ref", "calibration_population_id": "cal",
                      "random_state": 2026, "threshold_quantile": 0.99,
                      "chunk_shape": [64, 64, 256]}
            reused = MODULE._reuse_completed_model_run(root, source_identity=source_identity,
                shape=shape, reference_bounds=(0, 1, 0, 1), calibration_bounds=(1, 2, 1, 2),
                fit_sample_bounds=(0, 2), config=config, window_samples=9,
                min_support_fraction=0.8, calibration_design={"review": review})
            self.assertIsNotNone(reused)
            self.assertTrue(reused["cache_hit"])
            self.assertEqual(reused["run_directory"], str(run))


if __name__ == "__main__":
    unittest.main()
