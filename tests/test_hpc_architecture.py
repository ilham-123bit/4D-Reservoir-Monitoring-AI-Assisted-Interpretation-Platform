"""HPC workflow contract tests; no model training or competition data are used."""

import importlib.util
import json
from pathlib import Path
import tomllib

import pytest

from reservoir_time_machine.ai.siamese import Siamese3DConfig, build_torch_siamese_3d
from reservoir_time_machine.ai.torch_models import ConvAutoencoder3DConfig, build_conv_autoencoder_3d
from reservoir_time_machine.hpc.environment import collect_environment


ROOT = Path(__file__).resolve().parents[1]


def test_hpc_configs_are_templates_and_do_not_embed_actual_dataset_paths():
    configs = [ROOT / "config/hpc/siamese.training.example.json",
               ROOT / "config/hpc/autoencoder.training.example.json",
               ROOT / "config/hpc/inference.example.json"]
    for path in configs:
        contents = path.read_text(encoding="utf-8")
        parsed = json.loads(contents)
        assert parsed
        assert "/data/" not in contents
        assert "Challenge3_InputData" not in contents
    siamese = json.loads(configs[0].read_text())
    assert siamese["training"]["label_source"] is None
    assert siamese["data"]["adapter"] == "team_adapter.module:function"
    autoencoder = json.loads(configs[1].read_text())
    assert autoencoder["training"]["reference_population_id"] is None


def test_model_configs_validate_without_importing_torch():
    assert Siamese3DConfig().channels == (8, 16, 32)
    assert ConvAutoencoder3DConfig().channels == (8, 16, 32, 64)
    with pytest.raises(ValueError, match="channel"):
        ConvAutoencoder3DConfig(channels=())
    if importlib.util.find_spec("torch") is None:
        with pytest.raises(RuntimeError, match="PyTorch is required"):
            build_torch_siamese_3d(Siamese3DConfig())
        with pytest.raises(RuntimeError, match="PyTorch is required"):
            build_conv_autoencoder_3d()
    else:
        import torch
        pair_model = build_torch_siamese_3d(Siamese3DConfig(channels=(4, 8))).eval()
        ae_model = build_conv_autoencoder_3d(ConvAutoencoder3DConfig(channels=(4, 8))).eval()
        values = torch.ones((1, 1, 16, 16, 16))
        with torch.inference_mode():
            assert pair_model(values, values).shape == values.shape[:1] + (1,) + values.shape[2:]
            assert ae_model(values).shape == values.shape


def test_environment_probe_reports_runtime_without_assuming_an_h200():
    report = collect_environment()
    assert report["purpose"] == "read_only_hpc_environment_validation"
    assert report["python"]["version"]
    assert "pytorch" in report and "tensorflow" in report
    assert "nvidia_smi" in report
    assert "inside_slurm_job" in report["slurm_context"]
    assert "visible_devices" in report.get("pytorch", {}) or "import_or_probe_error" in report.get("pytorch", {})
    # Device identity comes only from live runtime probes, never a hard-coded H200 label.
    serialized = json.dumps(report).casefold()
    if report.get("pytorch", {}).get("available") and not report.get("pytorch", {}).get("cuda_available"):
        assert "visible_device_count" in report["pytorch"]


def test_hpc_entry_points_and_optional_framework_dependency_strategy_are_declared():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = project["project"]["scripts"]
    assert scripts["rtm-hpc-environment"] == "reservoir_time_machine.hpc.environment:main"
    assert scripts["rtm-hpc-train"] == "reservoir_time_machine.hpc.training:main"
    assert scripts["rtm-hpc-infer"] == "reservoir_time_machine.hpc.inference:main"
    assert "torch" not in project["project"]["dependencies"]
    assert "tensorflow" not in project["project"]["dependencies"]


def test_training_and_submission_scripts_are_fail_closed_templates():
    submit = (ROOT / "scripts/hpc/submit_training.sh").read_text(encoding="utf-8")
    job = (ROOT / "scripts/hpc/train_job.sh").read_text(encoding="utf-8")
    assert "RTM_SLURM_GPU_REQUEST" in submit
    assert "squeue" in submit and "flock" in submit
    assert "RTM_GPU_KERNEL_ACTIVE" in submit
    assert "--strict" in job
    assert "reservoir_time_machine.hpc.training" in job
    assert "gpu:h200" not in submit.casefold()
