# HPC workflow

## Environment boundary and current inspection status

Windows/Codex remains the primary source-code development environment. The only environment inspected in this task is the local Windows 10 workstation: Python 3.11.7, no PyTorch/TensorFlow/TensorBoard/Slurm client, and a GeForce MX330 with 2 GiB reported memory. This is not the competition allocation. The APGCE-specific manual was not available in the repository, so resource facts here follow the user-provided manual summary: Rocky Linux, a 24-thread/128-GB login node, Python 3.12 GPU via the SLURM kernel, NVIDIA H200 compute with participant MIG allocations, one active GPU kernel per team, and at most one running/two submitted Slurm jobs. We have not inspected the live HPC allocation or its actual GPU/MIG profile, CUDA, VRAM, Slurm configuration, or storage capacity.

Do not infer a physical H200 or a fixed VRAM size from the allocation description. The environment probe reports the device actually visible to the process, which may be a MIG slice. It does not load models or reserve GPU memory.

Use the login node for small metadata/configuration tasks. Run seismic-volume calculations, training, and inference only in the allocated GPU kernel or an authorized compute job; respect the 24-thread/128-GB login-node limits and team job quota.

## Read-only environment validation

Run inside the authorized Python 3.12 GPU SLURM kernel (or inside a GPU Slurm batch job):

```bash
export RTM_CHALLENGE3_ROOT='<authorized input directory confirmed on HPC>'
export RTM_HPC_OUTPUT_ROOT='<authorized output/scratch directory confirmed on HPC>'
python scripts/validate_hpc_environment.py \
  --output "$RTM_HPC_OUTPUT_ROOT/environment-$(date +%Y%m%dT%H%M%S).json" --strict
```

The script checks Python/platform, PyTorch and TensorFlow versions/builds, CUDA visibility, per-framework device details, visible GPU memory, `nvidia-smi` when available, Slurm allocation indicators, and relevant installed library versions. It is read-only. Review the JSON before training; do not paste reports containing restricted operational metadata into public issues.

## Data and output isolation

Set `RTM_CHALLENGE3_ROOT` and `RTM_HPC_OUTPUT_ROOT` to paths verified under the competition's authorized storage. The adapters read original files in place. Checkpoints, TensorBoard events, inference tiles, manifests, and derived attributes belong under the output root. Do not copy raw seismic/well files to Windows, external storage, or public GitHub. The documented scratch filesystem has no automatic backups; confirm retention policy and keep any required authoritative artifacts in authorized storage.

The training and inference entry points fail closed unless they run on Linux/Python 3.12 inside a Slurm allocation with visible CUDA and configured input/output roots. They never fall back to CPU. The private dataset adapter is deliberately not included: it must stream authorized data into fixed-shape PyTorch batches without writing duplicate raw files.

## Training and inference

No packages are installed by the workflow. Use the competition-provided Python 3.12 GPU environment and verify `torch.utils.tensorboard` is available. Do not pip-install Torch/TensorFlow/CUDA over the supplied environment.

1. Copy an example JSON from `config/hpc/` to a private HPC-only configuration location.
2. Set the trusted adapter import path. The training adapter returns `train_loader`, `validation_loader`, and a small provenance dictionary. Siamese training requires independently justified labels and validation labels; the example intentionally has null label fields. Autoencoder training requires a documented reference-population ID.
3. Run training from the current allocated GPU kernel with `python -m reservoir_time_machine.hpc.training --config <private-config.json>`, or use the Slurm batch wrapper after inspecting actual site resources.
4. Checkpoints (`last.pt`, `best.pt`), `experiment_manifest.json`, and TensorBoard event files are written under `RTM_HPC_OUTPUT_ROOT/experiments/<experiment>/<run>/`.
5. View TensorBoard only through an authorized HPC-local port-forward/session. Never publish event logs or checkpoints.
6. Configure `config/hpc/inference.example.json` and use `python -m reservoir_time_machine.hpc.inference --config <private-config.json>`. The data adapter yields bounded patches/tiles. Score maps are uncalibrated and saved only under the output root.

Checkpoint resume is at epoch boundaries. Model, optimizer, epoch, configuration hash and validation loss are saved; exact data-loader/sampler continuity depends on the private adapter and is not guaranteed by the runner.

## Slurm safeguards

`scripts/hpc/train_job.sh` is the batch entry script. `scripts/hpc/submit_training.sh` checks the selected account's current `squeue` state, refuses a submit if one job is running or two are already submitted, and refuses when `RTM_GPU_KERNEL_ACTIVE=1`. The account, partition, GPU/MIG GRES string, CPUs, memory and walltime are required environment values; the scripts do not guess them. Confirm the team's active Jupyter kernel and account-level jobs with teammates because the local wrapper cannot independently see a kernel that Slurm does not report. Do not submit anything until the user/team authorizes the specific job.

```bash
export RTM_SLURM_ACCOUNT='<confirmed team account>'
export RTM_SLURM_PARTITION='<confirmed partition>'
export RTM_SLURM_GPU_REQUEST='<confirmed GPU or MIG GRES request>'
export RTM_SLURM_CPUS='<approved CPU count>'
export RTM_SLURM_MEMORY='<approved memory request>'
export RTM_SLURM_TIME='<approved walltime>'
export RTM_CODE_ROOT='<code directory available on HPC>'
export RTM_EXPERIMENT_CONFIG='<private config path>'
# Only after team-wide kernel/job status and authorization are confirmed:
bash scripts/hpc/submit_training.sh
```

The wrapper's accounting check is a safety aid, not a replacement for the manual or team coordination. It serializes concurrent calls using a lock file under the team-shared output root, checks account-wide Slurm states, and requires an active-kernel declaration. Enforcement depends on all teammates using the same shared lock root and account. The wrapper has not been executed here; this Windows environment has no `sbatch`/`squeue`.
