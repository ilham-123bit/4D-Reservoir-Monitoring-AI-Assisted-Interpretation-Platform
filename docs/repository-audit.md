# Repository audit record

## Current development audit

- Active workspace: existing local Windows checkout (absolute workstation paths are not included in published documentation).
- Platform: Windows 10 local workstation, Python 3.11.7; not the competition HPC.
- Git branch: `main`; configured remote: `origin`.
- At the current zero-input architecture review: local branch is three commits behind `origin/main`; no pull or merge was performed.
- Working tree contained the prior uncommitted prototype changes; no staged changes were reported. They were preserved, and previous demo app/tests were copied into explicit legacy locations before active-path changes.
- `Challenge3_InputData/` is absent in the current workspace. A previous local copy was deleted after integrity investigation; no data were recreated.
- `sbatch` was unavailable. `nvidia-smi` reported a local GeForce MX330; this is not an HPC H200 allocation.
- NumPy, SciPy, pandas and pytest are installed. `segyio`, `lasio`, `openpyxl`, Streamlit, Plotly, OpenAI client and Ruff were not found in the inspected Python environment.
- Current task used no competition inputs. No commit, push, reset or branch deletion was performed.

## Historical data findings

A previous local read-only investigation found copied SEG-Y files incomplete. Those files and actual-data-derived audit artifacts are excluded from publication. No actual-data processing is claimed.

## Public repository security

The source repository must contain application code and tests only. `.gitignore` excludes data/input directories, seismic and well-data formats, volume arrays, caches and credentials. No actual dataset file appears in the current tracked/untracked project inventory. Review `git status` and staged files before any separately authorized publication.
