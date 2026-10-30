# Security and data management

The public repository must contain software, tests, and documentation only. Restricted SEG-Y, LAS, workbook, checkshot, deviation, marker files, full-resolution derivatives, and credentials must remain in authorized HPC storage. The audit manifest intentionally excludes raw sample arrays and coordinate values.

`.gitignore` excludes data/input directories, seismic and well formats, volume arrays, caches, and secret files. Before a separately authorized publication, inspect `git status`, the complete staged file list, and staged diffs. No publication is performed automatically.

Set `RTM_CHALLENGE3_ROOT` and `RTM_CACHE_DIR` explicitly. Do not copy data outside authorized HPC. The documented scratch filesystem has no automatic backups; follow competition retention rules and do not treat scratch as the sole archival copy. Never put API keys in source, docs, manifests, or notebook outputs.
