# Scalable 4D analytical foundations

## Current architecture

The analytical path is divided into input validation, exact spatial trace-key correspondence, preprocessing/registration, attributes, detection, and result provenance. Scientific functions remain outside the dashboard. The code does not require competition data at import or test time.

### Chunking and storage

`processing.chunks.iter_chunks` yields disjoint output cores with clipped read halos. `map_chunks` applies a shape-preserving NumPy/SciPy operation and writes a new `.npy` memory-mapped result; existing result files are never overwritten. The processor owns its edge rule. For example, a reflected Gaussian filter requires a halo at least as wide as its effective filter radius; internal chunk edges are cropped away, while physical volume edges use the operation's configured boundary mode. Too-small halos can produce seams and are the caller's responsibility to validate.

`attributes.chunked.calculate_attributes_chunked` computes signed difference, absolute and normalized difference, RMS and NRMS in inline/crossline tiles while retaining each sample axis intact. Its result arrays are separate `float32` NPY memmaps by default. Windowed NRMS uses the existing paired-finite-sample semantics and the same edge handling as the full-volume reference implementation. It does not alter source arrays.

`preprocessing.normalization.normalize_pair_to_memmap` writes an optional equalized monitor copy out-of-core. Global RMS normalization has a finite-sample first pass; trace RMS keeps only small inline/crossline scale maps plus a chunk buffer. Both methods can erase genuine amplitude changes and are opt-in. Raw volumes remain unchanged.

NPY memmap is the current built-in out-of-core format because it uses the existing NumPy dependency and supports local chunk access. Zarr was considered but is not a dependency and was not installed. A future storage adapter should define chunk shape, compressor, metadata, concurrent writer policy, checksums, and atomic completion markers before adding a Zarr backend. HDF5 is another candidate where the HPC deployment already provides a compatible library; benchmark actual data and shared-filesystem behavior before selecting a format.

### Provenance

`processing.provenance` provides streamed SHA-256, canonical configuration fingerprints, and exclusive JSON run manifests. Manifests record input references, parameters, outputs, software versions, UTC time, and the result classification as measurement/detection rather than geological conclusion. Callers must populate trusted input hashes and actual software/version details; the helper does not infer them. Existing manifests are rejected instead of replaced.

`processing.resumable.run_resumable_chunks` maintains an append-only, fsynced JSONL journal with input-halo and output-core checksums. A chunk is checkpointed only after its NPY output region is flushed. Interrupted final journal records are ignored; completed records are checksum-verified before skip/reuse. A whole-output digest is followed by an exclusive `.complete` sentinel. Concurrent writers to the same paths are unsupported. The journal avoids replace/rename dependence; operational durability still depends on filesystem fsync semantics.

## Spatial and temporal alignment

`registration.spatial.reindex_volume_to_axes` reorders only by explicit inline and crossline identifiers. Bins absent in the source, or already marked missing, become NaN and are carried in the missing-trace mask. It does not perform coordinate reprojection, interpolation, or infer a survey transform. A physical geometry mismatch still requires authoritative survey metadata and a separately validated resampling workflow.

`registration.temporal.estimate_temporal_registration` remains a bounded integer-sample, trace-wise diagnostic. Its optional correction is a derived copy with NaN edge padding. It does not do fractional-sample lag estimation, 3D spatial warping, or global smoothness regularization. Diagnostics-only mode returns the source monitor view and avoids a needless full-volume copy.

`registration.temporal.estimate_fractional_temporal_registration` adds parabolic sub-sample refinement around a bounded integer cross-correlation peak and optional linear correction with NaN support gating. It is still trace-wise; peak shape can bias fractional estimates, and no spatial smoothness or phase-specific model is applied.

`registration.transform.resample_volume_to_target_grid` accepts an explicit affine mapping from target physical x/y coordinates into source fractional grid indices and writes a derived memmap in sample chunks. It does not estimate the transform and requires documented coordinate-reference/transform provenance. Nearest/linear spatial interpolation is supported; outputs are always QC warnings until waveform/amplitude fidelity and residual misties have been examined.

## Multi-attribute screening

`detection.multivariate.local_robust_screen` calculates local median/MAD z-strength in an odd-sized 3D footprint and reports finite-support uncertainty. Reflective boundaries are used at physical edges. `local_robust_screen_chunked` computes the same result with a two-footprint-radius halo (the local median-of-MAD needs neighboring medians), writing score, mask, and uncertainty memmaps. SciPy's generic median callback remains computationally expensive; chunking bounds memory but does not yet establish practical multi-gigabyte throughput.

`detection.multivariate.fuse_attributes` independently centers each supplied attribute by its global median and scales by 1.4826 MAD. Weighted absolute robust z-scores are averaged over available attributes; a configurable minimum support is required. The mask uses a robust threshold on fused strength. The reported uncertainty is relative disagreement between normalized attributes, and missing support is separately available. Neither field is a calibrated probability. This is a transparent baseline and must be compared with simpler single-attribute screens on actual, authorized data before interpretation.

`detection.clustering.connected_zones_chunked` labels each core independently, reconciles face/edge/corner adjacencies according to connectivity using union-find, aggregates global voxel counts/bounds/strength, filters by global size, then writes compact global labels to a memmap. Its working label array is disk-backed, but the provisional component metadata and union-find table grow with component count; pathological salt-and-pepper masks can therefore use substantial RAM.

## Validation and tests

Tests use small mathematical arrays with exact constructed relationships; they are not synthetic seismic, reservoir scenarios, or geological validation. Chunked filtering is checked against an in-memory SciPy calculation, chunked attributes against the existing full-array calculation, spatial reindexing and affine mapping against known index values, and normalization against RMS invariants. Cross-chunk 6/18/26-connected components are checked against full-volume labeling. Restart after a controlled interruption verifies journal recovery and result reuse. Fractional lag uses a mathematically shifted 1D signal. The performance command `python -m reservoir_time_machine.evaluation.performance --shape 48 48 96` measures arithmetic throughput on a deterministic mathematical array; its `tracemalloc` result excludes many native NumPy allocations and OS page cache.

No actual Challenge 3 files are present. No processing algorithm in this phase has been validated against the competition surveys. Only controlled mathematical-array throughput is measured; production-scale peak memory, SEG-Y decoding, compression, and HPC shared-Lustre performance remain unmeasured. See [five-model detection](five_model_detection.md) for model interfaces, score meanings, calibration requirements, and AI limitations.

## Limitations and next steps

1. Benchmark chunk sizes and I/O throughput against authorized data on the target HPC; avoid running large jobs on the login node.
2. Add resumable/atomic output completion and per-chunk checksums for interrupted processing.
3. Implement chunked local robust filtering and chunk-boundary-reconciled 3D connected components.
4. Add fractional lag estimation and spatially coherent registration quality/rejection using measured survey geometry.
5. Add a versioned adapter for the selected chunked store only after actual volume and filesystem benchmarks.
6. Compare detectors, parameter stability, false-positive behavior, and memory/runtime on the real surveys with independent geophysical QC. Anomaly scores are not direct evidence of fluid or hydrocarbons.
