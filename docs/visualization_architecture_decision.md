# Visualization architecture decision

## Decision

Use a **hybrid DUG + custom Python platform**. DUG is an expert interpretation and QC companion if the actual competition installation has the required licence/features. The custom application remains the integrated reservoir intelligence experience for linked multi-vintage analysis, AI evidence, spatial clusters, uncertainty, candidate comparison, animation and reporting. This decision does not depend on DUG being present: Python remains the product path when it is absent.

The decision is architectural, not a claim of successful DUG or renderer integration. This Windows workspace has no DUG installation or prepared session, and Plotly/PyVista/VTK are not installed. No rendering benchmark has yet run against the actual files.

## Actual scale and performance constraints

The inspected complete trace records imply dense float32 display grids of roughly `334 × 1203 × 1301` and `333 × 1206 × 1301`, about **1.95 GiB per cube** before processing copies. The local GPU is an MX330 with 2 GiB VRAM, so a full cube plus rendering buffers is not a realistic GPU target here. Full-volume operations should use chunking or memmaps; visualization downsampling must be a separate derived copy.

Both local SEG-Y files are incomplete, so these dimensions describe the header-derived grid, not a validated complete volume. The monitor coordinate headers are also unreliable. Do not render an apparently aligned paired cube until the source/export and survey geometry pass validation.

## Option comparison

| Criterion | A. DUG-led visualization | B. Fully custom Python | C. Hybrid (selected) |
|---|---|---|---|
| 3D performance | Mature seismic viewing and survey-aware navigation are publicly documented; actual build/licence unverified | Plotly browser payloads become impractical for every voxel in a ~2 GiB cube; PyVista/VTK can render NumPy/ImageData volumes but need memory/GPU testing | DUG for expert full-volume QC if available; custom Python uses linked slices and selected downsampled 3D views |
| Scientific capabilities | Strong conventional interpretation tools where licensed | Highest control for custom repeatability diagnostics, detector outputs, uncertainty and ranking | DUG for established human workflows; Python for reproducible novel analysis |
| 4D animation and synchronized views | Must verify provided session interaction; no product-specific animation contract established | Flexible but large data and browser payload challenge | Python animates selected sections/endpoints and attributes; DUG cross-checks the endpoint views |
| AI anomaly overlays and spatial clusters | Import/display support must be checked; not assumed | Fully customizable | Python owns cluster geometry, evidence maps and overlays |
| Well integration | Mature manual workflows publicly described; availability unverified | Defensible only after CRS/datum/time references are validated | Use DUG as visual cross-check; Python owns source-preserving, explicit data joins |
| Geological interpretation | Established manual fault/horizon workflows | Rebuilding picking and QC UI would be expensive and risky | DUG-assisted interpretation plus custom evidence-to-decision panels |
| UX for judges | Professional expert tool, but not necessarily a guided competition narrative | Can be polished and task-specific, with significant engineering effort | Custom dashboard provides guided candidate stories and links to DUG checks |
| Interoperability | SEG-Y, DUGIO, text, horizon and fault exports are publicly documented; exact competition options unverified | Standard data adapters; no DUG session coupling | Explicit validated export/import boundary, no private session parsing |
| Development effort | Low for basic review, high if forced to act as the full product | Highest: interpretation, render, analytics and UX all ours | Balanced: reuse DUG where valuable and focus custom effort on analytical differentiation |
| Primary risk | Missing licence/session access could constrain the whole demo | Reimplementation/performance cost and duplicated mature tools | Export mistakes and misregistered overlays; controlled by gates and metadata checks |

## Plotly vs PyVista/VTK

- **Plotly in the browser:** best fit for Streamlit-linked controls, slice comparisons, compact surfaces, charts, candidate selection and timeline controls. Plotly's `Volume` trace supports 3D scalar rendering, but its interface uses flattened voxel coordinate/value arrays. Supplying x/y/z and values for the entire actual cube would exceed local memory and produce an excessive browser payload. Use Plotly for downsampled display copies and orthogonal slice surfaces rather than a full-resolution voxel cloud. [Plotly 3D volume documentation](https://plotly.com/python/3d-volume-plots/).
- **PyVista/VTK:** best candidate for an optional high-quality volume-rendered panel or exportable 3D scene. PyVista accepts 3D NumPy arrays and uniform `ImageData`; VTK exposes smart and GPU volume mappers. This is more suitable for seismic volume rendering than transmitting per-voxel coordinate arrays to the browser, but GPU texture size, memory copies, headless/HPC rendering, browser embedding and driver support must be benchmarked on the target system. [PyVista volume rendering](https://docs.pyvista.org/examples/02-plot/volume_rendering.html), [PyVista add-volume API](https://docs.pyvista.org/api/plotting/_autosummary/pyvista.plotter.add_volume), [VTK](https://docs.vtk.org/en/latest/).
- **Recommended initial dashboard:** Streamlit + Plotly for orthogonal sections, slice maps, matched-vintage panels, selected attribute overlays, cluster boundaries and user controls. Add an optional PyVista/VTK overview using a clearly labelled display-downsampled subvolume after a real performance test. Maintain original-resolution scientific calculations. Use DUG for expert full-volume review if available.

## Product division and experience

1. **DUG:** inspect full baseline/monitor datasets, wells, marker context, faults/horizons and selected standard attributes; use correction processes as review inputs rather than unquestioned preprocessing.
2. **Python engine:** own source audit, geometry gates, repeatability/registration diagnostics, multi-scale attributes, conventional/ML detectors, clustering, uncertainty and evidence-based prioritization.
3. **Custom dashboard:** display 1995 and 2006 endpoints with shared section indices; difference/NRMS and detector evidence; candidate zones; valid well/structure overlays; uncertainty and recommendations; exportable figures/report.
4. **Gemma:** narrate compact structured evidence and alternatives; it never supplies numerical seismic scores.

Animation uses only measured 1995 and 2006 endpoints. If smooth intermediate frames are used for a transition, label them as display interpolation, not measured vintages or proven flow trajectories.

## Implementation and validation gates

1. Resolve the incomplete local source exports and validate trace pairing before a paired view is considered scientific.
2. Verify DUG build, session, feature/licence state and export paths in the actual competition environment.
3. Confirm renderer dependencies and target GPU/driver. Benchmark one actual section, one downsampled subvolume and a linked comparison panel; record payload size, startup latency, interaction latency and peak RAM/VRAM.
4. For DUG exports, compare trace samples, indices, scalars, coordinate units, sample axis, masks and attribute values against Python.
5. Keep display downsampling separate from analysis arrays and preserve provenance in every exported figure/report.
