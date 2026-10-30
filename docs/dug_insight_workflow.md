# DUG Insight capability, installation, and interoperability audit

## Verified status for this workspace

This Windows workspace currently contains **no competition input files, DUG Insight installation, or prepared DUG session**. The competition installation, build, licence modules, session state, and exports therefore remain unverified. The public product and user manuals below establish what DUG documents, not what this team can currently run.

DUG cannot repair missing source samples. Once complete authorized inputs are available, verify their integrity and geometry in DUG and Python before paired analysis. DUG's SEG-Y exporter can split a volume by file size or inline, which may help avoid oversized single-file transfer problems; each exported segment still needs validation. [SEG-Y export options](https://help.dugeo.com/m/Insight/l/839349-exporting-to-seg-y).

## Capability matrix

The “publicly documented” column records vendor documentation only. Competition availability must be verified by opening the actual session and feature in the authorized installation.

| Capability | Publicly documented | Available in competition installation? | Licence/module status | Python role and recommendation | Expected benefit |
|---|---|---|---|---|---|
| 2D/3D seismic viewing and navigation | Base lists 2D/3D/pre-stack display and integrated 3D view | Unverified; no local DUG installation/session | Confirm Base entitlement/build | Use DUG for independent expert QC; custom app owns linked evidence views | Fast interpretation and visual reference without abandoning custom product |
| Baseline/monitor and 4D comparison | Base product page describes time-lapse seismic support; volume display and maths are documented | Unverified; test paired loading and matching survey setup | Confirm actual session configuration | Validate selected sections and exports in DUG; automate full repeatability and multi-attribute analysis in Python | Accelerates vintage QC; Python provides repeatable evidence chain |
| Inline, crossline, time/depth slices | 3D views and navigation are publicly described | Unverified | Confirm build and survey geometry | DUG for expert navigation; Plotly for synchronized section/slice controls | Familiar manual QC plus integrated candidate inspection |
| Wells, trajectories, logs, markers | Base lists Well Manager and marker/top tools; supported formats include LAS, ASCII, OWX and GEOH5 | Unverified; test import against actual available files | Confirm enabled well tools and format support | DUG is a reference view; Python owns source-preserving mapping and explicit CRS/depth transformations | Reduces time building manual correlation tooling; preserves reproducible joins |
| Fault/horizon interpretation | Public Base and user manuals describe fault and horizon workflows | Unverified | Confirm Base tools and export UI | Use DUG for human interpretation when available; Python consumes exported picks through verified adapters | Avoids reimplementing mature picking UX; provides structure context to detectors |
| Standard attributes and volume maths | Base lists amplitude extraction, trace arithmetic, volume maths, phase rotation, spectral analysis and resampling | Unverified | Some advanced attributes are module-specific; verify each enabled process | Cross-check selected calculations; keep parameterized, versioned Python algorithms as the reproducible pipeline | Fast exploratory QC plus testable custom multi-scale analysis |
| Time, phase, amplitude mistie correction | Public Base page lists mistie correction in time, phase and amplitude | Unverified; test on paired data and record module state | Confirm actual feature availability | Treat DUG output as a candidate correction/reference; quantify residuals and rejection criteria in Python before applying | Expert-guided comparison helps separate repeatability issues from apparent 4D response |
| Geobody detection and advanced attributes | Public Explorationist page lists geobody detection, volume interpretation, instantaneous attributes, spectral decomposition, coherence/semblance/curvature | Unverified | Assume Explorationist/other entitlement needed until confirmed | Reproduce selected analyses where needed; add multi-attribute evidence and evaluation in Python | Saves implementation time if licensed; Python adds transparent, benchmarkable detection |
| Volume export | Official manual documents SEG-Y, DUGIO and text export; SEG-Y can split by size or inline | Unverified in competition build | Confirm export permissions and output choices | Prefer SEG-Y for seismic, then validate headers and values; DUGIO is native/proprietary and not assumed readable in Python | Enables file-based, auditable bridge; split exports support bounded transfers |
| Horizons/faults export | Manuals document text/GEOH5/IL-CL horizon exports and fault CSV/GEOH5/IESX/GoCAD formats | Unverified | Confirm formats enabled | Prefer custom text/CSV for simple point products or GEOH5 after parser validation; preserve CRS/domain/units | Makes picked structure available to Python without private session parsing |
| Direct Python automation/API | No competition-specific supported API or automation path was established in this audit | Unverified | Ask organizer/support or inspect installed help | Start with explicit files and documented exports; do not reverse-engineer session files | Simple, inspectable interface; defer automation until a supported route is confirmed |

Sources: [DUG Insight Base](https://dug.com/dug-insight/base/), [Explorationist module](https://dug.com/dug-insight/modules/explorationist/), [accepted formats](https://help.dugeo.com/online-manual/user-manual/articles/references/195749-formats-accepted-by-insight-for-import-and-export.html), [SEG-Y export](https://help.dugeo.com/m/Insight/l/839349-exporting-to-seg-y), [horizon export formats](https://help.dugeo.com/m/Insight/l/438691-export-formats-horizon), [fault import/export](https://help.dugeo.com/online-manual/faq/articles/faults/69149-how-do-i-import-or-export-faults.html), and [volume overview](https://help.dugeo.com/m/Insight/l/438660-volumes-overview).

## Interoperability plan

Use explicit file exchange first:

1. Export a small, authorized SEG-Y subset (or one inline range) with geometry and headers retained. For large exports, split by inline or configured file size.
2. Validate one-to-one inline/crossline/sample values, trace ordering, coordinate scalar/units, sample domain and null behavior in Python before exporting full volumes.
3. Export interpreted horizons in IL/CL/time or depth text format when domain metadata is clear; export faults in CSV/IESX/GEOH5/GoCAD as actually supported. Read the exact schema and retain provenance.
4. Never treat DUGIO as open Python interchange unless a supported reader is confirmed. The DUG manual describes it as DUG's native, proprietary volume format.
5. For mistie corrections, export the correction table or corrected volume only after validating its units, sign convention, spatial scope and effect on a selected subset.

DUG's public SEG-Y export documentation states that survey X/Y can be written to ensemble X/Y headers, that coordinate/elevation scalars can be controlled, and that all-NaN traces may be dropped. Validate these export choices against the selected dataset's geometry before analysis. Do not export or publish any restricted result outside authorized storage.

## On-site installation and licence audit

On the competition workstation, record privately:

1. DUG build/version, operating system, project/session load status and whether `Challenge3_InputData` opens.
2. Licensed/enabled modules and disabled tools. Ask the licence administrator when a UI state is ambiguous.
3. Whether baseline and monitor can be displayed in corresponding survey geometry; inspect identical sections and compare against file headers.
4. Open each requested process (mistie, volume maths, amplitude/phase attributes, geobody/body tools) and record whether it runs, not merely whether a menu item exists.
5. Confirm well/marker/deviation imports and coordinate/domain display using the actual source references.
6. Export a small volume, horizon, fault, well/marker product, and correction table where available. Keep outputs on authorized storage and verify them in Python.
7. Check installed documentation for a supported automation interface. Until confirmed, use the documented file-exchange boundary.

## Decision and QC workflow

The product decision is hybrid: use DUG for mature interpretation and independent visual QC when the competition entitlement permits; keep Python responsible for repeatable 4D processing, detectors, uncertainty, evidence ranking and integrated custom visualization. See [visualization architecture decision](visualization_architecture_decision.md).

1. Verify source files are complete before loading them into DUG.
2. Inspect survey geometry and time/depth domain; compare inline, crossline and sample selections.
3. Check well overlays only after confirming the shared CRS and depth references.
4. Export a small representative subset and compare Python samples and attributes at the same bins.
5. If mistie tools are available, record suggested corrections and compare against bounded Python diagnostics; do not blindly apply a volume-wide warp.
6. Record discrepancies, settings and reviewer conclusions in authorized private storage. Do not publish raw coordinates, screenshots or restricted derived outputs.
