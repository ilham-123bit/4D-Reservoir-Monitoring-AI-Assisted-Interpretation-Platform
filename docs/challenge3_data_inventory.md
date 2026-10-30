# Challenge 3 data inventory status

## Current zero-input status

**No actual competition dataset is currently present in the workspace.** `Challenge3_InputData/` is absent, and the previously copied incomplete seismic files were deleted. The application must remain in its explicit no-data state until authorized replacement inputs are provided. No synthetic substitute is generated.

Historical local inspection artifacts are excluded from the public repository because they contain actual-data-derived metadata and visualizations. They do not establish current file availability, provenance, or validation. Do not recreate files from reports or figures.

## Expected input inventory (contract template, not verified files)

The competition input contract anticipates two seismic vintages, a well inventory, and optional well support files. Actual names, counts, coverage, formats, CRS, and geometry must be checked against the replacement dataset; none are asserted here.

| Data group | Required for | Format contract | Current availability |
|---|---|---|---|
| Baseline and monitor seismic | Paired 4D analysis | SEG-Y with verified sample encoding, headers, bin geometry, sample domain, coordinates, and integrity | Not supplied |
| Well inventory | Well metadata | XLSX or documented CSV with explicit columns and coordinate reference | Not supplied |
| LAS logs | Log context | LAS 2.0/3.0 with curve units, null value, depth reference, and source provenance | Not supplied |
| Deviations | Trajectory | Documented delimited table with measured depth, angle units/conventions, datum, and CRS | Not supplied |
| Checkshots | Depth/time relationship | Documented table with depth/time conventions, datum, units, and well ID | Not supplied |
| Markers | Stratigraphic context | Documented delimited table with marker names, reference depth, datum, and units | Not supplied |
| DUG attributes and interpretations | Optional integration | Validated SEG-Y, CSV/text, or supported exported product plus geometry/domain metadata | Not supplied |

See [data contract](data_contract.md) for required, optional, and conditionally required fields, formats, limitations, and output provenance.

## Read-only inventory utility

`scripts/inspect_challenge3.py` and the CLI `status`/`audit` commands can report availability without inventing data. A missing root or missing files must remain an explicit unavailable state. Run the audit only where authorized inputs are available and keep restricted reports in authorized storage.
