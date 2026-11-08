# Challenge 3 data dictionary

This dictionary describes the fields used by the analysis. A field may remain unavailable when the source file does not provide a verified value.

## Seismic survey

| Field | Meaning |
|---|---|
| Baseline / Monitor | The two acquired survey vintages compared in a 4D workflow. |
| Amplitude | Recorded seismic sample value; it is not a direct measurement of fluid saturation or production. |
| Inline / crossline | Survey trace identifiers. They are grid coordinates, not geographic positions. |
| Sample index | Position on the stored sample axis. It is not time or depth unless source metadata establishes that domain and origin. |
| Sample interval | Interval declared by seismic headers or authoritative metadata, with original units retained. |
| Missing-trace mask | Locations without valid paired measurements. These are excluded from paired attributes and identified in coverage views. |
| Coordinate reference | Coordinate values, scalar and units from headers. A CRS or transformation is used only when verified. |

## Calculated attributes

- **Signed difference:** Monitor amplitude minus Baseline amplitude.
- **Absolute difference:** Magnitude of the signed difference.
- **RMS:** Root mean square amplitude over a stated sample window and valid support.
- **NRMS:** Normalized repeatability measure calculated from paired samples under the documented implementation.
- **Support count:** Number of valid paired samples contributing to a windowed calculation.

Attributes depend on the selected window, masks and preprocessing; they are not additional acquired measurements.

## Well evidence

Well inventory fields preserve source identifiers and declared metadata. LAS curves retain mnemonic, unit, description, depth index and null handling. Marker, checkshot and deviation records retain their source columns and units. Formation meaning, datum, coordinate reference and seismic tie are not inferred when undocumented.
