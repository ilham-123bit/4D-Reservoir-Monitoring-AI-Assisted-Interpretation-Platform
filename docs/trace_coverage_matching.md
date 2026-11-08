# Trace coverage and pairing

Baseline and Monitor traces are matched using their declared inline and crossline identifiers. The paired arrays use the Baseline grid and preserve the association between each trace key and its array location.

## Coverage states

- **Paired:** a trace key exists in both vintages and has valid samples for the calculation.
- **Baseline-only:** the Baseline grid contains a trace without a matching Monitor trace.
- **Monitor-only:** a Monitor trace has no corresponding Baseline-grid location; it remains in the inventory.
- **Missing or invalid samples:** a trace or sample window lacks usable paired measurements and is excluded from the relevant attribute.

Coverage masks are carried separately from amplitudes. A missing cell is not a zero-amplitude observation. Attribute support counts show how many paired samples contributed to window calculations.

## Coordinate meaning

Inline and crossline IDs identify survey bins; they are not map coordinates. A shared trace key does not prove that the same subsurface location was sampled in both vintages. Physical comparison requires verified coordinate headers, CRS, survey geometry and registration quality. Until then, paired differences are preliminary repeatability screening.

## Display

Full-resolution coverage and anomaly masks are preserved analytically. Display aggregation conservatively indicates a bin when any source cell is missing or flagged. Region labels are not numerically averaged during downsampling; exact component membership remains in the finding record.
