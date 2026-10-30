"""Chunked paired-survey waveform and amplitude repeatability summaries."""

from __future__ import annotations

from typing import Any
import numpy as np

from ..data.validation import ValidationFinding, ValidationReport


def repeatability_report(baseline: np.ndarray, monitor: np.ndarray, *, inline_chunk: int = 8,
                         inline_tile: int = 8, crossline_tile: int = 8,
                         min_samples_per_trace: int = 8) -> dict[str, Any]:
    """Compute global and spatial-tile summaries with pairwise finite-sample masking.

    Correlation is Pearson waveform correlation after mean removal. NRMS is
    ``200 * sqrt(mean((B-M)^2)) / (RMS(B)+RMS(M))`` over paired finite samples.
    Zero energy returns NRMS 0; undefined/constant-trace correlations are reported as null.
    No universal repeatability pass threshold is applied: these are diagnostics, not geological
    classifications. Processing is chunked along inline to bound temporary memory.
    """
    before, after = np.asanyarray(baseline), np.asanyarray(monitor)
    findings: list[ValidationFinding] = []
    inputs = ("baseline_array", "monitor_array")
    if before.ndim != 3 or after.ndim != 3:
        report = ValidationReport((ValidationFinding("array_axes", "fail", "Repeatability inputs must be 3D (inline,crossline,sample) arrays.", inputs,
            {"baseline_ndim": before.ndim, "monitor_ndim": after.ndim}, "Provide two 3D arrays using the declared axis order.", True),))
        return {"validation": report.to_dict(), "metrics": {}}
    if before.shape != after.shape:
        report = ValidationReport((ValidationFinding("dimensions", "fail", "Repeatability inputs must have matching dimensions.", inputs,
            {"baseline_shape": before.shape, "monitor_shape": after.shape}, "Establish and validate a common geometry before comparison.", True),))
        return {"validation": report.to_dict(), "metrics": {}}
    if min(inline_chunk, inline_tile, crossline_tile, min_samples_per_trace) < 1:
        raise ValueError("chunk, tile, and minimum-sample settings must be positive")

    ni, nc, ns = before.shape
    trace_correlations: list[float] = []
    trace_nrms: list[float] = []
    valid_trace_count = zero_energy_trace_count = 0
    finite_pair_count = 0
    sums = np.zeros(3, dtype=np.float64)  # sum B², sum M², sum (B-M)²
    corr_count = 0
    mean_b = mean_m = m2_b = m2_m = covariance = 0.0
    tile_values: dict[tuple[int, int], dict[str, list[float] | int]] = {}
    for i0 in range(0, ni, inline_chunk):
        bblock = np.asarray(before[i0:i0+inline_chunk], dtype=np.float64)
        mblock = np.asarray(after[i0:i0+inline_chunk], dtype=np.float64)
        for local_i in range(bblock.shape[0]):
            inline = i0 + local_i
            for crossline in range(nc):
                btrace, mtrace = bblock[local_i, crossline], mblock[local_i, crossline]
                valid = np.isfinite(btrace) & np.isfinite(mtrace)
                count = int(valid.sum())
                if not count:
                    continue
                bvalues, mvalues = btrace[valid], mtrace[valid]
                valid_trace_count += int(count >= min_samples_per_trace)
                finite_pair_count += count
                sum_b2, sum_m2 = float(np.dot(bvalues, bvalues)), float(np.dot(mvalues, mvalues))
                diff_energy = float(np.dot(bvalues - mvalues, bvalues - mvalues))
                sums += (sum_b2, sum_m2, diff_energy)
                centered_b, centered_m = bvalues - bvalues.mean(), mvalues - mvalues.mean()
                trace_m2_b, trace_m2_m = float(np.dot(centered_b, centered_b)), float(np.dot(centered_m, centered_m))
                trace_covariance = float(np.dot(centered_b, centered_m))
                trace_denominator = np.sqrt(trace_m2_b * trace_m2_m)
                trace_corr = trace_covariance / trace_denominator if count >= 2 and trace_denominator > 0 else None
                if trace_corr is not None:
                    trace_correlations.append(float(trace_corr))
                block_mean_b, block_mean_m = float(bvalues.mean()), float(mvalues.mean())
                combined_count = corr_count + count
                delta_b, delta_m = block_mean_b - mean_b, block_mean_m - mean_m
                scale = corr_count * count / combined_count
                m2_b += trace_m2_b + delta_b * delta_b * scale
                m2_m += trace_m2_m + delta_m * delta_m * scale
                covariance += trace_covariance + delta_b * delta_m * scale
                mean_b += delta_b * count / combined_count
                mean_m += delta_m * count / combined_count
                corr_count = combined_count
                denominator = np.sqrt(sum_b2 / count) + np.sqrt(sum_m2 / count)
                if denominator <= 0:
                    trace_nrms.append(0.0)
                    zero_energy_trace_count += 1
                else:
                    trace_nrms.append(float(200 * np.sqrt(diff_energy / count) / denominator))
                tile = (inline // inline_tile, crossline // crossline_tile)
                item = tile_values.setdefault(tile, {"correlations": [], "nrms": [], "valid_traces": 0})
                item["valid_traces"] = int(item["valid_traces"]) + int(count >= min_samples_per_trace)
                if trace_corr is not None:
                    item["correlations"].append(float(trace_corr))
                item["nrms"].append(trace_nrms[-1])

    total_samples = int(before.size)
    total_pairs = int(finite_pair_count)
    if total_pairs == 0:
        findings.append(ValidationFinding("paired_finite_samples", "fail", "No finite sample pairs are available for repeatability calculation.", inputs,
            {"total_samples": total_samples, "valid_pairs": 0}, "Inspect source data and validity masks; no metrics can be computed.", True))
        return {"validation": ValidationReport(tuple(findings)).to_dict(), "metrics": {}}
    sum_b2, sum_m2, diff_energy = map(float, sums)
    global_corr_denominator = float(np.sqrt(max(0.0, m2_b) * max(0.0, m2_m)))
    global_corr = covariance / global_corr_denominator if global_corr_denominator > 0 else None
    global_denominator = np.sqrt(sum_b2 / total_pairs) + np.sqrt(sum_m2 / total_pairs)
    global_nrms = float(200 * np.sqrt(diff_energy / total_pairs) / global_denominator) if global_denominator > 0 else 0.0
    if total_pairs < total_samples:
        findings.append(ValidationFinding("paired_finite_samples", "warning", "Some samples are non-finite in one or both surveys and are excluded pairwise.", inputs,
            {"valid_pairs": total_pairs, "total_samples": total_samples, "paired_finite_fraction": total_pairs / total_samples},
            "Propagate the joint validity mask and report coverage; do not replace invalid samples with zero."))
    else:
        findings.append(ValidationFinding("paired_finite_samples", "pass", "All samples are finite in both surveys.", inputs,
            {"valid_pairs": total_pairs, "total_samples": total_samples}))
    if global_corr is None:
        findings.append(ValidationFinding("global_waveform_correlation", "warning", "Global waveform correlation is undefined because one or both inputs have zero variance.", inputs,
            {"correlation": None}, "Inspect amplitude energy and interpret NRMS separately."))
    else:
        findings.append(ValidationFinding("global_waveform_correlation", "pass", "Global waveform correlation was computed over paired finite samples.", inputs,
            {"correlation": float(global_corr)}))

    tiles = []
    for (tile_i, tile_x), values in sorted(tile_values.items()):
        correlations, nrms_values = values["correlations"], values["nrms"]
        tiles.append({
            "inline_tile": tile_i,
            "crossline_tile": tile_x,
            "valid_trace_count": values["valid_traces"],
            "median_trace_correlation": float(np.median(correlations)) if correlations else None,
            "median_trace_nrms_percent": float(np.median(nrms_values)) if nrms_values else None,
        })
    metrics = {
        "dimensions": list(before.shape),
        "paired_finite_samples": total_pairs,
        "paired_finite_fraction": total_pairs / total_samples,
        "global_waveform_correlation": float(global_corr) if global_corr is not None else None,
        "global_nrms_percent": global_nrms,
        "baseline_rms": float(np.sqrt(sum_b2 / total_pairs)),
        "monitor_rms": float(np.sqrt(sum_m2 / total_pairs)),
        "valid_trace_count": valid_trace_count,
        "total_trace_count": ni * nc,
        "trace_correlation_count": len(trace_correlations),
        "trace_correlation_quantiles": ({str(q): float(np.percentile(trace_correlations, q)) for q in (10, 50, 90)}
                                         if trace_correlations else {}),
        "trace_nrms_quantiles_percent": ({str(q): float(np.percentile(trace_nrms, q)) for q in (10, 50, 90)}
                                          if trace_nrms else {}),
        "zero_energy_trace_count": zero_energy_trace_count,
        "local_tiles": tiles,
        "definitions": {
            "global_waveform_correlation": "Pearson correlation over all paired finite samples after global mean removal.",
            "trace_waveform_correlation": "Pearson correlation per trace after per-trace mean removal; constant traces are undefined.",
            "nrms_percent": "200 * RMS(B-M) / (RMS(B)+RMS(M)) over paired finite samples; zero denominator returns 0.",
            "local_summary": f"Trace metrics aggregated in {inline_tile}x{crossline_tile} inline/crossline tiles; not a spatially calibrated threshold.",
        },
    }
    findings.append(ValidationFinding("repeatability_calculation", "pass", "Repeatability summaries were calculated; no universal accept/reject threshold is implied.", inputs,
        {"inline_chunk": inline_chunk, "inline_tile": inline_tile, "crossline_tile": crossline_tile,
         "minimum_samples_per_trace": min_samples_per_trace}))
    return {"validation": ValidationReport(tuple(findings), {"metric_scope": "diagnostic_only"}).to_dict(), "metrics": metrics}
