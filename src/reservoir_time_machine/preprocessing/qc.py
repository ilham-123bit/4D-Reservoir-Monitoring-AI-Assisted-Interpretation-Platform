"""NaN-aware volume and trace quality summaries."""

import numpy as np
from ..data.validation import ValidationFinding, ValidationReport


def quality_report(volume: np.ndarray, *, inline_chunk: int = 8) -> dict[str, object]:
    data = np.asanyarray(volume)
    if data.ndim != 3:
        raise ValueError("volume must use (inline, crossline, sample) axes")
    if inline_chunk < 1:
        raise ValueError("inline_chunk must be positive")
    finite_count = 0
    value_sum = value_square_sum = 0.0
    minimum, maximum = float("inf"), float("-inf")
    trace_rms_sum = trace_rms_square_sum = 0.0
    available_trace_count = missing_trace_count = 0
    nan_count = positive_inf_count = negative_inf_count = traces_with_nonfinite = zero_energy_trace_count = 0
    for start in range(0, data.shape[0], inline_chunk):
        block = np.asarray(data[start:start+inline_chunk], dtype=float)
        finite = np.isfinite(block)
        values = block[finite]
        finite_count += int(values.size)
        nan_count += int(np.isnan(block).sum())
        positive_inf_count += int(np.isposinf(block).sum())
        negative_inf_count += int(np.isneginf(block).sum())
        if values.size:
            value_sum += float(values.sum(dtype=np.float64))
            value_square_sum += float(np.square(values, dtype=np.float64).sum(dtype=np.float64))
            minimum, maximum = min(minimum, float(values.min())), max(maximum, float(values.max()))
        per_trace_count = finite.sum(axis=2)
        traces_with_nonfinite += int((~finite.all(axis=2)).sum())
        per_trace_energy = np.sum(np.where(finite, block, 0.0)**2, axis=2)
        trace_rms = np.sqrt(np.divide(per_trace_energy, per_trace_count,
            out=np.zeros(per_trace_energy.shape, dtype=float), where=per_trace_count > 0))
        present = per_trace_count > 0
        missing_trace_count += int((~present).sum())
        zero_energy_trace_count += int((present & (trace_rms == 0)).sum())
        available_trace_count += int(present.sum())
        selected = trace_rms[present]
        trace_rms_sum += float(selected.sum())
        trace_rms_square_sum += float(np.square(selected).sum())
    voxel_count = int(data.size)
    mean = value_sum / finite_count if finite_count else float("nan")
    rms = float(np.sqrt(value_square_sum / finite_count)) if finite_count else float("nan")
    std = float(np.sqrt(max(0.0, value_square_sum / finite_count - mean**2))) if finite_count else float("nan")
    trace_mean = trace_rms_sum / available_trace_count if available_trace_count else float("nan")
    trace_std = float(np.sqrt(max(0.0, trace_rms_square_sum / available_trace_count - trace_mean**2))) if available_trace_count else float("nan")
    findings = []
    if not finite_count:
        findings.append(ValidationFinding("finite_sample_availability", "fail", "Volume contains no finite samples.", ("volume",),
            {"finite_samples": 0, "total_samples": voxel_count}, "Check file decoding, invalid masks, and source integrity.", True))
    elif finite_count < voxel_count:
        findings.append(ValidationFinding("finite_sample_availability", "warning", "Volume contains NaN or infinite samples.", ("volume",),
            {"finite_samples": finite_count, "total_samples": voxel_count, "nan_samples": nan_count,
             "positive_infinity_samples": positive_inf_count, "negative_infinity_samples": negative_inf_count},
            "Preserve a validity mask and exclude invalid samples from calculations."))
    else:
        findings.append(ValidationFinding("finite_sample_availability", "pass", "All volume samples are finite.", ("volume",),
            {"finite_samples": finite_count, "total_samples": voxel_count}))
    if missing_trace_count:
        findings.append(ValidationFinding("missing_traces", "warning", "Some traces contain no finite samples.", ("volume",),
            {"missing_trace_count": missing_trace_count}, "Retain missing-trace masks and report coverage."))
    else:
        findings.append(ValidationFinding("missing_traces", "pass", "Every trace contains at least one finite sample.", ("volume",)))
    if zero_energy_trace_count:
        findings.append(ValidationFinding("zero_energy_traces", "warning", "Some available traces have zero RMS energy.", ("volume",),
            {"zero_energy_trace_count": zero_energy_trace_count}, "Review these traces as possible dead traces; do not infer their cause."))
    return {
        "dimensions": tuple(data.shape),
        "voxel_count": voxel_count,
        "finite_voxels": finite_count,
        "availability_fraction": finite_count / voxel_count if voxel_count else 0.0,
        "missing_trace_count": missing_trace_count,
        "traces_with_nonfinite_samples": traces_with_nonfinite,
        "nonfinite_sample_count": voxel_count - finite_count,
        "nan_sample_count": nan_count,
        "positive_infinity_sample_count": positive_inf_count,
        "negative_infinity_sample_count": negative_inf_count,
        "zero_energy_trace_count": zero_energy_trace_count,
        "trace_rms_mean": trace_mean,
        "trace_rms_std": trace_std,
        "amplitude_min": minimum if finite_count else float("nan"),
        "amplitude_max": maximum if finite_count else float("nan"),
        "amplitude_rms": rms,
        "signal_to_noise_proxy": rms / (std + 1e-12) if finite_count else float("nan"),
        "validation": ValidationReport(tuple(findings)).to_dict(),
    }
