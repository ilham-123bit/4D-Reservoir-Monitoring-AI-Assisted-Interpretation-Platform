"""Controlled mathematical-array throughput/memory benchmark for chunk primitives."""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import time
import tracemalloc

import numpy as np

from ..processing.chunks import map_chunks, iter_chunks


def benchmark_chunked_map(*, shape: tuple[int, int, int] = (48, 48, 96),
                          chunk_shape: tuple[int, int, int] = (16, 16, 32),
                          dtype: str = "float32") -> dict[str, object]:
    """Measure a simple exact arithmetic map; this is not seismic performance validation."""
    if len(shape) != 3 or any(int(value) < 1 for value in shape):
        raise ValueError("shape must be three positive dimensions")
    array = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)
    array = (array % 101).astype(dtype, copy=False)
    maximum_block_elements = max(int(np.prod(tuple(part.stop-part.start for part in chunk.expanded)))
                                 for chunk in iter_chunks(shape, chunk_shape))
    input_bytes = int(array.nbytes)
    output_bytes = input_bytes
    with tempfile.TemporaryDirectory(prefix="rtm_math_benchmark_") as temporary:
        output_path = Path(temporary) / "mapped.npy"
        tracemalloc.start()
        start = time.perf_counter()
        result = map_chunks(array, lambda block: block * 1.25 + 3.0,
                            output_path, chunk_shape=chunk_shape, output_dtype=dtype)
        elapsed = time.perf_counter() - start
        _, python_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        expected = array * 1.25 + 3.0
        exact = bool(np.allclose(result, expected, rtol=1e-6, atol=1e-6))
        result.flush()
        result._mmap.close()
        del result
    voxels = int(np.prod(shape))
    return {
        "benchmark_kind": "mathematical_array_arithmetic_not_seismic",
        "shape": list(shape), "chunk_shape": list(chunk_shape), "dtype": dtype,
        "input_bytes": input_bytes, "output_bytes": output_bytes,
        "maximum_input_block_bytes": maximum_block_elements * np.dtype(dtype).itemsize,
        "full_reference_bytes": input_bytes,
        "estimated_input_plus_mmap_output_plus_reference_plus_block_bytes": 3*input_bytes + maximum_block_elements*np.dtype(dtype).itemsize,
        "python_tracemalloc_peak_bytes": int(python_peak),
        "tracemalloc_note": "Python allocator measurement does not capture all NumPy native allocations or OS page cache.",
        "elapsed_seconds": elapsed, "voxels_per_second": voxels / elapsed if elapsed else None,
        "output_matches_full_array_reference": exact,
        "production_scale_validated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", nargs=3, type=int, default=(48, 48, 96))
    parser.add_argument("--chunk-shape", nargs=3, type=int, default=(16, 16, 32))
    parser.add_argument("--dtype", default="float32")
    args = parser.parse_args()
    import json
    print(json.dumps(benchmark_chunked_map(shape=tuple(args.shape), chunk_shape=tuple(args.chunk_shape),
                                          dtype=args.dtype), indent=2))


if __name__ == "__main__":
    main()
