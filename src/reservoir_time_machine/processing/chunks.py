"""Bounded-memory N-dimensional chunk processing with explicit halos.

The processor receives a clipped input block including its halo and must return an
array with the same shape. Only the requested core is written to a `.npy` memmap.
The source is never modified. At physical volume boundaries, processors should
use the documented boundary mode (typically reflect); internal chunk seams are
shielded by the halo and cropped before output is written.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain, product
from pathlib import Path
from typing import Callable, Iterator, Sequence

import numpy as np


@dataclass(frozen=True)
class Chunk:
    core: tuple[slice, ...]
    expanded: tuple[slice, ...]
    core_in_expanded: tuple[slice, ...]


def _shape_tuple(values: Sequence[int], ndim: int, name: str, *, allow_zero: bool = False) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if len(result) != ndim or any(value < (0 if allow_zero else 1) for value in result):
        raise ValueError(f"{name} must contain {ndim} {'non-negative' if allow_zero else 'positive'} integers")
    return result


def iter_chunks(shape: Sequence[int], chunk_shape: Sequence[int], *, halo: int | Sequence[int] = 0) -> Iterator[Chunk]:
    """Yield non-overlapping output cores and clipped input regions."""
    shape = tuple(int(value) for value in shape)
    if not shape or any(value < 1 for value in shape):
        raise ValueError("shape must contain positive dimensions")
    chunk_shape = _shape_tuple(chunk_shape, len(shape), "chunk_shape")
    halo = _shape_tuple((int(halo),) * len(shape) if isinstance(halo, int) else halo,
                        len(shape), "halo", allow_zero=True)
    starts = [range(0, size, step) for size, step in zip(shape, chunk_shape)]
    for origin in product(*starts):
        core = tuple(slice(start, min(start + step, size)) for start, step, size in zip(origin, chunk_shape, shape))
        expanded = tuple(slice(max(0, part.start - width), min(size, part.stop + width))
                         for part, width, size in zip(core, halo, shape))
        core_in_expanded = tuple(slice(part.start - ext.start, part.stop - ext.start)
                                 for part, ext in zip(core, expanded))
        yield Chunk(core, expanded, core_in_expanded)


def map_chunks(source: np.ndarray, processor: Callable[[np.ndarray], np.ndarray], output_path: str | Path, *,
               chunk_shape: Sequence[int], halo: int | Sequence[int] = 0,
               output_dtype: np.dtype | str | None = None) -> np.memmap:
    """Apply a shape-preserving function in bounded chunks and write a new NPY memmap.

    Existing output paths are rejected. Intermediate output is retained if a processor
    fails so the caller can inspect or remove the explicitly named incomplete artifact.
    """
    array = np.asanyarray(source)
    if array.ndim < 1:
        raise ValueError("source must have at least one dimension")
    # Validate chunk arguments before creating an output file.
    iterator = iter_chunks(array.shape, chunk_shape, halo=halo)
    first_chunk = next(iterator)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite analytical output: {output}")
    dtype = np.dtype(output_dtype) if output_dtype is not None else array.dtype
    result = np.lib.format.open_memmap(output, mode="w+", dtype=dtype, shape=array.shape)
    try:
        for chunk in chain((first_chunk,), iterator):
            block = np.asarray(array[chunk.expanded])
            processed = np.asarray(processor(block))
            if processed.shape != block.shape:
                raise ValueError(f"processor changed block shape from {block.shape} to {processed.shape}")
            result[chunk.core] = processed[chunk.core_in_expanded]
        result.flush()
    except Exception:
        result.flush()
        raise
    return result
