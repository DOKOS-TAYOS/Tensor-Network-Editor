"""Shared memory-dtype helpers for contraction analysis surfaces."""

from __future__ import annotations

DEFAULT_MEMORY_DTYPE = "float64"
SUPPORTED_MEMORY_DTYPES: tuple[str, ...] = (
    "float16",
    "float32",
    "float64",
    "complex64",
    "complex128",
)

_DTYPE_SIZE_BY_NAME: dict[str, int] = {
    "float16": 2,
    "float32": 4,
    "float64": 8,
    "complex64": 8,
    "complex128": 16,
}


def dtype_size_in_bytes(memory_dtype: str) -> int:
    """Return the element width used for memory estimates."""
    return _DTYPE_SIZE_BY_NAME.get(
        memory_dtype,
        _DTYPE_SIZE_BY_NAME[DEFAULT_MEMORY_DTYPE],
    )
