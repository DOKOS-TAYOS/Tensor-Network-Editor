"""Tensor-data lowering helpers for live Python imports."""

from __future__ import annotations

import math
from typing import Protocol, TypeAlias, TypeGuard, cast

from ...models import TensorDataMode, TensorDataSpec
from ..models._model_tensor_data import TensorComplexLiteral, TensorScalarLiteral

_LITERAL_DATA_ELEMENT_LIMIT = 4096
RuntimeTensorNumericLiteral: TypeAlias = (
    TensorScalarLiteral | list["RuntimeTensorNumericLiteral"]
)


class _ItemLike(Protocol):
    def item(self) -> object: ...


class _ToListLike(Protocol):
    def tolist(self) -> object: ...


def lower_runtime_tensor_data(
    data: object,
    *,
    shape: tuple[int, ...],
    tensor_name: str,
) -> tuple[TensorDataSpec | None, str | None]:
    """Lower one runtime tensor payload into the editor's tensor-data formats."""
    if data is None:
        return None, None
    element_count = shape_cardinality(shape)
    if element_count == 0:
        return (
            None,
            f"Dropped tensor data for tensor {tensor_name} because empty runtime tensors are not preserved.",
        )
    try:
        literal_values = coerce_tensor_literal(data)
    except TypeError:
        return (
            None,
            f"Dropped tensor data for tensor {tensor_name} because live import only preserves finite real or complex tensor values.",
        )
    flattened_values = flatten_tensor_literal(literal_values)
    if len(flattened_values) != element_count:
        return (
            None,
            f"Dropped tensor data for tensor {tensor_name} because the runtime values do not match the reported tensor shape.",
        )
    if flattened_values and all(value == 1 for value in flattened_values):
        return TensorDataSpec(mode=TensorDataMode.ONES), None
    if flattened_values and all_values_identical(flattened_values):
        return (
            TensorDataSpec(
                mode=TensorDataMode.FILL,
                fill_value=flattened_values[0],
            ),
            None,
        )
    if element_count <= _LITERAL_DATA_ELEMENT_LIMIT:
        return (
            TensorDataSpec(
                mode=TensorDataMode.LITERAL,
                values=literal_values,
            ),
            None,
        )
    return (
        None,
        f"Dropped tensor data for tensor {tensor_name} because literal runtime data is limited to {_LITERAL_DATA_ELEMENT_LIMIT} elements.",
    )


def shape_cardinality(shape: tuple[int, ...]) -> int:
    """Return the total number of elements implied by ``shape``."""
    cardinality = 1
    for dimension in shape:
        cardinality *= dimension
    return cardinality


def all_values_identical(values: list[TensorScalarLiteral]) -> bool:
    """Return whether every value in one flat literal sequence matches."""
    first_value = values[0]
    return all(value == first_value for value in values[1:])


def coerce_tensor_literal(data: object) -> RuntimeTensorNumericLiteral:
    """Convert runtime tensor data into finite real or complex Python literals."""
    if isinstance(data, bool):
        raise TypeError("Tensor literals must be numeric.")
    if isinstance(data, int):
        return data
    if isinstance(data, float):
        if not math.isfinite(data):
            raise TypeError("Tensor literals must be finite.")
        return data
    if isinstance(data, complex):
        if not math.isfinite(data.real) or not math.isfinite(data.imag):
            raise TypeError("Tensor literals must be finite.")
        return {"real": float(data.real), "imag": float(data.imag)}
    if _is_item_like(data):
        try:
            scalar_value = data.item()
        except (TypeError, ValueError):
            scalar_value = None
        else:
            return coerce_tensor_literal(scalar_value)
    if _is_tolist_like(data):
        return coerce_tensor_literal(data.tolist())
    if isinstance(data, tuple):
        return [coerce_tensor_literal(item) for item in data]
    if isinstance(data, list):
        return [coerce_tensor_literal(item) for item in data]
    raise TypeError("Unsupported tensor literal.")


def flatten_tensor_literal(
    values: RuntimeTensorNumericLiteral,
) -> list[TensorScalarLiteral]:
    """Flatten one nested tensor literal tree into a simple numeric list."""
    if isinstance(values, (int, float)) or _is_tensor_complex_literal(values):
        return [values]
    flattened_values: list[TensorScalarLiteral] = []
    for item in cast(list[RuntimeTensorNumericLiteral], values):
        flattened_values.extend(flatten_tensor_literal(item))
    return flattened_values


def _is_tensor_complex_literal(value: object) -> TypeGuard[TensorComplexLiteral]:
    """Return whether ``value`` is the portable complex literal mapping."""
    return isinstance(value, dict) and set(value) == {"real", "imag"}


def _is_item_like(value: object) -> TypeGuard[_ItemLike]:
    """Return whether ``value`` exposes one callable ``item()`` method."""
    return callable(getattr(value, "item", None))


def _is_tolist_like(value: object) -> TypeGuard[_ToListLike]:
    """Return whether ``value`` exposes one callable ``tolist()`` method."""
    return callable(getattr(value, "tolist", None))
