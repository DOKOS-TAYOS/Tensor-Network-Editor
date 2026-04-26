"""Tensor-data lowering helpers for live Python imports."""

from __future__ import annotations

import math
from typing import Protocol, TypeAlias, TypeGuard, cast

from ...models import TensorDataMode, TensorDataSpec
from ..models._model_tensor_data import TensorNumericLiteral

_LITERAL_DATA_ELEMENT_LIMIT = 4096
RealTensorNumericLiteral: TypeAlias = int | float | list["RealTensorNumericLiteral"]


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
            f"Dropped tensor data for tensor {tensor_name} because live import only preserves finite real tensor values.",
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
                fill_value=float(flattened_values[0]),
            ),
            None,
        )
    if element_count <= _LITERAL_DATA_ELEMENT_LIMIT:
        return (
            TensorDataSpec(
                mode=TensorDataMode.LITERAL,
                values=cast(TensorNumericLiteral, literal_values),
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


def all_values_identical(values: list[int | float]) -> bool:
    """Return whether every value in one flat literal sequence matches."""
    first_value = values[0]
    return all(value == first_value for value in values[1:])


def coerce_tensor_literal(data: object) -> RealTensorNumericLiteral:
    """Convert runtime tensor data into finite real Python literals."""
    if isinstance(data, bool):
        raise TypeError("Tensor literals must be numeric.")
    if isinstance(data, int):
        return data
    if isinstance(data, float):
        if not math.isfinite(data):
            raise TypeError("Tensor literals must be finite.")
        return data
    if isinstance(data, complex):
        raise TypeError("Complex tensor literals are not supported.")
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


def flatten_tensor_literal(values: RealTensorNumericLiteral) -> list[int | float]:
    """Flatten one nested tensor literal tree into a simple numeric list."""
    if isinstance(values, (int, float)):
        return [values]
    flattened_values: list[int | float] = []
    for item in values:
        flattened_values.extend(flatten_tensor_literal(item))
    return flattened_values


def _is_item_like(value: object) -> TypeGuard[_ItemLike]:
    """Return whether ``value`` exposes one callable ``item()`` method."""
    return callable(getattr(value, "item", None))


def _is_tolist_like(value: object) -> TypeGuard[_ToListLike]:
    """Return whether ``value`` exposes one callable ``tolist()`` method."""
    return callable(getattr(value, "tolist", None))
