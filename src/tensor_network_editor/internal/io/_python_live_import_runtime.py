"""Shared runtime-shape helpers for live Python imports."""

from __future__ import annotations

import operator
from collections.abc import Iterable, Mapping
from typing import SupportsIndex, SupportsInt, cast

from ...errors import SerializationError


def get_attribute(value: object, attribute_name: str) -> object | None:
    """Return one attribute value when present."""
    try:
        return cast(object, getattr(value, attribute_name))
    except AttributeError:
        return None


def module_name(value: object) -> str:
    """Return the runtime module path for one object type."""
    return type(value).__module__


def coerce_iterable_items(value: object | None) -> list[object]:
    """Return the items from one runtime iterable."""
    if value is None:
        return []
    if isinstance(value, (str, bytes, bytearray)):
        return []
    if isinstance(value, Mapping):
        return list(value.values())
    if isinstance(value, Iterable):
        return list(value)
    return []


def candidate_collection_items(
    value: object,
    attribute_name: str,
) -> list[list[object]]:
    """Return the collection views that should be inspected for runtime objects."""
    collection_candidates: list[list[object]] = []
    if isinstance(value, Mapping):
        collection_candidates.append(list(value.values()))
    elif not isinstance(value, (str, bytes, bytearray)):
        direct_items = coerce_iterable_items(value)
        if direct_items:
            collection_candidates.append(direct_items)
    attribute_value = get_attribute(value, attribute_name)
    attribute_items = coerce_iterable_items(attribute_value)
    if attribute_items or attribute_value is not None:
        collection_candidates.append(attribute_items)
    return collection_candidates


def coerce_shape(shape_value: object, *, context: str) -> tuple[int, ...]:
    """Convert one runtime shape object into a tuple of dimensions."""
    if shape_value is None:
        raise SerializationError(
            f"Live import could not determine the shape for tensor '{context}'."
        )
    if isinstance(shape_value, (str, bytes, bytearray)):
        raise SerializationError(
            f"Live import recovered an invalid shape for tensor '{context}'."
        )
    if not isinstance(shape_value, Iterable):
        raise SerializationError(
            f"Live import recovered an invalid shape for tensor '{context}'."
        )
    shape: list[int] = []
    for raw_dimension in shape_value:
        dimension = coerce_optional_axis_position(raw_dimension)
        if dimension is None:
            raise SerializationError(
                f"Live import recovered a non-numeric dimension for tensor '{context}'."
            )
        shape.append(dimension)
    return tuple(shape)


def shape_from_runtime_tensor(data: object | None) -> object | None:
    """Return the shape attribute from runtime tensor data when present."""
    if data is None:
        return None
    return get_attribute(data, "shape")


def coerce_label_tuple(
    labels_value: object,
    *,
    expected_length: int,
    fallback_prefix: str,
    allow_fallback: bool,
) -> tuple[str, ...]:
    """Convert runtime labels into a stable tuple of strings."""
    if labels_value is None:
        if allow_fallback:
            return tuple(
                f"{fallback_prefix}_{index + 1}" for index in range(expected_length)
            )
        raise SerializationError(
            "Live import requires runtime indices or axis names for the selected object."
        )
    if isinstance(labels_value, (str, bytes, bytearray)):
        raise SerializationError(
            "Live import requires runtime indices or axis names to be sequences."
        )
    if not isinstance(labels_value, Iterable):
        raise SerializationError(
            "Live import requires runtime indices or axis names to be sequences."
        )
    labels = [str(label) for label in labels_value]
    if len(labels) != expected_length:
        if allow_fallback:
            return tuple(
                f"{fallback_prefix}_{index + 1}" for index in range(expected_length)
            )
        raise SerializationError(
            "Live import requires tensor shapes to match their runtime labels."
        )
    return tuple(labels)


def coerce_optional_axis_position(value: object) -> int | None:
    """Convert one runtime axis position into an ``int`` when possible."""
    if value is None or isinstance(value, bool):
        return None
    axis_position: int | None = None
    if isinstance(value, int):
        axis_position = value
    elif isinstance(value, (str, bytes, bytearray)):
        try:
            axis_position = int(value)
        except ValueError:
            return None
    elif hasattr(value, "__index__"):
        try:
            axis_position = operator.index(cast(SupportsIndex, value))
        except TypeError:
            return None
    elif hasattr(value, "__int__"):
        try:
            axis_position = int(cast(SupportsInt, value))
        except (TypeError, ValueError):
            return None
    if axis_position is None or axis_position < 0:
        return None
    return axis_position
