"""Periodic-mode enums and coercion helpers for graph models."""

from __future__ import annotations

from enum import StrEnum

from ..io._payloads import coerce_int, coerce_string


class LinearPeriodicCellName(StrEnum):
    """Named cells available in the linear periodic-chain editor mode."""

    INITIAL = "initial"
    PERIODIC = "periodic"
    FINAL = "final"


class LinearPeriodicTensorRole(StrEnum):
    """Special editor-only roles used by virtual boundary tensors."""

    PREVIOUS = "previous"
    NEXT = "next"


class GridPeriodicCellName(StrEnum):
    """Named cells available in the bidimensional periodic-grid editor mode."""

    TOP_LEFT = "top_left"
    TOP = "top"
    TOP_RIGHT = "top_right"
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM = "bottom"
    BOTTOM_RIGHT = "bottom_right"


class GridPeriodicTensorRole(StrEnum):
    """Special editor-only roles used by 2D virtual boundary tensors."""

    UP = "up"
    RIGHT = "right"
    DOWN = "down"
    LEFT = "left"


class TreePeriodicCellName(StrEnum):
    """Named cells available in the tree periodic editor mode."""

    ROOT = "root"
    BRANCH = "branch"
    LEAF = "leaf"


class TreePeriodicTensorRole(StrEnum):
    """Special editor-only roles used by tree virtual boundary tensors."""

    PARENT = "parent"
    CHILD = "child"


def coerce_linear_periodic_cell_name(
    value: object,
    *,
    field_name: str,
) -> LinearPeriodicCellName:
    """Coerce a serialized value to a valid linear periodic cell name."""
    try:
        return LinearPeriodicCellName(coerce_string(value, field_name=field_name))
    except ValueError as exc:
        raise TypeError(
            f"{field_name} must be a valid linear periodic cell name."
        ) from exc


def coerce_linear_periodic_tensor_role(
    value: object,
    *,
    field_name: str,
) -> LinearPeriodicTensorRole | None:
    """Coerce a serialized value to a valid linear periodic tensor role."""
    if value is None:
        return None
    try:
        return LinearPeriodicTensorRole(coerce_string(value, field_name=field_name))
    except ValueError as exc:
        raise TypeError(
            f"{field_name} must be a valid linear periodic tensor role."
        ) from exc


def coerce_grid_periodic_cell_name(
    value: object,
    *,
    field_name: str,
) -> GridPeriodicCellName:
    """Coerce a serialized value to a valid grid periodic cell name."""
    try:
        return GridPeriodicCellName(coerce_string(value, field_name=field_name))
    except ValueError as exc:
        raise TypeError(
            f"{field_name} must be a valid grid periodic cell name."
        ) from exc


def coerce_grid_periodic_tensor_role(
    value: object,
    *,
    field_name: str,
) -> GridPeriodicTensorRole | None:
    """Coerce a serialized value to a valid grid periodic tensor role."""
    if value is None:
        return None
    try:
        return GridPeriodicTensorRole(coerce_string(value, field_name=field_name))
    except ValueError as exc:
        raise TypeError(
            f"{field_name} must be a valid grid periodic tensor role."
        ) from exc


def coerce_tree_periodic_cell_name(
    value: object,
    *,
    field_name: str,
) -> TreePeriodicCellName:
    """Coerce a serialized value to a valid tree periodic cell name."""
    try:
        return TreePeriodicCellName(coerce_string(value, field_name=field_name))
    except ValueError as exc:
        raise TypeError(
            f"{field_name} must be a valid tree periodic cell name."
        ) from exc


def coerce_tree_periodic_tensor_role(
    value: object,
    *,
    field_name: str,
) -> TreePeriodicTensorRole | None:
    """Coerce a serialized value to a valid tree periodic tensor role."""
    if value is None:
        return None
    try:
        return TreePeriodicTensorRole(coerce_string(value, field_name=field_name))
    except ValueError as exc:
        raise TypeError(
            f"{field_name} must be a valid tree periodic tensor role."
        ) from exc


def coerce_optional_int(value: object, *, field_name: str) -> int | None:
    """Coerce an optional integer payload field."""
    if value is None:
        return None
    return coerce_int(value, field_name=field_name)
