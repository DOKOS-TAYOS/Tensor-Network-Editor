"""Geometry data models shared by canvas and contraction views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from ._payloads import coerce_float
from .types import JSONValue


@dataclass(slots=True)
class CanvasPosition:
    """A 2D position on the editor canvas."""

    x: float = 120.0
    y: float = 120.0

    def to_dict(self) -> dict[str, JSONValue]:
        """Serialize the position to a JSON-compatible mapping."""
        return {"x": self.x, "y": self.y}

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> Self:
        """Build a position from a serialized mapping."""
        return cls(
            x=coerce_float(payload["x"], field_name="x"),
            y=coerce_float(payload["y"], field_name="y"),
        )


@dataclass(slots=True)
class TensorSize:
    """The rendered width and height of a tensor node."""

    width: float = 180.0
    height: float = 108.0

    def to_dict(self) -> dict[str, JSONValue]:
        """Serialize the size to a JSON-compatible mapping."""
        return {"width": self.width, "height": self.height}

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> Self:
        """Build a size from a serialized mapping."""
        return cls(
            width=coerce_float(payload["width"], field_name="width"),
            height=coerce_float(payload["height"], field_name="height"),
        )
