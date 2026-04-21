from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Self, TypeAlias, cast

from ...types import JSONValue
from ..io._payloads import coerce_float, coerce_string, require_dict

TensorNumericLiteral: TypeAlias = int | float | list["TensorNumericLiteral"]


class TensorDataMode(StrEnum):
    """Supported deterministic tensor-data definitions."""

    ONES = "ones"
    FILL = "fill"
    LITERAL = "literal"


@dataclass(slots=True)
class TensorDataSpec:
    """Describe how one tensor should be initialized in generated code."""

    mode: TensorDataMode
    fill_value: float | None = None
    values: TensorNumericLiteral | None = None

    def __post_init__(self) -> None:
        """Reject inconsistent mode-specific fields."""
        if self.mode is TensorDataMode.ONES:
            if self.fill_value is not None or self.values is not None:
                raise ValueError("TensorDataMode.ONES does not accept extra values.")
            return
        if self.mode is TensorDataMode.FILL:
            if self.fill_value is None or self.values is not None:
                raise ValueError(
                    "TensorDataMode.FILL requires 'fill_value' and forbids 'values'."
                )
            if not math.isfinite(self.fill_value):
                raise ValueError("TensorDataMode.FILL requires a finite fill_value.")
            return
        if self.values is None or self.fill_value is not None:
            raise ValueError(
                "TensorDataMode.LITERAL requires 'values' and forbids 'fill_value'."
            )

    def to_dict(self) -> dict[str, JSONValue]:
        """Serialize the tensor-data description to a JSON-compatible mapping."""
        payload: dict[str, JSONValue] = {"mode": self.mode.value}
        if self.mode is TensorDataMode.FILL:
            payload["fill_value"] = self.fill_value
        if self.mode is TensorDataMode.LITERAL:
            payload["values"] = cast(JSONValue, self.values)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        """Build tensor-data settings from a serialized mapping."""
        data_payload = require_dict(payload, field_name="tensor_data")
        mode = TensorDataMode(
            coerce_string(data_payload["mode"], field_name="tensor_data.mode")
        )
        if mode is TensorDataMode.ONES:
            return cls(mode=mode)
        if mode is TensorDataMode.FILL:
            return cls(
                mode=mode,
                fill_value=coerce_float(
                    data_payload["fill_value"],
                    field_name="tensor_data.fill_value",
                ),
            )
        return cls(
            mode=mode,
            values=_coerce_tensor_numeric_literal(
                data_payload["values"],
                field_name="tensor_data.values",
            ),
        )


def _coerce_tensor_numeric_literal(
    value: object, *, field_name: str
) -> TensorNumericLiteral:
    """Require a nested list/scalar tree containing only finite numbers."""
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must contain only numeric values.")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError(f"{field_name} must contain only numeric values.")
        return value
    if isinstance(value, list):
        return [
            _coerce_tensor_numeric_literal(item, field_name=field_name)
            for item in value
        ]
    raise TypeError(f"{field_name} must contain only numeric values.")
