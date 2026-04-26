from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Self, TypeAlias, TypedDict, cast

from ...types import JSONValue
from ..io._payloads import coerce_float, coerce_int, coerce_string, require_dict


class TensorComplexLiteral(TypedDict):
    """Portable JSON representation for one complex scalar."""

    real: float
    imag: float


TensorScalarLiteral: TypeAlias = int | float | TensorComplexLiteral
TensorNumericLiteral: TypeAlias = TensorScalarLiteral | list["TensorNumericLiteral"]


class TensorDataMode(StrEnum):
    """Supported deterministic tensor-data definitions."""

    ZEROS = "zeros"
    ONES = "ones"
    FILL = "fill"
    LITERAL = "literal"
    IDENTITY = "identity"
    COPY = "copy"
    RANDOM = "random"


class TensorDataDType(StrEnum):
    """Portable dtype labels supported by generated tensor initializers."""

    FLOAT32 = "float32"
    FLOAT64 = "float64"
    COMPLEX64 = "complex64"
    COMPLEX128 = "complex128"


class TensorDataRandomDistribution(StrEnum):
    """Supported seeded random tensor initializer distributions."""

    NORMAL = "normal"
    UNIFORM = "uniform"


@dataclass(slots=True)
class TensorDataSpec:
    """Describe how one tensor should be initialized in generated code."""

    mode: TensorDataMode
    fill_value: TensorScalarLiteral | None = None
    values: TensorNumericLiteral | None = None
    dtype: TensorDataDType | None = None
    seed: int | None = None
    distribution: TensorDataRandomDistribution | None = None

    def __post_init__(self) -> None:
        """Reject inconsistent mode-specific fields."""
        self.mode = TensorDataMode(self.mode)
        if self.dtype is not None:
            self.dtype = TensorDataDType(self.dtype)
        if self.distribution is not None:
            self.distribution = TensorDataRandomDistribution(self.distribution)

        if self.mode in {
            TensorDataMode.ZEROS,
            TensorDataMode.ONES,
            TensorDataMode.IDENTITY,
            TensorDataMode.COPY,
        }:
            if (
                self.fill_value is not None
                or self.values is not None
                or self.seed is not None
                or self.distribution is not None
            ):
                raise ValueError(
                    f"TensorDataMode.{self.mode.name} does not accept extra values."
                )
            return
        if self.mode is TensorDataMode.FILL:
            if (
                self.fill_value is None
                or self.values is not None
                or self.seed is not None
                or self.distribution is not None
            ):
                raise ValueError(
                    "TensorDataMode.FILL requires 'fill_value' and forbids "
                    "'values', 'seed', and 'distribution'."
                )
            self.fill_value = _coerce_tensor_scalar_literal(
                self.fill_value,
                field_name="tensor_data.fill_value",
            )
            return
        if self.mode is TensorDataMode.RANDOM:
            if self.fill_value is not None or self.values is not None:
                raise ValueError(
                    "TensorDataMode.RANDOM forbids 'fill_value' and 'values'."
                )
            if self.seed is None:
                self.seed = 0
            if isinstance(self.seed, bool) or not isinstance(self.seed, int):
                raise ValueError("TensorDataMode.RANDOM requires an integer seed.")
            if self.seed < 0:
                raise ValueError("TensorDataMode.RANDOM requires a non-negative seed.")
            if self.distribution is None:
                self.distribution = TensorDataRandomDistribution.NORMAL
            return
        if (
            self.values is None
            or self.fill_value is not None
            or self.seed is not None
            or self.distribution is not None
        ):
            raise ValueError(
                "TensorDataMode.LITERAL requires 'values' and forbids "
                "'fill_value', 'seed', and 'distribution'."
            )

    def to_dict(self) -> dict[str, JSONValue]:
        """Serialize the tensor-data description to a JSON-compatible mapping."""
        payload: dict[str, JSONValue] = {"mode": self.mode.value}
        if self.dtype is not None:
            payload["dtype"] = self.dtype.value
        if self.mode is TensorDataMode.FILL:
            payload["fill_value"] = cast(JSONValue, self.fill_value)
        if self.mode is TensorDataMode.LITERAL:
            payload["values"] = cast(JSONValue, self.values)
        if self.mode is TensorDataMode.RANDOM:
            payload["seed"] = self.seed
            payload["distribution"] = cast(
                TensorDataRandomDistribution,
                self.distribution,
            ).value
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        """Build tensor-data settings from a serialized mapping."""
        data_payload = require_dict(payload, field_name="tensor_data")
        mode = TensorDataMode(
            coerce_string(data_payload["mode"], field_name="tensor_data.mode")
        )
        raw_dtype = data_payload.get("dtype")
        dtype = (
            TensorDataDType(coerce_string(raw_dtype, field_name="tensor_data.dtype"))
            if raw_dtype is not None
            else None
        )
        if mode in {
            TensorDataMode.ZEROS,
            TensorDataMode.ONES,
            TensorDataMode.IDENTITY,
            TensorDataMode.COPY,
        }:
            return cls(mode=mode, dtype=dtype)
        if mode is TensorDataMode.FILL:
            return cls(
                mode=mode,
                dtype=dtype,
                fill_value=_coerce_tensor_scalar_literal(
                    data_payload["fill_value"],
                    field_name="tensor_data.fill_value",
                ),
            )
        if mode is TensorDataMode.RANDOM:
            raw_distribution = data_payload.get("distribution")
            distribution = (
                TensorDataRandomDistribution(
                    coerce_string(
                        raw_distribution,
                        field_name="tensor_data.distribution",
                    )
                )
                if raw_distribution is not None
                else None
            )
            return cls(
                mode=mode,
                dtype=dtype,
                seed=coerce_int(data_payload.get("seed", 0), field_name="seed"),
                distribution=distribution,
            )
        return cls(
            mode=mode,
            dtype=dtype,
            values=_coerce_tensor_numeric_literal(
                data_payload["values"],
                field_name="tensor_data.values",
            ),
        )


def _coerce_tensor_scalar_literal(
    value: object, *, field_name: str
) -> TensorScalarLiteral:
    """Require one finite real or portable complex scalar."""
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must contain only numeric values.")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError(f"{field_name} must contain only numeric values.")
        return value
    if isinstance(value, Mapping):
        value_payload = require_dict(value, field_name=field_name)
        return {
            "real": coerce_float(
                value_payload["real"], field_name=f"{field_name}.real"
            ),
            "imag": coerce_float(
                value_payload["imag"], field_name=f"{field_name}.imag"
            ),
        }
    raise TypeError(f"{field_name} must contain only numeric values.")


def _coerce_tensor_numeric_literal(
    value: object, *, field_name: str
) -> TensorNumericLiteral:
    """Require a nested list/scalar tree containing only finite numbers."""
    if isinstance(value, bool) or isinstance(value, (int, float, Mapping)):
        return _coerce_tensor_scalar_literal(value, field_name=field_name)
    if isinstance(value, list):
        return [
            _coerce_tensor_numeric_literal(item, field_name=field_name)
            for item in value
        ]
    raise TypeError(f"{field_name} must contain only numeric values.")
