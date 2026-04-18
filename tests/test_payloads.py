from __future__ import annotations

import pytest

from tensor_network_editor._payloads import (
    coerce_float,
    coerce_int,
    coerce_metadata,
    new_identifier,
    require_dict,
    require_list,
)


def test_new_identifier_uses_prefix_and_unique_suffix() -> None:
    first = new_identifier("tensor")
    second = new_identifier("tensor")

    assert first.startswith("tensor_")
    assert second.startswith("tensor_")
    assert first != second


@pytest.mark.parametrize(
    ("value", "expected"),
    [(1, 1.0), (2.5, 2.5), ("3.5", 3.5)],
)
def test_coerce_float_accepts_numeric_values(value: object, expected: float) -> None:
    assert coerce_float(value, field_name="value") == expected


def test_coerce_float_rejects_booleans() -> None:
    with pytest.raises(TypeError, match="value must be a number"):
        coerce_float(True, field_name="value")


@pytest.mark.parametrize("value", ["not-a-number", "nan", "inf", "-inf"])
def test_coerce_float_rejects_non_finite_or_invalid_strings(value: str) -> None:
    with pytest.raises(TypeError, match="value must be a number"):
        coerce_float(value, field_name="value")


def test_coerce_float_rejects_values_that_overflow_float_conversion() -> None:
    with pytest.raises(TypeError, match="value must be a number"):
        coerce_float(10**10_000, field_name="value")


@pytest.mark.parametrize(
    ("value", "expected"),
    [(1, 1), (2.0, 2), ("3", 3)],
)
def test_coerce_int_accepts_integer_like_values(value: object, expected: int) -> None:
    assert coerce_int(value, field_name="value") == expected


def test_coerce_int_rejects_booleans() -> None:
    with pytest.raises(TypeError, match="value must be an integer"):
        coerce_int(False, field_name="value")


def test_coerce_int_preserves_large_integer_string_precision() -> None:
    value = "123456789012345678901234567890"

    assert coerce_int(value, field_name="value") == int(value)


@pytest.mark.parametrize("value", [2.5, "3.5"])
def test_coerce_int_rejects_non_integral_numeric_values(value: object) -> None:
    with pytest.raises(TypeError, match="value must be an integer"):
        coerce_int(value, field_name="value")


def test_require_dict_rejects_non_mapping_values() -> None:
    with pytest.raises(TypeError, match="payload must be an object"):
        require_dict([], field_name="payload")


def test_require_list_rejects_non_list_values() -> None:
    with pytest.raises(TypeError, match="payload must be a list"):
        require_list({}, field_name="payload")


def test_coerce_metadata_returns_plain_dictionary() -> None:
    payload = {"engine": "quimb", "enabled": True}

    assert coerce_metadata(payload, field_name="metadata") == payload


def test_coerce_metadata_rejects_non_string_keys() -> None:
    with pytest.raises(TypeError, match="metadata must contain string keys"):
        coerce_metadata({1: "value"}, field_name="metadata")


def test_coerce_metadata_rejects_non_json_compatible_values() -> None:
    with pytest.raises(TypeError, match="metadata must contain JSON-compatible values"):
        coerce_metadata({"object": object()}, field_name="metadata")


def test_coerce_metadata_rejects_non_finite_float_values() -> None:
    with pytest.raises(TypeError, match="metadata must contain JSON-compatible values"):
        coerce_metadata({"score": float("nan")}, field_name="metadata")
