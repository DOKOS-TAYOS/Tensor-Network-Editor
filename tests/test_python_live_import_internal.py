from __future__ import annotations

from importlib import import_module

import pytest

from tensor_network_editor.models import TensorDataMode


class _ScalarLike:
    def __init__(self, value: float) -> None:
        self._value = value

    def item(self) -> float:
        return self._value


class _ArrayLike:
    def __init__(self, values: list[list[float]]) -> None:
        self._values = values

    def tolist(self) -> list[list[float]]:
        return self._values


class _ComplexArrayLike:
    def __init__(self, values: list[list[complex]]) -> None:
        self._values = values

    def tolist(self) -> list[list[complex]]:
        return self._values


def test_python_live_import_tensor_data_helpers_preserve_literal_payloads() -> None:
    tensor_data_module = import_module(
        "tensor_network_editor.internal.io._python_live_import_tensor_data"
    )

    tensor_data, warning = tensor_data_module.lower_runtime_tensor_data(
        [[1.0, 2.0], [3.0, 4.0]],
        shape=(2, 2),
        tensor_name="A",
    )

    assert warning is None
    assert tensor_data is not None
    assert tensor_data.mode is TensorDataMode.LITERAL
    assert tensor_data.values == [[1.0, 2.0], [3.0, 4.0]]


def test_python_live_import_tensor_data_helpers_preserve_complex_literals() -> None:
    tensor_data_module = import_module(
        "tensor_network_editor.internal.io._python_live_import_tensor_data"
    )

    scalar_literal = tensor_data_module.coerce_tensor_literal(1.5 - 2j)
    tensor_data, warning = tensor_data_module.lower_runtime_tensor_data(
        _ComplexArrayLike([[1.0 + 2.0j, 3.0 - 4.0j]]),
        shape=(1, 2),
        tensor_name="ComplexA",
    )

    assert scalar_literal == {"real": 1.5, "imag": -2.0}
    assert warning is None
    assert tensor_data is not None
    assert tensor_data.mode is TensorDataMode.LITERAL
    assert tensor_data.values == [
        [{"real": 1.0, "imag": 2.0}, {"real": 3.0, "imag": -4.0}]
    ]


def test_python_live_import_tensor_data_warning_mentions_complex_values() -> None:
    tensor_data_module = import_module(
        "tensor_network_editor.internal.io._python_live_import_tensor_data"
    )

    tensor_data, warning = tensor_data_module.lower_runtime_tensor_data(
        ["not numeric"],
        shape=(1,),
        tensor_name="Bad",
    )

    assert tensor_data is None
    assert warning is not None
    assert "finite real or complex tensor values" in warning


def test_python_live_import_tensor_data_helpers_accept_numpy_like_values() -> None:
    tensor_data_module = import_module(
        "tensor_network_editor.internal.io._python_live_import_tensor_data"
    )

    scalar_literal = tensor_data_module.coerce_tensor_literal(_ScalarLike(3.5))
    tensor_data, warning = tensor_data_module.lower_runtime_tensor_data(
        _ArrayLike([[1.0, 2.0], [3.0, 4.0]]),
        shape=(2, 2),
        tensor_name="B",
    )

    assert scalar_literal == 3.5
    assert warning is None
    assert tensor_data is not None
    assert tensor_data.mode is TensorDataMode.LITERAL
    assert tensor_data.values == [[1.0, 2.0], [3.0, 4.0]]


def test_python_live_import_tensor_data_helpers_preserve_large_uniform_ones() -> None:
    tensor_data_module = import_module(
        "tensor_network_editor.internal.io._python_live_import_tensor_data"
    )

    tensor_data, warning = tensor_data_module.lower_runtime_tensor_data(
        [[1.0] * 65 for _ in range(65)],
        shape=(65, 65),
        tensor_name="LargeOnes",
    )

    assert warning is None
    assert tensor_data is not None
    assert tensor_data.mode is TensorDataMode.ONES


def test_python_live_import_runner_rejects_non_live_source_profiles_before_exec() -> (
    None
):
    runner_module = import_module(
        "tensor_network_editor.internal.io._python_live_import_runner"
    )

    with pytest.raises(
        ValueError,
        match="supports only 'auto', 'quimb', or 'tensornetwork'",
    ):
        runner_module._run_request(
            {
                "code": "raise RuntimeError('should not execute')",
                "filename": "demo_live_import.py",
                "source_profile": "generated",
            }
        )


def test_python_live_import_runtime_helpers_coerce_axis_positions() -> None:
    runtime_module = import_module(
        "tensor_network_editor.internal.io._python_live_import_runtime"
    )

    assert runtime_module.coerce_optional_axis_position("3") == 3
    assert runtime_module.coerce_optional_axis_position(-1) is None
