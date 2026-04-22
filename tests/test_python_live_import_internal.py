from __future__ import annotations

from importlib import import_module

from tensor_network_editor.models import TensorDataMode


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


def test_python_live_import_runtime_helpers_coerce_axis_positions() -> None:
    runtime_module = import_module(
        "tensor_network_editor.internal.io._python_live_import_runtime"
    )

    assert runtime_module.coerce_optional_axis_position("3") == 3
    assert runtime_module.coerce_optional_axis_position(-1) is None
