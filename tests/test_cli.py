from __future__ import annotations

from unittest.mock import patch

from tensor_network_editor.cli import main
from tensor_network_editor.models import EngineName, NetworkSpec


def test_main_uses_expected_defaults() -> None:
    with patch("tensor_network_editor.cli.launch_tensor_network_editor") as launch_mock:
        exit_code = main([])

    assert exit_code == 0
    launch_mock.assert_called_once()
    assert launch_mock.call_args.kwargs == {
        "initial_spec": None,
        "default_engine": EngineName.TENSORNETWORK,
        "open_browser": True,
        "print_code": False,
        "code_path": None,
    }


def test_main_loads_spec_and_passes_output_flags(sample_spec: NetworkSpec) -> None:
    with (
        patch(
            "tensor_network_editor.cli.load_spec", return_value=sample_spec
        ) as load_mock,
        patch("tensor_network_editor.cli.launch_tensor_network_editor") as launch_mock,
    ):
        exit_code = main(
            [
                "--engine",
                EngineName.EINSUM_NUMPY.value,
                "--load",
                "saved-network.json",
                "--save-code",
                "generated.py",
                "--print-code",
                "--no-browser",
            ]
        )

    assert exit_code == 0
    load_mock.assert_called_once_with("saved-network.json")
    launch_mock.assert_called_once()
    assert launch_mock.call_args.kwargs == {
        "initial_spec": sample_spec,
        "default_engine": EngineName.EINSUM_NUMPY,
        "open_browser": False,
        "print_code": True,
        "code_path": "generated.py",
    }


def test_main_returns_130_on_keyboard_interrupt() -> None:
    with patch(
        "tensor_network_editor.cli.launch_tensor_network_editor",
        side_effect=KeyboardInterrupt,
    ) as launch_mock:
        exit_code = main([])

    assert exit_code == 130
    launch_mock.assert_called_once()
