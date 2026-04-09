from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from tensor_network_editor.cli import main
from tensor_network_editor.diffing import DiffEntityChanges, SpecDiffResult
from tensor_network_editor.linting import LintIssue, LintReport
from tensor_network_editor.models import EngineName, NetworkSpec, ValidationIssue


def test_main_uses_expected_defaults() -> None:
    with patch("tensor_network_editor.cli.launch_tensor_network_editor") as launch_mock:
        exit_code = main([])

    assert exit_code == 0
    launch_mock.assert_called_once()
    assert launch_mock.call_args.kwargs == {
        "initial_spec": None,
        "default_engine": EngineName.TENSORKROWCH,
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


def test_edit_subcommand_matches_legacy_mode(sample_spec: NetworkSpec) -> None:
    with (
        patch(
            "tensor_network_editor.cli.load_spec", return_value=sample_spec
        ) as load_mock,
        patch("tensor_network_editor.cli.launch_tensor_network_editor") as launch_mock,
    ):
        exit_code = main(["edit", "--load", "saved-network.json", "--no-browser"])

    assert exit_code == 0
    load_mock.assert_called_once_with("saved-network.json")
    assert launch_mock.call_args.kwargs["initial_spec"] is sample_spec
    assert launch_mock.call_args.kwargs["open_browser"] is False


def test_validate_subcommand_returns_json_and_exit_code_1(
    sample_spec: NetworkSpec,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with (
        patch(
            "tensor_network_editor.cli.load_spec", return_value=sample_spec
        ) as load_mock,
        patch(
            "tensor_network_editor.cli.validate_spec",
            return_value=[
                ValidationIssue(
                    code="bad-name",
                    message="Tensor name is invalid.",
                    path="tensors.tensor_a.name",
                )
            ],
        ),
    ):
        exit_code = main(["validate", "saved-network.json", "--format", "json"])

    assert exit_code == 1
    load_mock.assert_called_once_with("saved-network.json")
    payload = json.loads(capsys.readouterr().out)
    assert payload["issues"][0]["code"] == "bad-name"


def test_lint_subcommand_fails_on_warning_when_requested(
    sample_spec: NetworkSpec,
) -> None:
    with (
        patch(
            "tensor_network_editor.cli.load_spec_for_lint", return_value=sample_spec
        ) as load_mock,
        patch(
            "tensor_network_editor.cli.lint_spec",
            return_value=LintReport(
                issues=[
                    LintIssue(
                        severity="warning",
                        code="suspicious-open-index",
                        message="Open leg looks suspicious.",
                        path="tensors.tensor_a.indices.tensor_a_i",
                    )
                ]
            ),
        ),
    ):
        exit_code = main(
            ["lint", "saved-network.json", "--fail-on", "warning", "--format", "json"]
        )

    assert exit_code == 1
    load_mock.assert_called_once_with("saved-network.json")


def test_analyze_subcommand_prints_json_report(
    sample_spec: NetworkSpec,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with patch("tensor_network_editor.cli.load_spec", return_value=sample_spec):
        exit_code = main(["analyze", "saved-network.json", "--format", "json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["network"]["tensor_count"] == 2
    assert "contraction" in payload


def test_export_subcommand_calls_generate_code_with_requested_output(
    sample_spec: NetworkSpec,
) -> None:
    with (
        patch(
            "tensor_network_editor.cli.load_spec", return_value=sample_spec
        ) as load_mock,
        patch("tensor_network_editor.cli.generate_code") as generate_mock,
    ):
        exit_code = main(
            [
                "export",
                "saved-network.json",
                "--engine",
                EngineName.EINSUM_NUMPY.value,
                "--output",
                "generated.py",
            ]
        )

    assert exit_code == 0
    load_mock.assert_called_once_with("saved-network.json")
    generate_mock.assert_called_once()
    assert generate_mock.call_args.kwargs["path"] == "generated.py"
    assert generate_mock.call_args.kwargs["print_code"] is False


def test_diff_subcommand_prints_json(
    sample_spec: NetworkSpec,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with (
        patch(
            "tensor_network_editor.cli.load_spec",
            side_effect=[sample_spec, sample_spec],
        ) as load_mock,
        patch(
            "tensor_network_editor.cli.diff_specs",
            return_value=SpecDiffResult(
                tensor=DiffEntityChanges(changed=["tensor_a"]),
            ),
        ),
    ):
        exit_code = main(["diff", "before.json", "after.json", "--format", "json"])

    assert exit_code == 0
    assert load_mock.call_count == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["tensor"]["changed"] == ["tensor_a"]


def test_template_list_subcommand_prints_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["template", "list", "--format", "json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert list(payload) == ["mps", "mpo", "peps_2x2", "mera", "binary_tree"]


def test_template_build_subcommand_prints_json_when_no_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "template",
            "build",
            "mps",
            "--graph-size",
            "5",
            "--bond-dimension",
            "7",
            "--physical-dimension",
            "11",
            "--format",
            "json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["network"]["name"] == "MPS (5 sites)"
