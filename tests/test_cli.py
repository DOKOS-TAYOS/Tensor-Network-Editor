from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest

from tensor_network_editor.cli import build_command_parser, main
from tensor_network_editor.diffing import DiffEntityChanges, SpecDiffResult
from tensor_network_editor.internal.analysis._contraction_analysis_types import (
    AutomaticContractionPlanAnalysis,
    AutomaticContractionSummary,
    ContractionAnalysisResult,
    ContractionComparison,
    ManualContractionPlanAnalysis,
    ManualContractionSummary,
)
from tensor_network_editor.internal.cli._cli_formatters import (
    _coerce_int,
    _format_label_list,
    _format_shape,
)
from tensor_network_editor.internal.models._headless_models import (
    NetworkSummary,
    SemanticDiffEntry,
    SemanticFieldChange,
    SemanticSpecDiffResult,
    SpecAnalysisReport,
)
from tensor_network_editor.linting import LintIssue, LintReport
from tensor_network_editor.models import EngineName, NetworkSpec, ValidationIssue

REPO_ROOT = Path(__file__).resolve().parents[1]


def build_analysis_report(memory_dtype: str = "float64") -> SpecAnalysisReport:
    return SpecAnalysisReport(
        network=NetworkSummary(
            tensor_count=4,
            edge_count=3,
            group_count=0,
            note_count=0,
            open_index_count=2,
        ),
        contraction=ContractionAnalysisResult(
            network_output_shape=(2, 2),
            manual=ManualContractionPlanAnalysis(
                status="complete",
                steps=[],
                summary=ManualContractionSummary(
                    total_estimated_flops=1600,
                    total_estimated_macs=800,
                    peak_intermediate_size=100,
                    final_shape=(2, 2),
                    completion_status="complete",
                    remaining_operand_ids=("step_abcd",),
                ),
            ),
            automatic_full=AutomaticContractionPlanAnalysis(
                status="complete",
                steps=[],
                summary=AutomaticContractionSummary(
                    total_estimated_flops=1224,
                    total_estimated_macs=612,
                    peak_intermediate_size=6,
                ),
            ),
            automatic_future=AutomaticContractionPlanAnalysis(
                status="complete",
                steps=[],
                summary=AutomaticContractionSummary(
                    total_estimated_flops=140,
                    total_estimated_macs=70,
                    peak_intermediate_size=14,
                ),
            ),
            automatic_past=AutomaticContractionPlanAnalysis(
                status="complete",
                steps=[],
                summary=AutomaticContractionSummary(
                    total_estimated_flops=576,
                    total_estimated_macs=288,
                    peak_intermediate_size=12,
                ),
            ),
            comparisons={
                "manual_vs_automatic_full": ContractionComparison(
                    status="complete",
                    baseline_label="manual",
                    candidate_label="automatic_full",
                    memory_dtype=memory_dtype,
                    baseline_peak_intermediate_bytes=800,
                    candidate_peak_intermediate_bytes=48,
                    delta_total_estimated_flops=-376,
                    delta_total_estimated_macs=-188,
                    delta_peak_intermediate_size=-94,
                    delta_peak_intermediate_bytes=-752,
                    baseline_peak_step_id="step_bcd",
                    candidate_peak_step_id="auto_full_step_1",
                    baseline_bottleneck_labels=("x", "y", "z"),
                    candidate_bottleneck_labels=("i", "j"),
                ),
                "manual_subtrees_vs_automatic_past": ContractionComparison(
                    status="complete",
                    baseline_label="manual_subtrees",
                    candidate_label="automatic_past",
                    memory_dtype=memory_dtype,
                    baseline_peak_intermediate_bytes=192,
                    candidate_peak_intermediate_bytes=96,
                    delta_total_estimated_flops=-24,
                    delta_total_estimated_macs=-12,
                    delta_peak_intermediate_size=-12,
                    delta_peak_intermediate_bytes=-96,
                    baseline_peak_step_id="step_ab",
                    candidate_peak_step_id="step_ab",
                    baseline_bottleneck_labels=("x", "y"),
                    candidate_bottleneck_labels=("x",),
                ),
            },
            automatic_strategy="greedy",
        ),
    )


def test_main_requires_a_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    with patch("tensor_network_editor.cli.launch_tensor_network_editor") as launch_mock:
        exit_code = main([])

    assert exit_code == 2
    launch_mock.assert_not_called()
    assert "the following arguments are required: command" in capsys.readouterr().err


def test_global_log_level_is_accepted_before_subcommand() -> None:
    parser = build_command_parser()

    parsed_args = parser.parse_args(["--log-level", "debug", "edit", "--no-browser"])

    assert parsed_args.log_level == "debug"
    assert parsed_args.command == "edit"
    assert parsed_args.no_browser is True


def test_cli_modules_pass_targeted_mypy_check() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "src/tensor_network_editor/internal/cli/_cli_handlers.py",
            "src/tensor_network_editor/cli.py",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_format_shape_accepts_list_and_rejects_invalid_entries() -> None:
    assert _format_shape([2, "3", 4.0]) == "2 x 3 x 4"
    assert _format_shape([2, "bad"]) == "n/a"


def test_format_label_list_accepts_list_sequences() -> None:
    assert _format_label_list(["x", "y"]) == "x, y"


def test_edit_subcommand_uses_expected_defaults() -> None:
    with patch("tensor_network_editor.cli.launch_tensor_network_editor") as launch_mock:
        exit_code = main(["edit"])

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
                "edit",
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
        "code_path": Path("saved-network.json").resolve().parent / "generated.py",
        "template_catalog_path": Path("saved-network.json").resolve().parent
        / ".tensor-network-editor"
        / "templates.json",
    }


def test_edit_subcommand_uses_loaded_spec_directory_for_template_catalog(
    tmp_path: Path,
    sample_spec: NetworkSpec,
) -> None:
    design_path = tmp_path / "project_a" / "saved-network.json"
    design_path.parent.mkdir(parents=True)
    design_path.write_text("{}", encoding="utf-8")

    with (
        patch(
            "tensor_network_editor.cli.load_spec", return_value=sample_spec
        ) as load_mock,
        patch("tensor_network_editor.cli.launch_tensor_network_editor") as launch_mock,
    ):
        exit_code = main(["edit", "--load", str(design_path), "--no-browser"])

    assert exit_code == 0
    load_mock.assert_called_once_with(str(design_path))
    assert launch_mock.call_args.kwargs["template_catalog_path"] == (
        design_path.parent / ".tensor-network-editor" / "templates.json"
    )


def test_edit_subcommand_anchors_relative_save_code_to_loaded_spec_directory(
    tmp_path: Path,
    sample_spec: NetworkSpec,
) -> None:
    design_path = tmp_path / "project_a" / "saved-network.json"
    design_path.parent.mkdir(parents=True)
    design_path.write_text("{}", encoding="utf-8")

    with (
        patch(
            "tensor_network_editor.cli.load_spec", return_value=sample_spec
        ) as load_mock,
        patch("tensor_network_editor.cli.launch_tensor_network_editor") as launch_mock,
    ):
        exit_code = main(
            [
                "edit",
                "--load",
                str(design_path),
                "--save-code",
                "generated.py",
                "--no-browser",
            ]
        )

    assert exit_code == 0
    load_mock.assert_called_once_with(str(design_path))
    assert (
        launch_mock.call_args.kwargs["code_path"] == design_path.parent / "generated.py"
    )


def test_main_returns_130_on_keyboard_interrupt() -> None:
    with patch(
        "tensor_network_editor.cli.launch_tensor_network_editor",
        side_effect=KeyboardInterrupt,
    ) as launch_mock:
        exit_code = main(["edit"])

    assert exit_code == 130
    launch_mock.assert_called_once()


def test_edit_subcommand_loads_initial_spec(sample_spec: NetworkSpec) -> None:
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
    assert launch_mock.call_args.kwargs["template_catalog_path"] == (
        Path("saved-network.json").resolve().parent
        / ".tensor-network-editor"
        / "templates.json"
    )


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


def test_analyze_subcommand_passes_dtype_to_analysis(
    sample_spec: NetworkSpec,
) -> None:
    with (
        patch("tensor_network_editor.cli.load_spec", return_value=sample_spec),
        patch(
            "tensor_network_editor.cli.analyze_spec",
            return_value=build_analysis_report("float32"),
        ) as analyze_mock,
    ):
        exit_code = main(["analyze", "saved-network.json", "--dtype", "float32"])

    assert exit_code == 0
    analyze_mock.assert_called_once_with(sample_spec, memory_dtype="float32")


def test_analyze_subcommand_text_output_includes_comparison_details(
    sample_spec: NetworkSpec,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with (
        patch("tensor_network_editor.cli.load_spec", return_value=sample_spec),
        patch(
            "tensor_network_editor.cli.analyze_spec",
            return_value=build_analysis_report("float32"),
        ),
    ):
        exit_code = main(["analyze", "saved-network.json", "--dtype", "float32"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Manual: status=complete" in output
    assert "Automatic full: status=complete" in output
    assert "Automatic future: status=complete" in output
    assert "Automatic past: status=complete" in output
    assert "manual vs automatic full" in output.lower()
    assert "FLOP down by 376" in output
    assert "Peak memory down by 752 bytes (float32)" in output
    assert "Peak steps: manual=step_bcd, automatic_full=auto_full_step_1" in output
    assert "Bottlenecks: manual=x, y, z | automatic_full=i, j" in output


def test_coerce_int_accepts_integer_like_float() -> None:
    assert _coerce_int(42.0) == 42


def test_coerce_int_rejects_non_integer_float() -> None:
    with pytest.raises(
        ValueError,
        match="Expected an integer-like value, got non-integer float 3.5.",
    ):
        _coerce_int(3.5)


def test_coerce_int_rejects_non_integer_string() -> None:
    with pytest.raises(
        ValueError,
        match=r"Expected an integer-like string, got '12\.4'\.",
    ):
        _coerce_int("12.4")


def test_analyze_subcommand_reports_integer_metric_errors(
    sample_spec: NetworkSpec,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = build_analysis_report()
    assert report.contraction is not None
    summary = cast(Any, report.contraction.manual.summary)
    summary.total_estimated_flops = 1.5
    with (
        patch("tensor_network_editor.cli.load_spec", return_value=sample_spec),
        patch("tensor_network_editor.cli.analyze_spec", return_value=report),
    ):
        exit_code = main(["analyze", "saved-network.json"])

    assert exit_code == 2
    assert (
        "Invalid integer metric value 1.5: Expected an integer-like value, got non-integer float 1.5."
        in capsys.readouterr().out
    )


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


def test_diff_subcommand_prints_semantic_json(
    sample_spec: NetworkSpec,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with (
        patch(
            "tensor_network_editor.cli.load_spec",
            side_effect=[sample_spec, sample_spec],
        ) as load_mock,
        patch(
            "tensor_network_editor.cli.semantic_diff_specs",
            return_value=SemanticSpecDiffResult(
                entries=[
                    SemanticDiffEntry(
                        entity_type="tensor",
                        entity_id="tensor_a",
                        change_type="changed",
                        summary="Tensor changed.",
                        field_changes=[
                            SemanticFieldChange(
                                path="name",
                                before="A",
                                after="A prime",
                            )
                        ],
                    )
                ]
            ),
        ),
    ):
        exit_code = main(
            [
                "diff",
                "before.json",
                "after.json",
                "--semantic",
                "--format",
                "json",
            ]
        )

    assert exit_code == 0
    assert load_mock.call_count == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["entries"] == [
        {
            "entity_type": "tensor",
            "entity_id": "tensor_a",
            "change_type": "changed",
            "summary": "Tensor changed.",
            "field_changes": [{"path": "name", "before": "A", "after": "A prime"}],
        }
    ]


def test_diff_subcommand_prints_semantic_text(
    sample_spec: NetworkSpec,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with (
        patch(
            "tensor_network_editor.cli.load_spec",
            side_effect=[sample_spec, sample_spec],
        ),
        patch(
            "tensor_network_editor.cli.semantic_diff_specs",
            return_value=SemanticSpecDiffResult(
                entries=[
                    SemanticDiffEntry(
                        entity_type="tensor",
                        entity_id="tensor_a",
                        change_type="changed",
                        summary="Tensor fields changed: name.",
                        field_changes=[
                            SemanticFieldChange(
                                path="name",
                                before="A",
                                after="A prime",
                            )
                        ],
                    )
                ]
            ),
        ),
    ):
        exit_code = main(["diff", "before.json", "after.json", "--semantic"])

    assert exit_code == 0
    assert capsys.readouterr().out == (
        'Tensors:\n- tensor_a: Tensor fields changed: name.\n  name: "A" -> "A prime"\n'
    )


def test_canonicalize_subcommand_writes_output(
    sample_spec: NetworkSpec,
) -> None:
    canonical_spec = NetworkSpec(id="network_001", name=sample_spec.name)
    with (
        patch(
            "tensor_network_editor.cli.load_spec", return_value=sample_spec
        ) as load_mock,
        patch(
            "tensor_network_editor.cli.canonicalize_spec",
            return_value=canonical_spec,
        ) as canonicalize_mock,
        patch("tensor_network_editor.cli.save_spec") as save_mock,
    ):
        exit_code = main(
            [
                "canonicalize",
                "before.json",
                "--output",
                "canonical.json",
                "--deterministic-ids",
            ]
        )

    assert exit_code == 0
    load_mock.assert_called_once_with("before.json")
    canonicalize_mock.assert_called_once_with(sample_spec, deterministic_ids=True)
    save_mock.assert_called_once_with(canonical_spec, "canonical.json")


def test_template_list_subcommand_preserves_json_and_text_output_formats(
    capsys: pytest.CaptureFixture[str],
) -> None:
    definitions = {
        "mps": {"display_name": "Matrix Product State"},
        "mera": {"display_name": "MERA"},
    }
    with (
        patch(
            "tensor_network_editor.cli.serialize_template_definitions",
            return_value=definitions,
        ) as definitions_mock,
        patch(
            "tensor_network_editor.cli.list_template_names",
            return_value=["mps", "mera"],
        ),
    ):
        json_exit_code = main(["template", "list", "--format", "json"])
        json_payload = json.loads(capsys.readouterr().out)

        text_exit_code = main(["template", "list", "--format", "text"])
        text_output = capsys.readouterr().out

    assert json_exit_code == 0
    assert json_payload == definitions
    assert text_exit_code == 0
    assert text_output == "mps: Matrix Product State\nmera: MERA\n"
    assert definitions_mock.call_count == 2


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
