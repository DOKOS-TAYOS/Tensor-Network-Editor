from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest

from tensor_network_editor.cli import build_command_parser, main
from tensor_network_editor.editor import EditorLaunchOptions
from tensor_network_editor.internal.analysis._contraction_analysis_types import (
    AutomaticContractionPlanAnalysis,
    AutomaticContractionSummary,
    ContractionAnalysisResult,
    ContractionComparison,
    ManualContractionPlanAnalysis,
    ManualContractionSummary,
)
from tensor_network_editor.internal.cli._cli_doctor import build_doctor_report
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
from tensor_network_editor.io import SCHEMA_VERSION, PythonLoadOptions, serialize_spec
from tensor_network_editor.linting import LintIssue, LintReport
from tensor_network_editor.models import (
    DiffEntityChanges,
    EngineName,
    NetworkSpec,
    SpecDiffResult,
    ValidationIssue,
)
from tensor_network_editor.rendering import (
    DotRenderOptions,
    SvgRenderOptions,
    TikzRenderOptions,
)
from tests.factories import build_sample_spec, build_three_tensor_hyperedge_spec

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


def no_validation_issues(_spec: NetworkSpec) -> list[ValidationIssue]:
    return []


def empty_lint_report(_spec: NetworkSpec) -> LintReport:
    return LintReport()


def test_main_requires_a_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    with patch("tensor_network_editor.cli.open_editor") as open_editor_mock:
        exit_code = main([])

    assert exit_code == 2
    open_editor_mock.assert_not_called()
    assert "the following arguments are required: command" in capsys.readouterr().err


def test_global_log_level_is_accepted_before_subcommand() -> None:
    parser = build_command_parser()

    parsed_args = parser.parse_args(["--log-level", "debug", "edit", "--no-browser"])

    assert parsed_args.log_level == "debug"
    assert parsed_args.command == "edit"
    assert parsed_args.no_browser is True


def test_global_python_import_arguments_are_accepted_before_subcommand() -> None:
    parser = build_command_parser()

    parsed_args = parser.parse_args(
        [
            "--python-import-mode",
            "live",
            "--python-reconstruction-level",
            "simple",
            "--python-object",
            "network",
            "edit",
            "--no-browser",
        ]
    )

    assert parsed_args.python_import_mode == "live"
    assert parsed_args.python_reconstruction_level == "simple"
    assert parsed_args.python_object == "network"
    assert parsed_args.command == "edit"


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
    with patch("tensor_network_editor.cli.open_editor") as open_editor_mock:
        exit_code = main(["edit"])

    assert exit_code == 0
    open_editor_mock.assert_called_once_with(
        spec=None,
        options=EditorLaunchOptions(),
    )


def test_edit_subcommand_accepts_theme() -> None:
    with patch("tensor_network_editor.cli.open_editor") as open_editor_mock:
        exit_code = main(["edit", "--theme", "light", "--no-browser"])

    assert exit_code == 0
    open_editor_mock.assert_called_once_with(
        spec=None,
        options=EditorLaunchOptions(theme="light", open_browser=False),
    )


def test_edit_subcommand_rejects_unknown_theme(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with patch("tensor_network_editor.cli.open_editor") as open_editor_mock:
        exit_code = main(["edit", "--theme", "sepia"])

    assert exit_code == 2
    open_editor_mock.assert_not_called()
    assert "invalid choice: 'sepia'" in capsys.readouterr().err


def test_edit_subcommand_passes_live_python_import_options(
    sample_spec: NetworkSpec,
) -> None:
    with (
        patch(
            "tensor_network_editor.cli.load_spec", return_value=sample_spec
        ) as load_mock,
        patch("tensor_network_editor.cli.open_editor") as open_editor_mock,
    ):
        exit_code = main(
            [
                "--python-import-mode",
                "live",
                "--python-reconstruction-level",
                "simple",
                "--python-object",
                "network",
                "edit",
                "--load",
                "saved-network.py",
                "--no-browser",
            ]
        )

    assert exit_code == 0
    load_mock.assert_called_once_with(
        "saved-network.py",
        python=PythonLoadOptions(
            import_mode="live",
            reconstruction_level="simple",
            object_name="network",
        ),
    )
    open_editor_mock.assert_called_once_with(
        spec=sample_spec,
        options=EditorLaunchOptions(
            open_browser=False,
            template_catalog_path=Path("saved-network.py").resolve().parent
            / ".tensor-network-editor"
            / "templates.json",
            subnetwork_catalog_path=Path("saved-network.py").resolve().parent
            / ".tensor-network-editor"
            / "subnetworks.json",
        ),
    )


def test_main_loads_spec_and_passes_output_flags(sample_spec: NetworkSpec) -> None:
    with (
        patch(
            "tensor_network_editor.cli.load_spec", return_value=sample_spec
        ) as load_mock,
        patch("tensor_network_editor.cli.open_editor") as open_editor_mock,
    ):
        exit_code = main(
            [
                "edit",
                "--engine",
                EngineName.EINSUM_NUMPY.value,
                "--theme",
                "contrast",
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
    open_editor_mock.assert_called_once_with(
        spec=sample_spec,
        options=EditorLaunchOptions(
            default_engine=EngineName.EINSUM_NUMPY,
            theme="contrast",
            open_browser=False,
            print_code=True,
            code_path=Path("saved-network.json").resolve().parent / "generated.py",
            template_catalog_path=Path("saved-network.json").resolve().parent
            / ".tensor-network-editor"
            / "templates.json",
            subnetwork_catalog_path=Path("saved-network.json").resolve().parent
            / ".tensor-network-editor"
            / "subnetworks.json",
        ),
    )


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
        patch("tensor_network_editor.cli.open_editor") as open_editor_mock,
    ):
        exit_code = main(["edit", "--load", str(design_path), "--no-browser"])

    assert exit_code == 0
    load_mock.assert_called_once_with(str(design_path))
    options = open_editor_mock.call_args.kwargs["options"]
    assert isinstance(options, EditorLaunchOptions)
    assert options.template_catalog_path is not None
    assert options.subnetwork_catalog_path is not None
    assert Path(options.template_catalog_path).resolve() == (
        design_path.parent / ".tensor-network-editor" / "templates.json"
    ).resolve()
    assert Path(options.subnetwork_catalog_path).resolve() == (
        design_path.parent / ".tensor-network-editor" / "subnetworks.json"
    ).resolve()


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
        patch("tensor_network_editor.cli.open_editor") as open_editor_mock,
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
    options = open_editor_mock.call_args.kwargs["options"]
    assert isinstance(options, EditorLaunchOptions)
    assert options.code_path is not None
    assert Path(options.code_path).resolve() == (design_path.parent / "generated.py").resolve()


def test_main_returns_130_on_keyboard_interrupt() -> None:
    with patch(
        "tensor_network_editor.cli.open_editor",
        side_effect=KeyboardInterrupt,
    ) as open_editor_mock:
        exit_code = main(["edit"])

    assert exit_code == 130
    open_editor_mock.assert_called_once()


def test_edit_subcommand_loads_initial_spec(sample_spec: NetworkSpec) -> None:
    with (
        patch(
            "tensor_network_editor.cli.load_spec", return_value=sample_spec
        ) as load_mock,
        patch("tensor_network_editor.cli.open_editor") as open_editor_mock,
    ):
        exit_code = main(["edit", "--load", "saved-network.json", "--no-browser"])

    assert exit_code == 0
    load_mock.assert_called_once_with("saved-network.json")
    assert open_editor_mock.call_args.kwargs["spec"] is sample_spec
    options = open_editor_mock.call_args.kwargs["options"]
    assert isinstance(options, EditorLaunchOptions)
    assert options.open_browser is False
    assert options.template_catalog_path == (
        Path("saved-network.json").resolve().parent
        / ".tensor-network-editor"
        / "templates.json"
    )
    assert options.subnetwork_catalog_path == (
        Path("saved-network.json").resolve().parent
        / ".tensor-network-editor"
        / "subnetworks.json"
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


def test_build_command_parser_accepts_doctor_subcommand() -> None:
    parser = build_command_parser()

    parsed_args = parser.parse_args(
        [
            "doctor",
            "saved-network.json",
            "--dtype",
            "float32",
            "--format",
            "json",
        ]
    )

    assert parsed_args.command == "doctor"
    assert parsed_args.path == "saved-network.json"
    assert parsed_args.dtype == "float32"
    assert parsed_args.format == "json"


def test_doctor_subcommand_text_includes_diagnostic_sections(
    sample_spec: NetworkSpec,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with (
        patch("tensor_network_editor.cli.load_spec_for_lint", return_value=sample_spec),
        patch("tensor_network_editor.cli.validate_spec", return_value=[]),
        patch("tensor_network_editor.cli.lint_spec", return_value=LintReport()),
        patch(
            "tensor_network_editor.cli.analyze_spec",
            return_value=build_analysis_report("float32"),
        ),
    ):
        exit_code = main(["doctor", "saved-network.json", "--dtype", "float32"])

    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Validation" in output
    assert "Lint" in output
    assert "Analysis" in output
    assert "Benchmark" in output
    assert "Backends/Extras" in output
    assert "Suggestions" in output


def test_doctor_subcommand_json_has_stable_shape(
    sample_spec: NetworkSpec,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with (
        patch("tensor_network_editor.cli.load_spec_for_lint", return_value=sample_spec),
        patch("tensor_network_editor.cli.validate_spec", return_value=[]),
        patch("tensor_network_editor.cli.lint_spec", return_value=LintReport()),
        patch(
            "tensor_network_editor.cli.analyze_spec",
            return_value=build_analysis_report(),
        ),
    ):
        exit_code = main(["doctor", "saved-network.json", "--format", "json"])

    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert list(payload) == [
        "ok",
        "validation",
        "lint",
        "analysis",
        "benchmark",
        "backends",
        "warnings",
        "suggestions",
    ]
    assert payload["ok"] is True
    assert payload["validation"]["issue_count"] == 0
    assert payload["lint"]["issue_count"] == 0
    assert payload["benchmark"]["memory_dtype"] == "float64"
    assert "numpy" in payload["backends"]


def test_doctor_subcommand_returns_1_for_validation_errors(
    sample_spec: NetworkSpec,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with (
        patch("tensor_network_editor.cli.load_spec_for_lint", return_value=sample_spec),
        patch(
            "tensor_network_editor.cli.validate_spec",
            return_value=[
                ValidationIssue(
                    code="bad-index",
                    message="Index dimension is invalid.",
                    path="tensors.tensor_a.indices.tensor_a_i",
                )
            ],
        ),
        patch("tensor_network_editor.cli.lint_spec", return_value=LintReport()),
    ):
        exit_code = main(["doctor", "saved-network.json"])

    output = capsys.readouterr().out

    assert exit_code == 1
    assert "Validation" in output
    assert "bad-index" in output
    assert "Analysis" in output
    assert "skipped" in output.lower()


def test_doctor_subcommand_load_failure_returns_2(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with patch(
        "tensor_network_editor.cli.load_spec_for_lint",
        side_effect=ValueError("Could not read saved-network.json"),
    ):
        exit_code = main(["doctor", "saved-network.json"])

    assert exit_code == 2
    assert "Could not read saved-network.json" in capsys.readouterr().out


def test_doctor_subcommand_reports_hyperedges_as_warning(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with patch(
        "tensor_network_editor.cli.load_spec_for_lint",
        return_value=build_three_tensor_hyperedge_spec(),
    ):
        exit_code = main(["doctor", "saved-network.json"])

    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Hyperedges are analyzed as generated copy tensors" in output
    assert "Validation" in output


def test_doctor_report_recommends_available_backend_and_auto_full(
    sample_spec: NetworkSpec,
) -> None:
    report = build_doctor_report(
        sample_spec,
        validate_spec=no_validation_issues,
        lint_spec=empty_lint_report,
        analyze_spec=lambda _spec, memory_dtype: build_analysis_report(memory_dtype),
        find_spec=lambda import_name: (
            object() if import_name in {"numpy", "opt_einsum"} else None
        ),
    )

    suggestions_text = "\n".join(report.suggestions)

    assert "Recommended backend: einsum_numpy" in suggestions_text
    assert "Auto full is cheaper than the saved manual plan" in suggestions_text
    assert "FLOP 1600 vs 1224" in suggestions_text
    assert "No immediate fixes needed." not in report.suggestions


def test_doctor_report_suggests_auto_future_for_incomplete_manual_plan(
    sample_spec: NetworkSpec,
) -> None:
    analysis_report = build_analysis_report()
    assert analysis_report.contraction is not None
    analysis_report.contraction.manual.status = "partial"
    analysis_report.contraction.manual.summary.completion_status = "partial"
    analysis_report.contraction.manual.summary.remaining_operand_ids = (
        "step_ab",
        "tensor_c",
    )

    def analyze_incomplete_manual_plan(
        _spec: NetworkSpec,
        *,
        memory_dtype: str = "float64",
    ) -> SpecAnalysisReport:
        assert memory_dtype
        return analysis_report

    report = build_doctor_report(
        sample_spec,
        validate_spec=no_validation_issues,
        lint_spec=empty_lint_report,
        analyze_spec=analyze_incomplete_manual_plan,
        find_spec=lambda import_name: object() if import_name == "opt_einsum" else None,
    )

    assert any(
        "Auto future can complete the remaining manual frontier" in suggestion
        for suggestion in report.suggestions
    )


def test_doctor_report_adds_model_level_suggestions_from_lint(
    sample_spec: NetworkSpec,
) -> None:
    lint_report = LintReport(
        issues=[
            LintIssue(
                severity="warning",
                code="suspicious-open-index",
                message="Index 'x' is open and looks like a missing connection.",
                path="tensors.tensor_a.indices.tensor_a_x",
                suggestion="Connect it, rename it to reflect an output leg, or document it in metadata.",
            ),
            LintIssue(
                severity="warning",
                code="large-tensor-cardinality",
                message="Tensor 'A' spans many elements.",
                path="tensors.tensor_a",
                suggestion="Check dimensions, decomposition choices, or raise the threshold for this workflow.",
            ),
        ]
    )

    report = build_doctor_report(
        sample_spec,
        validate_spec=no_validation_issues,
        lint_spec=lambda _spec: lint_report,
        analyze_spec=lambda _spec, memory_dtype: build_analysis_report(memory_dtype),
        find_spec=lambda import_name: object() if import_name == "opt_einsum" else None,
    )

    suggestions_text = "\n".join(report.suggestions)

    assert "Review suspicious open indices" in suggestions_text
    assert "Inspect large tensor dimensions" in suggestions_text
    assert "No immediate fixes needed." not in report.suggestions


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
    assert generate_mock.call_args.kwargs["output_path"] == "generated.py"
    assert generate_mock.call_args.kwargs["print_code"] is False
    assert (
        generate_mock.call_args.kwargs["external_data_base_path"]
        == Path("saved-network.json").resolve().parent
    )


def test_render_subcommand_writes_svg_output(sample_spec: NetworkSpec) -> None:
    with (
        patch(
            "tensor_network_editor.cli.load_spec", return_value=sample_spec
        ) as load_mock,
        patch(
            "tensor_network_editor.cli.render_spec_svg", return_value="<svg />"
        ) as render_mock,
    ):
        exit_code = main(["render", "saved-network.json", "--output", "figure.svg"])

    assert exit_code == 0
    load_mock.assert_called_once_with("saved-network.json")
    render_mock.assert_called_once()
    assert render_mock.call_args.args == (sample_spec,)
    assert render_mock.call_args.kwargs["output_path"] == "figure.svg"


def test_render_subcommand_passes_svg_label_options(sample_spec: NetworkSpec) -> None:
    with (
        patch(
            "tensor_network_editor.cli.load_spec", return_value=sample_spec
        ) as load_mock,
        patch(
            "tensor_network_editor.cli.render_spec_svg", return_value="<svg />"
        ) as render_mock,
    ):
        exit_code = main(
            [
                "render",
                "saved-network.json",
                "--hide-tensor-names",
                "--hide-index-names",
                "--hide-bond-names",
            ]
        )

    assert exit_code == 0
    load_mock.assert_called_once_with("saved-network.json")
    options = render_mock.call_args.kwargs["options"]
    assert isinstance(options, SvgRenderOptions)
    assert options.show_tensor_labels is False
    assert options.show_index_labels is False
    assert options.show_edge_labels is False


def test_render_subcommand_writes_png_output(sample_spec: NetworkSpec) -> None:
    with (
        patch(
            "tensor_network_editor.cli.load_spec", return_value=sample_spec
        ) as load_mock,
        patch(
            "tensor_network_editor.cli.render_spec_png",
            return_value=b"\x89PNG\r\n\x1a\n",
        ) as render_mock,
    ):
        exit_code = main(
            [
                "render",
                "saved-network.json",
                "--format",
                "png",
                "--output",
                "figure.png",
            ]
        )

    assert exit_code == 0
    load_mock.assert_called_once_with("saved-network.json")
    render_mock.assert_called_once()
    assert render_mock.call_args.args == (sample_spec,)
    assert render_mock.call_args.kwargs["output_path"] == "figure.png"


def test_render_subcommand_writes_pdf_output(sample_spec: NetworkSpec) -> None:
    with (
        patch(
            "tensor_network_editor.cli.load_spec", return_value=sample_spec
        ) as load_mock,
        patch(
            "tensor_network_editor.cli.render_spec_pdf",
            return_value=b"%PDF-1.4",
        ) as render_mock,
    ):
        exit_code = main(
            [
                "render",
                "saved-network.json",
                "--format",
                "pdf",
                "--output",
                "figure.pdf",
            ]
        )

    assert exit_code == 0
    load_mock.assert_called_once_with("saved-network.json")
    render_mock.assert_called_once()
    assert render_mock.call_args.args == (sample_spec,)
    assert render_mock.call_args.kwargs["output_path"] == "figure.pdf"


def test_render_subcommand_writes_tikz_output(sample_spec: NetworkSpec) -> None:
    with (
        patch(
            "tensor_network_editor.cli.load_spec", return_value=sample_spec
        ) as load_mock,
        patch(
            "tensor_network_editor.cli.render_spec_tikz",
            return_value=r"\begin{tikzpicture}\end{tikzpicture}",
        ) as render_mock,
    ):
        exit_code = main(
            [
                "render",
                "saved-network.json",
                "--format",
                "tikz",
                "--output",
                "figure.tex",
            ]
        )

    assert exit_code == 0
    load_mock.assert_called_once_with("saved-network.json")
    render_mock.assert_called_once()
    assert render_mock.call_args.args == (sample_spec,)
    assert render_mock.call_args.kwargs["output_path"] == "figure.tex"


def test_render_subcommand_passes_tikz_label_options(
    sample_spec: NetworkSpec,
) -> None:
    with (
        patch(
            "tensor_network_editor.cli.load_spec", return_value=sample_spec
        ) as load_mock,
        patch(
            "tensor_network_editor.cli.render_spec_tikz",
            return_value=r"\begin{tikzpicture}\end{tikzpicture}",
        ) as render_mock,
    ):
        exit_code = main(
            [
                "render",
                "saved-network.json",
                "--format",
                "tikz",
                "--hide-tensor-names",
                "--hide-index-names",
                "--hide-bond-names",
            ]
        )

    assert exit_code == 0
    load_mock.assert_called_once_with("saved-network.json")
    options = render_mock.call_args.kwargs["options"]
    assert isinstance(options, TikzRenderOptions)
    assert options.show_tensor_labels is False
    assert options.show_index_labels is False
    assert options.show_edge_labels is False


def test_render_subcommand_writes_dot_output(sample_spec: NetworkSpec) -> None:
    with (
        patch(
            "tensor_network_editor.cli.load_spec", return_value=sample_spec
        ) as load_mock,
        patch(
            "tensor_network_editor.cli.render_spec_dot",
            return_value='graph "network_demo" {}',
        ) as render_mock,
    ):
        exit_code = main(
            [
                "render",
                "saved-network.json",
                "--format",
                "dot",
                "--output",
                "graph.dot",
            ]
        )

    assert exit_code == 0
    load_mock.assert_called_once_with("saved-network.json")
    render_mock.assert_called_once()
    assert render_mock.call_args.args == (sample_spec,)
    assert render_mock.call_args.kwargs["output_path"] == "graph.dot"


def test_render_subcommand_passes_dot_label_options(sample_spec: NetworkSpec) -> None:
    with (
        patch(
            "tensor_network_editor.cli.load_spec", return_value=sample_spec
        ) as load_mock,
        patch(
            "tensor_network_editor.cli.render_spec_dot",
            return_value='graph "network_demo" {}',
        ) as render_mock,
    ):
        exit_code = main(
            [
                "render",
                "saved-network.json",
                "--format",
                "dot",
                "--hide-tensor-names",
                "--hide-index-names",
                "--hide-bond-names",
            ]
        )

    assert exit_code == 0
    load_mock.assert_called_once_with("saved-network.json")
    options = render_mock.call_args.kwargs["options"]
    assert isinstance(options, DotRenderOptions)
    assert options.show_tensor_labels is False
    assert options.show_index_labels is False
    assert options.show_edge_labels is False


def test_render_subcommand_rejects_png_without_output(
    sample_spec: NetworkSpec,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with patch("tensor_network_editor.cli.load_spec", return_value=sample_spec):
        exit_code = main(["render", "saved-network.json", "--format", "png"])

    assert exit_code == 2
    assert "PNG render requires --output" in capsys.readouterr().out


def test_render_subcommand_rejects_pdf_without_output(
    sample_spec: NetworkSpec,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with patch("tensor_network_editor.cli.load_spec", return_value=sample_spec):
        exit_code = main(["render", "saved-network.json", "--format", "pdf"])

    assert exit_code == 2
    assert "PDF render requires --output" in capsys.readouterr().out


def test_render_subcommand_prints_tikz_when_no_output(
    sample_spec: NetworkSpec,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with (
        patch("tensor_network_editor.cli.load_spec", return_value=sample_spec),
        patch(
            "tensor_network_editor.cli.render_spec_tikz",
            return_value=r"\begin{tikzpicture}network\end{tikzpicture}",
        ),
    ):
        exit_code = main(["render", "saved-network.json", "--format", "tikz"])

    assert exit_code == 0
    assert capsys.readouterr().out == "\\begin{tikzpicture}network\\end{tikzpicture}\n"


def test_render_subcommand_prints_dot_when_no_output(
    sample_spec: NetworkSpec,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with (
        patch("tensor_network_editor.cli.load_spec", return_value=sample_spec),
        patch(
            "tensor_network_editor.cli.render_spec_dot",
            return_value='graph "network_demo" {}',
        ),
    ):
        exit_code = main(["render", "saved-network.json", "--format", "dot"])

    assert exit_code == 0
    assert capsys.readouterr().out == 'graph "network_demo" {}\n'


def test_render_subcommand_prints_svg_when_no_output(
    sample_spec: NetworkSpec,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with (
        patch("tensor_network_editor.cli.load_spec", return_value=sample_spec),
        patch(
            "tensor_network_editor.cli.render_spec_svg",
            return_value="<svg>network</svg>",
        ),
    ):
        exit_code = main(["render", "saved-network.json"])

    assert exit_code == 0
    assert capsys.readouterr().out == "<svg>network</svg>\n"


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
    save_mock.assert_called_once_with(canonical_spec, path="canonical.json")


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
    assert payload["network"]["metadata"]["initial_state"] == "zeros"


@pytest.mark.parametrize(
    ("template_name", "expected_name"),
    [
        ("ttn", "TTN depth 3"),
        ("pepo", "PEPO 3x3"),
        ("mpo", "MPO"),
    ],
)
def test_template_build_subcommand_generates_new_v3_templates(
    template_name: str,
    expected_name: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "template",
            "build",
            template_name,
            "--format",
            "json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["network"]["name"] == expected_name


def test_template_build_subcommand_accepts_new_mps_configuration_flags(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "template",
            "build",
            "mps",
            "--graph-size",
            "4",
            "--bond-dimension",
            "3",
            "--physical-dimension",
            "2",
            "--boundary-condition",
            "periodic",
            "--symmetry",
            "z2",
            "--initial-state",
            "neel",
            "--format",
            "json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["network"]["metadata"]["boundary_condition"] == "periodic"
    assert payload["network"]["metadata"]["symmetry"] == "z2"
    assert payload["network"]["metadata"]["initial_state"] == "neel"
    assert len(payload["network"]["edges"]) == 4


def test_template_build_subcommand_accepts_new_mpo_and_ttn_configuration_flags(
    capsys: pytest.CaptureFixture[str],
) -> None:
    mpo_exit_code = main(
        [
            "template",
            "build",
            "mpo",
            "--graph-size",
            "4",
            "--bond-dimension",
            "3",
            "--physical-dimension",
            "2",
            "--boundary-condition",
            "periodic",
            "--j",
            "1.5",
            "--h",
            "0.25",
            "--format",
            "json",
        ]
    )
    mpo_payload = json.loads(capsys.readouterr().out)

    ttn_exit_code = main(
        [
            "template",
            "build",
            "ttn",
            "--depth",
            "4",
            "--bond-dimension",
            "3",
            "--physical-dimension",
            "2",
            "--root-open-leg",
            "--no-leaf-physical-legs",
            "--isometric",
            "--format",
            "json",
        ]
    )
    ttn_payload = json.loads(capsys.readouterr().out)

    assert mpo_exit_code == 0
    assert mpo_payload["network"]["metadata"]["boundary_condition"] == "periodic"
    assert mpo_payload["network"]["metadata"]["j"] == 1.5
    assert mpo_payload["network"]["metadata"]["h"] == 0.25
    assert len(mpo_payload["network"]["edges"]) == 4
    assert ttn_exit_code == 0
    assert ttn_payload["network"]["metadata"]["depth"] == 4
    assert ttn_payload["network"]["metadata"]["root_open_leg"] is True
    assert ttn_payload["network"]["metadata"]["leaf_physical_legs"] is False
    assert ttn_payload["network"]["metadata"]["isometric"] is True


def test_subnetwork_list_subcommand_prints_project_catalog(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    catalog_path = tmp_path / ".tensor-network-editor" / "subnetworks.json"
    spec_path = tmp_path / "network.json"
    sample_spec = build_sample_spec()
    spec_path.write_text(
        json.dumps(serialize_spec(sample_spec), indent=2), encoding="utf-8"
    )

    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "subnetworks": [
                    {
                        "name": "project_pair",
                        "display_name": "Project Pair",
                        "tags": ["alpha", "project"],
                        "spec": {
                            "schema_version": SCHEMA_VERSION,
                            "network": sample_spec.to_dict(),
                        },
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    exit_code = main(["subnetwork", "list", str(spec_path)])

    assert exit_code == 0
    assert "project_pair: Project Pair [alpha, project]" in capsys.readouterr().out


def test_subnetwork_save_subcommand_persists_project_catalog(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec_path = tmp_path / "network.json"
    sample_spec = build_sample_spec()
    spec_path.write_text(
        json.dumps(serialize_spec(sample_spec), indent=2), encoding="utf-8"
    )

    exit_code = main(
        [
            "subnetwork",
            "save",
            str(spec_path),
            "--tensor-ids",
            "tensor_a",
            "tensor_b",
            "--name",
            "project_pair",
            "--tags",
            "alpha",
            "project",
        ]
    )

    catalog_path = tmp_path / ".tensor-network-editor" / "subnetworks.json"

    assert exit_code == 0
    saved_payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    assert saved_payload["schema_version"] == 1
    assert saved_payload["subnetworks"][0]["name"] == "project_pair"
    assert saved_payload["subnetworks"][0]["tags"] == ["alpha", "project"]
    assert [
        tensor["id"]
        for tensor in saved_payload["subnetworks"][0]["spec"]["network"]["tensors"]
    ] == [
        "tensor_a",
        "tensor_b",
    ]
    assert "Saved reusable subnetwork 'project_pair'" in capsys.readouterr().out


def test_subnetwork_export_subcommand_writes_selected_entry_to_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec_path = tmp_path / "network.json"
    output_path = tmp_path / "project_pair.json"
    sample_spec = build_sample_spec()
    spec_path.write_text(
        json.dumps(serialize_spec(sample_spec), indent=2), encoding="utf-8"
    )
    save_exit_code = main(
        [
            "subnetwork",
            "save",
            str(spec_path),
            "--tensor-ids",
            "tensor_a",
            "tensor_b",
            "--name",
            "project_pair",
        ]
    )

    export_exit_code = main(
        [
            "subnetwork",
            "export",
            str(spec_path),
            "project_pair",
            "--output",
            str(output_path),
        ]
    )

    assert save_exit_code == 0
    assert export_exit_code == 0
    exported_payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert exported_payload["schema_version"] == SCHEMA_VERSION
    assert [tensor["id"] for tensor in exported_payload["network"]["tensors"]] == [
        "tensor_a",
        "tensor_b",
    ]
    assert (
        f"Wrote reusable subnetwork 'project_pair' to {output_path}"
        in capsys.readouterr().out
    )
