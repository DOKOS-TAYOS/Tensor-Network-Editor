from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from tensor_network_editor.cli import build_command_parser, main
from tensor_network_editor.internal.analysis._contraction_analysis_types import (
    AutomaticContractionPlanAnalysis,
    AutomaticContractionSummary,
    ContractionAnalysisResult,
    ManualContractionPlanAnalysis,
    ManualContractionSummary,
)
from tensor_network_editor.internal.cli._cli_benchmark import (
    build_benchmark_report,
    serialize_benchmark_report_csv,
    serialize_benchmark_report_latex,
    serialize_benchmark_report_text,
)
from tensor_network_editor.models import NetworkSpec
from tests.factories import build_three_tensor_hyperedge_spec


def build_benchmark_analysis(
    memory_dtype: str = "float64",
) -> ContractionAnalysisResult:
    return ContractionAnalysisResult(
        network_output_shape=(2, 2),
        manual=ManualContractionPlanAnalysis(
            status="complete",
            steps=[],
            summary=ManualContractionSummary(
                total_estimated_flops=1_600,
                total_estimated_macs=800,
                peak_intermediate_size=100,
                peak_intermediate_bytes=800,
                final_shape=(2, 2),
                completion_status="complete",
                remaining_operand_ids=("step_abcd",),
            ),
        ),
        automatic_full=AutomaticContractionPlanAnalysis(
            status="complete",
            steps=[],
            summary=AutomaticContractionSummary(
                total_estimated_flops=1_224,
                total_estimated_macs=612,
                peak_intermediate_size=6,
                peak_intermediate_bytes=48,
            ),
        ),
        automatic_future=AutomaticContractionPlanAnalysis(
            status="unavailable",
            steps=[],
            summary=AutomaticContractionSummary(
                total_estimated_flops=0,
                total_estimated_macs=0,
                peak_intermediate_size=0,
                peak_intermediate_bytes=0,
            ),
            message="Install opt_einsum in the current .venv to enable Auto full, Auto future, and Auto past.",
        ),
        automatic_past=AutomaticContractionPlanAnalysis(
            status="complete",
            steps=[],
            summary=AutomaticContractionSummary(
                total_estimated_flops=576,
                total_estimated_macs=288,
                peak_intermediate_size=12,
                peak_intermediate_bytes=96,
            ),
        ),
        memory_dtype=memory_dtype,
        comparisons={},
        automatic_strategy="greedy",
    )


def test_build_command_parser_accepts_benchmark_subcommand() -> None:
    parser = build_command_parser()

    parsed_args = parser.parse_args(
        [
            "benchmark",
            "saved-network.json",
            "--dtype",
            "float32",
            "--format",
            "csv",
            "--output",
            "benchmark.csv",
        ]
    )

    assert parsed_args.command == "benchmark"
    assert parsed_args.path == "saved-network.json"
    assert parsed_args.dtype == "float32"
    assert parsed_args.format == "csv"
    assert parsed_args.output == "benchmark.csv"


def test_build_benchmark_report_uses_stable_row_names_and_null_metrics_for_unavailable() -> (
    None
):
    report = build_benchmark_report(build_benchmark_analysis("float32"))

    assert report.memory_dtype == "float32"
    assert [row.key for row in report.rows] == [
        "manual",
        "auto_full",
        "auto_future",
        "auto_past",
    ]
    assert [row.name for row in report.rows] == [
        "Manual",
        "Auto full",
        "Auto future",
        "Auto past",
    ]
    assert report.rows[2].status == "unavailable"
    assert report.rows[2].message == (
        "Install opt_einsum in the current .venv to enable Auto full, Auto future, and Auto past."
    )
    assert report.rows[2].flop is None
    assert report.rows[2].peak_memory is None


def test_serialize_benchmark_report_csv_uses_stable_columns() -> None:
    report = build_benchmark_report(build_benchmark_analysis("float32"))

    csv_output = serialize_benchmark_report_csv(report)

    assert csv_output.splitlines()[0] == "Name,FLOP,MAC,Peak,Peak Memory"
    assert "Manual,1600,800,100,800 bytes" in csv_output
    assert "Auto future,-,-,-,-" in csv_output


def test_serialize_benchmark_report_text_marks_unavailable_metrics_with_dash() -> None:
    report = build_benchmark_report(build_benchmark_analysis("float32"))

    text_output = serialize_benchmark_report_text(report)

    assert "Name" in text_output
    assert "Auto future" in text_output
    assert "Auto future  -     -    -     -" in text_output


def test_serialize_benchmark_report_latex_escapes_names() -> None:
    analysis = build_benchmark_analysis("float32")
    report = build_benchmark_report(analysis)
    report.rows[0].name = "Manual & baseline"

    latex_output = serialize_benchmark_report_latex(report)

    assert "\\begin{tabular}{lrrrr}" in latex_output
    assert "Manual \\& baseline" in latex_output
    assert "Auto future & - & - & - & -" in latex_output


def test_benchmark_subcommand_passes_dtype_to_analysis(
    sample_spec: NetworkSpec,
) -> None:
    with (
        patch("tensor_network_editor.cli.load_spec", return_value=sample_spec),
        patch(
            "tensor_network_editor.cli.analyze_contraction",
            return_value=build_benchmark_analysis("float32"),
        ) as analyze_mock,
    ):
        exit_code = main(["benchmark", "saved-network.json", "--dtype", "float32"])

    assert exit_code == 0
    analyze_mock.assert_called_once_with(sample_spec, memory_dtype="float32")


def test_benchmark_subcommand_prints_json_report(
    sample_spec: NetworkSpec,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with (
        patch("tensor_network_editor.cli.load_spec", return_value=sample_spec),
        patch(
            "tensor_network_editor.cli.analyze_contraction",
            return_value=build_benchmark_analysis("float32"),
        ),
    ):
        exit_code = main(["benchmark", "saved-network.json", "--format", "json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["memory_dtype"] == "float32"
    assert [row["key"] for row in payload["rows"]] == [
        "manual",
        "auto_full",
        "auto_future",
        "auto_past",
    ]
    assert payload["rows"][2]["status"] == "unavailable"
    assert payload["rows"][2]["message"] == (
        "Install opt_einsum in the current .venv to enable Auto full, Auto future, and Auto past."
    )
    assert payload["rows"][2]["peak_memory"] is None


def test_benchmark_subcommand_writes_latex_output_file(
    tmp_path: Path,
    sample_spec: NetworkSpec,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_path = tmp_path / "benchmark.tex"

    with (
        patch("tensor_network_editor.cli.load_spec", return_value=sample_spec),
        patch(
            "tensor_network_editor.cli.analyze_contraction",
            return_value=build_benchmark_analysis("float32"),
        ),
    ):
        exit_code = main(
            [
                "benchmark",
                "saved-network.json",
                "--format",
                "latex",
                "--output",
                str(output_path),
            ]
        )

    assert exit_code == 0
    assert output_path.exists()
    assert "\\begin{tabular}{lrrrr}" in output_path.read_text(encoding="utf-8")
    assert f"Wrote benchmark report to {output_path}" in capsys.readouterr().out


def test_benchmark_subcommand_accepts_hyperedges_with_warning(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with patch(
        "tensor_network_editor.cli.load_spec",
        return_value=build_three_tensor_hyperedge_spec(),
    ):
        exit_code = main(["benchmark", "saved-network.json"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Name" in output
    assert "Hyperedges are analyzed as generated copy tensors" in output
