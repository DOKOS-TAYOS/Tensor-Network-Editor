"""Helpers for headless benchmark table construction and serialization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

from ...types import JSONValue
from ..analysis._contraction_analysis_types import (
    AutomaticContractionPlanAnalysis,
    ContractionAnalysisResult,
    ManualContractionPlanAnalysis,
)

_BENCHMARK_HEADERS: tuple[str, ...] = (
    "Name",
    "FLOP",
    "MAC",
    "Peak",
    "Peak Memory",
)


@dataclass(slots=True)
class BenchmarkRow:
    """One benchmark comparison row."""

    key: str
    name: str
    status: str
    message: str | None = None
    flop: int | None = None
    mac: int | None = None
    peak: int | None = None
    peak_memory: int | None = None

    def to_dict(self) -> dict[str, JSONValue]:
        """Serialize the benchmark row to a JSON-compatible mapping."""
        return {
            "key": self.key,
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "flop": self.flop,
            "mac": self.mac,
            "peak": self.peak,
            "peak_memory": self.peak_memory,
        }


@dataclass(slots=True)
class BenchmarkReport:
    """Serializable benchmark table for one analyzed specification."""

    memory_dtype: str
    rows: list[BenchmarkRow] = field(default_factory=list)

    def to_dict(self) -> dict[str, JSONValue]:
        """Serialize the benchmark report to a JSON-compatible mapping."""
        return {
            "memory_dtype": self.memory_dtype,
            "rows": cast(JSONValue, [row.to_dict() for row in self.rows]),
        }


def build_benchmark_report(
    analysis: ContractionAnalysisResult,
) -> BenchmarkReport:
    """Build a stable benchmark table from one contraction analysis result."""
    return BenchmarkReport(
        memory_dtype=analysis.memory_dtype,
        rows=[
            _build_manual_row("manual", "Manual", analysis.manual),
            _build_automatic_row("auto_full", "Auto full", analysis.automatic_full),
            _build_automatic_row(
                "auto_future",
                "Auto future",
                analysis.automatic_future,
            ),
            _build_automatic_row("auto_past", "Auto past", analysis.automatic_past),
        ],
    )


def _build_manual_row(
    key: str,
    name: str,
    analysis: ManualContractionPlanAnalysis,
) -> BenchmarkRow:
    """Build one benchmark row from a manual analysis payload."""
    summary = analysis.summary
    if analysis.status == "unavailable":
        return BenchmarkRow(
            key=key,
            name=name,
            status=analysis.status,
            message=analysis.message,
        )
    return BenchmarkRow(
        key=key,
        name=name,
        status=analysis.status,
        message=analysis.message,
        flop=summary.total_estimated_flops,
        mac=summary.total_estimated_macs,
        peak=summary.peak_intermediate_size,
        peak_memory=summary.peak_intermediate_bytes,
    )


def _build_automatic_row(
    key: str,
    name: str,
    analysis: AutomaticContractionPlanAnalysis,
) -> BenchmarkRow:
    """Build one benchmark row from an automatic analysis payload."""
    summary = analysis.summary
    if analysis.status == "unavailable":
        return BenchmarkRow(
            key=key,
            name=name,
            status=analysis.status,
            message=analysis.message,
        )
    return BenchmarkRow(
        key=key,
        name=name,
        status=analysis.status,
        message=analysis.message,
        flop=summary.total_estimated_flops,
        mac=summary.total_estimated_macs,
        peak=summary.peak_intermediate_size,
        peak_memory=summary.peak_intermediate_bytes,
    )


def serialize_benchmark_report_csv(report: BenchmarkReport) -> str:
    """Serialize the benchmark report to a CSV table."""
    rows = [_BENCHMARK_HEADERS, *[_benchmark_display_row(row) for row in report.rows]]
    return "\n".join(
        ",".join(_escape_benchmark_csv_cell(value) for value in row) for row in rows
    )


def serialize_benchmark_report_text(report: BenchmarkReport) -> str:
    """Serialize the benchmark report to a plain-text aligned table."""
    rows = [_BENCHMARK_HEADERS, *[_benchmark_display_row(row) for row in report.rows]]
    column_widths = [
        max(len(str(row[column_index])) for row in rows)
        for column_index in range(len(_BENCHMARK_HEADERS))
    ]

    def format_row(row: tuple[str, ...]) -> str:
        return "  ".join(
            str(value).ljust(column_widths[column_index])
            for column_index, value in enumerate(row)
        )

    separator = "  ".join("-" * width for width in column_widths)
    return "\n".join(
        [format_row(rows[0]), separator] + [format_row(row) for row in rows[1:]]
    )


def serialize_benchmark_report_latex(report: BenchmarkReport) -> str:
    """Serialize the benchmark report to a compact LaTeX tabular block."""
    rows = [_BENCHMARK_HEADERS, *[_benchmark_display_row(row) for row in report.rows]]

    def format_row(row: tuple[str, ...]) -> str:
        return " & ".join(_escape_benchmark_latex_cell(value) for value in row) + r" \\"

    return "\n".join(
        [
            r"\begin{tabular}{lrrrr}",
            r"\hline",
            format_row(rows[0]),
            r"\hline",
            *[format_row(row) for row in rows[1:]],
            r"\hline",
            r"\end{tabular}",
        ]
    )


def _benchmark_display_row(row: BenchmarkRow) -> tuple[str, ...]:
    """Return the display strings for one benchmark row."""
    return (
        row.name,
        _format_benchmark_metric(row.flop),
        _format_benchmark_metric(row.mac),
        _format_benchmark_metric(row.peak),
        _format_benchmark_bytes(row.peak_memory),
    )


def _format_benchmark_metric(value: int | None) -> str:
    """Return one metric display value for text-oriented exports."""
    return "-" if value is None else str(value)


def _format_benchmark_bytes(value: int | None) -> str:
    """Return one byte metric display value for text-oriented exports."""
    return "-" if value is None else f"{value} bytes"


def _escape_benchmark_csv_cell(value: str) -> str:
    """Escape one CSV cell according to simple RFC4180 rules."""
    if any(character in value for character in ('"', ",", "\n", "\r")):
        return '"' + value.replace('"', '""') + '"'
    return value


def _escape_benchmark_latex_cell(value: str) -> str:
    """Escape one LaTeX table cell conservatively."""
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )
