"""Friendly project diagnostics for the tensor-network CLI."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import cast

from ...models import NetworkSpec, ValidationIssue
from ...types import JSONValue
from ..analysis._memory_dtypes import DEFAULT_MEMORY_DTYPE
from ..models._headless_models import LintReport, SpecAnalysisReport
from ._cli_benchmark import build_benchmark_report, serialize_benchmark_report_text

_BACKEND_IMPORTS: tuple[tuple[str, str], ...] = (
    ("numpy", "numpy"),
    ("torch", "torch"),
    ("tensornetwork", "tensornetwork"),
    ("quimb", "quimb"),
    ("tensorkrowch", "tensorkrowch"),
    ("opt_einsum", "opt_einsum"),
    ("PIL", "PIL"),
)


@dataclass(slots=True)
class DoctorReport:
    """Structured diagnostic report for one saved project."""

    ok: bool
    validation: dict[str, JSONValue]
    lint: dict[str, JSONValue]
    analysis: dict[str, JSONValue]
    benchmark: dict[str, JSONValue]
    backends: dict[str, JSONValue]
    warnings: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, JSONValue]:
        """Serialize the report with stable public keys."""
        return {
            "ok": self.ok,
            "validation": cast(JSONValue, self.validation),
            "lint": cast(JSONValue, self.lint),
            "analysis": cast(JSONValue, self.analysis),
            "benchmark": cast(JSONValue, self.benchmark),
            "backends": cast(JSONValue, self.backends),
            "warnings": cast(JSONValue, list(self.warnings)),
            "suggestions": cast(JSONValue, list(self.suggestions)),
        }


def build_doctor_report(
    spec: NetworkSpec,
    *,
    memory_dtype: str = DEFAULT_MEMORY_DTYPE,
    validate_spec: Callable[[NetworkSpec], list[ValidationIssue]],
    lint_spec: Callable[..., LintReport],
    analyze_spec: Callable[..., SpecAnalysisReport],
    find_spec: Callable[[str], object | None] = importlib.util.find_spec,
) -> DoctorReport:
    """Run validation, lint, analysis, benchmark, and backend checks."""
    validation_issues = validate_spec(spec)
    validation = _validation_payload(validation_issues)
    backends = _backend_payload(find_spec=find_spec)
    warnings: list[str] = []
    suggestions: list[str] = []

    if validation_issues:
        suggestions.append(
            "Fix validation errors before running analysis or benchmark."
        )
        return DoctorReport(
            ok=False,
            validation=validation,
            lint=_skipped_payload("Validation errors must be fixed first."),
            analysis=_skipped_payload("Validation errors must be fixed first."),
            benchmark=_skipped_benchmark_payload(memory_dtype),
            backends=backends,
            warnings=warnings,
            suggestions=suggestions,
        )

    lint_report = lint_spec(spec)
    lint = _lint_payload(lint_report)
    suggestions.extend(_lint_suggestions(lint_report))

    analysis_report = analyze_spec(spec, memory_dtype=memory_dtype)
    analysis = {
        "status": "complete",
        "report": cast(JSONValue, analysis_report.to_dict()),
    }
    if analysis_report.contraction is None:
        benchmark = _skipped_benchmark_payload(memory_dtype)
    else:
        benchmark_report = build_benchmark_report(analysis_report.contraction)
        benchmark = benchmark_report.to_dict()
        warnings.extend(benchmark_report.warnings)

    warnings.extend(_backend_warnings(backends))
    suggestions.extend(_backend_suggestions(backends))
    if not suggestions:
        suggestions.append("No immediate fixes needed.")

    return DoctorReport(
        ok=True,
        validation=validation,
        lint=lint,
        analysis=analysis,
        benchmark=benchmark,
        backends=backends,
        warnings=warnings,
        suggestions=suggestions,
    )


def format_doctor_report_text(report: DoctorReport, *, path: str) -> str:
    """Return a readable text rendering of a doctor report."""
    lines = [f"Doctor report for {path}", ""]
    lines.extend(_format_section("Validation", _format_validation(report.validation)))
    lines.extend(_format_section("Lint", _format_lint(report.lint)))
    lines.extend(_format_section("Analysis", _format_analysis(report.analysis)))
    lines.extend(_format_section("Benchmark", _format_benchmark(report.benchmark)))
    lines.extend(_format_section("Backends/Extras", _format_backends(report.backends)))
    if report.warnings:
        lines.extend(
            _format_section("Warnings", [f"- {warning}" for warning in report.warnings])
        )
    lines.extend(
        _format_section("Suggestions", [f"- {item}" for item in report.suggestions])
    )
    return "\n".join(lines)


def _validation_payload(issues: list[ValidationIssue]) -> dict[str, JSONValue]:
    """Return the validation subsection payload."""
    return {
        "status": "failed" if issues else "passed",
        "issue_count": len(issues),
        "issues": cast(
            JSONValue, [_validation_issue_to_dict(issue) for issue in issues]
        ),
    }


def _validation_issue_to_dict(issue: ValidationIssue) -> dict[str, JSONValue]:
    """Serialize a validation issue for doctor JSON output."""
    return {"code": issue.code, "message": issue.message, "path": issue.path}


def _lint_payload(report: LintReport) -> dict[str, JSONValue]:
    """Return the lint subsection payload."""
    status = "warnings" if report.has_warnings else "passed"
    if not report.issues:
        status = "passed"
    return {
        "status": status,
        "issue_count": len(report.issues),
        "issues": cast(JSONValue, [issue.to_dict() for issue in report.issues]),
    }


def _skipped_payload(reason: str) -> dict[str, JSONValue]:
    """Return a generic skipped subsection payload."""
    return {"status": "skipped", "reason": reason}


def _skipped_benchmark_payload(
    memory_dtype: str,
    reason: str = "Contraction analysis is unavailable.",
) -> dict[str, JSONValue]:
    """Return a skipped benchmark subsection with the usual public fields."""
    return {
        "status": "skipped",
        "reason": reason,
        "memory_dtype": memory_dtype,
        "warnings": cast(JSONValue, []),
        "rows": cast(JSONValue, []),
    }


def _backend_payload(
    *,
    find_spec: Callable[[str], object | None],
) -> dict[str, JSONValue]:
    """Return import availability for optional and core backends."""
    return {
        name: {
            "import_name": import_name,
            "available": find_spec(import_name) is not None,
        }
        for name, import_name in _BACKEND_IMPORTS
    }


def _backend_warnings(backends: dict[str, JSONValue]) -> list[str]:
    """Return warnings for commonly useful missing optional dependencies."""
    warnings: list[str] = []
    for backend_name in ("opt_einsum",):
        backend = backends.get(backend_name)
        if isinstance(backend, dict) and not backend.get("available"):
            warnings.append(
                f"{backend_name} is not installed; automatic contraction planning may be limited."
            )
    return warnings


def _backend_suggestions(backends: dict[str, JSONValue]) -> list[str]:
    """Return friendly suggestions based on optional dependency availability."""
    suggestions: list[str] = []
    pillow = backends.get("PIL")
    if isinstance(pillow, dict) and not pillow.get("available"):
        suggestions.append(
            "Install tensor-network-editor[png] to enable headless PNG rendering."
        )
    return suggestions


def _lint_suggestions(report: LintReport) -> list[str]:
    """Return suggestions copied from lint issues."""
    suggestions: list[str] = []
    for issue in report.issues:
        if issue.suggestion and issue.suggestion not in suggestions:
            suggestions.append(issue.suggestion)
    return suggestions


def _format_section(title: str, body: list[str]) -> list[str]:
    """Format a text section with a blank line after it."""
    return [f"{title}:", *body, ""]


def _format_validation(payload: dict[str, JSONValue]) -> list[str]:
    """Format the validation subsection."""
    if payload["status"] == "passed":
        return ["  OK: no validation errors."]
    lines = [f"  Failed with {payload['issue_count']} issue(s)."]
    for issue in cast(list[dict[str, JSONValue]], payload["issues"]):
        lines.append(f"  - [{issue['code']}] {issue['message']} ({issue['path']})")
    return lines


def _format_lint(payload: dict[str, JSONValue]) -> list[str]:
    """Format the lint subsection."""
    if payload["status"] == "skipped":
        return [f"  Skipped: {payload['reason']}"]
    if payload["issue_count"] == 0:
        return ["  OK: no lint issues."]
    lines = [f"  Reported {payload['issue_count']} issue(s)."]
    for issue in cast(list[dict[str, JSONValue]], payload["issues"]):
        lines.append(f"  - [{issue['severity']}:{issue['code']}] {issue['message']}")
    return lines


def _format_analysis(payload: dict[str, JSONValue]) -> list[str]:
    """Format the analysis subsection."""
    if payload["status"] == "skipped":
        return [f"  Skipped: {payload['reason']}"]
    report = cast(dict[str, JSONValue], payload["report"])
    network = cast(dict[str, JSONValue], report["network"])
    return [
        "  OK:"
        f" tensors={network['tensor_count']},"
        f" edges={network['edge_count']},"
        f" open_indices={network['open_index_count']}"
    ]


def _format_benchmark(payload: dict[str, JSONValue]) -> list[str]:
    """Format the benchmark subsection."""
    if payload.get("status") == "skipped":
        return [f"  Skipped: {payload['reason']}"]
    return [
        "  " + line
        for line in serialize_benchmark_report_text_from_payload(payload).splitlines()
    ]


def serialize_benchmark_report_text_from_payload(payload: dict[str, JSONValue]) -> str:
    """Rebuild a text benchmark table from a benchmark payload."""
    from ._cli_benchmark import BenchmarkReport, BenchmarkRow

    rows = [
        BenchmarkRow(
            key=str(row["key"]),
            name=str(row["name"]),
            status=str(row["status"]),
            message=cast(str | None, row["message"]),
            flop=cast(int | None, row["flop"]),
            mac=cast(int | None, row["mac"]),
            peak=cast(int | None, row["peak"]),
            peak_memory=cast(int | None, row["peak_memory"]),
        )
        for row in cast(list[dict[str, JSONValue]], payload["rows"])
    ]
    report = BenchmarkReport(
        memory_dtype=str(payload["memory_dtype"]),
        warnings=list(cast(list[str], payload.get("warnings", []))),
        rows=rows,
    )
    return serialize_benchmark_report_text(report)


def _format_backends(payload: dict[str, JSONValue]) -> list[str]:
    """Format the backend availability subsection."""
    lines: list[str] = []
    for name, backend in payload.items():
        backend_payload = cast(dict[str, JSONValue], backend)
        status = "available" if backend_payload["available"] else "missing"
        lines.append(f"  - {name}: {status}")
    return lines
