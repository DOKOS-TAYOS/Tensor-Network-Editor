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
    _extend_unique(suggestions, _lint_suggestions(lint_report))
    _extend_unique(suggestions, _model_suggestions_from_lint(lint_report))

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
        _extend_unique(
            suggestions,
            _contraction_suggestions(analysis_report.contraction),
        )

    warnings.extend(_backend_warnings(backends))
    _extend_unique(suggestions, _backend_suggestions(backends, spec=spec))
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


def _backend_suggestions(
    backends: dict[str, JSONValue],
    *,
    spec: NetworkSpec,
) -> list[str]:
    """Return friendly suggestions based on backend dependency availability."""
    suggestions: list[str] = []
    recommended_backend = _recommended_backend_suggestion(backends, spec=spec)
    if recommended_backend is not None:
        suggestions.append(recommended_backend)
    pillow = backends.get("PIL")
    if isinstance(pillow, dict) and not pillow.get("available"):
        suggestions.append(
            "Pillow is missing; reinstall the package or add Pillow to enable headless PNG/PDF rendering."
        )
    return suggestions


def _recommended_backend_suggestion(
    backends: dict[str, JSONValue],
    *,
    spec: NetworkSpec,
) -> str | None:
    """Return a conservative backend recommendation for the current environment."""
    is_periodic = (
        spec.linear_periodic_chain is not None
        or spec.grid_periodic_grid is not None
        or spec.tree_periodic_tree is not None
    )
    if is_periodic and _backend_available(backends, "quimb"):
        return (
            "Recommended backend: quimb is available and is a good fit for "
            "periodic array-style exports."
        )
    if _backend_available(backends, "quimb"):
        return (
            "Recommended backend: quimb is available for tensor-network-oriented "
            "exports and inspection."
        )
    if _backend_available(backends, "numpy"):
        return (
            "Recommended backend: einsum_numpy is available and is the lightest "
            "portable array-code export."
        )
    if _backend_available(backends, "torch"):
        return (
            "Recommended backend: einsum_torch is available when you want a "
            "PyTorch-based array export."
        )
    if _backend_available(backends, "tensornetwork"):
        return (
            "Recommended backend: tensornetwork is available for graph-style "
            "tensor-network exports."
        )
    if _backend_available(backends, "tensorkrowch"):
        return (
            "Recommended backend: tensorkrowch is available for graph-style "
            "tensor-network exports."
        )
    return None


def _backend_available(backends: dict[str, JSONValue], backend_name: str) -> bool:
    """Return whether one backend payload reports an available import."""
    backend = backends.get(backend_name)
    return isinstance(backend, dict) and bool(backend.get("available"))


def _lint_suggestions(report: LintReport) -> list[str]:
    """Return suggestions copied from lint issues."""
    suggestions: list[str] = []
    for issue in report.issues:
        if issue.suggestion and issue.suggestion not in suggestions:
            suggestions.append(issue.suggestion)
    return suggestions


def _model_suggestions_from_lint(report: LintReport) -> list[str]:
    """Return higher-level model suggestions derived from lint signals."""
    codes = {issue.code for issue in report.issues}
    suggestions: list[str] = []
    if "suspicious-open-index" in codes or "bond-leg-open-index" in codes:
        suggestions.append(
            "Review suspicious open indices; connect accidental dangling bonds, "
            "or rename/document intentional output legs in metadata."
        )
    if "disconnected-components" in codes:
        suggestions.append(
            "Review disconnected components; connect them before comparing contraction "
            "plans, or split independent components into separate specs."
        )
    if "large-tensor-rank" in codes or "large-tensor-cardinality" in codes:
        suggestions.append(
            "Inspect large tensor dimensions; a smaller bond dimension or a tensor "
            "decomposition may make exports and contractions easier."
        )
    return suggestions


def _contraction_suggestions(analysis: object) -> list[str]:
    """Return suggestions from manual-vs-automatic contraction analysis."""
    from ..analysis._contraction_analysis_types import ContractionAnalysisResult

    if not isinstance(analysis, ContractionAnalysisResult):
        return []
    suggestions: list[str] = []
    manual = analysis.manual
    automatic_full = analysis.automatic_full
    if manual.status != "unavailable" and automatic_full.status == "complete":
        comparison = _manual_auto_full_comparison_suggestion(analysis)
        if comparison is not None:
            suggestions.append(comparison)
    if (
        manual.status != "unavailable"
        and manual.summary.completion_status != "complete"
        and len(manual.summary.remaining_operand_ids) > 1
        and analysis.automatic_future.status == "complete"
    ):
        suggestions.append(
            "Auto future can complete the remaining manual frontier "
            f"({len(manual.summary.remaining_operand_ids)} active operands). "
            "Consider using it to finish the contraction suffix."
        )
    return suggestions


def _manual_auto_full_comparison_suggestion(
    analysis: object,
) -> str | None:
    """Return one suggestion when auto-full is clearly cheaper than manual."""
    from ..analysis._contraction_analysis_types import ContractionAnalysisResult

    if not isinstance(analysis, ContractionAnalysisResult):
        return None
    manual_summary = analysis.manual.summary
    automatic_summary = analysis.automatic_full.summary
    cheaper_metrics: list[str] = []
    if _manual_metric_is_worse(
        manual_summary.total_estimated_flops,
        automatic_summary.total_estimated_flops,
    ):
        cheaper_metrics.append(
            "FLOP "
            f"{manual_summary.total_estimated_flops} vs {automatic_summary.total_estimated_flops}"
        )
    if _manual_metric_is_worse(
        manual_summary.total_estimated_macs,
        automatic_summary.total_estimated_macs,
    ):
        cheaper_metrics.append(
            "MAC "
            f"{manual_summary.total_estimated_macs} vs {automatic_summary.total_estimated_macs}"
        )
    if _manual_metric_is_worse(
        manual_summary.peak_intermediate_bytes,
        automatic_summary.peak_intermediate_bytes,
    ):
        cheaper_metrics.append(
            "peak memory "
            f"{manual_summary.peak_intermediate_bytes} vs {automatic_summary.peak_intermediate_bytes} bytes"
        )
    if not cheaper_metrics:
        return None
    return (
        "Auto full is cheaper than the saved manual plan ("
        + ", ".join(cheaper_metrics)
        + "). Consider replacing the manual plan or using auto-full as a baseline."
    )


def _manual_metric_is_worse(manual_value: int, automatic_value: int) -> bool:
    """Return whether manual is at least 25 percent worse than automatic."""
    return automatic_value > 0 and manual_value >= automatic_value * 1.25


def _extend_unique(target: list[str], values: list[str]) -> None:
    """Append each value once, preserving order."""
    for value in values:
        if value not in target:
            target.append(value)


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
