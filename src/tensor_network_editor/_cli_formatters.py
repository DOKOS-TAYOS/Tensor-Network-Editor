"""Formatting and output helpers for the tensor-network CLI."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from ._headless_models import (
    SemanticDiffEntry,
    SemanticSpecDiffResult,
    SpecAnalysisReport,
    SpecDiffResult,
)
from ._memory_dtypes import DEFAULT_MEMORY_DTYPE
from .linting import LintReport
from .models import ValidationIssue


def print_json(payload: object) -> None:
    """Print a JSON payload with deterministic formatting."""
    print(json.dumps(payload, indent=2))


def print_validation_result(
    issues: list[ValidationIssue],
    *,
    output_format: str,
) -> None:
    """Print validation results in text or JSON format."""
    if output_format == "json":
        print_json(
            {
                "issues": [
                    {"code": issue.code, "message": issue.message, "path": issue.path}
                    for issue in issues
                ]
            }
        )
        return
    if not issues:
        print("Specification is valid.")
        return
    print(f"Validation found {len(issues)} issue(s):")
    for issue in issues:
        print(f"- [{issue.code}] {issue.message} ({issue.path})")


def print_lint_result(report: LintReport, *, output_format: str) -> None:
    """Print lint results in text or JSON format."""
    if output_format == "json":
        print_json(report.to_dict())
        return
    if not report.issues:
        print("No lint issues found.")
        return
    print(f"Linter reported {len(report.issues)} issue(s):")
    for issue in report.issues:
        print(f"- [{issue.severity}:{issue.code}] {issue.message} ({issue.path})")


def print_analysis_text(report: SpecAnalysisReport) -> None:
    """Print a readable text summary for analyze results."""
    network = report.network
    print(
        "Network:"
        f" tensors={network.tensor_count},"
        f" edges={network.edge_count},"
        f" open_indices={network.open_index_count}"
    )
    contraction = report.contraction
    if contraction is None:
        return
    manual = contraction.manual
    print(
        "Manual:"
        f" status={manual.status},"
        f" flops={_format_metric(manual.summary.total_estimated_flops)},"
        f" macs={_format_metric(manual.summary.total_estimated_macs)},"
        f" peak={_format_metric(manual.summary.peak_intermediate_size)},"
        f" shape={_format_shape(manual.summary.final_shape)}"
    )
    _print_automatic_analysis_text("Automatic full", contraction.automatic_full)
    _print_automatic_analysis_text("Automatic future", contraction.automatic_future)
    _print_automatic_analysis_text("Automatic past", contraction.automatic_past)
    _print_comparison_text(
        "Manual vs automatic full",
        contraction.comparisons.get("manual_vs_automatic_full"),
    )
    _print_comparison_text(
        "Manual subtrees vs automatic past",
        contraction.comparisons.get("manual_subtrees_vs_automatic_past"),
    )


def _print_automatic_analysis_text(label: str, analysis: object) -> None:
    """Print one automatic analysis summary line."""
    status = getattr(analysis, "status", "unknown")
    summary = getattr(analysis, "summary", None)
    message = getattr(analysis, "message", None)
    if summary is None:
        print(f"{label}: status={status}")
        return
    line = (
        f"{label}:"
        f" status={status},"
        f" flops={_format_metric(getattr(summary, 'total_estimated_flops', 0))},"
        f" macs={_format_metric(getattr(summary, 'total_estimated_macs', 0))},"
        f" peak={_format_metric(getattr(summary, 'peak_intermediate_size', 0))}"
    )
    if isinstance(message, str) and message:
        line += f", note={message}"
    print(line)


def _print_comparison_text(label: str, comparison: object | None) -> None:
    """Print one contraction comparison in a readable text block."""
    if comparison is None:
        return
    status = getattr(comparison, "status", "unknown")
    print(f"Comparison {label}:")
    if status != "complete":
        message = getattr(comparison, "message", None)
        if isinstance(message, str) and message:
            print(f"  Status: {status} ({message})")
        else:
            print(f"  Status: {status}")
        return
    baseline_label = str(getattr(comparison, "baseline_label", "baseline"))
    candidate_label = str(getattr(comparison, "candidate_label", "candidate"))
    memory_dtype = str(getattr(comparison, "memory_dtype", DEFAULT_MEMORY_DTYPE))
    print(
        "  FLOP"
        f" {_describe_delta(getattr(comparison, 'delta_total_estimated_flops', 0))}"
    )
    print(
        f"  MAC {_describe_delta(getattr(comparison, 'delta_total_estimated_macs', 0))}"
    )
    print(
        "  Peak size"
        f" {_describe_delta(getattr(comparison, 'delta_peak_intermediate_size', 0))}"
    )
    print(
        "  Peak memory"
        f" {_describe_delta(getattr(comparison, 'delta_peak_intermediate_bytes', 0), unit='bytes')}"
        f" ({memory_dtype})"
    )
    print(
        "  Peak bytes:"
        f" {baseline_label}={_format_metric(getattr(comparison, 'baseline_peak_intermediate_bytes', 0))} bytes,"
        f" {candidate_label}={_format_metric(getattr(comparison, 'candidate_peak_intermediate_bytes', 0))} bytes"
    )
    print(
        "  Peak steps:"
        f" {baseline_label}={_format_text_value(getattr(comparison, 'baseline_peak_step_id', None))},"
        f" {candidate_label}={_format_text_value(getattr(comparison, 'candidate_peak_step_id', None))}"
    )
    print(
        "  Bottlenecks:"
        f" {baseline_label}={_format_label_list(getattr(comparison, 'baseline_bottleneck_labels', ()))}"
        f" | {candidate_label}={_format_label_list(getattr(comparison, 'candidate_bottleneck_labels', ()))}"
    )


def _coerce_int(value: object) -> int:
    """Normalize integer-like values used by text summaries."""
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(
                f"Expected an integer-like value, got non-integer float {value}."
            )
        return int(value)
    if isinstance(value, str):
        normalized_value = value.strip()
        if not normalized_value:
            raise ValueError("Expected an integer-like string, got an empty string.")
        if re.fullmatch(r"[+-]?\d+", normalized_value) is None:
            raise ValueError(f"Expected an integer-like string, got {value!r}.")
        return int(normalized_value)
    raise TypeError(f"Expected an integer-like value, got {type(value).__name__}.")


def _format_metric(value: object) -> str:
    """Format a numeric metric with grouping separators."""
    try:
        normalized_value = _coerce_int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer metric value {value!r}: {exc}") from exc
    return f"{normalized_value:,}"


def _format_shape(shape: object) -> str:
    """Format a result shape for text output."""
    if shape == ():
        return "scalar"
    if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes)) or not shape:
        return "n/a"
    dimensions: list[str] = []
    for dimension in shape:
        try:
            dimensions.append(str(int(dimension)))
        except (TypeError, ValueError):
            return "n/a"
    return " x ".join(dimensions)


def _describe_delta(value: object, *, unit: str = "") -> str:
    """Describe whether one metric went up, down, or stayed unchanged."""
    try:
        normalized_value = _coerce_int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer delta value {value!r}: {exc}") from exc
    if normalized_value == 0:
        return "is unchanged"
    direction = "down" if normalized_value < 0 else "up"
    magnitude = _format_metric(abs(normalized_value))
    suffix = f" {unit}" if unit else ""
    return f"{direction} by {magnitude}{suffix}"


def _format_text_value(value: object | None) -> str:
    """Format optional text values for analysis output."""
    if value is None:
        return "n/a"
    return str(value)


def _format_label_list(labels: object) -> str:
    """Format bottleneck labels for analysis output."""
    if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)):
        return "n/a"
    if not labels:
        return "none"
    return ", ".join(str(label) for label in labels)


def print_diff_text(result: SpecDiffResult) -> None:
    """Print a compact text summary for diff results."""
    changes_by_entity = {
        "tensor": result.tensor,
        "edge": result.edge,
        "group": result.group,
        "note": result.note,
        "plan": result.plan,
    }
    for entity_name, changes in changes_by_entity.items():
        if not (changes.added or changes.removed or changes.changed):
            continue
        print(
            f"{entity_name}:"
            f" added={len(changes.added)},"
            f" removed={len(changes.removed)},"
            f" changed={len(changes.changed)}"
        )


def print_semantic_diff_text(result: SemanticSpecDiffResult) -> None:
    """Print semantic diff entries grouped by entity type."""
    if not result.entries:
        print("No semantic changes found.")
        return
    grouped_entries: dict[str, list[SemanticDiffEntry]] = {}
    for entry in result.entries:
        grouped_entries.setdefault(entry.entity_type, []).append(entry)
    for entity_type in (
        "tensor",
        "index",
        "edge",
        "group",
        "note",
        "plan",
        "step",
        "linear_periodic_chain",
    ):
        entries = grouped_entries.get(entity_type)
        if not entries:
            continue
        print(f"{_semantic_entity_heading(entity_type)}:")
        for entry in entries:
            print(f"- {entry.entity_id}: {entry.summary}")
            for field_change in entry.field_changes:
                print(
                    f"  {field_change.path}: "
                    f"{json.dumps(field_change.before)} -> {json.dumps(field_change.after)}"
                )


def _semantic_entity_heading(entity_type: str) -> str:
    """Return the display heading for one semantic diff entity group."""
    return {
        "tensor": "Tensors",
        "index": "Indices",
        "edge": "Edges",
        "group": "Groups",
        "note": "Notes",
        "plan": "Contraction Plans",
        "step": "Contraction Steps",
        "linear_periodic_chain": "Linear Periodic Chain",
    }.get(entity_type, entity_type.replace("_", " ").title())
