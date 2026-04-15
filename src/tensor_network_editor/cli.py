"""Command-line interface for editor and headless tensor-network workflows."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, cast

from ._headless_models import (
    SemanticDiffEntry,
    SemanticSpecDiffResult,
    SpecAnalysisReport,
    SpecDiffResult,
)
from ._memory_dtypes import DEFAULT_MEMORY_DTYPE, SUPPORTED_MEMORY_DTYPES
from .analysis import analyze_spec
from .api import generate_code, launch_tensor_network_editor, load_spec, save_spec
from .canonicalization import canonicalize_spec
from .diffing import diff_specs, semantic_diff_specs
from .errors import (
    CodeGenerationError,
    PackageIOError,
    SerializationError,
    SpecValidationError,
)
from .linting import LintReport, lint_spec
from .models import EngineName, NetworkSpec, TensorCollectionFormat, ValidationIssue
from .serialization import (
    deserialize_spec,
    deserialize_spec_from_python_code,
    serialize_spec,
)
from .templates import (
    build_template_spec,
    list_template_names,
    parse_template_parameters,
    serialize_template_definitions,
)
from .validation import validate_spec


class _CommandHandler(Protocol):
    """Callable stored on parsed subcommands."""

    def __call__(self, args: argparse.Namespace) -> int:
        """Run one parsed subcommand and return its process exit code."""
        ...


class _CommandNamespace(argparse.Namespace):
    """Parsed namespace for subcommands that install a handler."""

    handler: _CommandHandler


def build_command_parser() -> argparse.ArgumentParser:
    """Build the parser used by headless CLI subcommands.

    Returns:
        The fully configured top-level CLI parser.
    """
    parser = argparse.ArgumentParser(
        prog="tensor-network-editor",
        description="Work with tensor-network specs from scripts, terminals, and pipelines.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    edit_parser = subparsers.add_parser(
        "edit", help="Launch the local editor in the browser."
    )
    _add_edit_arguments(edit_parser)
    edit_parser.set_defaults(handler=_handle_edit)

    validate_parser = subparsers.add_parser(
        "validate", help="Validate a saved spec or supported generated Python file."
    )
    validate_parser.add_argument("path", type=str)
    _add_output_format_argument(validate_parser)
    validate_parser.set_defaults(handler=_handle_validate)

    lint_parser = subparsers.add_parser(
        "lint", help="Run soft diagnostics on a saved spec or generated Python file."
    )
    lint_parser.add_argument("path", type=str)
    lint_parser.add_argument("--max-tensor-rank", type=int, default=6)
    lint_parser.add_argument("--max-tensor-cardinality", type=int, default=4096)
    lint_parser.add_argument(
        "--fail-on",
        choices=["none", "warning"],
        default="none",
        help="Return exit code 1 when warnings are present.",
    )
    _add_output_format_argument(lint_parser)
    lint_parser.set_defaults(handler=_handle_lint)

    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze structure and contraction metrics for a saved spec."
    )
    analyze_parser.add_argument("path", type=str)
    analyze_parser.add_argument(
        "--dtype",
        choices=list(SUPPORTED_MEMORY_DTYPES),
        default=DEFAULT_MEMORY_DTYPE,
    )
    _add_output_format_argument(analyze_parser)
    analyze_parser.set_defaults(handler=_handle_analyze)

    export_parser = subparsers.add_parser(
        "export", help="Generate backend Python code from a saved spec."
    )
    export_parser.add_argument("path", type=str)
    export_parser.add_argument(
        "--engine",
        choices=[engine.value for engine in EngineName],
        required=True,
    )
    export_parser.add_argument(
        "--collection-format",
        choices=[
            collection_format.value for collection_format in TensorCollectionFormat
        ],
        default=TensorCollectionFormat.LIST.value,
    )
    export_parser.add_argument("--output", type=str)
    export_parser.set_defaults(handler=_handle_export)

    diff_parser = subparsers.add_parser(
        "diff", help="Compare two specs and summarize entity-level changes."
    )
    diff_parser.add_argument("before", type=str)
    diff_parser.add_argument("after", type=str)
    diff_parser.add_argument(
        "--semantic",
        action="store_true",
        help="Report field-level semantic changes instead of id-level summaries.",
    )
    _add_output_format_argument(diff_parser)
    diff_parser.set_defaults(handler=_handle_diff)

    canonicalize_parser = subparsers.add_parser(
        "canonicalize",
        help="Canonicalize a saved spec with stable ordering and optional deterministic ids.",
    )
    canonicalize_parser.add_argument("path", type=str)
    canonicalize_parser.add_argument("--output", type=str)
    canonicalize_parser.add_argument(
        "--deterministic-ids",
        action="store_true",
        help="Rewrite ids deterministically in canonical order.",
    )
    canonicalize_parser.set_defaults(handler=_handle_canonicalize)

    template_parser = subparsers.add_parser(
        "template", help="Inspect or build the built-in template catalog."
    )
    template_subparsers = template_parser.add_subparsers(
        dest="template_command", required=True
    )

    template_list_parser = template_subparsers.add_parser(
        "list", help="List the built-in template definitions."
    )
    _add_output_format_argument(template_list_parser)
    template_list_parser.set_defaults(handler=_handle_template_list)

    template_build_parser = template_subparsers.add_parser(
        "build", help="Build a spec from a built-in template."
    )
    template_build_parser.add_argument("template_name", type=str)
    template_build_parser.add_argument("--graph-size", type=int)
    template_build_parser.add_argument("--bond-dimension", type=int)
    template_build_parser.add_argument("--physical-dimension", type=int)
    template_build_parser.add_argument("--output", type=str)
    _add_output_format_argument(template_build_parser)
    template_build_parser.set_defaults(handler=_handle_template_build)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process-friendly exit code.

    Args:
        argv: Optional command-line arguments. When omitted, ``sys.argv[1:]`` is
            used.

    Returns:
        A shell-friendly exit code.
    """
    args_list = list(argv) if argv is not None else sys.argv[1:]
    try:
        parsed_args = cast(
            _CommandNamespace, build_command_parser().parse_args(args_list)
        )
        return _dispatch_command(parsed_args)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 2
    except KeyboardInterrupt:
        return 130
    except SpecValidationError as exc:
        _print_validation_result(exc.issues, output_format="text")
        return 1
    except (CodeGenerationError, PackageIOError, SerializationError, ValueError) as exc:
        print(str(exc))
        return 2


def _dispatch_command(args: _CommandNamespace) -> int:
    """Run the command handler stored on the parsed namespace."""
    return args.handler(args)


def _add_edit_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach editor-launch arguments to the provided parser."""
    parser.add_argument(
        "--engine",
        choices=[engine.value for engine in EngineName],
        default=EngineName.TENSORKROWCH.value,
        help="Default target engine shown in the editor.",
    )
    parser.add_argument(
        "--load",
        type=str,
        help="Optional path to a saved JSON design to preload.",
    )
    parser.add_argument(
        "--save-code",
        type=str,
        help="Optional output path for the generated Python code when the editor is confirmed.",
    )
    parser.add_argument(
        "--print-code",
        action="store_true",
        help="Print generated code to stdout when the editor is confirmed.",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Start the local server without opening the browser automatically.",
    )


def _add_output_format_argument(parser: argparse.ArgumentParser) -> None:
    """Attach a standard text/json output selector to ``parser``."""
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
    )


def _handle_edit(args: argparse.Namespace) -> int:
    """Launch the browser editor using explicit edit arguments."""
    initial_spec = load_spec(args.load) if args.load else None
    launch_tensor_network_editor(
        initial_spec=initial_spec,
        default_engine=EngineName(args.engine),
        open_browser=not args.no_browser,
        print_code=args.print_code,
        code_path=args.save_code,
    )
    return 0


def _handle_validate(args: argparse.Namespace) -> int:
    """Validate a spec file and emit text or JSON results."""
    spec = load_spec(args.path)
    issues = validate_spec(spec)
    _print_validation_result(issues, output_format=args.format)
    return 1 if issues else 0


def _handle_lint(args: argparse.Namespace) -> int:
    """Run the soft linter against a spec file."""
    spec = load_spec_for_lint(args.path)
    report = lint_spec(
        spec,
        max_tensor_rank=args.max_tensor_rank,
        max_tensor_cardinality=args.max_tensor_cardinality,
    )
    _print_lint_result(report, output_format=args.format)
    if args.fail_on == "warning" and report.has_warnings:
        return 1
    return 0


def _handle_analyze(args: argparse.Namespace) -> int:
    """Analyze structure and contraction metrics for a saved spec."""
    spec = load_spec(args.path)
    report = analyze_spec(spec, memory_dtype=args.dtype)
    if args.format == "json":
        _print_json(report.to_dict())
    else:
        _print_analysis_text(report)
    return 0


def _handle_export(args: argparse.Namespace) -> int:
    """Generate backend code from a saved spec without launching the editor."""
    spec = load_spec(args.path)
    generate_code(
        spec,
        engine=EngineName(args.engine),
        collection_format=TensorCollectionFormat(args.collection_format),
        print_code=args.output is None,
        path=args.output,
    )
    if args.output is not None:
        print(f"Wrote generated code to {args.output}")
    return 0


def _handle_diff(args: argparse.Namespace) -> int:
    """Compare two specs and print the resulting structured diff."""
    before = load_spec(args.before)
    after = load_spec(args.after)
    if args.semantic:
        result = semantic_diff_specs(before, after)
        if args.format == "json":
            _print_json(result.to_dict())
        else:
            _print_semantic_diff_text(result)
        return 0
    result = diff_specs(before, after)
    if args.format == "json":
        _print_json(result.to_dict())
    else:
        _print_diff_text(result)
    return 0


def _handle_canonicalize(args: argparse.Namespace) -> int:
    """Canonicalize a spec and print or save the normalized result."""
    spec = load_spec(args.path)
    canonical_spec = canonicalize_spec(spec, deterministic_ids=args.deterministic_ids)
    if args.output is not None:
        save_spec(canonical_spec, args.output)
        print(f"Wrote canonical spec to {args.output}")
        return 0
    _print_json(serialize_spec(canonical_spec))
    return 0


def _handle_template_list(args: argparse.Namespace) -> int:
    """Print the built-in template definitions."""
    definitions = serialize_template_definitions()
    if args.format == "json":
        _print_json(definitions)
    else:
        for template_name in list_template_names():
            definition = definitions[template_name]
            print(f"{template_name}: {definition['display_name']}")
    return 0


def _handle_template_build(args: argparse.Namespace) -> int:
    """Build a template spec and print or save the resulting serialized spec."""
    raw_parameters = {
        key: value
        for key, value in {
            "graph_size": args.graph_size,
            "bond_dimension": args.bond_dimension,
            "physical_dimension": args.physical_dimension,
        }.items()
        if value is not None
    }
    parameters = parse_template_parameters(
        args.template_name,
        raw_parameters if raw_parameters else None,
    )
    spec = build_template_spec(args.template_name, parameters)
    if args.output is not None:
        save_spec(spec, args.output)
        print(f"Wrote template spec to {args.output}")
        return 0
    _print_json(serialize_spec(spec))
    return 0


def load_spec_for_lint(path: str) -> NetworkSpec:
    """Load a spec for linting without enforcing hard validation first.

    Args:
        path: Path to a serialized JSON design or supported generated Python
            file.

    Returns:
        The deserialized network specification.

    Raises:
        SerializationError: If the payload cannot be parsed into a valid spec
            shape.
    """
    from ._io import read_utf8_text

    source_path = Path(path)
    if source_path.suffix.lower() == ".py":
        return deserialize_spec_from_python_code(
            read_utf8_text(path, description="generated Python code"),
            validate=False,
        )
    try:
        payload = json.loads(
            read_utf8_text(path, description="network specification JSON")
        )
    except json.JSONDecodeError as exc:
        raise SerializationError("Could not parse network specification JSON.") from exc
    if not isinstance(payload, dict):
        raise SerializationError("Serialized network must be a JSON object.")
    return deserialize_spec(payload, validate=False)


def _print_json(payload: object) -> None:
    """Print a JSON payload with deterministic formatting."""
    print(json.dumps(payload, indent=2))


def _print_validation_result(
    issues: list[ValidationIssue],
    *,
    output_format: str,
) -> None:
    """Print validation results in text or JSON format."""
    if output_format == "json":
        _print_json(
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


def _print_lint_result(report: LintReport, *, output_format: str) -> None:
    """Print lint results in text or JSON format."""
    if output_format == "json":
        _print_json(report.to_dict())
        return
    if not report.issues:
        print("No lint issues found.")
        return
    print(f"Linter reported {len(report.issues)} issue(s):")
    for issue in report.issues:
        print(f"- [{issue.severity}:{issue.code}] {issue.message} ({issue.path})")


def _print_analysis_text(report: SpecAnalysisReport) -> None:
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
    if not isinstance(shape, tuple) or not shape:
        return "n/a"
    return " x ".join(str(int(dimension)) for dimension in shape)


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
    if not isinstance(labels, tuple):
        return "n/a"
    if not labels:
        return "none"
    return ", ".join(str(label) for label in labels)


def _print_diff_text(result: SpecDiffResult) -> None:
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


def _print_semantic_diff_text(result: SemanticSpecDiffResult) -> None:
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
