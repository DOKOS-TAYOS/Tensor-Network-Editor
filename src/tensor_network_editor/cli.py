"""Command-line interface for editor and headless tensor-network workflows."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, cast

from ._headless_models import SpecAnalysisReport, SpecDiffResult
from .analysis import analyze_spec
from .api import generate_code, launch_tensor_network_editor, load_spec, save_spec
from .diffing import diff_specs
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
    TemplateParameters,
    build_template_spec,
    list_template_names,
    parse_template_parameters,
    serialize_template_definitions,
)
from .validation import validate_spec


class _CommandHandler(Protocol):
    """Callable stored on parsed subcommands."""

    def __call__(self, args: argparse.Namespace) -> int: ...


class _CommandNamespace(argparse.Namespace):
    """Parsed namespace for subcommands that install a handler."""

    handler: _CommandHandler


def build_parser() -> argparse.ArgumentParser:
    """Build the legacy parser used when no subcommand is provided."""
    parser = argparse.ArgumentParser(
        prog="tensor-network-editor",
        description="Launch the local tensor network editor in your browser.",
    )
    _add_edit_arguments(parser)
    return parser


def build_command_parser() -> argparse.ArgumentParser:
    """Build the parser used by headless CLI subcommands."""
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
    analyze_parser.add_argument("--dtype", default="float64")
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
    _add_output_format_argument(diff_parser)
    diff_parser.set_defaults(handler=_handle_diff)

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
    """Run the CLI and return a process-friendly exit code."""
    args_list = list(argv) if argv is not None else sys.argv[1:]
    try:
        if _should_use_legacy_parser(args_list):
            return _handle_edit(build_parser().parse_args(args_list))
        parsed_args = cast(
            _CommandNamespace, build_command_parser().parse_args(args_list)
        )
        return _dispatch_command(parsed_args)
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
        default=EngineName.TENSORNETWORK.value,
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


def _should_use_legacy_parser(args_list: list[str]) -> bool:
    """Return ``True`` when CLI arguments should launch the editor directly."""
    return not args_list or args_list[0].startswith("-")


def _handle_edit(args: argparse.Namespace) -> int:
    """Launch the browser editor using the legacy or explicit edit arguments."""
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
    report = analyze_spec(spec)
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
    result = diff_specs(before, after)
    if args.format == "json":
        _print_json(result.to_dict())
    else:
        _print_diff_text(result)
    return 0


def _handle_template_list(args: argparse.Namespace) -> int:
    """Print the built-in template definitions."""
    if args.format == "json":
        _print_json(serialize_template_definitions())
    else:
        definitions = serialize_template_definitions()
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
    parameters = _parse_template_cli_parameters(
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


def _parse_template_cli_parameters(
    template_name: str,
    raw_parameters: dict[str, int] | None,
) -> TemplateParameters:
    """Parse CLI template parameters using the public template helper."""
    return parse_template_parameters(template_name, raw_parameters)


def load_spec_for_lint(path: str) -> NetworkSpec:
    """Load a spec for linting without enforcing hard validation first."""
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
    """Print a compact text summary for analyze results."""
    network = report.network
    contraction = report.contraction
    print(
        "Network:"
        f" tensors={network.tensor_count},"
        f" edges={network.edge_count},"
        f" open_indices={network.open_index_count}"
    )
    if contraction is not None:
        print(
            "Contraction:"
            f" manual_flops={contraction.manual.summary.total_estimated_flops},"
            f" auto_strategy={contraction.automatic_strategy}"
        )


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
