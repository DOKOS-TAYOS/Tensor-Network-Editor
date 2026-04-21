"""Parser-building helpers for the tensor-network CLI."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass

from ...models import EngineName, TensorCollectionFormat
from ..analysis._memory_dtypes import DEFAULT_MEMORY_DTYPE, SUPPORTED_MEMORY_DTYPES
from ._logging import LOG_LEVEL_NAMES

CommandHandler = Callable[[argparse.Namespace], int]


@dataclass(slots=True, frozen=True)
class CliHandlerBindings:
    """Handler callbacks attached to CLI subcommands."""

    handle_edit: CommandHandler
    handle_validate: CommandHandler
    handle_lint: CommandHandler
    handle_analyze: CommandHandler
    handle_benchmark: CommandHandler
    handle_export: CommandHandler
    handle_diff: CommandHandler
    handle_canonicalize: CommandHandler
    handle_template_list: CommandHandler
    handle_template_build: CommandHandler


def build_command_parser(handlers: CliHandlerBindings) -> argparse.ArgumentParser:
    """Build the parser used by headless CLI subcommands."""
    parser = argparse.ArgumentParser(
        prog="tensor-network-editor",
        description="Work with tensor-network specs from scripts, terminals, and pipelines.",
    )
    parser.add_argument(
        "--log-level",
        choices=list(LOG_LEVEL_NAMES),
        default=None,
        help="Enable package logs at the requested severity.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    edit_parser = subparsers.add_parser(
        "edit", help="Launch the local editor in the browser."
    )
    _add_edit_arguments(edit_parser)
    edit_parser.set_defaults(handler=handlers.handle_edit)

    validate_parser = subparsers.add_parser(
        "validate", help="Validate a saved spec or supported generated Python file."
    )
    validate_parser.add_argument("path", type=str)
    _add_output_format_argument(validate_parser)
    validate_parser.set_defaults(handler=handlers.handle_validate)

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
    lint_parser.set_defaults(handler=handlers.handle_lint)

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
    analyze_parser.set_defaults(handler=handlers.handle_analyze)

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Compare manual and automatic contraction variants for a saved spec.",
    )
    benchmark_parser.add_argument("path", type=str)
    benchmark_parser.add_argument(
        "--dtype",
        choices=list(SUPPORTED_MEMORY_DTYPES),
        default=DEFAULT_MEMORY_DTYPE,
    )
    benchmark_parser.add_argument(
        "--format",
        choices=["text", "json", "csv", "latex"],
        default="text",
    )
    benchmark_parser.add_argument("--output", type=str)
    benchmark_parser.set_defaults(handler=handlers.handle_benchmark)

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
    export_parser.set_defaults(handler=handlers.handle_export)

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
    diff_parser.set_defaults(handler=handlers.handle_diff)

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
    canonicalize_parser.set_defaults(handler=handlers.handle_canonicalize)

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
    template_list_parser.set_defaults(handler=handlers.handle_template_list)

    template_build_parser = template_subparsers.add_parser(
        "build", help="Build a spec from a built-in template."
    )
    template_build_parser.add_argument("template_name", type=str)
    template_build_parser.add_argument("--graph-size", type=int)
    template_build_parser.add_argument("--bond-dimension", type=int)
    template_build_parser.add_argument("--physical-dimension", type=int)
    template_build_parser.add_argument("--output", type=str)
    _add_output_format_argument(template_build_parser)
    template_build_parser.set_defaults(handler=handlers.handle_template_build)
    return parser


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
