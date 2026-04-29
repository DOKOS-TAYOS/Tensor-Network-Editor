"""Parser-building helpers for the tensor-network CLI."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass

from ..._themes import DEFAULT_EDITOR_THEME, SUPPORTED_EDITOR_THEMES
from ...models import EngineName, TensorCollectionFormat
from ..analysis._memory_dtypes import DEFAULT_MEMORY_DTYPE, SUPPORTED_MEMORY_DTYPES
from ._logging import (
    DEFAULT_LOG_FILE_BACKUP_COUNT,
    DEFAULT_LOG_FILE_MAX_BYTES,
    LOG_LEVEL_NAMES,
)

CommandHandler = Callable[[argparse.Namespace], int]


@dataclass(slots=True, frozen=True)
class CliHandlerBindings:
    """Handler callbacks attached to CLI subcommands."""

    handle_edit: CommandHandler
    handle_validate: CommandHandler
    handle_lint: CommandHandler
    handle_analyze: CommandHandler
    handle_benchmark: CommandHandler
    handle_doctor: CommandHandler
    handle_export: CommandHandler
    handle_render: CommandHandler
    handle_diff: CommandHandler
    handle_canonicalize: CommandHandler
    handle_template_list: CommandHandler
    handle_template_build: CommandHandler
    handle_subnetwork_list: CommandHandler
    handle_subnetwork_save: CommandHandler
    handle_subnetwork_export: CommandHandler


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
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Write package logs to the requested file path.",
    )
    parser.add_argument(
        "--log-max-bytes",
        type=_build_positive_int_parser("--log-max-bytes"),
        default=DEFAULT_LOG_FILE_MAX_BYTES,
        help="Maximum size in bytes before --log-file rotates to a backup copy.",
    )
    parser.add_argument(
        "--log-backup-count",
        type=_build_positive_int_parser("--log-backup-count"),
        default=DEFAULT_LOG_FILE_BACKUP_COUNT,
        help="Number of rotated backup copies retained for --log-file.",
    )
    parser.add_argument(
        "--python-import-mode",
        choices=["static", "live"],
        default="static",
        help="Choose how Python files are imported when a command loads a .py source.",
    )
    parser.add_argument(
        "--python-reconstruction-level",
        choices=["auto", "simple", "best_available"],
        default="auto",
        help="Choose how much editor metadata to reconstruct from Python imports.",
    )
    parser.add_argument(
        "--python-object",
        type=str,
        default=None,
        help="Optional global object name used by live Python imports.",
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

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Run friendly diagnostics for a saved spec and local environment.",
    )
    doctor_parser.add_argument("path", type=str)
    doctor_parser.add_argument(
        "--dtype",
        choices=list(SUPPORTED_MEMORY_DTYPES),
        default=DEFAULT_MEMORY_DTYPE,
    )
    doctor_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
    )
    doctor_parser.set_defaults(handler=handlers.handle_doctor)

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

    render_parser = subparsers.add_parser(
        "render", help="Render a saved spec as a static image."
    )
    render_parser.add_argument("path", type=str)
    render_parser.add_argument(
        "--format",
        choices=["svg", "png", "pdf", "tikz", "dot"],
        default="svg",
    )
    render_parser.add_argument("--output", type=str)
    render_parser.add_argument(
        "--hide-tensor-names",
        dest="show_tensor_names",
        action="store_false",
        help="Hide tensor names in TikZ and DOT renders.",
    )
    render_parser.add_argument(
        "--hide-index-names",
        dest="show_index_names",
        action="store_false",
        help="Hide index names in TikZ and DOT renders.",
    )
    render_parser.add_argument(
        "--hide-bond-names",
        dest="show_bond_names",
        action="store_false",
        help="Hide bond names in TikZ and DOT renders.",
    )
    render_parser.set_defaults(
        show_tensor_names=True,
        show_index_names=True,
        show_bond_names=True,
    )
    render_parser.set_defaults(handler=handlers.handle_render)

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
    template_build_parser.add_argument("--depth", type=int)
    template_build_parser.add_argument("--bond-dimension", type=int)
    template_build_parser.add_argument("--physical-dimension", type=int)
    template_build_parser.add_argument(
        "--boundary-condition",
        choices=("open", "periodic"),
    )
    template_build_parser.add_argument("--j", type=float)
    template_build_parser.add_argument("--h", type=float)
    template_build_parser.add_argument(
        "--symmetry",
        choices=("none", "u1", "z2"),
    )
    template_build_parser.add_argument(
        "--initial-state",
        choices=("zeros", "random", "all_up", "all_down", "neel"),
    )
    template_build_parser.add_argument(
        "--leaf-physical-legs",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    template_build_parser.add_argument(
        "--root-open-leg",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    template_build_parser.add_argument(
        "--isometric",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    template_build_parser.add_argument("--output", type=str)
    _add_output_format_argument(template_build_parser)
    template_build_parser.set_defaults(handler=handlers.handle_template_build)

    subnetwork_parser = subparsers.add_parser(
        "subnetwork",
        help="Inspect, save, and export reusable subnetworks from project catalogs.",
    )
    subnetwork_subparsers = subnetwork_parser.add_subparsers(
        dest="subnetwork_command",
        required=True,
    )

    subnetwork_list_parser = subnetwork_subparsers.add_parser(
        "list",
        help="List reusable subnetworks available for one project spec path.",
    )
    subnetwork_list_parser.add_argument(
        "path",
        type=str,
        help="Saved spec path used to resolve the project catalog directory.",
    )
    subnetwork_list_parser.add_argument(
        "--shared-catalog-path",
        type=str,
        help="Optional shared reusable-subnetwork catalog path.",
    )
    _add_output_format_argument(subnetwork_list_parser)
    subnetwork_list_parser.set_defaults(handler=handlers.handle_subnetwork_list)

    subnetwork_save_parser = subnetwork_subparsers.add_parser(
        "save",
        help="Extract tensors from a spec and persist them into the project catalog.",
    )
    subnetwork_save_parser.add_argument(
        "path",
        type=str,
        help="Saved spec path to extract from.",
    )
    subnetwork_save_parser.add_argument(
        "--tensor-ids",
        nargs="+",
        required=True,
        help="Tensor ids to include in the saved reusable subnetwork.",
    )
    subnetwork_save_parser.add_argument(
        "--name",
        required=True,
        help="Catalog name for the saved reusable subnetwork.",
    )
    subnetwork_save_parser.add_argument(
        "--tags",
        nargs="*",
        default=[],
        help="Optional tags attached to the saved reusable subnetwork.",
    )
    subnetwork_save_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing project subnetwork with the same name.",
    )
    subnetwork_save_parser.set_defaults(handler=handlers.handle_subnetwork_save)

    subnetwork_export_parser = subnetwork_subparsers.add_parser(
        "export",
        help="Export one reusable subnetwork from the project/shared catalog.",
    )
    subnetwork_export_parser.add_argument(
        "path",
        type=str,
        help="Saved spec path used to resolve the project catalog directory.",
    )
    subnetwork_export_parser.add_argument(
        "subnetwork_name",
        type=str,
        help="Catalog name of the reusable subnetwork to export.",
    )
    subnetwork_export_parser.add_argument(
        "--shared-catalog-path",
        type=str,
        help="Optional shared reusable-subnetwork catalog path.",
    )
    subnetwork_export_parser.add_argument("--output", type=str)
    subnetwork_export_parser.set_defaults(handler=handlers.handle_subnetwork_export)
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
        "--theme",
        choices=list(SUPPORTED_EDITOR_THEMES),
        default=DEFAULT_EDITOR_THEME,
        help="Visual theme used by the browser editor.",
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


def _build_positive_int_parser(option_name: str) -> Callable[[str], int]:
    """Return an ``argparse`` parser for positive integer options."""

    def parse(raw_value: str) -> int:
        try:
            value = int(raw_value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{option_name} must be > 0.") from exc
        if value <= 0:
            raise argparse.ArgumentTypeError(f"{option_name} must be > 0.")
        return value

    return parse
