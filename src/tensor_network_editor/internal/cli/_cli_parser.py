"""Parser-building helpers for the tensor-network CLI."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..._themes import DEFAULT_EDITOR_THEME, SUPPORTED_EDITOR_THEMES
from ...models import EngineName, TensorCollectionFormat
from ..analysis._memory_dtypes import DEFAULT_MEMORY_DTYPE, SUPPORTED_MEMORY_DTYPES
from ._logging import (
    DEFAULT_LOG_FILE_BACKUP_COUNT,
    DEFAULT_LOG_FILE_MAX_BYTES,
    LOG_LEVEL_NAMES,
)

CommandHandler = Callable[[argparse.Namespace], int]
COMMAND_ARGUMENT_QUICK_REFERENCE = """\
Command argument quick reference:
  tensor-network-editor edit [--ui browser|pywebview|server] [--load PATH]
  tensor-network-editor validate PATH [--format text|json]
  tensor-network-editor lint PATH [--fail-on none|warning] [--format text|json]
  tensor-network-editor analyze PATH [--dtype DTYPE] [--format text|json]
  tensor-network-editor benchmark PATH [--format text|json|csv|latex] [--output FILE]
  tensor-network-editor doctor PATH [--format text|json]
  tensor-network-editor export PATH --engine ENGINE [--output FILE]
  tensor-network-editor render PATH [--format svg|png|pdf|tikz|dot|mermaid] [--output FILE]
  tensor-network-editor diff BEFORE AFTER [--semantic] [--format text|json]
  tensor-network-editor canonicalize PATH [--deterministic-ids] [--output FILE]
  tensor-network-editor template list [--format text|json]
  tensor-network-editor template build TEMPLATE_NAME [options]
  tensor-network-editor subnetwork list PATH [--format text|json]
  tensor-network-editor subnetwork save PATH --tensor-ids ID... --name NAME
  tensor-network-editor subnetwork export PATH SUBNETWORK_NAME [--output FILE]

Global options, such as --log-level and --python-import-mode, go before the
command. Run 'tensor-network-editor <command> --help' or
'tensor-network-editor <command> <subcommand> --help' for the full argument list.
"""


class _CliHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    """Render readable CLI help with defaults and preserved epilog layout."""

    def _get_help_string(self, action: argparse.Action) -> str | None:
        """Hide unhelpful ``None`` defaults while keeping meaningful defaults."""
        if action.default is None or action.required:
            return action.help
        return super()._get_help_string(action)


class _CliArgumentParser(argparse.ArgumentParser):
    """ArgumentParser variant used for every CLI parser level."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Use the project help formatter unless a caller overrides it."""
        kwargs.setdefault("formatter_class", _CliHelpFormatter)
        super().__init__(*args, **kwargs)


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
    parser = _CliArgumentParser(
        prog="tensor-network-editor",
        description="Work with tensor-network specs from scripts, terminals, and pipelines.",
        epilog=COMMAND_ARGUMENT_QUICK_REFERENCE,
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
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        parser_class=_CliArgumentParser,
    )

    edit_parser = subparsers.add_parser(
        "edit",
        help="Launch the local editor in a browser, desktop window, or server-only mode.",
    )
    _add_edit_arguments(edit_parser)
    edit_parser.set_defaults(handler=handlers.handle_edit)

    validate_parser = subparsers.add_parser(
        "validate", help="Validate a saved spec or supported generated Python file."
    )
    validate_parser.add_argument(
        "path",
        type=str,
        help="Saved JSON design or supported generated Python file to validate.",
    )
    _add_output_format_argument(validate_parser)
    validate_parser.set_defaults(handler=handlers.handle_validate)

    lint_parser = subparsers.add_parser(
        "lint", help="Run soft diagnostics on a saved spec or generated Python file."
    )
    lint_parser.add_argument(
        "path",
        type=str,
        help="Saved JSON design or supported generated Python file to lint.",
    )
    lint_parser.add_argument(
        "--max-tensor-rank",
        type=int,
        default=6,
        help="Warn when a tensor has more indices than this rank.",
    )
    lint_parser.add_argument(
        "--max-tensor-cardinality",
        type=int,
        default=4096,
        help="Warn when a tensor has more elements than this count.",
    )
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
    analyze_parser.add_argument(
        "path",
        type=str,
        help="Saved JSON design or supported generated Python file to analyze.",
    )
    analyze_parser.add_argument(
        "--dtype",
        choices=list(SUPPORTED_MEMORY_DTYPES),
        default=DEFAULT_MEMORY_DTYPE,
        help="Numeric dtype used when estimating memory in bytes.",
    )
    _add_output_format_argument(analyze_parser)
    analyze_parser.set_defaults(handler=handlers.handle_analyze)

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Compare manual and automatic contraction variants for a saved spec.",
    )
    benchmark_parser.add_argument(
        "path",
        type=str,
        help="Saved JSON design or supported generated Python file to benchmark.",
    )
    benchmark_parser.add_argument(
        "--dtype",
        choices=list(SUPPORTED_MEMORY_DTYPES),
        default=DEFAULT_MEMORY_DTYPE,
        help="Numeric dtype used when estimating memory in bytes.",
    )
    benchmark_parser.add_argument(
        "--format",
        choices=["text", "json", "csv", "latex"],
        default="text",
        help="Output format for the benchmark report.",
    )
    benchmark_parser.add_argument(
        "--output",
        type=str,
        help="Write the benchmark report to a file instead of stdout.",
    )
    benchmark_parser.set_defaults(handler=handlers.handle_benchmark)

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Run friendly diagnostics for a saved spec and local environment.",
    )
    doctor_parser.add_argument(
        "path",
        type=str,
        help="Saved JSON design or supported generated Python file to inspect.",
    )
    doctor_parser.add_argument(
        "--dtype",
        choices=list(SUPPORTED_MEMORY_DTYPES),
        default=DEFAULT_MEMORY_DTYPE,
        help="Numeric dtype used when estimating memory in bytes.",
    )
    doctor_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format for the diagnostic report.",
    )
    doctor_parser.set_defaults(handler=handlers.handle_doctor)

    export_parser = subparsers.add_parser(
        "export", help="Generate backend Python code from a saved spec."
    )
    export_parser.add_argument(
        "path",
        type=str,
        help="Saved JSON design or supported generated Python file to export.",
    )
    export_parser.add_argument(
        "--engine",
        choices=[engine.value for engine in EngineName],
        required=True,
        help="Backend used for generated Python code.",
    )
    export_parser.add_argument(
        "--collection-format",
        choices=[
            collection_format.value for collection_format in TensorCollectionFormat
        ],
        default=TensorCollectionFormat.LIST.value,
        help="Container style used for generated tensor collections.",
    )
    export_parser.add_argument(
        "--output",
        type=str,
        help="Write generated code to a file instead of stdout.",
    )
    export_parser.set_defaults(handler=handlers.handle_export)

    render_parser = subparsers.add_parser(
        "render", help="Render a saved spec as a static image."
    )
    render_parser.add_argument(
        "path",
        type=str,
        help="Saved JSON design or supported generated Python file to render.",
    )
    render_parser.add_argument(
        "--format",
        choices=["svg", "png", "pdf", "tikz", "dot", "mermaid"],
        default="svg",
        help="Static render format to produce.",
    )
    render_parser.add_argument(
        "--output",
        type=str,
        help="Write the rendering to a file instead of stdout when supported.",
    )
    render_parser.add_argument(
        "--hide-tensor-names",
        dest="show_tensor_names",
        action="store_false",
        help="Hide tensor names in TikZ, DOT, and Mermaid renders.",
    )
    render_parser.add_argument(
        "--hide-index-names",
        dest="show_index_names",
        action="store_false",
        help="Hide index names in TikZ, DOT, and Mermaid renders.",
    )
    render_parser.add_argument(
        "--hide-bond-names",
        dest="show_bond_names",
        action="store_false",
        help="Hide bond names in TikZ, DOT, and Mermaid renders.",
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
    diff_parser.add_argument("before", type=str, help="Baseline spec path.")
    diff_parser.add_argument("after", type=str, help="Candidate spec path.")
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
    canonicalize_parser.add_argument(
        "path",
        type=str,
        help="Saved JSON design to canonicalize.",
    )
    canonicalize_parser.add_argument(
        "--output",
        type=str,
        help="Write canonical JSON to a file instead of stdout.",
    )
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
        dest="template_command",
        required=True,
        parser_class=_CliArgumentParser,
    )

    template_list_parser = template_subparsers.add_parser(
        "list", help="List the built-in template definitions."
    )
    _add_output_format_argument(template_list_parser)
    template_list_parser.set_defaults(handler=handlers.handle_template_list)

    template_build_parser = template_subparsers.add_parser(
        "build", help="Build a spec from a built-in template."
    )
    template_build_parser.add_argument(
        "template_name",
        type=str,
        help="Built-in template name to instantiate.",
    )
    template_build_parser.add_argument(
        "--graph-size",
        type=int,
        help="Override the graph size parameter when the template supports it.",
    )
    template_build_parser.add_argument(
        "--depth",
        type=int,
        help="Override the tree or hierarchy depth when the template supports it.",
    )
    template_build_parser.add_argument(
        "--bond-dimension",
        type=int,
        help="Override the bond dimension when the template supports it.",
    )
    template_build_parser.add_argument(
        "--physical-dimension",
        type=int,
        help="Override the physical dimension when the template supports it.",
    )
    template_build_parser.add_argument(
        "--boundary-condition",
        choices=("open", "periodic"),
        help="Choose open or periodic boundaries when supported.",
    )
    template_build_parser.add_argument(
        "--j",
        type=float,
        help="Override the template coupling parameter J when supported.",
    )
    template_build_parser.add_argument(
        "--h",
        type=float,
        help="Override the template field parameter h when supported.",
    )
    template_build_parser.add_argument(
        "--symmetry",
        choices=("none", "u1", "z2"),
        help="Select the symmetry label when the template supports it.",
    )
    template_build_parser.add_argument(
        "--initial-state",
        choices=("zeros", "random", "all_up", "all_down", "neel"),
        help="Select the initial-state pattern when the template supports it.",
    )
    template_build_parser.add_argument(
        "--leaf-physical-legs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable physical legs on tree leaves when supported.",
    )
    template_build_parser.add_argument(
        "--root-open-leg",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable an open root leg when supported.",
    )
    template_build_parser.add_argument(
        "--isometric",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable isometric tensor annotations when supported.",
    )
    template_build_parser.add_argument(
        "--output",
        type=str,
        help="Write the built template spec to a file instead of stdout.",
    )
    _add_output_format_argument(template_build_parser)
    template_build_parser.set_defaults(handler=handlers.handle_template_build)

    subnetwork_parser = subparsers.add_parser(
        "subnetwork",
        help="Inspect, save, and export reusable subnetworks from project catalogs.",
    )
    subnetwork_subparsers = subnetwork_parser.add_subparsers(
        dest="subnetwork_command",
        required=True,
        parser_class=_CliArgumentParser,
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
    subnetwork_export_parser.add_argument(
        "--output",
        type=str,
        help="Write the reusable subnetwork spec to a file instead of stdout.",
    )
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
        help="Visual theme used by the editor UI.",
    )
    parser.add_argument(
        "--ui",
        choices=["browser", "pywebview", "server"],
        help="Choose whether to open the editor in the browser, a pywebview window, or server-only mode.",
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
        help="Legacy alias for --ui server: start the local server without opening a UI automatically.",
    )


def _add_output_format_argument(parser: argparse.ArgumentParser) -> None:
    """Attach a standard text/json output selector to ``parser``."""
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Choose text or JSON output.",
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
