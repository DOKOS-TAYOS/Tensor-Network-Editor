"""Command-line interface for editor and headless tensor-network workflows."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from typing import Protocol, cast

from ._public_codegen import generate_code
from .analysis import analyze_contraction, analyze_spec
from .canonicalization import canonicalize_spec
from .editor import open_editor
from .errors import (
    CodeGenerationError,
    PackageIOError,
    SerializationError,
    SpecValidationError,
)
from .internal._logging import (
    bind_log_context,
    format_log_message,
    log_branch,
    log_operation,
)
from .internal.cli._cli_formatters import (
    print_analysis_text,
    print_benchmark_report_text,
    print_diff_text,
    print_json,
    print_lint_result,
    print_semantic_diff_text,
    print_validation_result,
)
from .internal.cli._cli_handlers import (
    handle_analyze_command,
    handle_benchmark_command,
    handle_canonicalize_command,
    handle_diff_command,
    handle_doctor_command,
    handle_edit_command,
    handle_export_command,
    handle_lint_command,
    handle_render_command,
    handle_subnetwork_export_command,
    handle_subnetwork_list_command,
    handle_subnetwork_save_command,
    handle_template_build_command,
    handle_template_list_command,
    handle_validate_command,
    load_spec_for_lint,
)
from .internal.cli._cli_parser import CliHandlerBindings
from .internal.cli._cli_parser import build_command_parser as build_parser
from .internal.cli._logging import (
    configure_package_logging,
    emit_runtime_diagnostics,
)
from .internal.diffing._diffing import diff_specs, semantic_diff_specs
from .io import load_spec, save_spec
from .linting import lint_spec
from .rendering import (
    render_spec_dot,
    render_spec_mermaid,
    render_spec_pdf,
    render_spec_png,
    render_spec_svg,
    render_spec_tikz,
)
from .templates import (
    build_template_spec,
    list_template_names,
    parse_template_parameters,
    serialize_template_definitions,
)
from .validation import validate_spec

LOGGER = logging.getLogger(__name__)


class _CommandHandler(Protocol):
    """Callable stored on parsed subcommands."""

    def __call__(self, args: argparse.Namespace) -> int:
        """Run one parsed subcommand and return its process exit code."""
        ...


class _CommandNamespace(argparse.Namespace):
    """Parsed namespace for subcommands that install a handler."""

    handler: _CommandHandler


def build_command_parser() -> argparse.ArgumentParser:
    """Build the parser used by headless CLI subcommands."""
    return build_parser(
        CliHandlerBindings(
            handle_edit=_handle_edit,
            handle_validate=_handle_validate,
            handle_lint=_handle_lint,
            handle_analyze=_handle_analyze,
            handle_benchmark=_handle_benchmark,
            handle_doctor=_handle_doctor,
            handle_export=_handle_export,
            handle_render=_handle_render,
            handle_diff=_handle_diff,
            handle_canonicalize=_handle_canonicalize,
            handle_template_list=_handle_template_list,
            handle_template_build=_handle_template_build,
            handle_subnetwork_list=_handle_subnetwork_list,
            handle_subnetwork_save=_handle_subnetwork_save,
            handle_subnetwork_export=_handle_subnetwork_export,
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process-friendly exit code."""
    args_list = list(argv) if argv is not None else sys.argv[1:]
    try:
        parsed_args = cast(
            _CommandNamespace, build_command_parser().parse_args(args_list)
        )
        command_context = _build_cli_command_context(parsed_args)
        with configure_package_logging(
            parsed_args.log_level,
            log_file_path=parsed_args.log_file,
            log_file_max_bytes=parsed_args.log_max_bytes,
            log_file_backup_count=parsed_args.log_backup_count,
        ) as runtime_config:
            with bind_log_context(**command_context):
                with log_operation(LOGGER, "CLI command", context=command_context):
                    emit_runtime_diagnostics(
                        runtime_config.level_name
                        if runtime_config is not None
                        else None
                    )
                    log_branch(
                        LOGGER,
                        "Resolved CLI command arguments",
                        context=_build_cli_debug_context(parsed_args),
                    )
                    return _dispatch_command(parsed_args)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 2
    except KeyboardInterrupt:
        return 130
    except SpecValidationError as exc:
        _log_handled_cli_failure(exc)
        print_validation_result(exc.issues, output_format="text")
        return 1
    except (
        CodeGenerationError,
        PackageIOError,
        RuntimeError,
        SerializationError,
        ValueError,
    ) as exc:
        _log_handled_cli_failure(exc)
        print(str(exc))
        return 2


def _dispatch_command(args: _CommandNamespace) -> int:
    """Run the command handler stored on the parsed namespace."""
    return args.handler(args)


def _build_cli_command_context(args: argparse.Namespace) -> dict[str, object]:
    command = getattr(args, "command", None)
    template_command = getattr(args, "template_command", None)
    subnetwork_command = getattr(args, "subnetwork_command", None)
    command_name = str(command) if isinstance(command, str) else "unknown"
    if isinstance(template_command, str):
        command_name = f"{command_name}.{template_command}"
    if isinstance(subnetwork_command, str):
        command_name = f"{command_name}.{subnetwork_command}"
    context: dict[str, object] = {"command": command_name}
    path_value = getattr(args, "path", None)
    if isinstance(path_value, str):
        context["path"] = path_value
    return context


def _build_cli_debug_context(args: argparse.Namespace) -> dict[str, object]:
    debug_context: dict[str, object] = {}
    for field_name in (
        "engine",
        "format",
        "dtype",
        "python_import_mode",
        "python_reconstruction_level",
        "python_object",
        "output",
        "save_code",
    ):
        value = getattr(args, field_name, None)
        if value is not None:
            normalized_field_name = (
                "output_path" if field_name in {"output", "save_code"} else field_name
            )
            debug_context[normalized_field_name] = value
    return debug_context


def _log_handled_cli_failure(exc: Exception) -> None:
    LOGGER.warning(
        format_log_message(f"CLI command failed: {exc}", context={"outcome": "error"}),
        exc_info=LOGGER.isEnabledFor(logging.DEBUG),
    )


def _handle_edit(args: argparse.Namespace) -> int:
    """Launch the browser editor using explicit edit arguments."""
    return handle_edit_command(
        args,
        load_spec=load_spec,
        open_editor=open_editor,
    )


def _handle_validate(args: argparse.Namespace) -> int:
    """Validate a spec file and emit text or JSON results."""
    return handle_validate_command(
        args,
        load_spec=load_spec,
        validate_spec=validate_spec,
        print_validation_result=print_validation_result,
    )


def _handle_lint(args: argparse.Namespace) -> int:
    """Run the soft linter against a spec file."""
    return handle_lint_command(
        args,
        load_spec_for_lint=load_spec_for_lint,
        lint_spec=lint_spec,
        print_lint_result=print_lint_result,
    )


def _handle_analyze(args: argparse.Namespace) -> int:
    """Analyze structure and contraction metrics for a saved spec."""
    return handle_analyze_command(
        args,
        load_spec=load_spec,
        analyze_spec=analyze_spec,
        print_json=print_json,
        print_analysis_text=print_analysis_text,
    )


def _handle_benchmark(args: argparse.Namespace) -> int:
    """Compare manual and automatic contraction variants for one saved spec."""
    from .internal.io._io import write_utf8_text

    return handle_benchmark_command(
        args,
        load_spec=load_spec,
        analyze_contraction=analyze_contraction,
        print_json=print_json,
        print_benchmark_report_text=print_benchmark_report_text,
        write_utf8_text=lambda path, content: write_utf8_text(
            path,
            content,
            description="benchmark report",
        ),
    )


def _handle_doctor(args: argparse.Namespace) -> int:
    """Run validation, lint, analysis, benchmark, and dependency diagnostics."""
    return handle_doctor_command(
        args,
        load_spec=load_spec_for_lint,
        validate_spec=validate_spec,
        lint_spec=lint_spec,
        analyze_spec=analyze_spec,
        print_json=print_json,
    )


def _handle_export(args: argparse.Namespace) -> int:
    """Generate backend code from a saved spec without launching the editor."""
    return handle_export_command(
        args,
        load_spec=load_spec,
        generate_code=generate_code,
    )


def _handle_render(args: argparse.Namespace) -> int:
    """Render a saved spec as a static image without launching the editor."""
    return handle_render_command(
        args,
        load_spec=load_spec,
        render_spec_dot=render_spec_dot,
        render_spec_mermaid=render_spec_mermaid,
        render_spec_pdf=render_spec_pdf,
        render_spec_png=render_spec_png,
        render_spec_svg=render_spec_svg,
        render_spec_tikz=render_spec_tikz,
    )


def _handle_diff(args: argparse.Namespace) -> int:
    """Compare two specs and print the resulting structured diff."""
    return handle_diff_command(
        args,
        load_spec=load_spec,
        diff_specs=diff_specs,
        semantic_diff_specs=semantic_diff_specs,
        print_json=print_json,
        print_diff_text=print_diff_text,
        print_semantic_diff_text=print_semantic_diff_text,
    )


def _handle_canonicalize(args: argparse.Namespace) -> int:
    """Canonicalize a spec and print or save the normalized result."""
    return handle_canonicalize_command(
        args,
        load_spec=load_spec,
        canonicalize_spec=canonicalize_spec,
        save_spec=save_spec,
        print_json=print_json,
    )


def _handle_template_list(args: argparse.Namespace) -> int:
    """Print the built-in template definitions."""
    return handle_template_list_command(
        args,
        serialize_template_definitions=serialize_template_definitions,
        list_template_names=list_template_names,
        print_json=print_json,
    )


def _handle_template_build(args: argparse.Namespace) -> int:
    """Build a template spec and print or save the resulting serialized spec."""
    return handle_template_build_command(
        args,
        parse_template_parameters=parse_template_parameters,
        build_template_spec=build_template_spec,
        save_spec=save_spec,
        print_json=print_json,
    )


def _handle_subnetwork_list(args: argparse.Namespace) -> int:
    """Print reusable-subnetwork entries for the requested project context."""
    return handle_subnetwork_list_command(
        args,
        print_json=print_json,
    )


def _handle_subnetwork_save(args: argparse.Namespace) -> int:
    """Save a reusable subnetwork from one source spec."""
    return handle_subnetwork_save_command(
        args,
        load_spec=load_spec,
    )


def _handle_subnetwork_export(args: argparse.Namespace) -> int:
    """Export a reusable subnetwork from the merged project/shared catalogs."""
    return handle_subnetwork_export_command(
        args,
        save_spec=save_spec,
        print_json=print_json,
    )
