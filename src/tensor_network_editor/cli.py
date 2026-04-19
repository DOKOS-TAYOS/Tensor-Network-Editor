"""Command-line interface for editor and headless tensor-network workflows."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import Protocol, cast

from ._cli_formatters import (
    print_analysis_text,
    print_diff_text,
    print_json,
    print_lint_result,
    print_semantic_diff_text,
    print_validation_result,
)
from ._cli_handlers import (
    handle_analyze_command,
    handle_canonicalize_command,
    handle_diff_command,
    handle_edit_command,
    handle_export_command,
    handle_lint_command,
    handle_template_build_command,
    handle_template_list_command,
    handle_validate_command,
    load_spec_for_lint,
)
from ._cli_parser import CliHandlerBindings
from ._cli_parser import build_command_parser as build_parser
from ._logging import configure_package_logging, emit_runtime_diagnostics
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
from .linting import lint_spec
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
    """Build the parser used by headless CLI subcommands."""
    return build_parser(
        CliHandlerBindings(
            handle_edit=_handle_edit,
            handle_validate=_handle_validate,
            handle_lint=_handle_lint,
            handle_analyze=_handle_analyze,
            handle_export=_handle_export,
            handle_diff=_handle_diff,
            handle_canonicalize=_handle_canonicalize,
            handle_template_list=_handle_template_list,
            handle_template_build=_handle_template_build,
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process-friendly exit code."""
    args_list = list(argv) if argv is not None else sys.argv[1:]
    try:
        parsed_args = cast(
            _CommandNamespace, build_command_parser().parse_args(args_list)
        )
        selected_log_level = configure_package_logging(parsed_args.log_level)
        emit_runtime_diagnostics(selected_log_level)
        return _dispatch_command(parsed_args)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 2
    except KeyboardInterrupt:
        return 130
    except SpecValidationError as exc:
        print_validation_result(exc.issues, output_format="text")
        return 1
    except (CodeGenerationError, PackageIOError, SerializationError, ValueError) as exc:
        print(str(exc))
        return 2


def _dispatch_command(args: _CommandNamespace) -> int:
    """Run the command handler stored on the parsed namespace."""
    return args.handler(args)


def _handle_edit(args: argparse.Namespace) -> int:
    """Launch the browser editor using explicit edit arguments."""
    return handle_edit_command(
        args,
        load_spec=load_spec,
        launch_tensor_network_editor=launch_tensor_network_editor,
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


def _handle_export(args: argparse.Namespace) -> int:
    """Generate backend code from a saved spec without launching the editor."""
    return handle_export_command(
        args,
        load_spec=load_spec,
        generate_code=generate_code,
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
