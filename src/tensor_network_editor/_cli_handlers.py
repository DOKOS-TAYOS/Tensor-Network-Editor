"""Command handlers for the tensor-network CLI."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

from .errors import SerializationError
from .models import EngineName, NetworkSpec, TensorCollectionFormat
from .serialization import (
    deserialize_spec,
    deserialize_spec_from_python_code,
    serialize_spec,
)


def handle_edit_command(
    args: argparse.Namespace,
    *,
    load_spec: Callable[[str], NetworkSpec],
    launch_tensor_network_editor: Callable[..., object],
) -> int:
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


def handle_validate_command(
    args: argparse.Namespace,
    *,
    load_spec: Callable[[str], NetworkSpec],
    validate_spec: Callable[[NetworkSpec], object],
    print_validation_result: Callable[..., None],
) -> int:
    """Validate a spec file and emit text or JSON results."""
    spec = load_spec(args.path)
    issues = validate_spec(spec)
    print_validation_result(issues, output_format=args.format)
    return 1 if issues else 0


def handle_lint_command(
    args: argparse.Namespace,
    *,
    load_spec_for_lint: Callable[[str], NetworkSpec],
    lint_spec: Callable[..., object],
    print_lint_result: Callable[..., None],
) -> int:
    """Run the soft linter against a spec file."""
    spec = load_spec_for_lint(args.path)
    report = lint_spec(
        spec,
        max_tensor_rank=args.max_tensor_rank,
        max_tensor_cardinality=args.max_tensor_cardinality,
    )
    print_lint_result(report, output_format=args.format)
    if args.fail_on == "warning" and report.has_warnings:
        return 1
    return 0


def handle_analyze_command(
    args: argparse.Namespace,
    *,
    load_spec: Callable[[str], NetworkSpec],
    analyze_spec: Callable[..., object],
    print_json: Callable[[object], None],
    print_analysis_text: Callable[[object], None],
) -> int:
    """Analyze structure and contraction metrics for a saved spec."""
    spec = load_spec(args.path)
    report = analyze_spec(spec, memory_dtype=args.dtype)
    if args.format == "json":
        print_json(report.to_dict())
    else:
        print_analysis_text(report)
    return 0


def handle_export_command(
    args: argparse.Namespace,
    *,
    load_spec: Callable[[str], NetworkSpec],
    generate_code: Callable[..., object],
) -> int:
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


def handle_diff_command(
    args: argparse.Namespace,
    *,
    load_spec: Callable[[str], NetworkSpec],
    diff_specs: Callable[[NetworkSpec, NetworkSpec], object],
    semantic_diff_specs: Callable[[NetworkSpec, NetworkSpec], object],
    print_json: Callable[[object], None],
    print_diff_text: Callable[[object], None],
    print_semantic_diff_text: Callable[[object], None],
) -> int:
    """Compare two specs and print the resulting structured diff."""
    before = load_spec(args.before)
    after = load_spec(args.after)
    if args.semantic:
        semantic_result = semantic_diff_specs(before, after)
        if args.format == "json":
            print_json(semantic_result.to_dict())
        else:
            print_semantic_diff_text(semantic_result)
        return 0
    diff_result = diff_specs(before, after)
    if args.format == "json":
        print_json(diff_result.to_dict())
    else:
        print_diff_text(diff_result)
    return 0


def handle_canonicalize_command(
    args: argparse.Namespace,
    *,
    load_spec: Callable[[str], NetworkSpec],
    canonicalize_spec: Callable[..., NetworkSpec],
    save_spec: Callable[[NetworkSpec, str], None],
    print_json: Callable[[object], None],
) -> int:
    """Canonicalize a spec and print or save the normalized result."""
    spec = load_spec(args.path)
    canonical_spec = canonicalize_spec(spec, deterministic_ids=args.deterministic_ids)
    if args.output is not None:
        save_spec(canonical_spec, args.output)
        print(f"Wrote canonical spec to {args.output}")
        return 0
    print_json(serialize_spec(canonical_spec))
    return 0


def handle_template_list_command(
    args: argparse.Namespace,
    *,
    serialize_template_definitions: Callable[[], dict[str, dict[str, object]]],
    list_template_names: Callable[[], list[str]],
    print_json: Callable[[object], None],
) -> int:
    """Print the built-in template definitions."""
    definitions = serialize_template_definitions()
    if args.format == "json":
        print_json(definitions)
    else:
        for template_name in list_template_names():
            definition = definitions[template_name]
            print(f"{template_name}: {definition['display_name']}")
    return 0


def handle_template_build_command(
    args: argparse.Namespace,
    *,
    parse_template_parameters: Callable[..., object],
    build_template_spec: Callable[[str, object], NetworkSpec],
    save_spec: Callable[[NetworkSpec, str], None],
    print_json: Callable[[object], None],
) -> int:
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
    print_json(serialize_spec(spec))
    return 0


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
