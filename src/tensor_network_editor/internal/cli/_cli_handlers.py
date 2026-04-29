"""Command handlers for the tensor-network CLI."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, cast

from ...editor import EditorLaunchOptions
from ...errors import SerializationError
from ...io import PythonLoadOptions
from ...models import EngineName, NetworkSpec, TensorCollectionFormat, ValidationIssue
from ...rendering import DotRenderOptions, SvgRenderOptions, TikzRenderOptions
from ...types import JSONValue, StrPath
from .._logging import log_branch, log_operation
from ..analysis._contraction_analysis_types import ContractionAnalysisResult
from ..io._serialization import (
    deserialize_spec,
    deserialize_spec_from_python_code_result,
    serialize_spec,
)
from ..models._headless_models import (
    LintReport,
    SemanticSpecDiffResult,
    SpecAnalysisReport,
    SpecDiffResult,
)
from ..subnetworks._catalog import (
    SubnetworkCatalog,
    SubnetworkCatalogEntry,
    append_project_subnetwork,
    load_project_subnetwork_catalog,
)
from ..subnetworks._subnetworks import extract_subnetwork_spec
from ..templates._template_catalog import TemplateParameters
from ._cli_benchmark import (
    BenchmarkReport,
    build_benchmark_report,
    serialize_benchmark_report_csv,
    serialize_benchmark_report_latex,
    serialize_benchmark_report_text,
)
from ._cli_doctor import build_doctor_report, format_doctor_report_text

LOGGER = logging.getLogger(__name__)


def handle_edit_command(
    args: argparse.Namespace,
    *,
    load_spec: Callable[..., NetworkSpec],
    open_editor: Callable[..., object],
) -> int:
    """Launch the browser editor using explicit edit arguments."""
    loaded_spec_path = Path(args.load).resolve() if args.load else None
    load_kwargs = _python_load_kwargs(args)
    initial_spec = load_spec(args.load, **load_kwargs) if args.load else None
    code_path: str | Path | None = args.save_code
    if loaded_spec_path is not None and args.save_code:
        candidate_code_path = Path(args.save_code)
        if not candidate_code_path.is_absolute():
            code_path = loaded_spec_path.parent / candidate_code_path
    open_editor_kwargs: dict[str, object] = {
        "spec": initial_spec,
        "options": EditorLaunchOptions(
            default_engine=EngineName(args.engine),
            theme=args.theme,
            open_browser=not args.no_browser,
            print_code=args.print_code,
            code_path=code_path,
            log_file_path=args.log_file,
            log_file_max_bytes=args.log_max_bytes,
            log_file_backup_count=args.log_backup_count,
        ),
    }
    if loaded_spec_path is not None:
        open_editor_kwargs["options"] = EditorLaunchOptions(
            default_engine=EngineName(args.engine),
            theme=args.theme,
            open_browser=not args.no_browser,
            print_code=args.print_code,
            code_path=code_path,
            log_file_path=args.log_file,
            log_file_max_bytes=args.log_max_bytes,
            log_file_backup_count=args.log_backup_count,
            template_catalog_path=(
                loaded_spec_path.parent / ".tensor-network-editor" / "templates.json"
            ),
            subnetwork_catalog_path=(
                loaded_spec_path.parent / ".tensor-network-editor" / "subnetworks.json"
            ),
        )
    open_editor(**open_editor_kwargs)
    return 0


def handle_validate_command(
    args: argparse.Namespace,
    *,
    load_spec: Callable[..., NetworkSpec],
    validate_spec: Callable[[NetworkSpec], list[ValidationIssue]],
    print_validation_result: Callable[..., None],
) -> int:
    """Validate a spec file and emit text or JSON results."""
    spec = load_spec(args.path, **_python_load_kwargs(args))
    issues = validate_spec(spec)
    print_validation_result(issues, output_format=args.format)
    return 1 if issues else 0


def handle_lint_command(
    args: argparse.Namespace,
    *,
    load_spec_for_lint: Callable[..., NetworkSpec],
    lint_spec: Callable[..., LintReport],
    print_lint_result: Callable[..., None],
) -> int:
    """Run the soft linter against a spec file."""
    spec = load_spec_for_lint(args.path, **_python_load_kwargs(args))
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
    load_spec: Callable[..., NetworkSpec],
    analyze_spec: Callable[..., SpecAnalysisReport],
    print_json: Callable[[object], None],
    print_analysis_text: Callable[[SpecAnalysisReport], None],
) -> int:
    """Analyze structure and contraction metrics for a saved spec."""
    spec = load_spec(args.path, **_python_load_kwargs(args))
    report = analyze_spec(spec, memory_dtype=args.dtype)
    if args.format == "json":
        print_json(report.to_dict())
    else:
        print_analysis_text(report)
    return 0


def handle_benchmark_command(
    args: argparse.Namespace,
    *,
    load_spec: Callable[..., NetworkSpec],
    analyze_contraction: Callable[..., ContractionAnalysisResult],
    print_json: Callable[[object], None],
    print_benchmark_report_text: Callable[[BenchmarkReport], None],
    write_utf8_text: Callable[[str, str], None],
) -> int:
    """Analyze and export a stable benchmark comparison table."""
    with log_operation(
        LOGGER,
        "Benchmark command",
        context={
            "memory_dtype": args.dtype,
            "export_format": args.format,
            "output_path": args.output,
        },
    ):
        spec = load_spec(args.path, **_python_load_kwargs(args))
        analysis = analyze_contraction(spec, memory_dtype=args.dtype)
        report = build_benchmark_report(analysis)
        log_branch(
            LOGGER,
            "Built benchmark report",
            context={
                "analysis_status": "ready",
                "scheme_count": len(report.rows),
                "warning_count": len(report.warnings),
                "manual_status": analysis.manual.status,
                "automatic_full_status": analysis.automatic_full.status,
                "automatic_future_status": analysis.automatic_future.status,
                "automatic_past_status": analysis.automatic_past.status,
            },
        )
        if args.output is not None:
            output_text = _serialize_benchmark_report(report, output_format=args.format)
            write_utf8_text(args.output, output_text)
            log_branch(
                LOGGER,
                "Serialized benchmark report to disk",
                context={
                    "export_format": args.format,
                    "output_path": args.output,
                    "scheme_count": len(report.rows),
                },
            )
            print(f"Wrote benchmark report to {args.output}")
            return 0
        if args.format == "json":
            print_json(report.to_dict())
        elif args.format == "text":
            print_benchmark_report_text(report)
        else:
            print(_serialize_benchmark_report(report, output_format=args.format))
        return 0


def handle_doctor_command(
    args: argparse.Namespace,
    *,
    load_spec: Callable[..., NetworkSpec],
    validate_spec: Callable[[NetworkSpec], list[ValidationIssue]],
    lint_spec: Callable[..., LintReport],
    analyze_spec: Callable[..., SpecAnalysisReport],
    print_json: Callable[[object], None],
) -> int:
    """Run a friendly diagnostic report for one saved spec."""
    spec = load_spec(args.path, **_python_load_kwargs(args))
    report = build_doctor_report(
        spec,
        memory_dtype=args.dtype,
        validate_spec=validate_spec,
        lint_spec=lint_spec,
        analyze_spec=analyze_spec,
    )
    if args.format == "json":
        print_json(report.to_dict())
    else:
        print(format_doctor_report_text(report, path=args.path))
    return 0 if report.ok else 1


def handle_export_command(
    args: argparse.Namespace,
    *,
    load_spec: Callable[..., NetworkSpec],
    generate_code: Callable[..., object],
) -> int:
    """Generate backend code from a saved spec without launching the editor."""
    spec = load_spec(args.path, **_python_load_kwargs(args))
    generate_code(
        spec,
        engine=EngineName(args.engine),
        collection_format=TensorCollectionFormat(args.collection_format),
        print_code=args.output is None,
        output_path=args.output,
        external_data_base_path=Path(args.path).resolve().parent,
    )
    if args.output is not None:
        print(f"Wrote generated code to {args.output}")
    return 0


def handle_render_command(
    args: argparse.Namespace,
    *,
    load_spec: Callable[..., NetworkSpec],
    render_spec_dot: Callable[..., str],
    render_spec_pdf: Callable[..., bytes],
    render_spec_svg: Callable[..., str],
    render_spec_tikz: Callable[..., str],
    render_spec_png: Callable[..., bytes],
) -> int:
    """Render a saved spec as a static image."""
    spec = load_spec(args.path, **_python_load_kwargs(args))
    svg_options = SvgRenderOptions(
        show_tensor_labels=args.show_tensor_names,
        show_index_labels=args.show_index_names,
        show_edge_labels=args.show_bond_names,
    )
    if args.format == "png":
        if args.output is None:
            raise ValueError("PNG render requires --output.")
        render_spec_png(spec, options=svg_options, output_path=args.output)
        print(f"Wrote PNG rendering to {args.output}")
        return 0
    if args.format == "pdf":
        if args.output is None:
            raise ValueError("PDF render requires --output.")
        render_spec_pdf(spec, options=svg_options, output_path=args.output)
        print(f"Wrote PDF rendering to {args.output}")
        return 0
    if args.format == "tikz":
        text = render_spec_tikz(
            spec,
            options=TikzRenderOptions(
                show_tensor_labels=args.show_tensor_names,
                show_index_labels=args.show_index_names,
                show_edge_labels=args.show_bond_names,
            ),
            output_path=args.output,
        )
        output_label = "TikZ"
    elif args.format == "dot":
        text = render_spec_dot(
            spec,
            options=DotRenderOptions(
                show_tensor_labels=args.show_tensor_names,
                show_index_labels=args.show_index_names,
                show_edge_labels=args.show_bond_names,
            ),
            output_path=args.output,
        )
        output_label = "Graphviz/DOT"
    else:
        text = render_spec_svg(spec, options=svg_options, output_path=args.output)
        output_label = "SVG"
    if args.output is None:
        print(text)
    else:
        print(f"Wrote {output_label} rendering to {args.output}")
    return 0


def handle_diff_command(
    args: argparse.Namespace,
    *,
    load_spec: Callable[..., NetworkSpec],
    diff_specs: Callable[[NetworkSpec, NetworkSpec], SpecDiffResult],
    semantic_diff_specs: Callable[[NetworkSpec, NetworkSpec], SemanticSpecDiffResult],
    print_json: Callable[[object], None],
    print_diff_text: Callable[[SpecDiffResult], None],
    print_semantic_diff_text: Callable[[SemanticSpecDiffResult], None],
) -> int:
    """Compare two specs and print the resulting structured diff."""
    load_kwargs = _python_load_kwargs(args)
    before = load_spec(args.before, **load_kwargs)
    after = load_spec(args.after, **load_kwargs)
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
    load_spec: Callable[..., NetworkSpec],
    canonicalize_spec: Callable[..., NetworkSpec],
    save_spec: _SaveSpecFunction,
    print_json: Callable[[object], None],
) -> int:
    """Canonicalize a spec and print or save the normalized result."""
    spec = load_spec(args.path, **_python_load_kwargs(args))
    canonical_spec = canonicalize_spec(spec, deterministic_ids=args.deterministic_ids)
    if args.output is not None:
        save_spec(canonical_spec, path=args.output)
        print(f"Wrote canonical spec to {args.output}")
        return 0
    print_json(serialize_spec(canonical_spec))
    return 0


def handle_template_list_command(
    args: argparse.Namespace,
    *,
    serialize_template_definitions: Callable[[], dict[str, dict[str, JSONValue]]],
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
    parse_template_parameters: Callable[..., TemplateParameters],
    build_template_spec: Callable[[str, TemplateParameters | None], NetworkSpec],
    save_spec: _SaveSpecFunction,
    print_json: Callable[[object], None],
) -> int:
    """Build a template spec and print or save the resulting serialized spec."""
    raw_parameters = {
        key: value
        for key, value in {
            "graph_size": args.graph_size,
            "depth": args.depth,
            "bond_dimension": args.bond_dimension,
            "physical_dimension": args.physical_dimension,
            "boundary_condition": args.boundary_condition,
            "j": args.j,
            "h": args.h,
            "symmetry": args.symmetry,
            "initial_state": args.initial_state,
            "leaf_physical_legs": args.leaf_physical_legs,
            "root_open_leg": args.root_open_leg,
            "isometric": args.isometric,
        }.items()
        if value is not None
    }
    parameters = parse_template_parameters(
        args.template_name,
        raw_parameters if raw_parameters else None,
    )
    spec = build_template_spec(args.template_name, parameters)
    if args.output is not None:
        save_spec(spec, path=args.output)
        print(f"Wrote template spec to {args.output}")
        return 0
    print_json(serialize_spec(spec))
    return 0


def handle_subnetwork_list_command(
    args: argparse.Namespace,
    *,
    print_json: Callable[[object], None],
) -> int:
    """Print reusable-subnetwork catalog entries for one project context."""
    project_catalog_path = _resolve_project_subnetwork_catalog_path_for_spec(args.path)
    catalog_payload, _ = _build_subnetwork_catalog_payload(
        project_catalog_path,
        shared_catalog_path=args.shared_catalog_path,
    )
    if args.format == "json":
        print_json(catalog_payload)
        return 0
    subnetwork_definitions = cast(
        dict[str, dict[str, JSONValue]],
        catalog_payload["subnetwork_definitions"],
    )
    for subnetwork_name in cast(list[str], catalog_payload["subnetworks"]):
        definition = subnetwork_definitions[subnetwork_name]
        tags = cast(list[str], definition["tags"])
        tag_suffix = f" [{', '.join(tags)}]" if tags else ""
        print(f"{subnetwork_name}: {definition['display_name']}{tag_suffix}")
    return 0


def handle_subnetwork_save_command(
    args: argparse.Namespace,
    *,
    load_spec: Callable[..., NetworkSpec],
) -> int:
    """Extract and save one reusable subnetwork into the project catalog."""
    spec = load_spec(args.path, **_python_load_kwargs(args))
    project_catalog_path = _resolve_project_subnetwork_catalog_path_for_spec(args.path)
    subnetwork_spec = extract_subnetwork_spec(
        spec,
        tensor_ids=list(cast(list[str], args.tensor_ids)),
    )
    append_project_subnetwork(
        project_catalog_path,
        str(args.name),
        subnetwork_spec,
        tags=list(cast(list[str], args.tags)),
        overwrite=bool(args.overwrite),
    )
    print(f"Saved reusable subnetwork '{args.name}' to {project_catalog_path}")
    return 0


def handle_subnetwork_export_command(
    args: argparse.Namespace,
    *,
    save_spec: _SaveSpecFunction,
    print_json: Callable[[object], None],
) -> int:
    """Export one reusable subnetwork from the merged project/shared catalogs."""
    project_catalog_path = _resolve_project_subnetwork_catalog_path_for_spec(args.path)
    _, merged_entries = _build_subnetwork_catalog_payload(
        project_catalog_path,
        shared_catalog_path=args.shared_catalog_path,
    )
    try:
        spec = merged_entries[str(args.subnetwork_name)].spec
    except KeyError as exc:
        raise ValueError(
            f"Unknown reusable subnetwork '{args.subnetwork_name}'."
        ) from exc
    if args.output is not None:
        save_spec(spec, path=args.output)
        print(f"Wrote reusable subnetwork '{args.subnetwork_name}' to {args.output}")
        return 0
    print_json(serialize_spec(spec))
    return 0


def _serialize_benchmark_report(report: object, *, output_format: str) -> str:
    """Return the serialized benchmark report for a non-JSON format."""
    if output_format == "text":
        return serialize_benchmark_report_text(cast(BenchmarkReport, report))
    if output_format == "csv":
        return serialize_benchmark_report_csv(cast(BenchmarkReport, report))
    if output_format == "latex":
        return serialize_benchmark_report_latex(cast(BenchmarkReport, report))
    if output_format == "json":
        return json.dumps(cast(BenchmarkReport, report).to_dict(), indent=2)
    raise ValueError(f"Unsupported benchmark output format: {output_format}")


def _resolve_project_subnetwork_catalog_path_for_spec(spec_path: str) -> Path:
    """Resolve the project reusable-subnetwork catalog path for one spec file."""
    return (
        Path(spec_path).resolve().parent / ".tensor-network-editor" / "subnetworks.json"
    )


def _build_subnetwork_catalog_payload(
    project_catalog_path: Path,
    *,
    shared_catalog_path: str | None = None,
) -> tuple[dict[str, JSONValue], dict[str, SubnetworkCatalogEntry]]:
    """Build merged reusable-subnetwork payload data for CLI commands."""
    project_catalog = load_project_subnetwork_catalog(project_catalog_path)
    shared_catalog = (
        load_project_subnetwork_catalog(shared_catalog_path)
        if shared_catalog_path is not None
        else SubnetworkCatalog(path=project_catalog_path, entries={}, warnings=[])
    )
    merged_entries: dict[str, SubnetworkCatalogEntry] = dict(project_catalog.entries)
    for subnetwork_name, entry in shared_catalog.entries.items():
        if subnetwork_name in merged_entries:
            continue
        merged_entries[subnetwork_name] = entry
    warnings = list(project_catalog.warnings)
    warnings.extend(shared_catalog.warnings)
    warnings.extend(
        _shadowed_shared_subnetwork_warnings(project_catalog, shared_catalog)
    )
    return (
        {
            "subnetworks": cast(JSONValue, list(merged_entries)),
            "subnetwork_definitions": cast(
                JSONValue,
                {
                    subnetwork_name: (
                        project_catalog.entries[subnetwork_name].to_definition(
                            source="project"
                        )
                        if subnetwork_name in project_catalog.entries
                        else shared_catalog.entries[subnetwork_name].to_definition(
                            source="shared"
                        )
                    )
                    for subnetwork_name in merged_entries
                },
            ),
            "subnetwork_catalog_warnings": cast(JSONValue, warnings),
        },
        merged_entries,
    )


def _shadowed_shared_subnetwork_warnings(
    project_catalog: SubnetworkCatalog,
    shared_catalog: SubnetworkCatalog,
) -> list[str]:
    """Return warnings for shared entries shadowed by project-local entries."""
    return [
        f"Project reusable subnetwork '{subnetwork_name}' shadows the shared catalog entry."
        for subnetwork_name in project_catalog.entries
        if subnetwork_name in shared_catalog.entries
    ]


def load_spec_for_lint(
    path: str,
    *,
    python: PythonLoadOptions | None = None,
) -> NetworkSpec:
    """Load a spec for linting without enforcing hard validation first."""
    from ..io._io import read_utf8_text

    options = python or PythonLoadOptions()
    source_path = Path(path)
    if source_path.suffix.lower() == ".py":
        return deserialize_spec_from_python_code_result(
            read_utf8_text(path, description="generated Python code"),
            validate=False,
            source_profile=options.source_profile,
            python_import_mode=options.import_mode,
            python_reconstruction_level=options.reconstruction_level,
            python_object_name=options.object_name,
            source_path=source_path,
        ).spec
    try:
        payload = json.loads(
            read_utf8_text(path, description="network specification JSON")
        )
    except json.JSONDecodeError as exc:
        raise SerializationError("Could not parse network specification JSON.") from exc
    if not isinstance(payload, dict):
        raise SerializationError("Serialized network must be a JSON object.")
    return deserialize_spec(payload, validate=False)


def _python_load_kwargs(args: argparse.Namespace) -> dict[str, object]:
    """Return the optional Python import arguments requested on the CLI."""
    options = PythonLoadOptions(
        import_mode=getattr(args, "python_import_mode", "static"),
        reconstruction_level=getattr(args, "python_reconstruction_level", "auto"),
        object_name=getattr(args, "python_object", None),
    )
    if options == PythonLoadOptions():
        return {}
    return {"python": options}


class _SaveSpecFunction(Protocol):
    """Callable protocol for public spec writers with keyword-only paths."""

    def __call__(self, spec: NetworkSpec, *, path: StrPath) -> None: ...
