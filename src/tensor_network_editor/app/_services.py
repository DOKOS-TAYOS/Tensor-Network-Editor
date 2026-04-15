"""Service-layer helpers shared by the local editor HTTP routes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .._annotation_catalog import serialize_annotation_definitions
from .._contraction_analysis_types import ContractionAnalysisResult
from ..analysis import analyze_contraction
from ..codegen.registry import (
    engine_name_to_text,
    list_generator_names,
)
from ..codegen.registry import (
    generate_code as generate_code_internal,
)
from ..errors import SpecValidationError
from ..models import (
    CanvasPosition,
    CodegenResult,
    EditorResult,
    EngineIdentifier,
    NetworkSpec,
    TensorCollectionFormat,
)
from ..serialization import SCHEMA_VERSION, deserialize_spec
from ..subnetworks import (
    extract_subnetwork_spec,
    prepare_subnetwork_for_insertion,
)
from ..templates import (
    TemplateParameters,
    build_template_spec,
    list_template_names,
    parse_template_parameters,
    serialize_template_definitions,
)
from ..validation import validate_spec

if TYPE_CHECKING:
    from .session import EditorSession


def build_bootstrap_payload(session: EditorSession) -> dict[str, object]:
    """Build the initial payload used to bootstrap the browser client."""
    return {
        "default_engine": engine_name_to_text(session.default_engine),
        "engines": list_generator_names(),
        "default_collection_format": session.default_collection_format.value,
        "collection_formats": [
            collection_format.value for collection_format in TensorCollectionFormat
        ],
        "schema_version": SCHEMA_VERSION,
        "templates": list_template_names(),
        "template_definitions": serialize_template_definitions(),
        "annotation_definitions": serialize_annotation_definitions(),
        "spec": {
            "schema_version": SCHEMA_VERSION,
            "network": session.initial_spec.to_dict(),
        },
    }


def generate_session_request(
    session: EditorSession,
    serialized_spec: dict[str, object],
    engine: EngineIdentifier,
    collection_format: TensorCollectionFormat | None = None,
) -> CodegenResult:
    """Generate preview code for one editor request."""
    spec = deserialize_spec(serialized_spec)
    return generate_code_internal(
        spec,
        engine,
        collection_format=_resolve_collection_format(session, collection_format),
    )


def complete_session_request(
    session: EditorSession,
    serialized_spec: dict[str, object],
    engine: EngineIdentifier,
    collection_format: TensorCollectionFormat | None = None,
) -> EditorResult:
    """Finalize a session request and optionally print or save generated code."""
    spec = deserialize_spec(serialized_spec)
    codegen_result = generate_code_internal(
        spec,
        engine,
        collection_format=_resolve_collection_format(session, collection_format),
    )
    if session.print_code:
        print(codegen_result.code)
    if session.code_path is not None:
        from .._io import write_utf8_text

        write_utf8_text(
            session.code_path,
            codegen_result.code,
            description="generated Python code",
        )
    return EditorResult(
        spec=spec, engine=engine, codegen=codegen_result, confirmed=True
    )


def build_template_from_payload(
    session: EditorSession,
    template_name: str,
    raw_parameters: object | None = None,
) -> NetworkSpec:
    """Build a validated template spec from raw API payload values."""
    del session
    parameters: TemplateParameters | None = parse_template_parameters(
        template_name,
        raw_parameters,
    )
    return build_template_spec(template_name, parameters)


def analyze_serialized_contraction(
    serialized_spec: dict[str, object],
) -> ContractionAnalysisResult:
    """Deserialize, validate, and analyze contraction data for one payload."""
    spec = deserialize_spec(serialized_spec, validate=False)
    issues = validate_spec(spec)
    if issues:
        raise SpecValidationError(issues)
    return analyze_contraction(spec)


def extract_serialized_subnetwork(
    serialized_spec: dict[str, object],
    *,
    tensor_ids: list[str],
) -> NetworkSpec:
    """Deserialize one payload and extract its selected tensor fragment."""
    spec = deserialize_spec(serialized_spec, validate=False)
    return extract_subnetwork_spec(spec, tensor_ids=tensor_ids)


def prepare_serialized_subnetwork_for_insertion(
    serialized_spec: dict[str, object],
    *,
    target_center: CanvasPosition,
) -> NetworkSpec:
    """Deserialize one payload and prepare it for editor insertion."""
    spec = deserialize_spec(serialized_spec, validate=False)
    return prepare_subnetwork_for_insertion(spec, target_center=target_center)


def _resolve_collection_format(
    session: EditorSession,
    collection_format: TensorCollectionFormat | None,
) -> TensorCollectionFormat:
    """Resolve a request collection format against the session default."""
    return collection_format or session.default_collection_format
