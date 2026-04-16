"""Request and response helpers for the local editor HTTP API."""

from __future__ import annotations

import json
from dataclasses import dataclass
from http import HTTPStatus
from typing import TypeAlias, cast

from ..codegen.registry import engine_name_to_text, resolve_registered_engine
from ..models import (
    CanvasPosition,
    CodegenResult,
    EditorResult,
    EngineIdentifier,
    NetworkSpec,
    TensorCollectionFormat,
    ValidationIssue,
)
from ..serialization import (
    SCHEMA_VERSION,
    deserialize_spec,
    deserialize_spec_from_python_code,
)
from ..types import JSONValue

_JsonArray: TypeAlias = list[JSONValue]
_JsonObject: TypeAlias = dict[str, JSONValue]

JsonDict: TypeAlias = _JsonObject
JsonList: TypeAlias = _JsonArray
JsonResponse: TypeAlias = tuple[int, JsonDict]


@dataclass(slots=True, frozen=True)
class CodegenRequest:
    """Normalized payload for generate and complete API requests."""

    serialized_spec: JsonDict
    engine: EngineIdentifier
    collection_format: TensorCollectionFormat


@dataclass(slots=True, frozen=True)
class TemplatePromoteRequest:
    """Normalized payload for project-template promotion requests."""

    serialized_spec: JsonDict
    tensor_ids: list[str]
    template_name: str
    overwrite: bool


@dataclass(slots=True, frozen=True)
class TemplateRenameRequest:
    """Normalized payload for project-template rename requests."""

    template_name: str
    new_template_name: str
    overwrite: bool


@dataclass(slots=True, frozen=True)
class TemplateDeleteRequest:
    """Normalized payload for project-template delete requests."""

    template_name: str


@dataclass(slots=True, frozen=True)
class SubnetworkSelectionRequest:
    """Normalized payload for subnetwork extract or promotion requests."""

    serialized_spec: JsonDict
    tensor_ids: list[str]


@dataclass(slots=True, frozen=True)
class SubnetworkPrepareInsertRequest:
    """Normalized payload for subnetwork insertion preparation requests."""

    serialized_spec: JsonDict
    target_center: CanvasPosition


def read_json(body: bytes) -> JsonDict:
    """Decode a request body into a JSON object payload."""
    if not body:
        return {}
    try:
        payload = cast(object, json.loads(body.decode("utf-8")))
    except json.JSONDecodeError as exc:
        raise ValueError("Request body contains invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object payload.")
    return cast(JsonDict, payload)


def require_serialized_spec(payload: JsonDict) -> JsonDict:
    """Return the serialized spec payload or raise a request error."""
    serialized_spec = payload.get("spec")
    if not isinstance(serialized_spec, dict):
        raise ValueError("Missing 'spec' payload.")
    return serialized_spec


def require_non_empty_string(payload: JsonDict, field_name: str) -> str:
    """Return a trimmed non-empty string field from ``payload``."""
    raw_value = payload.get(field_name)
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ValueError(f"Missing '{field_name}' payload.")
    return raw_value.strip()


def require_boolean(
    payload: JsonDict, field_name: str, *, default: bool = False
) -> bool:
    """Return a boolean field from ``payload`` or ``default`` when omitted."""
    raw_value = payload.get(field_name, default)
    if not isinstance(raw_value, bool):
        raise ValueError(f"'{field_name}' must be a boolean when provided.")
    return raw_value


def require_string_list(payload: JsonDict, field_name: str) -> list[str]:
    """Return a non-empty list of non-empty strings from ``payload``."""
    raw_values = payload.get(field_name)
    if not isinstance(raw_values, list):
        raise ValueError(f"'{field_name}' must be a non-empty list of values.")
    values: list[str] = []
    for raw_value in raw_values:
        if not isinstance(raw_value, str) or not raw_value.strip():
            raise ValueError(f"'{field_name}' must be a non-empty list of values.")
        values.append(raw_value)
    if not values:
        raise ValueError(f"'{field_name}' must be a non-empty list of values.")
    return values


def require_canvas_position(payload: JsonDict, field_name: str) -> CanvasPosition:
    """Return a parsed canvas position from ``payload``."""
    raw_value = payload.get(field_name)
    if not isinstance(raw_value, dict):
        raise ValueError(f"Missing '{field_name}' payload.")
    try:
        return CanvasPosition.from_dict(raw_value)
    except TypeError as exc:
        raise ValueError(str(exc)) from exc


def deserialize_validation_payload(payload: JsonDict) -> NetworkSpec:
    """Load validation input from serialized spec or supported Python code."""
    serialized_spec = payload.get("spec")
    if isinstance(serialized_spec, dict):
        return deserialize_spec_with_issues(serialized_spec)

    python_code = payload.get("python_code")
    if isinstance(python_code, str):
        if not python_code.strip():
            raise ValueError("Missing 'spec' or 'python_code' payload.")
        return deserialize_spec_from_python_code(python_code, validate=False)

    raise ValueError("Missing 'spec' or 'python_code' payload.")


def parse_codegen_request(
    payload: JsonDict,
    *,
    default_engine: EngineIdentifier,
    default_collection_format: TensorCollectionFormat = TensorCollectionFormat.LIST,
) -> CodegenRequest:
    """Normalize a generate or complete request payload."""
    return CodegenRequest(
        serialized_spec=require_serialized_spec(payload),
        engine=resolve_engine(payload, default_engine),
        collection_format=resolve_collection_format(payload, default_collection_format),
    )


def parse_template_promote_request(payload: JsonDict) -> TemplatePromoteRequest:
    """Normalize a template-promotion request payload."""
    return TemplatePromoteRequest(
        serialized_spec=require_serialized_spec(payload),
        tensor_ids=require_string_list(payload, "tensor_ids"),
        template_name=require_non_empty_string(payload, "template_name"),
        overwrite=require_boolean(payload, "overwrite", default=False),
    )


def parse_template_rename_request(payload: JsonDict) -> TemplateRenameRequest:
    """Normalize a template-rename request payload."""
    return TemplateRenameRequest(
        template_name=require_non_empty_string(payload, "template_name"),
        new_template_name=require_non_empty_string(payload, "new_template_name"),
        overwrite=require_boolean(payload, "overwrite", default=False),
    )


def parse_template_delete_request(payload: JsonDict) -> TemplateDeleteRequest:
    """Normalize a template-delete request payload."""
    return TemplateDeleteRequest(
        template_name=require_non_empty_string(payload, "template_name"),
    )


def parse_subnetwork_selection_request(payload: JsonDict) -> SubnetworkSelectionRequest:
    """Normalize a subnetwork-selection request payload."""
    return SubnetworkSelectionRequest(
        serialized_spec=require_serialized_spec(payload),
        tensor_ids=require_string_list(payload, "tensor_ids"),
    )


def parse_subnetwork_prepare_insert_request(
    payload: JsonDict,
) -> SubnetworkPrepareInsertRequest:
    """Normalize a subnetwork insertion-preparation request payload."""
    return SubnetworkPrepareInsertRequest(
        serialized_spec=require_serialized_spec(payload),
        target_center=require_canvas_position(payload, "target_center"),
    )


def resolve_engine(
    payload: JsonDict,
    default_engine: EngineIdentifier,
) -> EngineIdentifier:
    """Resolve the requested engine from a JSON payload."""
    engine_value = payload.get("engine", engine_name_to_text(default_engine))
    return resolve_registered_engine(str(engine_value))


def resolve_collection_format(
    payload: JsonDict,
    default_collection_format: TensorCollectionFormat,
) -> TensorCollectionFormat:
    """Resolve the requested tensor collection format from a JSON payload."""
    collection_format_value = payload.get(
        "collection_format", default_collection_format.value
    )
    try:
        return TensorCollectionFormat(str(collection_format_value))
    except ValueError as exc:
        raise ValueError(
            f"Unsupported collection format '{collection_format_value}'."
        ) from exc


def serialize_issues(issues: list[ValidationIssue]) -> JsonList:
    """Serialize validation issues for JSON responses."""
    return cast(
        JsonList,
        [
            {"code": issue.code, "message": issue.message, "path": issue.path}
            for issue in issues
        ],
    )


def ok_response(payload: JsonDict | None = None) -> JsonResponse:
    """Return a standard successful JSON response."""
    body: JsonDict = {"ok": True}
    if payload is not None:
        body.update(payload)
    return HTTPStatus.OK, body


def bad_request_response(message: str) -> JsonResponse:
    """Return a standard bad-request JSON response."""
    return HTTPStatus.BAD_REQUEST, {"ok": False, "message": message}


def not_found_response() -> JsonResponse:
    """Return a standard not-found JSON response."""
    return HTTPStatus.NOT_FOUND, {"ok": False, "message": "Not found."}


def internal_server_error_response() -> JsonResponse:
    """Return a standard internal-server-error JSON response."""
    return HTTPStatus.INTERNAL_SERVER_ERROR, {
        "ok": False,
        "message": "Internal server error.",
    }


def issues_response(issues: list[ValidationIssue]) -> JsonResponse:
    """Return a successful response that reports validation issues."""
    return HTTPStatus.OK, {"ok": False, "issues": serialize_issues(issues)}


def serialize_spec_payload(spec: NetworkSpec) -> JsonDict:
    """Serialize a ``NetworkSpec`` with the current schema wrapper."""
    return {"schema_version": SCHEMA_VERSION, "network": spec.to_dict()}


def serialize_codegen_result(result: CodegenResult) -> JsonDict:
    """Serialize a code-generation result for the API."""
    return {
        "engine": engine_name_to_text(result.engine),
        "code": result.code,
        "warnings": cast(JSONValue, list(result.warnings)),
        "artifacts": cast(JSONValue, dict(result.artifacts)),
    }


def serialize_editor_result(result: EditorResult) -> JsonDict:
    """Serialize the final editor result for the API."""
    return {
        "engine": engine_name_to_text(result.engine),
        "confirmed": result.confirmed,
    }


def deserialize_spec_with_issues(serialized_spec: JsonDict) -> NetworkSpec:
    """Deserialize a spec payload without raising on validation issues."""
    return deserialize_spec(serialized_spec, validate=False)
