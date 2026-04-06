from __future__ import annotations

import json
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, TypeAlias, cast

from ..models import (
    CodegenResult,
    EditorResult,
    EngineName,
    NetworkSpec,
    TensorCollectionFormat,
    ValidationIssue,
)
from ..serialization import (
    SCHEMA_VERSION,
    deserialize_spec,
    deserialize_spec_from_python_code,
)

JsonDict: TypeAlias = dict[str, Any]
JsonResponse: TypeAlias = tuple[int, JsonDict]


@dataclass(slots=True, frozen=True)
class CodegenRequest:
    serialized_spec: JsonDict
    engine: EngineName
    collection_format: TensorCollectionFormat


def read_json(body: bytes) -> JsonDict:
    if not body:
        return {}
    try:
        payload = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Request body contains invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object payload.")
    return payload


def require_serialized_spec(payload: JsonDict) -> JsonDict:
    serialized_spec = payload.get("spec")
    if not isinstance(serialized_spec, dict):
        raise ValueError("Missing 'spec' payload.")
    return serialized_spec


def deserialize_validation_payload(payload: JsonDict) -> NetworkSpec:
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
    default_engine: EngineName,
    default_collection_format: TensorCollectionFormat = TensorCollectionFormat.LIST,
) -> CodegenRequest:
    return CodegenRequest(
        serialized_spec=cast(JsonDict, require_serialized_spec(payload)),
        engine=resolve_engine(payload, default_engine),
        collection_format=resolve_collection_format(payload, default_collection_format),
    )


def resolve_engine(payload: JsonDict, default_engine: EngineName) -> EngineName:
    engine_value = payload.get("engine", default_engine.value)
    try:
        return EngineName(str(engine_value))
    except ValueError as exc:
        raise ValueError(f"Unsupported engine '{engine_value}'.") from exc


def resolve_collection_format(
    payload: JsonDict,
    default_collection_format: TensorCollectionFormat,
) -> TensorCollectionFormat:
    collection_format_value = payload.get(
        "collection_format", default_collection_format.value
    )
    try:
        return TensorCollectionFormat(str(collection_format_value))
    except ValueError as exc:
        raise ValueError(
            f"Unsupported collection format '{collection_format_value}'."
        ) from exc


def serialize_issues(issues: list[ValidationIssue]) -> list[JsonDict]:
    return [
        {"code": issue.code, "message": issue.message, "path": issue.path}
        for issue in issues
    ]


def ok_response(payload: JsonDict | None = None) -> JsonResponse:
    body = {"ok": True}
    if payload is not None:
        body.update(payload)
    return HTTPStatus.OK, body


def bad_request_response(message: str) -> JsonResponse:
    return HTTPStatus.BAD_REQUEST, {"ok": False, "message": message}


def not_found_response() -> JsonResponse:
    return HTTPStatus.NOT_FOUND, {"ok": False, "message": "Not found."}


def internal_server_error_response() -> JsonResponse:
    return HTTPStatus.INTERNAL_SERVER_ERROR, {
        "ok": False,
        "message": "Internal server error.",
    }


def issues_response(issues: list[ValidationIssue]) -> JsonResponse:
    return HTTPStatus.OK, {"ok": False, "issues": serialize_issues(issues)}


def serialize_spec_payload(spec: NetworkSpec) -> JsonDict:
    return {"schema_version": SCHEMA_VERSION, "network": spec.to_dict()}


def serialize_codegen_result(result: CodegenResult) -> JsonDict:
    return {
        "engine": result.engine.value,
        "code": result.code,
        "warnings": result.warnings,
        "artifacts": result.artifacts,
    }


def serialize_editor_result(result: EditorResult) -> JsonDict:
    return {"engine": result.engine.value, "confirmed": result.confirmed}


def deserialize_spec_with_issues(serialized_spec: JsonDict) -> NetworkSpec:
    return deserialize_spec(serialized_spec, validate=False)
