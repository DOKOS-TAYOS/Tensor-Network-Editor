from __future__ import annotations

from http import HTTPStatus
from typing import cast

import pytest

from tensor_network_editor.app._protocol import (
    CodegenRequest,
    JsonDict,
    bad_request_response,
    deserialize_spec_with_issues,
    issues_response,
    ok_response,
    parse_codegen_request,
    read_json,
    require_serialized_spec,
    resolve_collection_format,
    resolve_engine,
    serialize_codegen_result,
    serialize_editor_result,
    serialize_issues,
)
from tensor_network_editor.models import (
    CodegenResult,
    EditorResult,
    EngineName,
    NetworkSpec,
    TensorCollectionFormat,
    ValidationIssue,
)


def test_read_json_accepts_empty_body() -> None:
    assert read_json(b"") == {}


def test_read_json_rejects_invalid_json() -> None:
    with pytest.raises(ValueError, match="invalid JSON"):
        read_json(b"{not-json}")


def test_read_json_rejects_non_object_payload() -> None:
    with pytest.raises(ValueError, match="JSON object payload"):
        read_json(b"[]")


def test_require_serialized_spec_rejects_missing_spec() -> None:
    with pytest.raises(ValueError, match="Missing 'spec' payload"):
        require_serialized_spec({})


def test_resolve_engine_uses_default_when_payload_omits_engine() -> None:
    assert resolve_engine({}, EngineName.EINSUM_NUMPY) is EngineName.EINSUM_NUMPY


def test_resolve_engine_rejects_unknown_engine() -> None:
    with pytest.raises(ValueError, match="Unsupported engine"):
        resolve_engine({"engine": "unknown"}, EngineName.EINSUM_NUMPY)


def test_resolve_collection_format_uses_default_when_payload_omits_value() -> None:
    assert (
        resolve_collection_format({}, TensorCollectionFormat.MATRIX)
        is TensorCollectionFormat.MATRIX
    )


def test_resolve_collection_format_rejects_unknown_collection_format() -> None:
    with pytest.raises(ValueError, match="Unsupported collection format"):
        resolve_collection_format(
            {"collection_format": "invalid"},
            TensorCollectionFormat.LIST,
        )


def test_parse_codegen_request_uses_defaults_when_optional_fields_are_missing(
    serialized_sample_spec: dict[str, object],
) -> None:
    request = parse_codegen_request(
        {"spec": serialized_sample_spec},
        default_engine=EngineName.EINSUM_TORCH,
        default_collection_format=TensorCollectionFormat.DICT,
    )

    assert request == CodegenRequest(
        serialized_spec=cast(JsonDict, serialized_sample_spec),
        engine=EngineName.EINSUM_TORCH,
        collection_format=TensorCollectionFormat.DICT,
    )


def test_parse_codegen_request_honors_explicit_engine_and_collection_format(
    serialized_sample_spec: dict[str, object],
) -> None:
    request = parse_codegen_request(
        {
            "spec": serialized_sample_spec,
            "engine": EngineName.QUIMB.value,
            "collection_format": TensorCollectionFormat.MATRIX.value,
        },
        default_engine=EngineName.EINSUM_NUMPY,
        default_collection_format=TensorCollectionFormat.LIST,
    )

    assert request == CodegenRequest(
        serialized_spec=cast(JsonDict, serialized_sample_spec),
        engine=EngineName.QUIMB,
        collection_format=TensorCollectionFormat.MATRIX,
    )


def test_parse_codegen_request_rejects_missing_spec() -> None:
    with pytest.raises(ValueError, match="Missing 'spec' payload"):
        parse_codegen_request(
            {},
            default_engine=EngineName.EINSUM_NUMPY,
            default_collection_format=TensorCollectionFormat.LIST,
        )


def test_serialize_helpers_expose_public_response_shapes(
    sample_spec: NetworkSpec,
) -> None:
    codegen_result = CodegenResult(
        engine=EngineName.EINSUM_NUMPY,
        code="print('ok')\n",
        warnings=["warn"],
        artifacts={"format": "python"},
    )
    editor_result = EditorResult(
        spec=sample_spec,
        engine=EngineName.EINSUM_NUMPY,
        codegen=codegen_result,
        confirmed=True,
    )
    issues = [
        ValidationIssue(
            code="invalid-name",
            message="Tensor name cannot be empty.",
            path="tensors.tensor_a.name",
        )
    ]

    assert ok_response({"value": 1}) == (HTTPStatus.OK, {"ok": True, "value": 1})
    assert bad_request_response("bad") == (
        HTTPStatus.BAD_REQUEST,
        {"ok": False, "message": "bad"},
    )
    assert issues_response(issues) == (
        HTTPStatus.OK,
        {
            "ok": False,
            "issues": [
                {
                    "code": "invalid-name",
                    "message": "Tensor name cannot be empty.",
                    "path": "tensors.tensor_a.name",
                }
            ],
        },
    )
    assert serialize_issues(issues) == [
        {
            "code": "invalid-name",
            "message": "Tensor name cannot be empty.",
            "path": "tensors.tensor_a.name",
        }
    ]
    assert serialize_codegen_result(codegen_result) == {
        "engine": EngineName.EINSUM_NUMPY.value,
        "code": "print('ok')\n",
        "warnings": ["warn"],
        "artifacts": {"format": "python"},
    }
    assert serialize_editor_result(editor_result) == {
        "engine": EngineName.EINSUM_NUMPY.value,
        "confirmed": True,
    }


def test_deserialize_spec_with_issues_skips_validation(
    serialized_sample_spec: dict[str, object],
) -> None:
    payload = dict(serialized_sample_spec)
    network_payload = dict(cast(dict[str, object], payload["network"]))
    network_payload["name"] = "   "
    payload["network"] = network_payload

    restored = deserialize_spec_with_issues(payload)

    assert restored.name == "   "
