from __future__ import annotations

from http import HTTPStatus

import pytest

from tensor_network_editor.app._protocol import (
    bad_request_response,
    deserialize_spec_with_issues,
    handle_codegen_operation,
    issues_response,
    ok_response,
    read_json,
    require_serialized_spec,
    resolve_engine,
    serialize_codegen_result,
    serialize_editor_result,
    serialize_issues,
)
from tensor_network_editor.errors import SerializationError, SpecValidationError
from tensor_network_editor.models import (
    CodegenResult,
    EditorResult,
    EngineName,
    NetworkSpec,
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


def test_handle_codegen_operation_returns_success_payload(
    serialized_sample_spec: dict[str, object],
) -> None:
    def operation(
        serialized_spec: dict[str, object],
        engine: EngineName,
    ) -> CodegenResult:
        assert serialized_spec is serialized_sample_spec
        assert engine is EngineName.QUIMB
        return CodegenResult(engine=engine, code="generated\n")

    status, response = handle_codegen_operation(
        {"spec": serialized_sample_spec, "engine": EngineName.QUIMB.value},
        default_engine=EngineName.EINSUM_NUMPY,
        operation=operation,
        success_payload_builder=serialize_codegen_result,
    )

    assert status == HTTPStatus.OK
    assert response["ok"] is True
    assert response["engine"] == EngineName.QUIMB.value
    assert response["code"] == "generated\n"


def test_handle_codegen_operation_uses_default_engine_when_missing(
    serialized_sample_spec: dict[str, object],
) -> None:
    def operation(
        serialized_spec: dict[str, object],
        engine: EngineName,
    ) -> CodegenResult:
        assert serialized_spec is serialized_sample_spec
        return CodegenResult(engine=engine, code="generated\n")

    status, response = handle_codegen_operation(
        {"spec": serialized_sample_spec},
        default_engine=EngineName.EINSUM_TORCH,
        operation=operation,
        success_payload_builder=serialize_codegen_result,
    )

    assert status == HTTPStatus.OK
    assert response["engine"] == EngineName.EINSUM_TORCH.value


def test_handle_codegen_operation_maps_serialization_failures_to_bad_request(
    serialized_sample_spec: dict[str, object],
) -> None:
    def operation(
        serialized_spec: dict[str, object],
        engine: EngineName,
    ) -> CodegenResult:
        del serialized_spec
        del engine
        raise SerializationError("bad spec")

    status, response = handle_codegen_operation(
        {"spec": serialized_sample_spec},
        default_engine=EngineName.EINSUM_NUMPY,
        operation=operation,
        success_payload_builder=serialize_codegen_result,
    )

    assert status == HTTPStatus.BAD_REQUEST
    assert response == {"ok": False, "message": "bad spec"}


def test_handle_codegen_operation_maps_validation_failures_to_issue_payload(
    serialized_sample_spec: dict[str, object],
) -> None:
    issue = ValidationIssue(
        code="invalid-name",
        message="Tensor name cannot be empty.",
        path="tensors.tensor_a.name",
    )

    def operation(
        serialized_spec: dict[str, object],
        engine: EngineName,
    ) -> CodegenResult:
        del serialized_spec
        del engine
        raise SpecValidationError([issue])

    status, response = handle_codegen_operation(
        {"spec": serialized_sample_spec},
        default_engine=EngineName.EINSUM_NUMPY,
        operation=operation,
        success_payload_builder=serialize_codegen_result,
    )

    assert status == HTTPStatus.OK
    assert response == {
        "ok": False,
        "issues": [
            {
                "code": "invalid-name",
                "message": "Tensor name cannot be empty.",
                "path": "tensors.tensor_a.name",
            }
        ],
    }


def test_deserialize_spec_with_issues_skips_validation(
    serialized_sample_spec: dict[str, object],
) -> None:
    payload = dict(serialized_sample_spec)
    network_payload = dict(payload["network"])
    network_payload["name"] = "   "
    payload["network"] = network_payload

    restored = deserialize_spec_with_issues(payload)

    assert restored.name == "   "
