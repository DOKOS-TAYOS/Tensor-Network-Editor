from __future__ import annotations

import json
from unittest.mock import patch

from tensor_network_editor.app.server import EditorServer
from tensor_network_editor.app.session import EditorSession
from tensor_network_editor.models import EngineName
from tests.app_support import request_json, request_json_with_status
from tests.factories import build_sample_spec


def test_bootstrap_returns_session_contract(
    editor_server: EditorServer,
) -> None:
    payload = request_json(f"{editor_server.base_url}/api/bootstrap")

    assert payload["default_engine"] == EngineName.EINSUM_NUMPY.value
    assert payload["schema_version"] == 3
    assert payload["spec"]["network"]["id"] == "network_demo"
    assert set(payload["engines"]) == {engine.value for engine in EngineName}
    assert payload["templates"] == list(payload["template_definitions"])
    assert payload["template_definitions"]["mps"]["graph_size_label"] == "Sites"


def test_validate_route_reports_issues_and_echoes_serialized_spec(
    editor_server: EditorServer,
) -> None:
    invalid_spec = build_sample_spec()
    invalid_spec.edges.append(invalid_spec.edges[0])

    payload = request_json(
        f"{editor_server.base_url}/api/validate",
        method="POST",
        payload={"spec": {"schema_version": 3, "network": invalid_spec.to_dict()}},
    )

    assert payload["ok"] is False
    assert payload["spec"]["schema_version"] == 3
    assert payload["spec"]["network"]["id"] == invalid_spec.id
    assert "index-already-connected" in [issue["code"] for issue in payload["issues"]]


def test_validate_route_rejects_invalid_json_with_400(
    editor_server: EditorServer,
) -> None:
    status, payload = request_json_with_status(
        f"{editor_server.base_url}/api/validate",
        method="POST",
        raw_body=b"{not-json}",
    )

    assert status == 400
    assert payload == {"ok": False, "message": "Request body contains invalid JSON."}


def test_validate_route_rejects_non_object_json_payload_with_400(
    editor_server: EditorServer,
) -> None:
    status, payload = request_json_with_status(
        f"{editor_server.base_url}/api/validate",
        method="POST",
        raw_body=json.dumps(["not", "an", "object"]).encode("utf-8"),
    )

    assert status == 400
    assert payload == {"ok": False, "message": "Expected a JSON object payload."}


def test_validate_route_rejects_legacy_schema_versions(
    editor_server: EditorServer,
) -> None:
    status, payload = request_json_with_status(
        f"{editor_server.base_url}/api/validate",
        method="POST",
        payload={
            "spec": {"schema_version": 2, "network": build_sample_spec().to_dict()}
        },
    )

    assert status == 400
    assert payload["ok"] is False
    assert "Unsupported schema version" in payload["message"]


def test_generate_route_uses_default_engine_when_missing(
    editor_server: EditorServer,
    serialized_sample_spec: dict[str, object],
) -> None:
    payload = request_json(
        f"{editor_server.base_url}/api/generate",
        method="POST",
        payload={"spec": serialized_sample_spec},
    )

    assert payload["ok"] is True
    assert payload["engine"] == EngineName.EINSUM_NUMPY.value
    assert payload["code"]


def test_generate_route_rejects_missing_spec_with_400(
    editor_server: EditorServer,
) -> None:
    status, payload = request_json_with_status(
        f"{editor_server.base_url}/api/generate",
        method="POST",
        payload={"engine": EngineName.TENSORNETWORK.value},
    )

    assert status == 400
    assert payload == {"ok": False, "message": "Missing 'spec' payload."}


def test_generate_route_rejects_unsupported_engine_with_400(
    editor_server: EditorServer,
    serialized_sample_spec: dict[str, object],
) -> None:
    status, payload = request_json_with_status(
        f"{editor_server.base_url}/api/generate",
        method="POST",
        payload={"engine": "unknown-engine", "spec": serialized_sample_spec},
    )

    assert status == 400
    assert payload["ok"] is False
    assert "Unsupported engine" in payload["message"]


def test_generate_route_returns_validation_issues_for_invalid_spec(
    editor_server: EditorServer,
) -> None:
    invalid_spec = build_sample_spec()
    invalid_spec.edges.append(invalid_spec.edges[0])

    payload = request_json(
        f"{editor_server.base_url}/api/generate",
        method="POST",
        payload={
            "engine": EngineName.TENSORNETWORK.value,
            "spec": {"schema_version": 3, "network": invalid_spec.to_dict()},
        },
    )

    assert payload["ok"] is False
    assert "index-already-connected" in [issue["code"] for issue in payload["issues"]]


def test_complete_route_stores_result_in_session(
    editor_server: EditorServer,
    editor_session: EditorSession,
    serialized_sample_spec: dict[str, object],
) -> None:
    payload = request_json(
        f"{editor_server.base_url}/api/complete",
        method="POST",
        payload={
            "engine": EngineName.QUIMB.value,
            "spec": serialized_sample_spec,
        },
    )

    assert payload == {
        "ok": True,
        "engine": EngineName.QUIMB.value,
        "confirmed": True,
    }
    result = editor_session.wait_for_result(timeout=0.1)
    assert result is not None
    assert result.engine is EngineName.QUIMB
    assert result.codegen is not None


def test_cancel_route_ends_session_without_result(
    editor_server: EditorServer,
    editor_session: EditorSession,
) -> None:
    payload = request_json(
        f"{editor_server.base_url}/api/cancel",
        method="POST",
        payload={},
    )

    assert payload == {"ok": True}
    assert editor_session.wait_for_result(timeout=0.1) is None


def test_autolayout_route_is_not_available(editor_server: EditorServer) -> None:
    status, payload = request_json_with_status(
        f"{editor_server.base_url}/api/autolayout",
        method="POST",
        payload={
            "spec": {"schema_version": 3, "network": build_sample_spec().to_dict()}
        },
    )

    assert status == 404
    assert payload == {"ok": False, "message": "Not found."}


def test_template_route_returns_valid_serialized_spec(
    editor_server: EditorServer,
) -> None:
    payload = request_json(
        f"{editor_server.base_url}/api/template",
        method="POST",
        payload={"template": "mps"},
    )

    assert payload["ok"] is True
    assert payload["spec"]["schema_version"] == 3
    assert payload["spec"]["network"]["name"] == "MPS"
    assert payload["spec"]["network"]["tensors"]


def test_template_route_applies_requested_parameters(
    editor_server: EditorServer,
) -> None:
    payload = request_json(
        f"{editor_server.base_url}/api/template",
        method="POST",
        payload={
            "template": "peps_2x2",
            "parameters": {
                "graph_size": 3,
                "bond_dimension": 5,
                "physical_dimension": 7,
            },
        },
    )

    center_tensor = next(
        tensor
        for tensor in payload["spec"]["network"]["tensors"]
        if tensor["name"] == "B2"
    )

    assert payload["ok"] is True
    assert payload["spec"]["network"]["name"] == "PEPS 3x3"
    assert len(payload["spec"]["network"]["tensors"]) == 9
    assert len(payload["spec"]["network"]["edges"]) == 12
    assert {index["name"] for index in center_tensor["indices"]} == {
        "left",
        "right",
        "up",
        "down",
        "phys",
    }
    assert {
        index["dimension"]
        for tensor in payload["spec"]["network"]["tensors"]
        for index in tensor["indices"]
        if index["name"] == "phys"
    } == {7}


def test_template_route_rejects_invalid_template_parameters(
    editor_server: EditorServer,
) -> None:
    status, payload = request_json_with_status(
        f"{editor_server.base_url}/api/template",
        method="POST",
        payload={
            "template": "mps",
            "parameters": {
                "graph_size": 1,
                "bond_dimension": 0,
                "physical_dimension": 2,
            },
        },
    )

    assert status == 400
    assert payload["ok"] is False
    assert "graph_size" in payload["message"]


def test_analyze_contraction_route_returns_manual_summary(
    editor_server: EditorServer,
    serialized_sample_spec: dict[str, object],
) -> None:
    payload = request_json(
        f"{editor_server.base_url}/api/analyze-contraction",
        method="POST",
        payload={"spec": serialized_sample_spec},
    )

    assert payload["ok"] is True
    assert payload["automatic_strategy"] == "greedy"
    assert payload["network_output_shape"] == [2, 4]
    assert payload["manual"]["status"] == "complete"
    assert payload["manual"]["summary"]["total_estimated_flops"] == 48
    assert payload["manual"]["summary"]["total_estimated_macs"] == 24
    assert payload["manual"]["summary"]["final_shape"] == [2, 4]
    assert payload["manual"]["steps"][0]["estimated_flops"] == 48
    assert payload["manual"]["steps"][0]["estimated_macs"] == 24


def test_unexpected_server_errors_return_generic_500_payload(
    editor_server: EditorServer,
    serialized_sample_spec: dict[str, object],
) -> None:
    with patch(
        "tensor_network_editor.app.server.routes.handle_generate",
        side_effect=RuntimeError("boom"),
    ):
        status, payload = request_json_with_status(
            f"{editor_server.base_url}/api/generate",
            method="POST",
            payload={
                "engine": EngineName.TENSORNETWORK.value,
                "spec": serialized_sample_spec,
            },
        )

    assert status == 500
    assert payload == {"ok": False, "message": "Internal server error."}
