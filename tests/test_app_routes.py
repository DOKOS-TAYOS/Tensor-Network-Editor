from __future__ import annotations

import json
from collections.abc import Callable
from http.client import HTTPConnection
from pathlib import Path
from typing import cast
from unittest.mock import patch
from urllib.parse import urlparse

import pytest

from tensor_network_editor.analysis import analyze_contraction
from tensor_network_editor.api import generate_code
from tensor_network_editor.app._protocol import JsonDict
from tensor_network_editor.app.routes import handle_bootstrap
from tensor_network_editor.app.server import EditorServer
from tensor_network_editor.app.session import EditorSession
from tensor_network_editor.errors import PackageIOError
from tensor_network_editor.models import EngineName, NetworkSpec, TensorCollectionFormat
from tensor_network_editor.serialization import (
    SCHEMA_VERSION,
)
from tensor_network_editor.serialization import (
    deserialize_spec as deserialize_spec_impl,
)
from tests.app_support import request_json, request_json_with_status
from tests.factories import (
    build_linear_periodic_carry_chain_spec,
    build_linear_periodic_chain_spec,
    build_linear_periodic_partial_carry_chain_spec,
    build_outer_product_plan_spec,
    build_sample_spec,
    build_sample_spec_with_view_snapshots,
)


def test_bootstrap_returns_session_contract(
    editor_server: EditorServer,
) -> None:
    payload = request_json(f"{editor_server.base_url}/api/bootstrap")

    assert payload["default_engine"] == EngineName.EINSUM_NUMPY.value
    assert payload["default_collection_format"] == TensorCollectionFormat.LIST.value
    assert payload["collection_formats"] == [
        collection_format.value for collection_format in TensorCollectionFormat
    ]
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["spec"]["network"]["id"] == "network_demo"
    assert set(payload["engines"]) == {engine.value for engine in EngineName}
    assert payload["templates"] == list(payload["template_definitions"])
    assert payload["template_definitions"]["mps"]["graph_size_label"] == "Sites"
    assert payload["template_definitions"]["mps"]["source"] == "global"
    assert list(payload["annotation_definitions"]) == ["tensor", "index"]
    assert payload["annotation_definitions"]["tensor"][0]["key"] == "role"
    assert payload["annotation_definitions"]["index"][0]["key"] == "leg_kind"
    assert payload["app_metadata"] == {
        "repository_url": "https://github.com/DOKOS-TAYOS/Tensor-Network-Editor",
        "version": "0.3.0",
        "license_name": "MIT",
        "author_name": "Alejandro Mata Ali",
    }


def test_bootstrap_accepts_invalid_initial_spec_for_editing() -> None:
    status, payload = handle_bootstrap(
        EditorSession(
            initial_spec=NetworkSpec(id="network_invalid", name="   "),
            default_engine=EngineName.EINSUM_NUMPY,
        )
    )
    spec_payload = cast(JsonDict, payload["spec"])
    network_payload = cast(JsonDict, spec_payload["network"])

    assert status == 200
    assert network_payload["id"] == "network_invalid"
    assert network_payload["name"] == "   "


def test_validate_route_reports_issues_and_echoes_serialized_spec(
    editor_server: EditorServer,
) -> None:
    invalid_spec = build_sample_spec()
    invalid_spec.edges.append(invalid_spec.edges[0])

    payload = request_json(
        f"{editor_server.base_url}/api/validate",
        method="POST",
        payload={
            "spec": {
                "schema_version": SCHEMA_VERSION,
                "network": invalid_spec.to_dict(),
            }
        },
    )

    assert payload["ok"] is False
    assert payload["spec"]["schema_version"] == SCHEMA_VERSION
    assert payload["spec"]["network"]["id"] == invalid_spec.id
    assert "index-already-connected" in [issue["code"] for issue in payload["issues"]]


def test_validate_route_preserves_contraction_view_snapshots(
    editor_server: EditorServer,
) -> None:
    payload = request_json(
        f"{editor_server.base_url}/api/validate",
        method="POST",
        payload={
            "spec": {
                "schema_version": SCHEMA_VERSION,
                "network": build_sample_spec_with_view_snapshots().to_dict(),
            }
        },
    )

    snapshots = payload["spec"]["network"]["contraction_plan"]["view_snapshots"]

    assert payload["ok"] is True
    assert len(snapshots) == 2
    assert snapshots[1]["applied_step_count"] == 1
    assert snapshots[1]["operand_layouts"][0]["operand_id"] == "step_contract_ab"


def test_validate_route_accepts_generated_python_code_payload(
    editor_server: EditorServer,
) -> None:
    generated = generate_code(
        build_sample_spec(),
        engine=EngineName.QUIMB,
        collection_format=TensorCollectionFormat.DICT,
    )

    payload = request_json(
        f"{editor_server.base_url}/api/validate",
        method="POST",
        payload={"python_code": generated.code},
    )

    assert payload["ok"] is True
    assert payload["issues"] == []
    assert payload["spec"]["schema_version"] == SCHEMA_VERSION
    assert payload["spec"]["network"]["name"] == "Imported Python Network"
    assert [tensor["name"] for tensor in payload["spec"]["network"]["tensors"]] == [
        "A",
        "B",
    ]


def test_validate_route_rejects_linear_periodic_generated_python_with_clear_message(
    editor_server: EditorServer,
) -> None:
    generated = generate_code(
        build_linear_periodic_chain_spec(),
        engine=EngineName.TENSORNETWORK,
    )

    status, payload = request_json_with_status(
        f"{editor_server.base_url}/api/validate",
        method="POST",
        payload={"python_code": generated.code},
    )

    assert status == 400
    assert payload["ok"] is False
    assert "linear periodic mode" in payload["message"].lower()


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


def test_validate_route_rejects_non_integer_content_length_with_400(
    editor_server: EditorServer,
) -> None:
    status, payload = _post_validate_with_raw_content_length(
        editor_server,
        content_length="abc",
        body=b"{}",
    )

    assert status == 400
    assert payload == {"ok": False, "message": "Invalid Content-Length header."}


def test_validate_route_rejects_oversized_content_length_with_400(
    editor_server: EditorServer,
) -> None:
    status, payload = _post_validate_with_raw_content_length(
        editor_server,
        content_length="1048577",
        body=b"{}",
    )

    assert status == 400
    assert payload == {
        "ok": False,
        "message": "Request body exceeds maximum allowed size.",
    }


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
            "spec": {
                "schema_version": 1,
                "network": build_sample_spec().to_dict(),
            }
        },
    )

    assert status == 400
    assert payload["ok"] is False
    assert "Unsupported schema version" in payload["message"]


def _post_validate_with_raw_content_length(
    editor_server: EditorServer, content_length: str, body: bytes
) -> tuple[int, dict[str, object]]:
    parsed = urlparse(editor_server.base_url)
    host = parsed.hostname
    port = parsed.port
    if host is None or port is None:
        raise AssertionError(f"Unexpected editor base URL: {editor_server.base_url}")
    connection = HTTPConnection(host, port, timeout=5)
    try:
        connection.putrequest("POST", "/api/validate")
        connection.putheader("Content-Type", "application/json")
        connection.putheader("Content-Length", content_length)
        connection.endheaders()
        if body:
            connection.send(body)
        response = connection.getresponse()
        response_payload = json.loads(response.read().decode("utf-8"))
        return response.status, cast(dict[str, object], response_payload)
    finally:
        connection.close()


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


def test_generate_route_accepts_collection_format(
    editor_server: EditorServer,
    serialized_sample_spec: dict[str, object],
) -> None:
    payload = request_json(
        f"{editor_server.base_url}/api/generate",
        method="POST",
        payload={
            "engine": EngineName.EINSUM_NUMPY.value,
            "collection_format": TensorCollectionFormat.DICT.value,
            "spec": serialized_sample_spec,
        },
    )

    assert payload["ok"] is True
    assert payload["engine"] == EngineName.EINSUM_NUMPY.value
    assert "tensors_dict = {" in payload["code"]


def test_generate_route_returns_linear_periodic_carry_code(
    editor_server: EditorServer,
) -> None:
    payload = request_json(
        f"{editor_server.base_url}/api/generate",
        method="POST",
        payload={
            "engine": EngineName.TENSORNETWORK.value,
            "spec": {
                "schema_version": SCHEMA_VERSION,
                "network": build_linear_periodic_carry_chain_spec().to_dict(),
            },
        },
    )

    assert payload["ok"] is True
    assert payload["engine"] == EngineName.TENSORNETWORK.value
    assert "previous_payload = build_initial_cell()" in payload["code"]


@pytest.mark.parametrize(
    ("engine", "spec_factory", "expected_snippet"),
    [
        (
            EngineName.QUIMB,
            build_linear_periodic_chain_spec,
            "import quimb.tensor as qtn",
        ),
        (
            EngineName.QUIMB,
            build_linear_periodic_carry_chain_spec,
            "previous_payload = build_initial_cell()",
        ),
        (
            EngineName.EINSUM_NUMPY,
            build_linear_periodic_chain_spec,
            "result = np.einsum(",
        ),
        (
            EngineName.EINSUM_NUMPY,
            build_linear_periodic_carry_chain_spec,
            "results_list.append(np.einsum(",
        ),
        (
            EngineName.EINSUM_TORCH,
            build_linear_periodic_chain_spec,
            "result = torch.einsum(",
        ),
        (
            EngineName.EINSUM_TORCH,
            build_linear_periodic_carry_chain_spec,
            "results_list.append(torch.einsum(",
        ),
    ],
)
def test_generate_route_accepts_linear_periodic_for_remaining_backends(
    editor_server: EditorServer,
    engine: EngineName,
    spec_factory: Callable[[], NetworkSpec],
    expected_snippet: str,
) -> None:
    payload = request_json(
        f"{editor_server.base_url}/api/generate",
        method="POST",
        payload={
            "engine": engine.value,
            "spec": {
                "schema_version": SCHEMA_VERSION,
                "network": spec_factory().to_dict(),
            },
        },
    )

    assert payload["ok"] is True
    assert payload["engine"] == engine.value
    assert expected_snippet in payload["code"]


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
            "spec": {
                "schema_version": SCHEMA_VERSION,
                "network": invalid_spec.to_dict(),
            },
        },
    )

    assert payload["ok"] is False
    assert "index-already-connected" in [issue["code"] for issue in payload["issues"]]


def test_generate_route_returns_backend_codegen_error_message(
    editor_server: EditorServer,
) -> None:
    status, payload = request_json_with_status(
        f"{editor_server.base_url}/api/generate",
        method="POST",
        payload={
            "engine": EngineName.TENSORKROWCH.value,
            "spec": {
                "schema_version": SCHEMA_VERSION,
                "network": build_outer_product_plan_spec().to_dict(),
            },
        },
    )

    assert status == 400
    assert payload["ok"] is False
    assert "TensorKrowch" in payload["message"]
    assert "shared index" in payload["message"].lower()


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


def test_complete_route_accepts_collection_format(
    editor_server: EditorServer,
    editor_session: EditorSession,
    serialized_sample_spec: dict[str, object],
) -> None:
    payload = request_json(
        f"{editor_server.base_url}/api/complete",
        method="POST",
        payload={
            "engine": EngineName.EINSUM_NUMPY.value,
            "collection_format": TensorCollectionFormat.MATRIX.value,
            "spec": serialized_sample_spec,
        },
    )

    assert payload == {
        "ok": True,
        "engine": EngineName.EINSUM_NUMPY.value,
        "confirmed": True,
    }
    result = editor_session.wait_for_result(timeout=0.1)
    assert result is not None
    assert result.codegen is not None
    assert "tensor_rows = []" in result.codegen.code


def test_complete_route_reports_code_output_write_errors_as_bad_request(
    tmp_path: Path,
    serialized_sample_spec: dict[str, object],
) -> None:
    server = EditorServer(
        EditorSession(
            initial_spec=build_sample_spec(),
            default_engine=EngineName.EINSUM_NUMPY,
            code_path=tmp_path / "missing_parent" / "generated.py",
        )
    )
    server.start()
    try:
        status, payload = request_json_with_status(
            f"{server.base_url}/api/complete",
            method="POST",
            payload={
                "engine": EngineName.EINSUM_NUMPY.value,
                "spec": serialized_sample_spec,
            },
        )
    finally:
        server.stop()

    assert status == 400
    assert payload["ok"] is False
    assert "Could not write generated Python code" in payload["message"]


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
            "spec": {
                "schema_version": SCHEMA_VERSION,
                "network": build_sample_spec().to_dict(),
            }
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
    assert payload["spec"]["schema_version"] == SCHEMA_VERSION
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


def test_template_promote_route_persists_project_template_catalog(
    tmp_path: Path,
) -> None:
    catalog_path = tmp_path / ".tensor-network-editor" / "templates.json"
    server = EditorServer(
        EditorSession(
            initial_spec=build_sample_spec(),
            default_engine=EngineName.EINSUM_NUMPY,
            template_catalog_path=catalog_path,
        )
    )
    server.start()
    try:
        payload = request_json(
            f"{server.base_url}/api/template/promote",
            method="POST",
            payload={
                "spec": {
                    "schema_version": SCHEMA_VERSION,
                    "network": build_sample_spec().to_dict(),
                },
                "tensor_ids": ["tensor_a", "tensor_b"],
                "template_name": "project_pair",
            },
        )
    finally:
        server.stop()

    persisted_payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    reloaded_server = EditorServer(
        EditorSession(
            initial_spec=build_sample_spec(),
            default_engine=EngineName.EINSUM_NUMPY,
            template_catalog_path=catalog_path,
        )
    )
    reloaded_server.start()
    try:
        bootstrap_payload = request_json(f"{reloaded_server.base_url}/api/bootstrap")
    finally:
        reloaded_server.stop()

    assert payload["ok"] is True
    assert payload["selected_template"] == "project_pair"
    assert payload["templates"][0] == "project_pair"
    assert (
        payload["template_definitions"]["project_pair"]["supports_parameters"] is False
    )
    assert payload["template_definitions"]["project_pair"]["source"] == "project"
    assert payload["template_catalog_warnings"] == []
    assert persisted_payload["templates"][0]["name"] == "project_pair"
    assert persisted_payload["templates"][0]["spec"]["network"]["notes"] == []
    assert (
        persisted_payload["templates"][0]["spec"]["network"]["contraction_plan"] is None
    )
    assert bootstrap_payload["templates"][0] == "project_pair"


def test_template_promote_route_rejects_invalid_template_name(
    tmp_path: Path,
) -> None:
    server = EditorServer(
        EditorSession(
            initial_spec=build_sample_spec(),
            default_engine=EngineName.EINSUM_NUMPY,
            template_catalog_path=tmp_path
            / ".tensor-network-editor"
            / "templates.json",
        )
    )
    server.start()
    try:
        status, payload = request_json_with_status(
            f"{server.base_url}/api/template/promote",
            method="POST",
            payload={
                "spec": {
                    "schema_version": SCHEMA_VERSION,
                    "network": build_sample_spec().to_dict(),
                },
                "tensor_ids": ["tensor_a", "tensor_b"],
                "template_name": "Bad Name",
            },
        )
    finally:
        server.stop()

    assert status == 400
    assert payload["ok"] is False
    assert "lowercase letter" in payload["message"]


def test_template_promote_route_rejects_duplicate_template_name(
    tmp_path: Path,
) -> None:
    server = EditorServer(
        EditorSession(
            initial_spec=build_sample_spec(),
            default_engine=EngineName.EINSUM_NUMPY,
            template_catalog_path=tmp_path
            / ".tensor-network-editor"
            / "templates.json",
        )
    )
    server.start()
    try:
        first_payload = {
            "spec": {
                "schema_version": SCHEMA_VERSION,
                "network": build_sample_spec().to_dict(),
            },
            "tensor_ids": ["tensor_a", "tensor_b"],
            "template_name": "project_pair",
        }
        first_response = request_json(
            f"{server.base_url}/api/template/promote",
            method="POST",
            payload=first_payload,
        )
        status, payload = request_json_with_status(
            f"{server.base_url}/api/template/promote",
            method="POST",
            payload=first_payload,
        )
    finally:
        server.stop()

    assert first_response["ok"] is True
    assert status == 400
    assert payload["ok"] is False
    assert "already registered" in payload["message"]


def test_template_promote_route_allows_overwrite_for_project_templates(
    tmp_path: Path,
) -> None:
    catalog_path = tmp_path / ".tensor-network-editor" / "templates.json"
    server = EditorServer(
        EditorSession(
            initial_spec=build_sample_spec(),
            default_engine=EngineName.EINSUM_NUMPY,
            template_catalog_path=catalog_path,
        )
    )
    server.start()
    try:
        first_payload = {
            "spec": {
                "schema_version": SCHEMA_VERSION,
                "network": build_sample_spec().to_dict(),
            },
            "tensor_ids": ["tensor_a", "tensor_b"],
            "template_name": "project_pair",
        }
        first_response = request_json(
            f"{server.base_url}/api/template/promote",
            method="POST",
            payload=first_payload,
        )
        overwritten_response = request_json(
            f"{server.base_url}/api/template/promote",
            method="POST",
            payload={
                **first_payload,
                "tensor_ids": ["tensor_a"],
                "overwrite": True,
            },
        )
    finally:
        server.stop()

    persisted_payload = json.loads(catalog_path.read_text(encoding="utf-8"))

    assert first_response["ok"] is True
    assert overwritten_response["ok"] is True
    assert overwritten_response["selected_template"] == "project_pair"
    assert persisted_payload["templates"][0]["name"] == "project_pair"
    assert [
        tensor["id"]
        for tensor in persisted_payload["templates"][0]["spec"]["network"]["tensors"]
    ] == ["tensor_a"]


def test_template_promote_route_rejects_overwrite_of_global_template_name(
    tmp_path: Path,
) -> None:
    server = EditorServer(
        EditorSession(
            initial_spec=build_sample_spec(),
            default_engine=EngineName.EINSUM_NUMPY,
            template_catalog_path=tmp_path
            / ".tensor-network-editor"
            / "templates.json",
        )
    )
    server.start()
    try:
        status, payload = request_json_with_status(
            f"{server.base_url}/api/template/promote",
            method="POST",
            payload={
                "spec": {
                    "schema_version": SCHEMA_VERSION,
                    "network": build_sample_spec().to_dict(),
                },
                "tensor_ids": ["tensor_a", "tensor_b"],
                "template_name": "mps",
                "overwrite": True,
            },
        )
    finally:
        server.stop()

    assert status == 400
    assert payload["ok"] is False
    assert "global" in payload["message"]


def test_template_promote_route_reports_catalog_io_errors_as_bad_request(
    editor_server: EditorServer,
) -> None:
    with patch(
        "tensor_network_editor.app.routes.promote_serialized_subnetwork_to_template",
        side_effect=PackageIOError("Could not write project template catalog JSON."),
    ):
        status, payload = request_json_with_status(
            f"{editor_server.base_url}/api/template/promote",
            method="POST",
            payload={
                "spec": {
                    "schema_version": SCHEMA_VERSION,
                    "network": build_sample_spec().to_dict(),
                },
                "tensor_ids": ["tensor_a", "tensor_b"],
                "template_name": "project_pair",
            },
        )

    assert status == 400
    assert payload == {
        "ok": False,
        "message": "Could not write project template catalog JSON.",
    }


def test_template_rename_route_renames_project_template_and_updates_selection(
    tmp_path: Path,
) -> None:
    catalog_path = tmp_path / ".tensor-network-editor" / "templates.json"
    server = EditorServer(
        EditorSession(
            initial_spec=build_sample_spec(),
            default_engine=EngineName.EINSUM_NUMPY,
            template_catalog_path=catalog_path,
        )
    )
    server.start()
    try:
        request_json(
            f"{server.base_url}/api/template/promote",
            method="POST",
            payload={
                "spec": {
                    "schema_version": SCHEMA_VERSION,
                    "network": build_sample_spec().to_dict(),
                },
                "tensor_ids": ["tensor_a", "tensor_b"],
                "template_name": "project_pair",
            },
        )
        payload = request_json(
            f"{server.base_url}/api/template/rename",
            method="POST",
            payload={
                "template_name": "project_pair",
                "new_template_name": "renamed_pair",
            },
        )
    finally:
        server.stop()

    persisted_payload = json.loads(catalog_path.read_text(encoding="utf-8"))

    assert payload["ok"] is True
    assert payload["selected_template"] == "renamed_pair"
    assert payload["templates"][0] == "renamed_pair"
    assert payload["template_definitions"]["renamed_pair"]["source"] == "project"
    assert persisted_payload["templates"][0]["name"] == "renamed_pair"
    assert persisted_payload["templates"][0]["display_name"] == "Renamed Pair"
    assert (
        persisted_payload["templates"][0]["spec"]["network"]["name"] == "Renamed Pair"
    )


def test_template_rename_route_rejects_global_duplicate_and_missing_template(
    tmp_path: Path,
) -> None:
    catalog_path = tmp_path / ".tensor-network-editor" / "templates.json"
    server = EditorServer(
        EditorSession(
            initial_spec=build_sample_spec(),
            default_engine=EngineName.EINSUM_NUMPY,
            template_catalog_path=catalog_path,
        )
    )
    server.start()
    try:
        request_json(
            f"{server.base_url}/api/template/promote",
            method="POST",
            payload={
                "spec": {
                    "schema_version": SCHEMA_VERSION,
                    "network": build_sample_spec().to_dict(),
                },
                "tensor_ids": ["tensor_a", "tensor_b"],
                "template_name": "project_pair",
            },
        )
        global_status, global_payload = request_json_with_status(
            f"{server.base_url}/api/template/rename",
            method="POST",
            payload={
                "template_name": "project_pair",
                "new_template_name": "mps",
            },
        )
        missing_status, missing_payload = request_json_with_status(
            f"{server.base_url}/api/template/rename",
            method="POST",
            payload={
                "template_name": "missing_template",
                "new_template_name": "renamed_pair",
            },
        )
    finally:
        server.stop()

    assert global_status == 400
    assert global_payload["ok"] is False
    assert "global" in global_payload["message"]
    assert missing_status == 400
    assert missing_payload["ok"] is False
    assert "missing_template" in missing_payload["message"]


def test_template_rename_route_reports_catalog_io_errors_as_bad_request(
    editor_server: EditorServer,
) -> None:
    with patch(
        "tensor_network_editor.app.routes.rename_session_project_template",
        side_effect=PackageIOError("Could not write project template catalog JSON."),
    ):
        status, payload = request_json_with_status(
            f"{editor_server.base_url}/api/template/rename",
            method="POST",
            payload={
                "template_name": "project_pair",
                "new_template_name": "renamed_pair",
            },
        )

    assert status == 400
    assert payload == {
        "ok": False,
        "message": "Could not write project template catalog JSON.",
    }


def test_template_delete_route_deletes_project_template_and_keeps_selection_stable(
    tmp_path: Path,
) -> None:
    catalog_path = tmp_path / ".tensor-network-editor" / "templates.json"
    server = EditorServer(
        EditorSession(
            initial_spec=build_sample_spec(),
            default_engine=EngineName.EINSUM_NUMPY,
            template_catalog_path=catalog_path,
        )
    )
    server.start()
    try:
        for template_name, tensor_ids in (
            ("project_pair", ["tensor_a", "tensor_b"]),
            ("project_single", ["tensor_a"]),
        ):
            request_json(
                f"{server.base_url}/api/template/promote",
                method="POST",
                payload={
                    "spec": {
                        "schema_version": SCHEMA_VERSION,
                        "network": build_sample_spec().to_dict(),
                    },
                    "tensor_ids": tensor_ids,
                    "template_name": template_name,
                },
            )
        payload = request_json(
            f"{server.base_url}/api/template/delete",
            method="POST",
            payload={"template_name": "project_pair"},
        )
    finally:
        server.stop()

    persisted_payload = json.loads(catalog_path.read_text(encoding="utf-8"))

    assert payload["ok"] is True
    assert payload["selected_template"] == "project_single"
    assert payload["templates"][0] == "project_single"
    assert "project_pair" not in payload["templates"]
    assert [entry["name"] for entry in persisted_payload["templates"]] == [
        "project_single"
    ]


def test_template_delete_route_selects_next_surviving_project_template(
    tmp_path: Path,
) -> None:
    catalog_path = tmp_path / ".tensor-network-editor" / "templates.json"
    server = EditorServer(
        EditorSession(
            initial_spec=build_sample_spec(),
            default_engine=EngineName.EINSUM_NUMPY,
            template_catalog_path=catalog_path,
        )
    )
    server.start()
    try:
        for template_name, tensor_ids in (
            ("project_first", ["tensor_a"]),
            ("project_middle", ["tensor_a", "tensor_b"]),
            ("project_last", ["tensor_b"]),
        ):
            request_json(
                f"{server.base_url}/api/template/promote",
                method="POST",
                payload={
                    "spec": {
                        "schema_version": SCHEMA_VERSION,
                        "network": build_sample_spec().to_dict(),
                    },
                    "tensor_ids": tensor_ids,
                    "template_name": template_name,
                },
            )
        payload = request_json(
            f"{server.base_url}/api/template/delete",
            method="POST",
            payload={"template_name": "project_middle"},
        )
    finally:
        server.stop()

    assert payload["ok"] is True
    assert payload["selected_template"] == "project_last"
    assert payload["templates"][:2] == ["project_first", "project_last"]


def test_template_delete_route_falls_back_to_first_global_template_when_project_is_empty(
    tmp_path: Path,
) -> None:
    catalog_path = tmp_path / ".tensor-network-editor" / "templates.json"
    server = EditorServer(
        EditorSession(
            initial_spec=build_sample_spec(),
            default_engine=EngineName.EINSUM_NUMPY,
            template_catalog_path=catalog_path,
        )
    )
    server.start()
    try:
        request_json(
            f"{server.base_url}/api/template/promote",
            method="POST",
            payload={
                "spec": {
                    "schema_version": SCHEMA_VERSION,
                    "network": build_sample_spec().to_dict(),
                },
                "tensor_ids": ["tensor_a", "tensor_b"],
                "template_name": "project_pair",
            },
        )
        payload = request_json(
            f"{server.base_url}/api/template/delete",
            method="POST",
            payload={"template_name": "project_pair"},
        )
    finally:
        server.stop()

    assert payload["ok"] is True
    assert payload["selected_template"] == "mps"
    assert payload["templates"][0] == "mps"


def test_template_delete_route_rejects_global_and_missing_templates(
    tmp_path: Path,
) -> None:
    server = EditorServer(
        EditorSession(
            initial_spec=build_sample_spec(),
            default_engine=EngineName.EINSUM_NUMPY,
            template_catalog_path=tmp_path
            / ".tensor-network-editor"
            / "templates.json",
        )
    )
    server.start()
    try:
        global_status, global_payload = request_json_with_status(
            f"{server.base_url}/api/template/delete",
            method="POST",
            payload={"template_name": "mps"},
        )
        missing_status, missing_payload = request_json_with_status(
            f"{server.base_url}/api/template/delete",
            method="POST",
            payload={"template_name": "missing_template"},
        )
    finally:
        server.stop()

    assert global_status == 400
    assert global_payload["ok"] is False
    assert "global" in global_payload["message"]
    assert missing_status == 400
    assert missing_payload == {
        "ok": False,
        "message": "Unknown project template 'missing_template'.",
    }


def test_template_delete_route_reports_catalog_io_errors_as_bad_request(
    editor_server: EditorServer,
) -> None:
    with patch(
        "tensor_network_editor.app.routes.delete_session_project_template",
        side_effect=PackageIOError("Could not write project template catalog JSON."),
    ):
        status, payload = request_json_with_status(
            f"{editor_server.base_url}/api/template/delete",
            method="POST",
            payload={"template_name": "project_pair"},
        )

    assert status == 400
    assert payload == {
        "ok": False,
        "message": "Could not write project template catalog JSON.",
    }


def test_template_promote_route_rejects_linear_periodic_mode(
    tmp_path: Path,
) -> None:
    server = EditorServer(
        EditorSession(
            initial_spec=build_linear_periodic_chain_spec(),
            default_engine=EngineName.EINSUM_NUMPY,
            template_catalog_path=tmp_path
            / ".tensor-network-editor"
            / "templates.json",
        )
    )
    server.start()
    try:
        status, payload = request_json_with_status(
            f"{server.base_url}/api/template/promote",
            method="POST",
            payload={
                "spec": {
                    "schema_version": SCHEMA_VERSION,
                    "network": build_linear_periodic_chain_spec().to_dict(),
                },
                "tensor_ids": ["periodic_left_tensor"],
                "template_name": "periodic_fragment",
            },
        )
    finally:
        server.stop()

    assert status == 400
    assert payload["ok"] is False
    assert "normal graph mode" in payload["message"]


def test_subnetwork_extract_route_returns_serialized_fragment(
    editor_server: EditorServer,
) -> None:
    payload = request_json(
        f"{editor_server.base_url}/api/subnetwork/extract",
        method="POST",
        payload={
            "spec": {
                "schema_version": SCHEMA_VERSION,
                "network": build_sample_spec().to_dict(),
            },
            "tensor_ids": ["tensor_a", "tensor_b"],
        },
    )

    assert payload["ok"] is True
    assert payload["spec"]["network"]["notes"] == []
    assert payload["spec"]["network"]["contraction_plan"] is None
    assert [tensor["id"] for tensor in payload["spec"]["network"]["tensors"]] == [
        "tensor_a",
        "tensor_b",
    ]
    assert [edge["id"] for edge in payload["spec"]["network"]["edges"]] == ["edge_x"]
    assert [group["id"] for group in payload["spec"]["network"]["groups"]] == [
        "group_demo"
    ]


def test_subnetwork_extract_route_rejects_invalid_selection_payload(
    editor_server: EditorServer,
) -> None:
    status, payload = request_json_with_status(
        f"{editor_server.base_url}/api/subnetwork/extract",
        method="POST",
        payload={
            "spec": {
                "schema_version": SCHEMA_VERSION,
                "network": build_sample_spec().to_dict(),
            },
            "tensor_ids": [],
        },
    )

    assert status == 400
    assert payload["ok"] is False
    assert "tensor_ids" in payload["message"]


def test_subnetwork_extract_route_rejects_linear_periodic_mode(
    editor_server: EditorServer,
) -> None:
    status, payload = request_json_with_status(
        f"{editor_server.base_url}/api/subnetwork/extract",
        method="POST",
        payload={
            "spec": {
                "schema_version": SCHEMA_VERSION,
                "network": build_linear_periodic_chain_spec().to_dict(),
            },
            "tensor_ids": ["periodic_left_tensor"],
        },
    )

    assert status == 400
    assert payload["ok"] is False
    assert "normal graph mode" in payload["message"]


def test_subnetwork_prepare_insert_route_remaps_ids_and_centers_fragment(
    editor_server: EditorServer,
) -> None:
    payload = request_json(
        f"{editor_server.base_url}/api/subnetwork/prepare-insert",
        method="POST",
        payload={
            "spec": {
                "schema_version": SCHEMA_VERSION,
                "network": build_sample_spec().to_dict(),
            },
            "target_center": {"x": 500.0, "y": 420.0},
        },
    )

    tensors = payload["spec"]["network"]["tensors"]
    tensor_ids = {tensor["id"] for tensor in tensors}
    left = min(
        tensor["position"]["x"] - tensor["size"]["width"] / 2 for tensor in tensors
    )
    right = max(
        tensor["position"]["x"] + tensor["size"]["width"] / 2 for tensor in tensors
    )
    top = min(
        tensor["position"]["y"] - tensor["size"]["height"] / 2 for tensor in tensors
    )
    bottom = max(
        tensor["position"]["y"] + tensor["size"]["height"] / 2 for tensor in tensors
    )

    assert payload["ok"] is True
    assert tensor_ids.isdisjoint({"tensor_a", "tensor_b"})
    assert payload["spec"]["network"]["notes"] == []
    assert payload["spec"]["network"]["contraction_plan"] is None
    assert (
        payload["spec"]["network"]["groups"][0]["tensor_ids"] == list(tensor_ids)
        or set(payload["spec"]["network"]["groups"][0]["tensor_ids"]) == tensor_ids
    )
    assert (left + right) / 2 == pytest.approx(500.0)
    assert (top + bottom) / 2 == pytest.approx(420.0)


def test_subnetwork_prepare_insert_route_rejects_missing_target_center(
    editor_server: EditorServer,
) -> None:
    status, payload = request_json_with_status(
        f"{editor_server.base_url}/api/subnetwork/prepare-insert",
        method="POST",
        payload={
            "spec": {
                "schema_version": SCHEMA_VERSION,
                "network": build_sample_spec().to_dict(),
            }
        },
    )

    assert status == 400
    assert payload["ok"] is False
    assert "target_center" in payload["message"]


def test_subnetwork_library_save_route_persists_project_entry(
    tmp_path: Path,
) -> None:
    catalog_path = tmp_path / ".tensor-network-editor" / "subnetworks.json"
    server = EditorServer(
        EditorSession(
            initial_spec=build_sample_spec(),
            default_engine=EngineName.EINSUM_NUMPY,
            subnetwork_catalog_path=catalog_path,
        )
    )
    server.start()
    try:
        payload = request_json(
            f"{server.base_url}/api/subnetwork-library/save",
            method="POST",
            payload={
                "spec": {
                    "schema_version": SCHEMA_VERSION,
                    "network": build_sample_spec().to_dict(),
                },
                "tensor_ids": ["tensor_a", "tensor_b"],
                "subnetwork_name": "project_pair",
                "tags": [" alpha ", "project"],
            },
        )
    finally:
        server.stop()

    assert payload["ok"] is True
    assert payload["selected_subnetwork"] == "project_pair"
    assert payload["subnetwork_definitions"]["project_pair"]["source"] == "project"
    assert payload["subnetwork_definitions"]["project_pair"]["tags"] == [
        "alpha",
        "project",
    ]


def test_subnetwork_library_prepare_insert_route_inserts_saved_entry(
    tmp_path: Path,
) -> None:
    catalog_path = tmp_path / ".tensor-network-editor" / "subnetworks.json"
    server = EditorServer(
        EditorSession(
            initial_spec=build_sample_spec(),
            default_engine=EngineName.EINSUM_NUMPY,
            subnetwork_catalog_path=catalog_path,
        )
    )
    server.start()
    try:
        request_json(
            f"{server.base_url}/api/subnetwork-library/save",
            method="POST",
            payload={
                "spec": {
                    "schema_version": SCHEMA_VERSION,
                    "network": build_sample_spec().to_dict(),
                },
                "tensor_ids": ["tensor_a", "tensor_b"],
                "subnetwork_name": "project_pair",
            },
        )
        payload = request_json(
            f"{server.base_url}/api/subnetwork-library/prepare-insert",
            method="POST",
            payload={
                "subnetwork_name": "project_pair",
                "target_center": {"x": 640.0, "y": 480.0},
            },
        )
    finally:
        server.stop()

    tensors = payload["spec"]["network"]["tensors"]
    left = min(
        tensor["position"]["x"] - tensor["size"]["width"] / 2 for tensor in tensors
    )
    right = max(
        tensor["position"]["x"] + tensor["size"]["width"] / 2 for tensor in tensors
    )
    top = min(
        tensor["position"]["y"] - tensor["size"]["height"] / 2 for tensor in tensors
    )
    bottom = max(
        tensor["position"]["y"] + tensor["size"]["height"] / 2 for tensor in tensors
    )

    assert payload["ok"] is True
    assert {tensor["id"] for tensor in tensors}.isdisjoint({"tensor_a", "tensor_b"})
    assert (left + right) / 2 == pytest.approx(640.0)
    assert (top + bottom) / 2 == pytest.approx(480.0)


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
    assert payload["memory_dtype"] == "float64"
    assert payload["network_output_shape"] == [2, 4]
    assert "automatic_full" in payload
    assert "automatic_future" in payload
    assert "automatic_past" in payload
    assert "comparisons" in payload
    assert payload["manual"]["status"] == "complete"
    assert payload["manual"]["summary"]["total_estimated_flops"] == 48
    assert payload["manual"]["summary"]["total_estimated_macs"] == 24
    assert payload["manual"]["summary"]["peak_intermediate_bytes"] == 64
    assert payload["manual"]["summary"]["final_shape"] == [2, 4]
    assert payload["manual"]["steps"][0]["estimated_flops"] == 48
    assert payload["manual"]["steps"][0]["estimated_macs"] == 24
    assert (
        payload["comparisons"]["manual_vs_automatic_full"]["memory_dtype"] == "float64"
    )
    assert (
        payload["comparisons"]["manual_remaining_vs_automatic_future"]["status"]
        == "unavailable"
    )
    assert (
        payload["comparisons"]["manual_remaining_vs_automatic_future"]["baseline_label"]
        == "manual_remaining"
    )
    assert (
        payload["comparisons"]["manual_remaining_vs_automatic_future"][
            "candidate_label"
        ]
        == "automatic_future"
    )
    assert (
        payload["comparisons"]["manual_remaining_vs_automatic_future"]["memory_dtype"]
        == "float64"
    )


def test_analyze_contraction_route_uses_active_linear_periodic_cell(
    editor_server: EditorServer,
) -> None:
    payload = request_json(
        f"{editor_server.base_url}/api/analyze-contraction",
        method="POST",
        payload={
            "spec": {
                "schema_version": SCHEMA_VERSION,
                "network": build_linear_periodic_partial_carry_chain_spec().to_dict(),
            }
        },
    )

    assert payload["ok"] is True
    assert payload["network_output_shape"] == [11, 19, 13, 29]
    assert [step["step_id"] for step in payload["manual"]["steps"]] == [
        "periodic_from_previous_partial",
        "periodic_partial_carry",
    ]
    assert payload["automatic_full"]["status"] in {"complete", "unavailable"}
    assert payload["automatic_future"]["status"] in {"complete", "unavailable"}
    assert payload["comparisons"]["manual_vs_automatic_full"]["memory_dtype"] == (
        "float64"
    )


def test_analyze_contraction_route_uses_shared_service_helper(
    editor_server: EditorServer,
    serialized_sample_spec: dict[str, object],
) -> None:
    expected_result = analyze_contraction(build_sample_spec())

    with patch(
        "tensor_network_editor.app.routes.analyze_serialized_contraction",
        return_value=expected_result,
        create=True,
    ) as analyze_mock:
        payload = request_json(
            f"{editor_server.base_url}/api/analyze-contraction",
            method="POST",
            payload={"spec": serialized_sample_spec},
        )

    assert payload["ok"] is True
    analyze_mock.assert_called_once_with(serialized_sample_spec)


def test_analyze_contraction_route_deserializes_the_spec_once(
    editor_server: EditorServer,
    serialized_sample_spec: dict[str, object],
) -> None:
    deserialize_call_count = 0

    def counting_deserialize_spec(
        payload: dict[str, object], *, validate: bool = True
    ) -> NetworkSpec:
        nonlocal deserialize_call_count
        deserialize_call_count += 1
        return deserialize_spec_impl(payload, validate=validate)

    with (
        patch(
            "tensor_network_editor.app._protocol.deserialize_spec",
            side_effect=counting_deserialize_spec,
        ),
        patch(
            "tensor_network_editor.app._services.deserialize_spec",
            side_effect=counting_deserialize_spec,
        ),
    ):
        payload = request_json(
            f"{editor_server.base_url}/api/analyze-contraction",
            method="POST",
            payload={"spec": serialized_sample_spec},
        )

    assert payload["ok"] is True
    assert deserialize_call_count == 1


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
