from __future__ import annotations

import base64
import json
import logging
from collections.abc import Callable
from http.client import HTTPConnection
from pathlib import Path
from typing import cast
from unittest.mock import patch
from urllib.parse import urlparse

import pytest

from tensor_network_editor import generate_code
from tensor_network_editor.analysis import analyze_contraction
from tensor_network_editor.app._protocol import JsonDict
from tensor_network_editor.app.routes import handle_bootstrap
from tensor_network_editor.app.server import EditorServer
from tensor_network_editor.app.session import EditorSession
from tensor_network_editor.errors import PackageIOError, SerializationError
from tensor_network_editor.internal._logging import package_logging_scope
from tensor_network_editor.io import SCHEMA_VERSION
from tensor_network_editor.io import deserialize_spec as deserialize_spec_impl
from tensor_network_editor.models import EngineName, NetworkSpec, TensorCollectionFormat
from tests.app_support import request_json, request_json_with_status
from tests.factories import (
    build_linear_periodic_carry_chain_spec,
    build_linear_periodic_chain_spec,
    build_linear_periodic_partial_carry_chain_spec,
    build_outer_product_plan_spec,
    build_sample_spec,
    build_sample_spec_with_view_snapshots,
    build_three_tensor_hyperedge_spec,
)
from tests.optional_backends import require_light_optional_modules


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
    assert payload["frontend_logging"] == {
        "enabled": False,
        "level": "off",
        "persist": False,
        "transport_endpoint": "/api/client-log",
    }
    assert payload["app_metadata"] == {
        "repository_url": "https://github.com/DOKOS-TAYOS/Tensor-Network-Editor",
        "version": "0.4.0",
        "license_name": "MIT",
        "author_name": "Alejandro Mata Ali",
    }


def test_bootstrap_route_logs_success_with_session_and_timing(
    editor_server: EditorServer,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.DEBUG, logger="tensor_network_editor"):
        payload = request_json(f"{editor_server.base_url}/api/bootstrap")

    assert payload["default_engine"] == EngineName.EINSUM_NUMPY.value
    assert "Bootstrap route started" in caplog.text
    assert "Bootstrap route finished" in caplog.text
    assert f"session={editor_server.session_id}" in caplog.text
    assert "route=/api/bootstrap" in caplog.text
    assert "outcome=success" in caplog.text
    assert "elapsed_ms=" in caplog.text


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


def test_draft_routes_round_trip_project_local_draft(tmp_path: Path) -> None:
    session = EditorSession(
        initial_spec=build_sample_spec(),
        default_engine=EngineName.EINSUM_NUMPY,
        draft_path=tmp_path / "drafts" / "active.json",
    )
    server = EditorServer(session)
    server.start()
    try:
        spec = build_sample_spec()
        saved_payload = request_json(
            f"{server.base_url}/api/draft",
            method="POST",
            payload={
                "spec": {
                    "schema_version": SCHEMA_VERSION,
                    "network": spec.to_dict(),
                },
                "engine": EngineName.EINSUM_TORCH.value,
                "collection_format": TensorCollectionFormat.DICT.value,
            },
        )
        loaded_payload = request_json(f"{server.base_url}/api/draft")
        cleared_payload = request_json(
            f"{server.base_url}/api/draft/clear",
            method="POST",
            payload={},
        )
        empty_payload = request_json(f"{server.base_url}/api/draft")
    finally:
        server.stop()

    assert saved_payload["ok"] is True
    assert isinstance(saved_payload["draft"]["saved_at"], str)
    assert loaded_payload["draft"]["spec"]["network"]["id"] == spec.id
    assert loaded_payload["draft"]["engine"] == EngineName.EINSUM_TORCH.value
    assert loaded_payload["draft"]["collection_format"] == (
        TensorCollectionFormat.DICT.value
    )
    assert cleared_payload["ok"] is True
    assert empty_payload["draft"] is None


def test_draft_routes_log_lifecycle_and_persistence_context(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    draft_path = tmp_path / "drafts" / "active.json"
    session = EditorSession(
        initial_spec=build_sample_spec(),
        default_engine=EngineName.EINSUM_NUMPY,
        draft_path=draft_path,
    )
    server = EditorServer(session)
    server.start()
    try:
        spec = build_sample_spec()
        with caplog.at_level(logging.DEBUG, logger="tensor_network_editor"):
            saved_payload = request_json(
                f"{server.base_url}/api/draft",
                method="POST",
                payload={
                    "spec": {
                        "schema_version": SCHEMA_VERSION,
                        "network": spec.to_dict(),
                    },
                    "engine": EngineName.EINSUM_TORCH.value,
                    "collection_format": TensorCollectionFormat.DICT.value,
                },
            )
            loaded_payload = request_json(f"{server.base_url}/api/draft")
            cleared_payload = request_json(
                f"{server.base_url}/api/draft/clear",
                method="POST",
                payload={},
            )
            empty_payload = request_json(f"{server.base_url}/api/draft")
    finally:
        server.stop()

    assert saved_payload["ok"] is True
    assert loaded_payload["draft"]["engine"] == EngineName.EINSUM_TORCH.value
    assert cleared_payload["ok"] is True
    assert empty_payload["draft"] is None
    assert "Draft save route started" in caplog.text
    assert "Draft load route started" in caplog.text
    assert "Draft clear route started" in caplog.text
    assert "Draft persistence finished" in caplog.text
    assert "No project draft found on disk" in caplog.text
    assert f"path={draft_path}" in caplog.text
    assert f"session={session.session_id}" in caplog.text


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


def test_render_route_returns_academic_text_exports(
    editor_server: EditorServer,
) -> None:
    spec = build_sample_spec()
    serialized_spec = {
        "schema_version": SCHEMA_VERSION,
        "network": spec.to_dict(),
    }

    tikz_payload = request_json(
        f"{editor_server.base_url}/api/render",
        method="POST",
        payload={"format": "tikz", "spec": serialized_spec},
    )
    dot_payload = request_json(
        f"{editor_server.base_url}/api/render",
        method="POST",
        payload={"format": "dot", "spec": serialized_spec},
    )

    assert tikz_payload["format"] == "tikz"
    assert tikz_payload["content_type"] == "text/x-tex;charset=utf-8"
    assert tikz_payload["text"].startswith(r"\def\tneGlobalWidth")
    assert r"\begin{tikzpicture}" in tikz_payload["text"]
    assert r"\node[tne index]" not in tikz_payload["text"]
    assert r"\draw[tne edge]" in tikz_payload["text"]
    assert dot_payload["format"] == "dot"
    assert dot_payload["content_type"] == "text/vnd.graphviz;charset=utf-8"
    assert dot_payload["text"].startswith('graph "demo" {')
    assert '"tensor_a" -- "tensor_b" [label="bond_x / x=3"]' in dot_payload["text"]


def test_render_route_returns_mermaid_export(
    editor_server: EditorServer,
) -> None:
    spec = build_sample_spec()
    serialized_spec = {
        "schema_version": SCHEMA_VERSION,
        "network": spec.to_dict(),
    }

    mermaid_payload = request_json(
        f"{editor_server.base_url}/api/render",
        method="POST",
        payload={"format": "mermaid", "spec": serialized_spec},
    )

    assert mermaid_payload["format"] == "mermaid"
    assert mermaid_payload["content_type"] == "text/plain;charset=utf-8"
    assert mermaid_payload["text"].startswith("flowchart LR\n")
    assert 'tensor_tensor_a["A"]' in mermaid_payload["text"]


def test_render_route_applies_academic_label_options(
    editor_server: EditorServer,
) -> None:
    spec = build_sample_spec()
    serialized_spec = {
        "schema_version": SCHEMA_VERSION,
        "network": spec.to_dict(),
    }

    tikz_payload = request_json(
        f"{editor_server.base_url}/api/render",
        method="POST",
        payload={
            "format": "tikz",
            "spec": serialized_spec,
            "show_tensor_names": False,
            "show_index_names": False,
            "show_bond_names": False,
        },
    )
    dot_payload = request_json(
        f"{editor_server.base_url}/api/render",
        method="POST",
        payload={
            "format": "dot",
            "spec": serialized_spec,
            "show_tensor_names": False,
            "show_index_names": False,
            "show_bond_names": False,
        },
    )

    assert r"{A}" not in tikz_payload["text"]
    assert r"\node[tne index label]" not in tikz_payload["text"]
    assert r"bond\_x" not in tikz_payload["text"]
    assert '"tensor_a" [label="", shape="circle"]' in dot_payload["text"]
    assert '"open_tensor_a_i" [label="", shape="circle"]' in dot_payload["text"]
    assert '"tensor_a" -- "tensor_b";' in dot_payload["text"]
    assert "bond_x" not in dot_payload["text"]
    assert "x=3" not in dot_payload["text"]


def test_render_route_applies_mermaid_label_options(
    editor_server: EditorServer,
) -> None:
    spec = build_sample_spec()
    serialized_spec = {
        "schema_version": SCHEMA_VERSION,
        "network": spec.to_dict(),
    }

    mermaid_payload = request_json(
        f"{editor_server.base_url}/api/render",
        method="POST",
        payload={
            "format": "mermaid",
            "spec": serialized_spec,
            "show_tensor_names": False,
            "show_index_names": False,
            "show_bond_names": False,
        },
    )

    assert 'tensor_tensor_a["tensor_a"]' in mermaid_payload["text"]
    assert 'open_tensor_a_i["i (2)"]' not in mermaid_payload["text"]
    assert "bond_x" not in mermaid_payload["text"]
    assert "x=3" not in mermaid_payload["text"]


def test_render_route_returns_svg_png_and_pdf_exports(
    editor_server: EditorServer,
) -> None:
    pytest.importorskip("matplotlib")
    spec = build_sample_spec()
    serialized_spec = {
        "schema_version": SCHEMA_VERSION,
        "network": spec.to_dict(),
    }

    svg_payload = request_json(
        f"{editor_server.base_url}/api/render",
        method="POST",
        payload={"format": "svg", "spec": serialized_spec},
    )
    png_payload = request_json(
        f"{editor_server.base_url}/api/render",
        method="POST",
        payload={"format": "png", "spec": serialized_spec},
    )
    pdf_payload = request_json(
        f"{editor_server.base_url}/api/render",
        method="POST",
        payload={"format": "pdf", "spec": serialized_spec},
    )

    assert svg_payload["format"] == "svg"
    assert svg_payload["content_type"] == "image/svg+xml;charset=utf-8"
    assert svg_payload["text"].startswith('<?xml version="1.0" encoding="UTF-8"?>')
    assert png_payload["format"] == "png"
    assert png_payload["content_type"] == "image/png"
    assert base64.b64decode(png_payload["base64"]).startswith(b"\x89PNG\r\n\x1a\n")
    assert pdf_payload["format"] == "pdf"
    assert pdf_payload["content_type"] == "application/pdf"
    assert base64.b64decode(pdf_payload["base64"]).startswith(b"%PDF")


def test_render_route_applies_label_options_to_svg_png_and_pdf(
    editor_server: EditorServer,
) -> None:
    spec = build_sample_spec()
    serialized_spec = {
        "schema_version": SCHEMA_VERSION,
        "network": spec.to_dict(),
    }

    with (
        patch(
            "tensor_network_editor.app.routes.render_spec_svg",
            return_value="<?xml version='1.0'?><svg />",
        ) as render_svg_mock,
        patch(
            "tensor_network_editor.app.routes.render_spec_png",
            return_value=b"\x89PNG\r\n\x1a\n",
        ) as render_png_mock,
        patch(
            "tensor_network_editor.app.routes.render_spec_pdf",
            return_value=b"%PDF-1.4",
        ) as render_pdf_mock,
    ):
        for render_format in ("svg", "png", "pdf"):
            payload = request_json(
                f"{editor_server.base_url}/api/render",
                method="POST",
                payload={
                    "format": render_format,
                    "spec": serialized_spec,
                    "show_tensor_names": False,
                    "show_index_names": False,
                    "show_bond_names": False,
                },
            )
            assert payload["ok"] is True

    for render_mock in (render_svg_mock, render_png_mock, render_pdf_mock):
        options = render_mock.call_args.kwargs["options"]
        assert options.show_tensor_labels is False
        assert options.show_index_labels is False
        assert options.show_edge_labels is False


def test_render_route_rejects_unsupported_academic_format(
    editor_server: EditorServer,
) -> None:
    spec = build_sample_spec()

    status, payload = request_json_with_status(
        f"{editor_server.base_url}/api/render",
        method="POST",
        payload={
            "format": "json",
            "spec": {
                "schema_version": SCHEMA_VERSION,
                "network": spec.to_dict(),
            },
        },
    )

    assert status == 400
    assert payload["ok"] is False
    assert "Unsupported render format" in payload["message"]


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


@pytest.mark.optional_backend
def test_validate_route_returns_live_import_warnings(
    editor_server: EditorServer,
) -> None:
    require_light_optional_modules(("numpy", "quimb"))
    code = "\n".join(
        [
            "import numpy as np",
            "import quimb.tensor as qtn",
            "",
            "def build_network() -> qtn.TensorNetwork:",
            "    left = qtn.Tensor(np.arange(65 * 65, dtype=float).reshape(65, 65), inds=('i', 'bond_x'), tags=('A',))",
            "    right = qtn.Tensor(np.ones((65, 5), dtype=float), inds=('bond_x', 'j'), tags=('B',))",
            "    return qtn.TensorNetwork([left, right])",
            "",
            "network = build_network()",
        ]
    )

    payload = request_json(
        f"{editor_server.base_url}/api/validate",
        method="POST",
        payload={
            "python_code": code,
            "python_import_mode": "live",
            "source_profile": "quimb",
        },
    )

    assert payload["ok"] is True
    assert payload["issues"] == []
    assert payload["warnings"]
    assert "tensor data" in payload["warnings"][0].lower()
    assert payload["spec"]["network"]["tensors"][0]["tensor_data"] is None


def test_validate_route_falls_back_to_static_parser_for_generated_python_when_live_import_fails(
    editor_server: EditorServer,
) -> None:
    generated = generate_code(
        build_sample_spec(),
        engine=EngineName.TENSORKROWCH,
    )

    with patch(
        "tensor_network_editor.internal.io._serialization.import_live_python_source",
        side_effect=SerializationError("No module named 'torch'"),
    ):
        payload = request_json(
            f"{editor_server.base_url}/api/validate",
            method="POST",
            payload={
                "python_code": generated.code,
                "python_import_mode": "live",
            },
        )

    assert payload["ok"] is True
    assert payload["issues"] == []
    assert payload["warnings"]
    assert "static parser" in payload["warnings"][0].lower()
    assert "no module named 'torch'" in payload["warnings"][0].lower()
    assert [tensor["name"] for tensor in payload["spec"]["network"]["tensors"]] == [
        "A",
        "B",
    ]


def test_validate_route_accepts_linear_periodic_generated_python_with_marker(
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

    assert status == 200
    assert payload["ok"] is True
    assert (
        payload["spec"]["network"]["linear_periodic_chain"]["active_cell"] == "periodic"
    )


def test_validate_route_rejects_legacy_linear_periodic_generated_python_with_clear_message(
    editor_server: EditorServer,
) -> None:
    generated = generate_code(
        build_linear_periodic_chain_spec(),
        engine=EngineName.TENSORNETWORK,
    )
    markerless_code = "\n".join(
        line
        for line in generated.code.splitlines()
        if not line.startswith("# TNE_SPEC_B64:")
    )

    status, payload = request_json_with_status(
        f"{editor_server.base_url}/api/validate",
        method="POST",
        payload={"python_code": markerless_code},
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


def test_generate_route_logs_success_with_session_and_timing(
    editor_server: EditorServer,
    serialized_sample_spec: dict[str, object],
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.DEBUG, logger="tensor_network_editor"):
        payload = request_json(
            f"{editor_server.base_url}/api/generate",
            method="POST",
            payload={
                "engine": EngineName.TENSORNETWORK.value,
                "spec": serialized_sample_spec,
            },
        )

    assert payload["ok"] is True
    assert "Route request started" in caplog.text
    assert "Route request finished" in caplog.text
    assert f"session={editor_server.session_id}" in caplog.text
    assert "route=/api/generate" in caplog.text
    assert "outcome=success" in caplog.text
    assert "elapsed_ms=" in caplog.text


def test_invalid_json_request_logs_warning_with_route_context(
    editor_server: EditorServer,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.DEBUG, logger="tensor_network_editor"):
        status, payload = request_json_with_status(
            f"{editor_server.base_url}/api/validate",
            method="POST",
            raw_body=b"{not-json}",
        )

    assert status == 400
    assert payload == {"ok": False, "message": "Request body contains invalid JSON."}
    assert "Rejected malformed JSON request" in caplog.text
    assert f"session={editor_server.session_id}" in caplog.text
    assert "route=/api/validate" in caplog.text


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


@pytest.mark.parametrize("legacy_schema_version", [4, 5, 6])
def test_validate_route_rejects_legacy_schema_versions(
    editor_server: EditorServer,
    legacy_schema_version: int,
) -> None:
    status, payload = request_json_with_status(
        f"{editor_server.base_url}/api/validate",
        method="POST",
        payload={
            "spec": {
                "schema_version": legacy_schema_version,
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


def test_cancel_route_logs_success_with_session_context(
    editor_server: EditorServer,
    editor_session: EditorSession,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.DEBUG, logger="tensor_network_editor"):
        payload = request_json(
            f"{editor_server.base_url}/api/cancel",
            method="POST",
            payload={},
        )

    assert payload == {"ok": True}
    assert editor_session.wait_for_result(timeout=0.1) is None
    assert "Cancel route started" in caplog.text
    assert "Cancel route finished" in caplog.text
    assert "Session cancel started" in caplog.text
    assert f"session={editor_session.session_id}" in caplog.text
    assert "route=/api/cancel" in caplog.text
    assert "outcome=success" in caplog.text


def test_client_log_route_accepts_one_valid_batch(
    editor_server: EditorServer,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.DEBUG, logger="tensor_network_editor"):
        payload = request_json(
            f"{editor_server.base_url}/api/client-log",
            method="POST",
            payload={
                "events": [
                    {
                        "level": "debug",
                        "message": "Bootstrap finished",
                        "context": {
                            "session": editor_server.session_id,
                            "route": "/api/bootstrap",
                            "request_id": "req-1",
                            "outcome": "success",
                            "elapsed_ms": "25",
                            "client_ts_ms": "12345",
                        },
                    }
                ]
            },
        )

    assert payload["ok"] is True
    assert "Frontend client log route started" in caplog.text
    assert "Frontend client log route finished" in caplog.text
    assert "event_count=1" in caplog.text
    assert "Bootstrap finished" in caplog.text
    assert "request_id=req-1" in caplog.text
    assert "client_ts_ms=12345" in caplog.text


def test_client_log_route_rejects_malformed_payload(
    editor_server: EditorServer,
) -> None:
    status, payload = request_json_with_status(
        f"{editor_server.base_url}/api/client-log",
        method="POST",
        payload={
            "events": [
                {
                    "level": "debug",
                    "context": {},
                }
            ]
        },
    )

    assert status == 400
    assert payload["ok"] is False
    assert "message" in payload


def test_client_log_route_persists_to_rotating_log_file(
    editor_server: EditorServer,
    tmp_path: Path,
) -> None:
    log_file_path = tmp_path / "client-log-route.log"
    payload: JsonDict | None = None

    with package_logging_scope(
        "debug",
        log_file_path=log_file_path,
        enable_stderr=False,
        log_file_max_bytes=1024,
        log_file_backup_count=3,
    ):
        for index in range(5):
            payload = request_json(
                f"{editor_server.base_url}/api/client-log",
                method="POST",
                payload={
                    "events": [
                        {
                            "level": "warning",
                            "message": f"Planner refresh resolved {'x' * 220} {index}",
                            "context": {
                                "session": editor_server.session_id,
                                "route": "/api/analyze-contraction",
                                "request_id": f"req-{index}",
                                "outcome": "success",
                                "elapsed_ms": "25",
                                "client_ts_ms": str(10_000 + index),
                            },
                        }
                    ]
                },
            )

    assert payload is not None
    combined_log_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(tmp_path.glob("client-log-route.log*"))
        if path.is_file()
    )

    assert payload["ok"] is True
    assert log_file_path.exists()
    assert (tmp_path / "client-log-route.log.1").exists()
    assert "Frontend client log route finished" in combined_log_text
    assert "event_count=1" in combined_log_text
    assert "Planner refresh resolved" in combined_log_text


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


def test_template_route_applies_requested_mps_model_parameters(
    editor_server: EditorServer,
) -> None:
    payload = request_json(
        f"{editor_server.base_url}/api/template",
        method="POST",
        payload={
            "template": "mps",
            "parameters": {
                "graph_size": 4,
                "bond_dimension": 3,
                "physical_dimension": 2,
                "boundary_condition": "periodic",
                "symmetry": "z2",
                "initial_state": "neel",
            },
        },
    )

    assert payload["ok"] is True
    assert payload["spec"]["network"]["metadata"]["boundary_condition"] == "periodic"
    assert payload["spec"]["network"]["metadata"]["symmetry"] == "z2"
    assert payload["spec"]["network"]["metadata"]["initial_state"] == "neel"
    assert len(payload["spec"]["network"]["edges"]) == 4


def test_template_route_applies_requested_mpo_and_ttn_parameters(
    editor_server: EditorServer,
) -> None:
    mpo_payload = request_json(
        f"{editor_server.base_url}/api/template",
        method="POST",
        payload={
            "template": "mpo",
            "parameters": {
                "graph_size": 4,
                "bond_dimension": 3,
                "physical_dimension": 2,
                "boundary_condition": "periodic",
                "j": 1.5,
                "h": 0.25,
            },
        },
    )
    ttn_payload = request_json(
        f"{editor_server.base_url}/api/template",
        method="POST",
        payload={
            "template": "ttn",
            "parameters": {
                "depth": 4,
                "bond_dimension": 3,
                "physical_dimension": 2,
                "leaf_physical_legs": False,
                "root_open_leg": True,
                "isometric": True,
            },
        },
    )

    assert mpo_payload["ok"] is True
    assert (
        mpo_payload["spec"]["network"]["metadata"]["boundary_condition"] == "periodic"
    )
    assert mpo_payload["spec"]["network"]["metadata"]["j"] == 1.5
    assert mpo_payload["spec"]["network"]["metadata"]["h"] == 0.25
    assert len(mpo_payload["spec"]["network"]["edges"]) == 4
    assert ttn_payload["ok"] is True
    assert ttn_payload["spec"]["network"]["metadata"]["depth"] == 4
    assert ttn_payload["spec"]["network"]["metadata"]["leaf_physical_legs"] is False
    assert ttn_payload["spec"]["network"]["metadata"]["root_open_leg"] is True
    assert ttn_payload["spec"]["network"]["metadata"]["isometric"] is True


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


def test_template_route_rejects_spin_presets_for_non_spin_dimension(
    editor_server: EditorServer,
) -> None:
    status, payload = request_json_with_status(
        f"{editor_server.base_url}/api/template",
        method="POST",
        payload={
            "template": "mps",
            "parameters": {
                "graph_size": 4,
                "bond_dimension": 3,
                "physical_dimension": 3,
                "initial_state": "all_up",
            },
        },
    )

    assert status == 400
    assert payload["ok"] is False
    assert "physical_dimension" in payload["message"]


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


def test_template_promote_route_logs_catalog_trace(
    tmp_path: Path,
    serialized_sample_spec: JsonDict,
    caplog: pytest.LogCaptureFixture,
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
        with caplog.at_level(logging.DEBUG, logger="tensor_network_editor"):
            payload = request_json(
                f"{server.base_url}/api/template/promote",
                method="POST",
                payload={
                    "spec": serialized_sample_spec,
                    "tensor_ids": ["tensor_a", "tensor_b"],
                    "template_name": "project_pair",
                },
            )
    finally:
        server.stop()

    assert payload["selected_template"] == "project_pair"
    assert "Template promote route started" in caplog.text
    assert "Template promotion started" in caplog.text
    assert "Project template catalog save finished" in caplog.text
    assert "template_name=project_pair" in caplog.text
    assert f"path={catalog_path}" in caplog.text
    assert "outcome=success" in caplog.text


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


def test_subnetwork_library_save_route_logs_catalog_trace(
    tmp_path: Path,
    serialized_sample_spec: JsonDict,
    caplog: pytest.LogCaptureFixture,
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
        with caplog.at_level(logging.DEBUG, logger="tensor_network_editor"):
            payload = request_json(
                f"{server.base_url}/api/subnetwork-library/save",
                method="POST",
                payload={
                    "spec": serialized_sample_spec,
                    "tensor_ids": ["tensor_a", "tensor_b"],
                    "subnetwork_name": "project_pair",
                    "tags": [" alpha ", "project"],
                },
            )
    finally:
        server.stop()

    assert payload["selected_subnetwork"] == "project_pair"
    assert "Subnetwork library save route started" in caplog.text
    assert "Reusable subnetwork save started" in caplog.text
    assert "Reusable subnetwork catalog save finished" in caplog.text
    assert "subnetwork_name=project_pair" in caplog.text
    assert f"path={catalog_path}" in caplog.text
    assert "outcome=success" in caplog.text


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


def test_analyze_contraction_route_logs_semantic_summary(
    editor_server: EditorServer,
    serialized_sample_spec: dict[str, object],
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.DEBUG, logger="tensor_network_editor"):
        payload = request_json(
            f"{editor_server.base_url}/api/analyze-contraction",
            method="POST",
            payload={"spec": serialized_sample_spec},
        )

    assert payload["ok"] is True
    assert "Analyze contraction route started" in caplog.text
    assert "Serialized contraction analysis finished" in caplog.text
    assert "Analyze contraction route finished" in caplog.text
    assert "analysis_status=ready" in caplog.text
    assert "warning_count=0" in caplog.text
    assert "manual_step_count=1" in caplog.text
    assert "automatic_full_status=complete" in caplog.text
    assert "automatic_future_status=complete" in caplog.text
    assert "automatic_past_status=complete" in caplog.text


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


def test_analyze_contraction_route_logs_periodic_analysis_selection(
    editor_server: EditorServer,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.DEBUG, logger="tensor_network_editor"):
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
    assert "Using active linear periodic cell" in caplog.text
    assert "mode=linear_periodic" in caplog.text
    assert "analysis_status=ready" in caplog.text


def test_analyze_contraction_route_accepts_hyperedges(
    editor_server: EditorServer,
) -> None:
    status, payload = request_json_with_status(
        f"{editor_server.base_url}/api/analyze-contraction",
        method="POST",
        payload={
            "spec": {
                "schema_version": SCHEMA_VERSION,
                "network": build_three_tensor_hyperedge_spec().to_dict(),
            }
        },
    )

    assert status == 200
    assert payload["ok"] is True
    assert payload["warnings"] == [
        "Hyperedges are analyzed as generated copy tensors; the visual model is unchanged."
    ]
    assert (
        payload["synthetic_operands"][0]["operand_id"] == "hyperedge_copy_hyperedge_h"
    )


def test_analyze_contraction_route_logs_hyperedge_lowering_warning_count(
    editor_server: EditorServer,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.DEBUG, logger="tensor_network_editor"):
        payload = request_json(
            f"{editor_server.base_url}/api/analyze-contraction",
            method="POST",
            payload={
                "spec": {
                    "schema_version": SCHEMA_VERSION,
                    "network": build_three_tensor_hyperedge_spec().to_dict(),
                }
            },
        )

    assert payload["ok"] is True
    assert "Lowering hyperedges to pairwise analysis spec" in caplog.text
    assert "analysis_status=ready" in caplog.text
    assert "warning_count=1" in caplog.text


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


def test_analyze_contraction_route_logs_validation_issues(
    editor_server: EditorServer,
    caplog: pytest.LogCaptureFixture,
) -> None:
    invalid_spec = build_sample_spec()
    invalid_spec.edges.append(invalid_spec.edges[0])

    with caplog.at_level(logging.DEBUG, logger="tensor_network_editor"):
        payload = request_json(
            f"{editor_server.base_url}/api/analyze-contraction",
            method="POST",
            payload={
                "spec": {
                    "schema_version": SCHEMA_VERSION,
                    "network": invalid_spec.to_dict(),
                }
            },
        )

    assert payload["ok"] is False
    assert "Contraction-analysis payload failed validation" in caplog.text
    assert "analysis_status=issues" in caplog.text
    assert "issue_count=" in caplog.text


def test_unexpected_server_errors_return_enriched_500_payload(
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
    assert payload == {
        "ok": False,
        "message": "Unexpected internal error.",
        "guidance": (
            "Try again. If the problem continues, check the terminal output for "
            "this session or rerun with debug logging."
        ),
        "reference": editor_server.session_id,
    }
