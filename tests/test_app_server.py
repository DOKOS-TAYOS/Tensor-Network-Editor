from __future__ import annotations

import json
import threading
import unittest
from queue import Queue
from typing import Any
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from tensor_network_editor.api import launch_tensor_network_editor
from tensor_network_editor.app.server import EditorServer
from tensor_network_editor.app.session import EditorSession, wait_for_editor_result
from tensor_network_editor.models import EngineName
from tests.test_api import build_sample_spec


def request_json(
    url: str, method: str = "GET", payload: dict[str, Any] | None = None
) -> dict[str, Any]:
    status, response = request_json_with_status(url, method=method, payload=payload)
    if status >= 400:
        raise AssertionError(f"Expected success response for {url}, received {status}.")
    return response


def request_json_with_status(
    url: str,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    raw_body: bytes | None = None,
) -> tuple[int, dict[str, Any]]:
    data = None
    headers: dict[str, str] = {}
    if payload is not None and raw_body is not None:
        raise ValueError("payload and raw_body cannot be combined.")
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    elif raw_body is not None:
        data = raw_body
        headers["Content-Type"] = "application/json"
    request = Request(url=url, method=method, data=data, headers=headers)
    try:
        with urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def request_text(url: str) -> str:
    with urlopen(url, timeout=5) as response:
        return response.read().decode("utf-8")


def request_with_headers(url: str) -> tuple[str, dict[str, str]]:
    with urlopen(url, timeout=5) as response:
        body = response.read().decode("utf-8")
        headers = {key: value for key, value in response.headers.items()}
        return body, headers


class AppServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.session = EditorSession(
            initial_spec=build_sample_spec(),
            default_engine=EngineName.EINSUM,
        )
        self.server = EditorServer(self.session)
        self.server.start()
        self.addCleanup(self.server.stop)

    def test_bootstrap_returns_initial_spec_and_engine(self) -> None:
        payload = request_json(f"{self.server.base_url}/api/bootstrap")

        self.assertEqual(payload["default_engine"], EngineName.EINSUM.value)
        self.assertEqual(payload["spec"]["network"]["id"], "network_demo")
        self.assertIn(EngineName.QUIMB.value, payload["engines"])

    def test_root_serves_editor_shell(self) -> None:
        html = request_text(f"{self.server.base_url}/")

        self.assertIn("Tensor Network Editor", html)
        self.assertIn("app.js", html)
        self.assertIn("?v=", html)
        self.assertIn('class="title-actions"', html)
        self.assertIn('class="code-header-actions"', html)
        self.assertIn('class="code-output-shell"', html)
        self.assertIn('id="undo-button"', html)
        self.assertIn('id="redo-button"', html)
        self.assertIn('id="export-png-button"', html)
        self.assertIn('id="export-svg-button"', html)
        self.assertIn('id="help-button"', html)
        self.assertIn('id="generate-button"', html)
        self.assertIn('id="canvas-selection-box"', html)
        self.assertIn('id="minimap"', html)
        self.assertIn('id="help-modal"', html)
        self.assertNotIn('id="validate-button"', html)
        self.assertNotIn('id="status-message"', html)

    def test_vendor_asset_is_served_locally(self) -> None:
        asset_body = request_text(f"{self.server.base_url}/vendor/cytoscape.min.js")

        self.assertIn("cytoscape", asset_body)

    def test_frontend_uses_dark_color_scheme(self) -> None:
        css_body = request_text(f"{self.server.base_url}/app.css")

        self.assertIn("color-scheme: dark", css_body)
        self.assertIn("height: 100dvh", css_body)
        self.assertIn("overflow: hidden", css_body)
        self.assertIn(".apply-button", css_body)
        self.assertIn("#72d98c", css_body)
        self.assertIn(".field-row", css_body)
        self.assertIn(".compact-number-field", css_body)
        self.assertIn(".control-inline-color", css_body)
        self.assertIn(".canvas-overlay", css_body)
        self.assertIn(".selection-box", css_body)
        self.assertIn(".minimap-shell", css_body)
        self.assertIn(".canvas-titlebar", css_body)
        self.assertIn(".title-actions", css_body)
        self.assertIn(".code-header-actions", css_body)
        self.assertIn(".code-output-shell", css_body)
        self.assertIn(".code-copy-floating", css_body)
        self.assertIn(".icon-button", css_body)
        self.assertIn(".help-modal", css_body)
        self.assertIn(".help-shortcuts", css_body)

    def test_frontend_script_registers_unload_cancel_and_relative_port_behaviour(self) -> None:
        script_body = request_text(f"{self.server.base_url}/app.js")

        self.assertIn("/api/cancel", script_body)
        self.assertIn("viewportCenterPosition", script_body)
        self.assertIn("suggestTensorPosition", script_body)
        self.assertNotIn("parent: tensor.id", script_body)
        self.assertIn("clampIndexOffset", script_body)
        self.assertIn("stripImportLines", script_body)
        self.assertIn("bringTensorToFront", script_body)
        self.assertIn("type=\"color\"", script_body)
        self.assertIn("kind: \"index-label\"", script_body)
        self.assertIn("metadata.color", script_body)
        self.assertIn("stripImportLines(payload.code)", script_body)
        self.assertIn("labelElement.position(indexLabelPosition(absolutePosition))", script_body)
        self.assertIn("syncIndexLabelNodePosition(located.index, absolutePosition)", script_body)
        self.assertIn("syncIndexLabelNodePosition(index, absolutePosition)", script_body)
        self.assertIn("text-halign\": \"center\"", script_body)
        self.assertIn("\"text-margin-y\": 20", script_body)
        self.assertIn(
            'name: nextName("i", tensor.indices.map((index) => index.name))',
            script_body,
        )
        self.assertIn("class=\"field-row\"", script_body)
        self.assertIn("class=\"field-group compact-number-field\"", script_body)
        self.assertIn("class=\"control-inline-color\"", script_body)
        self.assertNotIn("Tensor color", script_body)
        self.assertNotIn("Port color", script_body)
        self.assertNotIn("locked: true", script_body)
        self.assertNotIn("Move Left", script_body)
        self.assertNotIn("apply-network-button", script_body)
        self.assertIn("selectionIds", script_body)
        self.assertIn("primarySelectionId", script_body)
        self.assertIn("undoStack", script_body)
        self.assertIn("redoStack", script_body)
        self.assertIn("userPanningEnabled: true", script_body)
        self.assertIn("userZoomingEnabled: true", script_body)
        self.assertIn("commitHistorySnapshot", script_body)
        self.assertIn("performUndo", script_body)
        self.assertIn("performRedo", script_body)
        self.assertIn("renderMultiSelectionProperties", script_body)
        self.assertIn("renderMinimap", script_body)
        self.assertIn("downloadSvgExport", script_body)
        self.assertIn("downloadPngExport", script_body)
        self.assertIn("toggleHelpModal", script_body)
        self.assertIn("startBoxSelection", script_body)
        self.assertIn("contextmenu", script_body)
        self.assertIn("Ctrl+Shift+Z", script_body)
        self.assertNotIn("spacePanPressed", script_body)
        self.assertNotIn("startCanvasPan", script_body)
        self.assertNotIn("updateCanvasPan", script_body)
        self.assertNotIn("Drag tensors directly on the canvas", script_body)
        self.assertNotIn("Hold <strong>Shift</strong>", script_body)

    def test_static_assets_disable_browser_cache(self) -> None:
        _, headers = request_with_headers(f"{self.server.base_url}/app.js")

        self.assertIn("no-store", headers["Cache-Control"])

    def test_validate_route_reports_issues_for_invalid_spec(self) -> None:
        invalid_spec = build_sample_spec()
        invalid_spec.edges.append(invalid_spec.edges[0])

        payload = request_json(
            f"{self.server.base_url}/api/validate",
            method="POST",
            payload={"spec": {"schema_version": 1, "network": invalid_spec.to_dict()}},
        )

        self.assertFalse(payload["ok"])
        self.assertIn(
            "index-already-connected", [issue["code"] for issue in payload["issues"]]
        )

    def test_validate_route_rejects_invalid_json_with_400(self) -> None:
        status, payload = request_json_with_status(
            f"{self.server.base_url}/api/validate",
            method="POST",
            raw_body=b"{not-json}",
        )

        self.assertEqual(status, 400)
        self.assertFalse(payload["ok"])
        self.assertIn("invalid JSON", payload["message"])

    def test_validate_route_rejects_non_object_json_payload_with_400(self) -> None:
        status, payload = request_json_with_status(
            f"{self.server.base_url}/api/validate",
            method="POST",
            raw_body=json.dumps(["not", "an", "object"]).encode("utf-8"),
        )

        self.assertEqual(status, 400)
        self.assertFalse(payload["ok"])
        self.assertIn("JSON object", payload["message"])

    def test_generate_route_returns_code(self) -> None:
        payload = request_json(
            f"{self.server.base_url}/api/generate",
            method="POST",
            payload={
                "engine": EngineName.TENSORNETWORK.value,
                "spec": {"schema_version": 1, "network": build_sample_spec().to_dict()},
            },
        )

        self.assertEqual(payload["engine"], EngineName.TENSORNETWORK.value)
        self.assertIn("import tensornetwork as tn", payload["code"])

    def test_generate_route_rejects_missing_spec_with_400(self) -> None:
        status, payload = request_json_with_status(
            f"{self.server.base_url}/api/generate",
            method="POST",
            payload={"engine": EngineName.TENSORNETWORK.value},
        )

        self.assertEqual(status, 400)
        self.assertFalse(payload["ok"])
        self.assertIn("Missing 'spec' payload.", payload["message"])

    def test_generate_route_rejects_unsupported_engine_with_400(self) -> None:
        status, payload = request_json_with_status(
            f"{self.server.base_url}/api/generate",
            method="POST",
            payload={
                "engine": "unknown-engine",
                "spec": {"schema_version": 1, "network": build_sample_spec().to_dict()},
            },
        )

        self.assertEqual(status, 400)
        self.assertFalse(payload["ok"])
        self.assertIn("Unsupported engine", payload["message"])

    def test_generate_route_returns_validation_issues_for_invalid_spec(self) -> None:
        invalid_spec = build_sample_spec()
        invalid_spec.edges.append(invalid_spec.edges[0])

        payload = request_json(
            f"{self.server.base_url}/api/generate",
            method="POST",
            payload={
                "engine": EngineName.TENSORNETWORK.value,
                "spec": {"schema_version": 1, "network": invalid_spec.to_dict()},
            },
        )

        self.assertFalse(payload["ok"])
        self.assertIn(
            "index-already-connected", [issue["code"] for issue in payload["issues"]]
        )

    def test_unexpected_server_errors_return_generic_500_payload(self) -> None:
        with patch(
            "tensor_network_editor.app.server.routes.handle_generate",
            side_effect=RuntimeError("boom"),
        ):
            status, payload = request_json_with_status(
                f"{self.server.base_url}/api/generate",
                method="POST",
                payload={
                    "engine": EngineName.TENSORNETWORK.value,
                    "spec": {
                        "schema_version": 1,
                        "network": build_sample_spec().to_dict(),
                    },
                },
            )

        self.assertEqual(status, 500)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["message"], "Internal server error.")

    def test_complete_route_stores_result_in_session(self) -> None:
        payload = request_json(
            f"{self.server.base_url}/api/complete",
            method="POST",
            payload={
                "engine": EngineName.QUIMB.value,
                "spec": {"schema_version": 1, "network": build_sample_spec().to_dict()},
            },
        )

        self.assertTrue(payload["ok"])
        result = self.session.wait_for_result(timeout=0.1)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.engine, EngineName.QUIMB)
        self.assertIsNotNone(result.codegen)

    def test_cancel_route_ends_session_without_result(self) -> None:
        payload = request_json(
            f"{self.server.base_url}/api/cancel",
            method="POST",
            payload={},
        )

        self.assertTrue(payload["ok"])
        self.assertIsNone(self.session.wait_for_result(timeout=0.1))


class LaunchEditorSessionTests(unittest.TestCase):
    def test_wait_for_editor_result_delegates_to_session_without_private_event_access(
        self,
    ) -> None:
        class FakeSession:
            def __init__(self) -> None:
                self.calls: list[float | None] = []

            def wait_for_result(self, timeout: float | None = None) -> object | None:
                self.calls.append(timeout)
                return None

        session = FakeSession()

        result = wait_for_editor_result(session, poll_interval=0.05)

        self.assertIsNone(result)
        self.assertEqual(session.calls, [None])

    def test_wait_for_editor_result_polls_until_a_result_is_available(self) -> None:
        session = EditorSession(initial_spec=build_sample_spec(), default_engine=EngineName.EINSUM)

        def finish_session() -> None:
            session.complete(
                serialized_spec={"schema_version": 1, "network": build_sample_spec().to_dict()},
                engine=EngineName.EINSUM,
            )

        timer = threading.Timer(0.2, finish_session)
        timer.start()
        self.addCleanup(timer.cancel)

        result = wait_for_editor_result(session, poll_interval=0.05)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.engine, EngineName.EINSUM)

    def test_launch_tensor_network_editor_waits_for_complete(self) -> None:
        ready_queue: Queue[str] = Queue()
        result_queue: Queue[object] = Queue()

        def run_editor() -> None:
            result = launch_tensor_network_editor(
                initial_spec=build_sample_spec(),
                default_engine=EngineName.EINSUM,
                open_browser=False,
                _on_server_ready=ready_queue.put,
            )
            result_queue.put(result)

        thread = threading.Thread(target=run_editor, daemon=True)
        thread.start()

        base_url = ready_queue.get(timeout=5)
        payload = request_json(
            f"{base_url}/api/complete",
            method="POST",
            payload={
                "engine": EngineName.EINSUM.value,
                "spec": {"schema_version": 1, "network": build_sample_spec().to_dict()},
            },
        )
        self.assertTrue(payload["ok"])

        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        result = result_queue.get(timeout=1)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.engine, EngineName.EINSUM)

    def test_launch_tensor_network_editor_returns_none_on_cancel(self) -> None:
        ready_queue: Queue[str] = Queue()
        result_queue: Queue[object] = Queue()

        def run_editor() -> None:
            result = launch_tensor_network_editor(
                initial_spec=build_sample_spec(),
                default_engine=EngineName.EINSUM,
                open_browser=False,
                _on_server_ready=ready_queue.put,
            )
            result_queue.put(result)

        thread = threading.Thread(target=run_editor, daemon=True)
        thread.start()

        base_url = ready_queue.get(timeout=5)
        payload = request_json(
            f"{base_url}/api/cancel",
            method="POST",
            payload={},
        )
        self.assertTrue(payload["ok"])

        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertIsNone(result_queue.get(timeout=1))


if __name__ == "__main__":
    unittest.main()
