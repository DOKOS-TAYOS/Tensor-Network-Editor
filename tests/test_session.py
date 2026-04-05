from __future__ import annotations

import threading
import unittest
from queue import Queue
from typing import cast

from tensor_network_editor.api import launch_tensor_network_editor
from tensor_network_editor.app.session import EditorSession, wait_for_editor_result
from tensor_network_editor.models import EditorResult, EngineName
from tests.app_support import request_json
from tests.test_api import build_sample_spec


class SessionTests(unittest.TestCase):
    def test_bootstrap_payload_includes_template_parameter_definitions(self) -> None:
        session = EditorSession(
            initial_spec=build_sample_spec(),
            default_engine=EngineName.EINSUM_NUMPY,
        )

        payload = session.bootstrap_payload()
        template_definitions = cast(dict[str, object], payload["template_definitions"])
        mps_definition = cast(dict[str, object], template_definitions["mps"])
        binary_tree_definition = cast(
            dict[str, object], template_definitions["binary_tree"]
        )
        binary_tree_defaults = cast(
            dict[str, object], binary_tree_definition["defaults"]
        )

        self.assertIn("template_definitions", payload)
        self.assertEqual(mps_definition["graph_size_label"], "Sites")
        self.assertEqual(binary_tree_defaults["graph_size"], 3)

    def test_wait_for_editor_result_delegates_to_session_without_private_event_access(
        self,
    ) -> None:
        class FakeSession:
            def __init__(self) -> None:
                self.calls: list[float | None] = []

            def wait_for_result(
                self, timeout: float | None = None
            ) -> EditorResult | None:
                self.calls.append(timeout)
                return None

        session = FakeSession()

        result = wait_for_editor_result(session, poll_interval=0.05)

        self.assertIsNone(result)
        self.assertEqual(session.calls, [None])

    def test_wait_for_editor_result_polls_until_a_result_is_available(self) -> None:
        session = EditorSession(
            initial_spec=build_sample_spec(),
            default_engine=EngineName.EINSUM_NUMPY,
        )

        def finish_session() -> None:
            session.complete(
                serialized_spec={
                    "schema_version": 3,
                    "network": build_sample_spec().to_dict(),
                },
                engine=EngineName.EINSUM_NUMPY,
            )

        timer = threading.Timer(0.2, finish_session)
        timer.start()
        self.addCleanup(timer.cancel)

        result = wait_for_editor_result(session, poll_interval=0.05)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.engine, EngineName.EINSUM_NUMPY)

    def test_launch_tensor_network_editor_waits_for_complete(self) -> None:
        ready_queue: Queue[str] = Queue()
        result_queue: Queue[EditorResult | None] = Queue()

        def run_editor() -> None:
            result = launch_tensor_network_editor(
                initial_spec=build_sample_spec(),
                default_engine=EngineName.EINSUM_NUMPY,
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
                "engine": EngineName.EINSUM_NUMPY.value,
                "spec": {"schema_version": 3, "network": build_sample_spec().to_dict()},
            },
        )
        self.assertTrue(payload["ok"])

        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        result = result_queue.get(timeout=1)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.engine, EngineName.EINSUM_NUMPY)

    def test_launch_tensor_network_editor_returns_none_on_cancel(self) -> None:
        ready_queue: Queue[str] = Queue()
        result_queue: Queue[EditorResult | None] = Queue()

        def run_editor() -> None:
            result = launch_tensor_network_editor(
                initial_spec=build_sample_spec(),
                default_engine=EngineName.EINSUM_NUMPY,
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
