from __future__ import annotations

import threading
from pathlib import Path
from queue import Queue
from typing import cast

import pytest

from tensor_network_editor.api import launch_tensor_network_editor
from tensor_network_editor.app.session import (
    EditorSession,
    build_blank_network_spec,
    wait_for_editor_result,
)
from tensor_network_editor.models import EditorResult, EngineName, NetworkSpec
from tests.app_support import request_json


def test_build_blank_network_spec_returns_empty_editor_state() -> None:
    spec = build_blank_network_spec()

    assert spec.name == "Untitled Network"
    assert spec.tensors == []
    assert spec.edges == []
    assert spec.groups == []
    assert spec.notes == []


def test_bootstrap_payload_includes_template_parameter_definitions(
    editor_session: EditorSession,
) -> None:
    payload = editor_session.bootstrap_payload()
    template_definitions = cast(dict[str, object], payload["template_definitions"])
    mps_definition = cast(dict[str, object], template_definitions["mps"])
    binary_tree_definition = cast(
        dict[str, object], template_definitions["binary_tree"]
    )
    binary_tree_defaults = cast(dict[str, object], binary_tree_definition["defaults"])
    spec_payload = cast(dict[str, object], payload["spec"])
    network_payload = cast(dict[str, object], spec_payload["network"])

    assert payload["default_engine"] == EngineName.EINSUM_NUMPY.value
    assert payload["schema_version"] == 3
    assert network_payload["id"] == "network_demo"
    assert mps_definition["graph_size_label"] == "Sites"
    assert binary_tree_defaults["graph_size"] == 3


def test_generate_returns_preview_without_finishing_session(
    editor_session: EditorSession,
    serialized_sample_spec: dict[str, object],
) -> None:
    result = editor_session.generate(
        serialized_sample_spec,
        EngineName.EINSUM_NUMPY,
    )

    assert result.engine is EngineName.EINSUM_NUMPY
    assert result.code
    assert editor_session.wait_for_result(timeout=0.01) is None


def test_complete_records_result_and_can_write_code(
    sample_spec: NetworkSpec,
    serialized_sample_spec: dict[str, object],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_path = tmp_path / "generated.py"
    session = EditorSession(
        initial_spec=sample_spec,
        default_engine=EngineName.EINSUM_NUMPY,
        print_code=True,
        code_path=output_path,
    )

    result = session.complete(serialized_sample_spec, EngineName.EINSUM_NUMPY)

    assert result.confirmed is True
    assert result.codegen is not None
    assert session.wait_for_result(timeout=0.01) == result
    assert output_path.read_text(encoding="utf-8") == result.codegen.code
    assert capsys.readouterr().out == f"{result.codegen.code}\n"


def test_cancel_marks_session_finished_without_result(
    editor_session: EditorSession,
) -> None:
    editor_session.cancel()

    assert editor_session.wait_for_result(timeout=0.01) is None


def test_wait_for_editor_result_delegates_to_session_once() -> None:
    class FakeSession:
        def __init__(self) -> None:
            self.calls: list[float | None] = []

        def wait_for_result(self, timeout: float | None = None) -> EditorResult | None:
            self.calls.append(timeout)
            return None

    session = FakeSession()

    result = wait_for_editor_result(session, poll_interval=0.05)

    assert result is None
    assert session.calls == [None]


def test_launch_tensor_network_editor_waits_for_complete(
    sample_spec: NetworkSpec,
    serialized_sample_spec: dict[str, object],
) -> None:
    ready_queue: Queue[str] = Queue()
    result_queue: Queue[EditorResult | None] = Queue()

    def run_editor() -> None:
        result = launch_tensor_network_editor(
            initial_spec=sample_spec,
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
            "spec": serialized_sample_spec,
        },
    )

    assert payload["ok"] is True

    thread.join(timeout=5)
    assert not thread.is_alive()
    result = result_queue.get(timeout=1)
    assert result is not None
    assert result.engine is EngineName.EINSUM_NUMPY


def test_launch_tensor_network_editor_returns_none_on_cancel(
    sample_spec: NetworkSpec,
) -> None:
    ready_queue: Queue[str] = Queue()
    result_queue: Queue[EditorResult | None] = Queue()

    def run_editor() -> None:
        result = launch_tensor_network_editor(
            initial_spec=sample_spec,
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

    assert payload["ok"] is True

    thread.join(timeout=5)
    assert not thread.is_alive()
    assert result_queue.get(timeout=1) is None
