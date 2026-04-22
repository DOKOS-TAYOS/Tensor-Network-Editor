from __future__ import annotations

import signal
import threading
from importlib import import_module
from pathlib import Path
from queue import Queue
from typing import cast
from unittest.mock import patch

import pytest

from tensor_network_editor.api import launch_tensor_network_editor
from tensor_network_editor.app._protocol import JsonDict
from tensor_network_editor.app.session import (
    EditorSession,
    build_blank_network_spec,
    wait_for_editor_result,
)
from tensor_network_editor.errors import CodeGenerationError
from tensor_network_editor.models import (
    EditorResult,
    EngineName,
    NetworkSpec,
    TensorCollectionFormat,
)
from tensor_network_editor.serialization import SCHEMA_VERSION
from tests.app_support import request_json
from tests.factories import build_outer_product_plan_spec, serialize_spec_payload


def test_build_blank_network_spec_returns_empty_editor_state() -> None:
    spec = build_blank_network_spec()

    assert spec.name == "Untitled Network"
    assert spec.tensors == []
    assert spec.edges == []
    assert spec.groups == []
    assert spec.notes == []


def test_editor_session_defaults_to_tensorkrowch() -> None:
    session = EditorSession()

    assert session.default_engine is EngineName.TENSORKROWCH
    assert session.default_collection_format is TensorCollectionFormat.LIST


def test_editor_session_exposes_short_logging_identifier() -> None:
    session = EditorSession()

    assert len(session.session_id) == 8
    assert session.session_id.isalnum()


def test_session_module_reuses_internal_runtime_helpers() -> None:
    session_module = import_module("tensor_network_editor.app.session")
    runtime_module = import_module("tensor_network_editor.app._session_runtime")

    assert (
        session_module.build_blank_network_spec
        is runtime_module.build_blank_network_spec
    )
    assert (
        session_module.wait_for_editor_result is runtime_module.wait_for_editor_result
    )


def test_bootstrap_payload_includes_template_parameter_definitions(
    editor_session: EditorSession,
) -> None:
    payload = editor_session.bootstrap_payload()
    template_definitions = cast(JsonDict, payload["template_definitions"])
    mps_definition = cast(JsonDict, template_definitions["mps"])
    peps_definition = cast(JsonDict, template_definitions["peps_2x2"])
    binary_tree_definition = cast(JsonDict, template_definitions["binary_tree"])
    peps_defaults = cast(JsonDict, peps_definition["defaults"])
    binary_tree_defaults = cast(JsonDict, binary_tree_definition["defaults"])
    spec_payload = cast(JsonDict, payload["spec"])
    network_payload = cast(JsonDict, spec_payload["network"])

    assert payload["default_engine"] == EngineName.EINSUM_NUMPY.value
    assert payload["default_collection_format"] == TensorCollectionFormat.LIST.value
    assert payload["collection_formats"] == [
        collection_format.value for collection_format in TensorCollectionFormat
    ]
    assert payload["schema_version"] == SCHEMA_VERSION
    assert network_payload["id"] == "network_demo"
    assert mps_definition["graph_size_label"] == "Sites"
    assert peps_defaults["graph_size"] == 3
    assert binary_tree_defaults["graph_size"] == 3


def test_generate_returns_preview_without_finishing_session(
    editor_session: EditorSession,
    serialized_sample_spec: JsonDict,
) -> None:
    result = editor_session.generate(
        serialized_sample_spec,
        EngineName.EINSUM_NUMPY,
        TensorCollectionFormat.DICT,
    )

    assert result.engine is EngineName.EINSUM_NUMPY
    assert result.code
    assert "tensors_dict = {" in result.code
    assert editor_session.wait_for_result(timeout=0.01) is None


def test_complete_records_result_and_can_write_code(
    sample_spec: NetworkSpec,
    serialized_sample_spec: JsonDict,
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


def test_complete_supports_collection_format_in_generated_output(
    sample_spec: NetworkSpec,
    serialized_sample_spec: JsonDict,
) -> None:
    session = EditorSession(
        initial_spec=sample_spec,
        default_engine=EngineName.EINSUM_NUMPY,
    )

    result = session.complete(
        serialized_sample_spec,
        EngineName.EINSUM_NUMPY,
        TensorCollectionFormat.MATRIX,
    )

    assert result.codegen is not None
    assert "tensor_rows = []" in result.codegen.code


def test_complete_returns_existing_result_when_called_after_finish(
    sample_spec: NetworkSpec,
    serialized_sample_spec: JsonDict,
) -> None:
    session = EditorSession(
        initial_spec=sample_spec,
        default_engine=EngineName.EINSUM_NUMPY,
    )

    first_result = session.complete(serialized_sample_spec, EngineName.EINSUM_NUMPY)
    second_result = session.complete(serialized_sample_spec, EngineName.QUIMB)

    assert second_result is first_result
    assert session.wait_for_result(timeout=0.01) is first_result
    assert first_result.engine is EngineName.EINSUM_NUMPY


def test_generate_propagates_codegen_errors_from_backend() -> None:
    spec = build_outer_product_plan_spec()
    session = EditorSession(
        initial_spec=spec,
        default_engine=EngineName.TENSORKROWCH,
    )

    with pytest.raises(CodeGenerationError, match="TensorKrowch"):
        session.generate(
            serialize_spec_payload(spec),
            EngineName.TENSORKROWCH,
        )


def test_cancel_marks_session_finished_without_result(
    editor_session: EditorSession,
) -> None:
    editor_session.cancel()

    assert editor_session.wait_for_result(timeout=0.01) is None


def test_cancel_does_not_override_completed_result(
    sample_spec: NetworkSpec,
    serialized_sample_spec: JsonDict,
) -> None:
    session = EditorSession(
        initial_spec=sample_spec,
        default_engine=EngineName.EINSUM_NUMPY,
    )

    result = session.complete(serialized_sample_spec, EngineName.EINSUM_NUMPY)
    session.cancel()

    assert session.wait_for_result(timeout=0.01) == result


def test_wait_for_editor_result_delegates_to_session_once() -> None:
    class FakeSession:
        def __init__(self) -> None:
            self.calls: list[float | None] = []

        def wait_for_result(self, timeout: float | None = None) -> EditorResult | None:
            self.calls.append(timeout)
            return None

    session = FakeSession()

    result = wait_for_editor_result(session)

    assert result is None
    assert session.calls == [None]


def test_wait_for_editor_result_warns_when_poll_interval_is_passed() -> None:
    class FakeSession:
        def wait_for_result(self, timeout: float | None = None) -> EditorResult | None:
            assert timeout is None
            return None

    session = FakeSession()

    with pytest.warns(
        DeprecationWarning,
        match="poll_interval=.*deprecated and has no effect",
    ):
        result = wait_for_editor_result(session, poll_interval=0.05)

    assert result is None


def test_launch_tensor_network_editor_waits_for_complete(
    sample_spec: NetworkSpec,
    serialized_sample_spec: JsonDict,
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


def test_launch_editor_session_start_failure_restores_sigint_and_does_not_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tensor_network_editor.app import session as session_module

    previous_handler = signal.default_int_handler
    installed_handlers: list[object] = []

    def fake_getsignal(sig: signal.Signals) -> object:
        assert sig is signal.SIGINT
        return previous_handler

    def fake_signal(sig: signal.Signals, handler: object) -> object:
        assert sig is signal.SIGINT
        installed_handlers.append(handler)
        return handler

    class FailingEditorServer:
        stop_calls = 0

        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def start(self) -> None:
            raise RuntimeError("boom")

        def stop(self) -> None:
            type(self).stop_calls += 1

    monkeypatch.setattr(signal, "getsignal", fake_getsignal)
    monkeypatch.setattr(signal, "signal", fake_signal)
    monkeypatch.setattr(
        "tensor_network_editor.app.server.EditorServer",
        FailingEditorServer,
    )

    with pytest.raises(RuntimeError, match="boom"):
        session_module.launch_editor_session(open_browser=False)

    assert len(installed_handlers) == 2
    assert callable(installed_handlers[0])
    assert installed_handlers[1] is previous_handler
    assert FailingEditorServer.stop_calls == 0


def test_launch_editor_session_prints_local_url_when_browser_is_disabled(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tensor_network_editor.app import session as session_module

    class FakeEditorServer:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            self.base_url = "http://127.0.0.1:43210"

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

    def fake_wait_for_editor_result(_session: object) -> None:
        return None

    class FakeThread:
        name = "worker"

    monkeypatch.setattr(
        "tensor_network_editor.app.server.EditorServer",
        FakeEditorServer,
    )
    monkeypatch.setattr(
        session_module,
        "wait_for_editor_result",
        fake_wait_for_editor_result,
    )
    monkeypatch.setattr(
        session_module.threading, "current_thread", lambda: FakeThread()
    )

    result = session_module.launch_editor_session(open_browser=False)

    assert result is None
    assert "http://127.0.0.1:43210" in capsys.readouterr().out


def test_launch_editor_session_prints_local_url_when_browser_open_fails(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tensor_network_editor.app import session as session_module

    class FakeEditorServer:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            self.base_url = "http://127.0.0.1:43210"

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

    def fake_wait_for_editor_result(_session: object) -> None:
        return None

    class FakeThread:
        name = "worker"

    monkeypatch.setattr(
        "tensor_network_editor.app.server.EditorServer",
        FakeEditorServer,
    )
    monkeypatch.setattr(
        session_module,
        "wait_for_editor_result",
        fake_wait_for_editor_result,
    )
    monkeypatch.setattr(
        session_module.threading, "current_thread", lambda: FakeThread()
    )

    with patch.object(
        session_module.webbrowser,
        "open",
        side_effect=OSError("no browser"),
    ):
        result = session_module.launch_editor_session(open_browser=True)

    assert result is None
    assert "http://127.0.0.1:43210" in capsys.readouterr().out


def test_launch_tensor_network_editor_passes_template_catalog_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_kwargs: dict[str, object] = {}

    def fake_launch_editor_session(*args: object, **kwargs: object) -> None:
        del args
        captured_kwargs.update(kwargs)
        return None

    monkeypatch.setattr(
        "tensor_network_editor.app.session.launch_editor_session",
        fake_launch_editor_session,
    )

    result = launch_tensor_network_editor(
        open_browser=False,
        template_catalog_path=tmp_path / ".tensor-network-editor" / "templates.json",
    )

    assert result is None
    assert captured_kwargs["template_catalog_path"] == (
        tmp_path / ".tensor-network-editor" / "templates.json"
    )


def test_launch_tensor_network_editor_passes_subnetwork_catalog_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_kwargs: dict[str, object] = {}

    def fake_launch_editor_session(*args: object, **kwargs: object) -> None:
        del args
        captured_kwargs.update(kwargs)
        return None

    monkeypatch.setattr(
        "tensor_network_editor.app.session.launch_editor_session",
        fake_launch_editor_session,
    )

    result = launch_tensor_network_editor(
        open_browser=False,
        subnetwork_catalog_path=tmp_path
        / ".tensor-network-editor"
        / "subnetworks.json",
        shared_subnetwork_catalog_path=tmp_path / "shared" / "subnetworks.json",
    )

    assert result is None
    assert captured_kwargs["subnetwork_catalog_path"] == (
        tmp_path / ".tensor-network-editor" / "subnetworks.json"
    )
    assert captured_kwargs["shared_subnetwork_catalog_path"] == (
        tmp_path / "shared" / "subnetworks.json"
    )
