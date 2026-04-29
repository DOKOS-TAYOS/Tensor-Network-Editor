from __future__ import annotations

import logging
import signal
import threading
from collections.abc import Iterator
from importlib import import_module
from pathlib import Path
from queue import Queue
from typing import Any, cast
from unittest.mock import patch

import pytest

from tensor_network_editor.app._protocol import JsonDict
from tensor_network_editor.app.session import (
    EditorSession,
    build_blank_network_spec,
    wait_for_editor_result,
)
from tensor_network_editor.editor import EditorLaunchOptions, open_editor
from tensor_network_editor.errors import CodeGenerationError
from tensor_network_editor.io import SCHEMA_VERSION
from tensor_network_editor.models import (
    EditorResult,
    EngineName,
    NetworkSpec,
    TensorCollectionFormat,
)
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
    ttn_definition = cast(JsonDict, template_definitions["ttn"])
    peps_defaults = cast(JsonDict, peps_definition["defaults"])
    ttn_defaults = cast(JsonDict, ttn_definition["defaults"])
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
    assert ttn_defaults["depth"] == 3


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


def test_generate_logs_preview_lifecycle_with_session_context(
    editor_session: EditorSession,
    serialized_sample_spec: JsonDict,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.DEBUG, logger="tensor_network_editor"):
        result = editor_session.generate(
            serialized_sample_spec,
            EngineName.EINSUM_NUMPY,
            TensorCollectionFormat.DICT,
        )

    assert result.engine is EngineName.EINSUM_NUMPY
    assert "Session preview generation started" in caplog.text
    assert "Session preview generation finished" in caplog.text
    assert f"session={editor_session.session_id}" in caplog.text
    assert "outcome=success" in caplog.text
    assert "elapsed_ms=" in caplog.text


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


def test_complete_logs_duplicate_request_branch_with_session_context(
    sample_spec: NetworkSpec,
    serialized_sample_spec: JsonDict,
    caplog: pytest.LogCaptureFixture,
) -> None:
    session = EditorSession(
        initial_spec=sample_spec,
        default_engine=EngineName.EINSUM_NUMPY,
    )

    with caplog.at_level(logging.DEBUG, logger="tensor_network_editor"):
        first_result = session.complete(serialized_sample_spec, EngineName.EINSUM_NUMPY)
        second_result = session.complete(serialized_sample_spec, EngineName.QUIMB)

    assert second_result is first_result
    assert "Session completion started" in caplog.text
    assert "Session completion finished" in caplog.text
    assert "Duplicate completion request ignored" in caplog.text
    assert f"session={session.session_id}" in caplog.text


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


def test_wait_for_editor_result_polls_until_the_session_finishes() -> None:
    class FakeSession:
        def __init__(self) -> None:
            self.calls: list[float | None] = []
            self._finished = False

        def wait_for_result(self, timeout: float | None = None) -> EditorResult | None:
            self.calls.append(timeout)
            self._finished = True
            return None

        def is_finished(self) -> bool:
            return self._finished

    session = FakeSession()

    result = wait_for_editor_result(session)

    assert result is None
    assert session.calls == [0.1]


def test_wait_for_editor_result_warns_when_poll_interval_is_passed() -> None:
    class FakeSession:
        def wait_for_result(self, timeout: float | None = None) -> EditorResult | None:
            assert timeout == 0.1
            return None

        def is_finished(self) -> bool:
            return True

    session = FakeSession()

    with pytest.warns(
        DeprecationWarning,
        match="poll_interval=.*deprecated and has no effect",
    ):
        result = wait_for_editor_result(session, poll_interval=0.05)

    assert result is None


def test_wait_for_editor_result_returns_completed_result_after_polling() -> None:
    completed_result = EditorResult(
        spec=build_blank_network_spec(),
        engine=EngineName.EINSUM_NUMPY,
        confirmed=False,
    )

    class FakeSession:
        def __init__(self) -> None:
            self.calls = 0

        def wait_for_result(self, timeout: float | None = None) -> EditorResult | None:
            assert timeout == 0.1
            self.calls += 1
            return completed_result if self.calls == 3 else None

        def is_finished(self) -> bool:
            return False

    session = FakeSession()

    result = wait_for_editor_result(session)

    assert result is completed_result
    assert session.calls == 3


def test_open_editor_waits_for_complete(
    sample_spec: NetworkSpec,
    serialized_sample_spec: JsonDict,
) -> None:
    ready_queue: Queue[str] = Queue()
    result_queue: Queue[EditorResult | None] = Queue()

    def run_editor() -> None:
        result = open_editor(
            sample_spec,
            options=EditorLaunchOptions(
                default_engine=EngineName.EINSUM_NUMPY,
                open_browser=False,
                _on_server_ready=ready_queue.put,
            ),
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


def test_open_editor_returns_none_on_cancel(
    sample_spec: NetworkSpec,
) -> None:
    ready_queue: Queue[str] = Queue()
    result_queue: Queue[EditorResult | None] = Queue()

    def run_editor() -> None:
        result = open_editor(
            sample_spec,
            options=EditorLaunchOptions(
                default_engine=EngineName.EINSUM_NUMPY,
                open_browser=False,
                _on_server_ready=ready_queue.put,
            ),
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
    captured = capsys.readouterr().out
    assert "http://127.0.0.1:43210" in captured
    assert "still running" not in captured.lower()


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
    captured = capsys.readouterr().out
    assert "could not open the browser automatically" in captured.lower()
    assert "server is still running" in captured.lower()
    assert "http://127.0.0.1:43210" in captured


def test_launch_editor_session_logs_browser_fallback_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
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

    with (
        caplog.at_level(logging.DEBUG, logger="tensor_network_editor"),
        patch.object(
            session_module.webbrowser,
            "open",
            side_effect=OSError("no browser"),
        ),
    ):
        result = session_module.launch_editor_session(open_browser=True)

    assert result is None
    assert "Editor session launch started" in caplog.text
    assert "Browser open attempt started" in caplog.text
    assert "Browser open attempt failed" in caplog.text
    assert "Editor session launch finished" in caplog.text
    assert "session=" in caplog.text


def test_launch_editor_session_prints_local_url_when_browser_open_is_not_acknowledged(
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

    with patch.object(session_module.webbrowser, "open", return_value=False):
        result = session_module.launch_editor_session(open_browser=True)

    assert result is None
    captured = capsys.readouterr().out
    assert "could not open the browser automatically" in captured.lower()
    assert "server is still running" in captured.lower()
    assert "http://127.0.0.1:43210" in captured


def test_open_editor_passes_template_catalog_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_kwargs: dict[str, object] = {}

    def fake_launch_editor_session(*args: object, **kwargs: object) -> None:
        del args
        captured_kwargs.update(kwargs)
        return None

    monkeypatch.setattr(
        "tensor_network_editor.editor.launch_editor_session",
        fake_launch_editor_session,
    )

    result = open_editor(
        options=EditorLaunchOptions(
            open_browser=False,
            template_catalog_path=(
                tmp_path / ".tensor-network-editor" / "templates.json"
            ),
        ),
    )

    assert result is None
    assert captured_kwargs["template_catalog_path"] == (
        tmp_path / ".tensor-network-editor" / "templates.json"
    )


def test_open_editor_passes_subnetwork_catalog_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_kwargs: dict[str, object] = {}

    def fake_launch_editor_session(*args: object, **kwargs: object) -> None:
        del args
        captured_kwargs.update(kwargs)
        return None

    monkeypatch.setattr(
        "tensor_network_editor.editor.launch_editor_session",
        fake_launch_editor_session,
    )

    result = open_editor(
        options=EditorLaunchOptions(
            open_browser=False,
            subnetwork_catalog_path=(
                tmp_path / ".tensor-network-editor" / "subnetworks.json"
            ),
            shared_subnetwork_catalog_path=(tmp_path / "shared" / "subnetworks.json"),
        ),
    )

    assert result is None
    assert captured_kwargs["subnetwork_catalog_path"] == (
        tmp_path / ".tensor-network-editor" / "subnetworks.json"
    )
    assert captured_kwargs["shared_subnetwork_catalog_path"] == (
        tmp_path / "shared" / "subnetworks.json"
    )


def test_open_editor_passes_log_file_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_kwargs: dict[str, object] = {}

    def fake_launch_editor_session(*args: object, **kwargs: object) -> None:
        del args
        captured_kwargs.update(kwargs)
        return None

    monkeypatch.setattr(
        "tensor_network_editor.editor.launch_editor_session",
        fake_launch_editor_session,
    )

    result = open_editor(
        options=EditorLaunchOptions(
            open_browser=False,
            log_file_path="session.log",
            log_file_max_bytes=2048,
            log_file_backup_count=7,
        ),
    )

    assert result is None
    assert captured_kwargs["log_file_path"] == "session.log"
    assert captured_kwargs["log_file_max_bytes"] == 2048
    assert captured_kwargs["log_file_backup_count"] == 7


def test_open_editor_logs_theme_and_spec_mode_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from contextlib import contextmanager

    from tests.factories import build_tree_periodic_tree_spec

    captured_contexts: list[dict[str, object]] = []

    @contextmanager
    def fake_log_operation(
        *_args: object,
        context: dict[str, object] | None = None,
        **_kwargs: object,
    ) -> Iterator[dict[str, object]]:
        captured_contexts.append(dict(context or {}))
        yield {}

    monkeypatch.setattr(
        "tensor_network_editor.editor.log_operation",
        fake_log_operation,
    )

    def fake_launch_editor_session(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(
        "tensor_network_editor.editor.launch_editor_session",
        fake_launch_editor_session,
    )

    open_editor(
        build_tree_periodic_tree_spec(),
        options=EditorLaunchOptions(theme="dark", open_browser=False),
    )

    assert len(captured_contexts) == 1
    assert captured_contexts[0]["engine"] is EngineName.TENSORKROWCH
    assert captured_contexts[0]["mode"] == "dark"
    assert captured_contexts[0]["spec_mode"] == "tree_periodic"


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("log_file_max_bytes", 0),
        ("log_file_backup_count", -1),
    ],
)
def test_editor_launch_options_reject_non_positive_log_rotation_settings(
    field_name: str,
    value: int,
) -> None:
    with pytest.raises(ValueError, match="must be > 0"):
        cast(Any, EditorLaunchOptions)(**{field_name: value})


def test_launch_editor_session_log_file_writes_trace_and_releases_handler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tensor_network_editor.app import session as session_module

    log_file_path = tmp_path / "editor-session.log"
    package_logger = logging.getLogger("tensor_network_editor")

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

    existing_handler_ids = {id(handler) for handler in package_logger.handlers}

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

    result = session_module.launch_editor_session(
        open_browser=False,
        log_file_path=log_file_path,
    )

    assert result is None
    assert log_file_path.exists()
    assert "Editor session launch started" in log_file_path.read_text(encoding="utf-8")
    assert all(
        id(handler) in existing_handler_ids
        or not isinstance(handler, logging.FileHandler)
        or Path(handler.baseFilename).resolve() != log_file_path.resolve()
        for handler in package_logger.handlers
    )


def test_launch_editor_session_log_file_rotates_and_releases_handler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tensor_network_editor.app import session as session_module

    log_file_path = tmp_path / "rotating-editor-session.log"
    package_logger = logging.getLogger("tensor_network_editor")

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

    existing_handler_ids = {id(handler) for handler in package_logger.handlers}

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

    first_result = session_module.launch_editor_session(
        open_browser=False,
        log_file_path=log_file_path,
        log_file_max_bytes=256,
        log_file_backup_count=2,
    )
    second_result = session_module.launch_editor_session(
        open_browser=False,
        log_file_path=log_file_path,
        log_file_max_bytes=256,
        log_file_backup_count=2,
    )

    combined_log_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(tmp_path.glob("rotating-editor-session.log*"))
        if path.is_file()
    )

    assert first_result is None
    assert second_result is None
    assert log_file_path.exists()
    assert (tmp_path / "rotating-editor-session.log.1").exists()
    assert "Editor session launch started" in combined_log_text
    assert all(
        id(handler) in existing_handler_ids
        or not isinstance(handler, logging.FileHandler)
        or Path(handler.baseFilename).resolve() != log_file_path.resolve()
        for handler in package_logger.handlers
    )
