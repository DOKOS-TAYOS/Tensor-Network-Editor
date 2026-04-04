from __future__ import annotations

import logging
import signal
import threading
import webbrowser
from collections.abc import Callable
from types import FrameType
from typing import Any

from .._io import write_utf8_text
from .._templates import build_template_spec, list_template_names
from ..codegen.registry import generate_code as generate_code_internal
from ..models import CodegenResult, EditorResult, EngineName, NetworkSpec
from ..serialization import SCHEMA_VERSION, serialize_spec
from ..types import StrPath

LOGGER = logging.getLogger(__name__)
SignalHandler = Callable[[int, FrameType | None], Any]


def build_blank_network_spec() -> NetworkSpec:
    return NetworkSpec(name="Untitled Network")


class EditorSession:
    def __init__(
        self,
        initial_spec: NetworkSpec | None = None,
        default_engine: EngineName = EngineName.TENSORNETWORK,
        *,
        print_code: bool = False,
        code_path: StrPath | None = None,
    ) -> None:
        self.initial_spec = initial_spec or build_blank_network_spec()
        self.default_engine = default_engine
        self.print_code = print_code
        self.code_path = code_path
        self._finished_event = threading.Event()
        self._result: EditorResult | None = None
        self._lock = threading.Lock()

    def bootstrap_payload(self) -> dict[str, object]:
        return {
            "default_engine": self.default_engine.value,
            "engines": [engine.value for engine in EngineName],
            "schema_version": SCHEMA_VERSION,
            "templates": list_template_names(),
            "spec": serialize_spec(self.initial_spec),
        }

    def generate(
        self, serialized_spec: dict[str, object], engine: EngineName
    ) -> CodegenResult:
        LOGGER.debug("Generating preview code for engine '%s'", engine.value)
        spec = deserialize_serialized_spec(serialized_spec)
        result = generate_code_internal(spec, engine)
        return result

    def complete(
        self, serialized_spec: dict[str, object], engine: EngineName
    ) -> EditorResult:
        LOGGER.info("Completing editor session with engine '%s'", engine.value)
        spec = deserialize_serialized_spec(serialized_spec)
        codegen_result = generate_code_internal(spec, engine)

        if self.print_code:
            print(codegen_result.code)
        if self.code_path is not None:
            write_utf8_text(
                self.code_path,
                codegen_result.code,
                description="generated Python code",
            )

        result = EditorResult(
            spec=spec, engine=engine, codegen=codegen_result, confirmed=True
        )
        with self._lock:
            self._result = result
            self._finished_event.set()
        return result

    def build_template(self, template_name: str) -> NetworkSpec:
        return build_template_spec(template_name)

    def cancel(self) -> None:
        LOGGER.info("Cancelling editor session")
        with self._lock:
            self._result = None
            self._finished_event.set()

    def wait_for_result(self, timeout: float | None = None) -> EditorResult | None:
        finished = self._finished_event.wait(timeout)
        if not finished:
            return None
        with self._lock:
            return self._result


def wait_for_editor_result(
    session: EditorSession,
    *,
    poll_interval: float = 0.2,
) -> EditorResult | None:
    del poll_interval
    return session.wait_for_result(timeout=None)


def deserialize_serialized_spec(serialized_spec: dict[str, object]) -> NetworkSpec:
    from ..serialization import deserialize_spec

    return deserialize_spec(serialized_spec)


def launch_editor_session(
    initial_spec: NetworkSpec | None = None,
    *,
    default_engine: EngineName = EngineName.TENSORNETWORK,
    open_browser: bool = True,
    host: str = "127.0.0.1",
    port: int = 0,
    print_code: bool = False,
    code_path: StrPath | None = None,
    _on_server_ready: Callable[[str], None] | None = None,
) -> EditorResult | None:
    from .server import EditorServer

    LOGGER.info("Starting editor session")
    session = EditorSession(
        initial_spec=initial_spec,
        default_engine=default_engine,
        print_code=print_code,
        code_path=code_path,
    )
    server = EditorServer(session=session, host=host, port=port)
    previous_sigint_handler: SignalHandler | int | None = None

    if threading.current_thread() is threading.main_thread():
        previous_sigint_handler = signal.getsignal(signal.SIGINT)

        def _handle_sigint(signum: int, frame: FrameType | None) -> None:
            session.cancel()
            raise KeyboardInterrupt

        signal.signal(signal.SIGINT, _handle_sigint)

    server.start()
    try:
        if _on_server_ready is not None:
            _on_server_ready(server.base_url)
        if open_browser:
            LOGGER.info("Opening browser at %s", server.base_url)
            try:
                opened = webbrowser.open(server.base_url)
            except Exception:  # pragma: no cover - platform dependent browser errors
                LOGGER.exception("Failed to open the system browser for the editor.")
            else:
                if not opened:
                    LOGGER.warning("Browser open request was not acknowledged.")
        return wait_for_editor_result(session)
    except KeyboardInterrupt:
        LOGGER.info("Editor session interrupted by keyboard input")
        session.cancel()
        raise
    finally:
        server.stop()
        if previous_sigint_handler is not None:
            signal.signal(signal.SIGINT, previous_sigint_handler)
