"""Session lifecycle helpers for the local browser editor."""

from __future__ import annotations

import logging
import signal
import threading
import webbrowser
from collections.abc import Callable
from types import FrameType
from typing import Any, Protocol

from .._templates import TemplateParameters
from ..models import (
    CodegenResult,
    EditorResult,
    EngineName,
    NetworkSpec,
    TensorCollectionFormat,
)
from ..types import StrPath
from ._services import (
    build_bootstrap_payload,
    build_template_from_payload,
    complete_session_request,
    generate_session_request,
)

LOGGER = logging.getLogger(__name__)
SignalHandler = Callable[[int, FrameType | None], Any]


class SupportsWaitForResult(Protocol):
    """Protocol implemented by session-like objects that can wait for results."""

    def wait_for_result(self, timeout: float | None = None) -> EditorResult | None:
        """Wait for the final editor result or ``None`` on timeout."""
        ...


def build_blank_network_spec() -> NetworkSpec:
    """Build the default empty network shown in a new editor session."""
    return NetworkSpec(name="Untitled Network")


class EditorSession:
    """Mutable session state shared between the HTTP server and the caller."""

    def __init__(
        self,
        initial_spec: NetworkSpec | None = None,
        default_engine: EngineName = EngineName.TENSORKROWCH,
        default_collection_format: TensorCollectionFormat = TensorCollectionFormat.LIST,
        *,
        print_code: bool = False,
        code_path: StrPath | None = None,
    ) -> None:
        self.initial_spec = initial_spec or build_blank_network_spec()
        self.default_engine = default_engine
        self.default_collection_format = default_collection_format
        self.print_code = print_code
        self.code_path = code_path
        self._finished_event = threading.Event()
        self._result: EditorResult | None = None
        self._lock = threading.Lock()

    def bootstrap_payload(self) -> dict[str, object]:
        """Return the bootstrap payload consumed by the browser client."""
        return build_bootstrap_payload(self)

    def generate(
        self,
        serialized_spec: dict[str, object],
        engine: EngineName,
        collection_format: TensorCollectionFormat | None = None,
    ) -> CodegenResult:
        """Generate preview code without finalizing the session."""
        LOGGER.debug("Generating preview code for engine '%s'", engine.value)
        return generate_session_request(
            self,
            serialized_spec,
            engine,
            collection_format,
        )

    def complete(
        self,
        serialized_spec: dict[str, object],
        engine: EngineName,
        collection_format: TensorCollectionFormat | None = None,
    ) -> EditorResult:
        """Finalize the session and store the resulting editor output."""
        LOGGER.info("Completing editor session with engine '%s'", engine.value)
        result = complete_session_request(
            self,
            serialized_spec,
            engine,
            collection_format,
        )
        with self._lock:
            self._result = result
            self._finished_event.set()
        return result

    def build_template(
        self,
        template_name: str,
        parameters: TemplateParameters | None = None,
    ) -> NetworkSpec:
        """Build a validated template spec for insertion into the session."""
        return build_template_from_payload(self, template_name, parameters)

    def cancel(self) -> None:
        """Cancel the session and unblock any waiter."""
        LOGGER.info("Cancelling editor session")
        with self._lock:
            self._result = None
            self._finished_event.set()

    def wait_for_result(self, timeout: float | None = None) -> EditorResult | None:
        """Wait for the session to finish and return its final result."""
        finished = self._finished_event.wait(timeout)
        if not finished:
            return None
        with self._lock:
            return self._result


def wait_for_editor_result(
    session: SupportsWaitForResult,
    *,
    poll_interval: float = 0.2,
) -> EditorResult | None:
    """Wait for an editor session result using the session's blocking API."""
    del poll_interval
    return session.wait_for_result(timeout=None)


def launch_editor_session(
    initial_spec: NetworkSpec | None = None,
    *,
    default_engine: EngineName = EngineName.TENSORKROWCH,
    default_collection_format: TensorCollectionFormat = TensorCollectionFormat.LIST,
    open_browser: bool = True,
    host: str = "127.0.0.1",
    port: int = 0,
    print_code: bool = False,
    code_path: StrPath | None = None,
    _on_server_ready: Callable[[str], None] | None = None,
) -> EditorResult | None:
    """Create the local server, optionally open the browser, and wait for a result."""
    from .server import EditorServer

    LOGGER.info("Starting editor session")
    session = EditorSession(
        initial_spec=initial_spec,
        default_engine=default_engine,
        default_collection_format=default_collection_format,
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
