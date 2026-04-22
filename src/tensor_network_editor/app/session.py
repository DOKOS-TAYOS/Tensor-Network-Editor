"""Session lifecycle helpers for the local browser editor."""

from __future__ import annotations

import logging
import signal
import threading
import webbrowser
from collections.abc import Callable, Mapping, Sequence
from types import FrameType
from typing import Any
from uuid import uuid4

from ..codegen.registry import engine_name_to_text
from ..internal.templates._templates import TemplateParameters
from ..models import (
    CodegenResult,
    EditorResult,
    EngineIdentifier,
    EngineName,
    NetworkSpec,
    TensorCollectionFormat,
)
from ..types import JSONValue, StrPath
from ._protocol import JsonDict
from ._services import (
    build_bootstrap_payload,
    build_template_from_payload,
    complete_session_request,
    generate_session_request,
)
from ._session_catalogs import SessionCatalogState
from ._session_runtime import build_blank_network_spec, wait_for_editor_result

LOGGER = logging.getLogger(__name__)
SignalHandler = Callable[[int, FrameType | None], Any]


def _print_editor_url(base_url: str) -> None:
    """Print the local editor URL for manual browser opening."""
    print(f"Open the editor at {base_url}", flush=True)


class EditorSession:
    """Mutable session state shared between the HTTP server and the caller."""

    def __init__(
        self,
        initial_spec: NetworkSpec | None = None,
        default_engine: EngineIdentifier = EngineName.TENSORKROWCH,
        default_collection_format: TensorCollectionFormat = TensorCollectionFormat.LIST,
        *,
        print_code: bool = False,
        code_path: StrPath | None = None,
        template_catalog_path: StrPath | None = None,
        subnetwork_catalog_path: StrPath | None = None,
        shared_subnetwork_catalog_path: StrPath | None = None,
    ) -> None:
        """Initialize one mutable editor session.

        Args:
            initial_spec: Optional network specification to preload.
            default_engine: Backend initially selected in the editor.
            default_collection_format: Initial tensor collection layout for
                generated code.
            print_code: Whether to print generated code after confirmation.
            code_path: Optional output path for generated code after
                confirmation.
            template_catalog_path: Optional per-project static template catalog
                path.
            subnetwork_catalog_path: Optional per-project reusable subnetwork
                catalog path.
            shared_subnetwork_catalog_path: Optional shared reusable subnetwork
                catalog path.
        """
        self.initial_spec = initial_spec or build_blank_network_spec()
        self.session_id = uuid4().hex[:8]
        self.default_engine = default_engine
        self.default_collection_format = default_collection_format
        self.print_code = print_code
        self.code_path = code_path
        self._catalog_state = SessionCatalogState.load(
            template_catalog_path=template_catalog_path,
            subnetwork_catalog_path=subnetwork_catalog_path,
            shared_subnetwork_catalog_path=shared_subnetwork_catalog_path,
        )
        self._finished_event = threading.Event()
        self._result: EditorResult | None = None
        self._lock = threading.Lock()
        LOGGER.debug(
            "[session=%s] Initialized editor session with engine '%s'",
            self.session_id,
            engine_name_to_text(self.default_engine),
        )

    @property
    def template_catalog_path(self) -> StrPath | None:
        """Return the resolved project-template catalog path."""
        return self._catalog_state.template_catalog_path

    @property
    def subnetwork_catalog_path(self) -> StrPath | None:
        """Return the resolved project-subnetwork catalog path."""
        return self._catalog_state.subnetwork_catalog_path

    @property
    def shared_subnetwork_catalog_path(self) -> StrPath | None:
        """Return the resolved shared-subnetwork catalog path."""
        return self._catalog_state.shared_subnetwork_catalog_path

    @property
    def project_template_entries(self) -> Mapping[str, object]:
        """Return the project-local static template entries keyed by name."""
        return self._catalog_state.project_template_entries

    @property
    def template_catalog_warnings(self) -> list[str]:
        """Return any warnings raised while loading the local template catalog."""
        return self._catalog_state.template_catalog_warnings

    @property
    def project_subnetwork_entries(self) -> Mapping[str, object]:
        """Return the project-local reusable subnetwork entries keyed by name."""
        return self._catalog_state.project_subnetwork_entries

    @property
    def shared_subnetwork_entries(self) -> Mapping[str, object]:
        """Return the shared reusable subnetwork entries keyed by name."""
        return self._catalog_state.shared_subnetwork_entries

    @property
    def subnetwork_catalog_warnings(self) -> list[str]:
        """Return any warnings raised while loading reusable-subnetwork catalogs."""
        return self._catalog_state.subnetwork_catalog_warnings

    def list_available_template_names(self) -> list[str]:
        """Return the merged project-local and globally registered templates."""
        return self._catalog_state.list_available_template_names()

    def list_global_template_names(self) -> list[str]:
        """Return the globally registered template names only."""
        return self._catalog_state.list_global_template_names()

    def serialize_available_template_definitions(
        self,
    ) -> dict[str, dict[str, JSONValue]]:
        """Return serialized template definitions for the current session."""
        return self._catalog_state.serialize_available_template_definitions()

    def list_available_subnetwork_names(self) -> list[str]:
        """Return merged project-local and shared reusable subnetworks."""
        return self._catalog_state.list_available_subnetwork_names()

    def serialize_available_subnetwork_definitions(
        self,
    ) -> dict[str, dict[str, JSONValue]]:
        """Return serialized reusable-subnetwork definitions for the editor."""
        return self._catalog_state.serialize_available_subnetwork_definitions()

    def has_project_template(self, template_name: str) -> bool:
        """Return whether the session exposes a project-local template name."""
        return self._catalog_state.has_project_template(template_name)

    def has_global_template(self, template_name: str) -> bool:
        """Return whether the session exposes a globally registered template."""
        return self._catalog_state.has_global_template(template_name)

    def has_project_subnetwork(self, subnetwork_name: str) -> bool:
        """Return whether the session exposes a project-local reusable subnetwork."""
        return self._catalog_state.has_project_subnetwork(subnetwork_name)

    def build_project_template(self, template_name: str) -> NetworkSpec:
        """Build a copied project-local template spec for insertion."""
        return self._catalog_state.build_project_template(template_name)

    def build_project_template_display_name(self, template_name: str) -> str:
        """Return the derived display name used for one promoted template."""
        return self._catalog_state.build_project_template_display_name(template_name)

    def build_saved_subnetwork(self, subnetwork_name: str) -> NetworkSpec:
        """Build a copied saved reusable subnetwork spec for insertion."""
        return self._catalog_state.build_saved_subnetwork(subnetwork_name)

    def save_project_subnetwork(
        self,
        subnetwork_name: str,
        spec: NetworkSpec,
        *,
        tags: Sequence[str] | None = None,
        overwrite: bool = False,
    ) -> None:
        """Persist one reusable subnetwork and reload the project catalog."""
        self._catalog_state.save_project_subnetwork(
            subnetwork_name,
            spec,
            tags=tags,
            overwrite=overwrite,
        )

    def rename_project_subnetwork(
        self,
        subnetwork_name: str,
        new_subnetwork_name: str,
        *,
        overwrite: bool = False,
    ) -> None:
        """Rename one project-local reusable subnetwork and reload the catalog."""
        self._catalog_state.rename_project_subnetwork(
            subnetwork_name,
            new_subnetwork_name,
            overwrite=overwrite,
        )

    def delete_project_subnetwork(self, subnetwork_name: str) -> None:
        """Delete one project-local reusable subnetwork and reload the catalog."""
        self._catalog_state.delete_project_subnetwork(subnetwork_name)

    def save_project_template(
        self,
        template_name: str,
        spec: NetworkSpec,
        *,
        overwrite: bool = False,
    ) -> None:
        """Persist one new project-local static template and reload the catalog."""
        self._catalog_state.save_project_template(
            template_name,
            spec,
            overwrite=overwrite,
        )

    def rename_project_template(
        self,
        template_name: str,
        new_template_name: str,
        *,
        overwrite: bool = False,
    ) -> None:
        """Rename one project-local static template and reload the catalog."""
        self._catalog_state.rename_project_template(
            template_name,
            new_template_name,
            overwrite=overwrite,
        )

    def delete_project_template(self, template_name: str) -> None:
        """Delete one project-local static template and reload the catalog."""
        self._catalog_state.delete_project_template(template_name)

    def bootstrap_payload(self) -> JsonDict:
        """Return the bootstrap payload consumed by the browser client."""
        return build_bootstrap_payload(self)

    def generate(
        self,
        serialized_spec: Mapping[str, object],
        engine: EngineIdentifier,
        collection_format: TensorCollectionFormat | None = None,
    ) -> CodegenResult:
        """Generate preview code without finalizing the session."""
        LOGGER.debug(
            "[session=%s] Generating preview code for engine '%s'",
            self.session_id,
            engine_name_to_text(engine),
        )
        return generate_session_request(
            self,
            serialized_spec,
            engine,
            collection_format,
        )

    def complete(
        self,
        serialized_spec: Mapping[str, object],
        engine: EngineIdentifier,
        collection_format: TensorCollectionFormat | None = None,
    ) -> EditorResult:
        """Finalize the session and store the resulting editor output."""
        with self._lock:
            if self._finished_event.is_set() and self._result is not None:
                LOGGER.debug(
                    "[session=%s] Ignoring duplicate completion request",
                    self.session_id,
                )
                return self._result
        LOGGER.info(
            "[session=%s] Completing editor session with engine '%s'",
            self.session_id,
            engine_name_to_text(engine),
        )
        result = complete_session_request(
            self,
            serialized_spec,
            engine,
            collection_format,
        )
        with self._lock:
            if self._finished_event.is_set() and self._result is not None:
                LOGGER.debug(
                    "[session=%s] Returning existing completed result",
                    self.session_id,
                )
                return self._result
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
        LOGGER.info("[session=%s] Cancelling editor session", self.session_id)
        with self._lock:
            if self._finished_event.is_set():
                LOGGER.debug(
                    "[session=%s] Ignoring cancel request for finished session",
                    self.session_id,
                )
                return
            self._result = None
            self._finished_event.set()

    def wait_for_result(self, timeout: float | None = None) -> EditorResult | None:
        """Wait for the session to finish and return its final result."""
        finished = self._finished_event.wait(timeout)
        if not finished:
            return None
        with self._lock:
            return self._result


def launch_editor_session(
    initial_spec: NetworkSpec | None = None,
    *,
    default_engine: EngineIdentifier = EngineName.TENSORKROWCH,
    default_collection_format: TensorCollectionFormat = TensorCollectionFormat.LIST,
    open_browser: bool = True,
    host: str = "127.0.0.1",
    port: int = 0,
    print_code: bool = False,
    code_path: StrPath | None = None,
    template_catalog_path: StrPath | None = None,
    subnetwork_catalog_path: StrPath | None = None,
    shared_subnetwork_catalog_path: StrPath | None = None,
    _on_server_ready: Callable[[str], None] | None = None,
) -> EditorResult | None:
    """Create the local server, optionally open the browser, and wait.

    Args:
        initial_spec: Optional network specification to preload.
        default_engine: Backend initially selected in the editor UI.
        default_collection_format: Initial tensor collection layout for
            generated code.
        open_browser: Whether to ask the system browser to open the local URL.
        host: Local host interface to bind.
        port: Local port to bind. Use ``0`` for an ephemeral port.
        print_code: Whether to print generated code after confirmation.
        code_path: Optional output path for generated code after confirmation.
        template_catalog_path: Optional per-project static template catalog
            path.
        subnetwork_catalog_path: Optional per-project reusable subnetwork
            catalog path.
        shared_subnetwork_catalog_path: Optional shared reusable subnetwork
            catalog path.
        _on_server_ready: Internal callback used by tests once the local URL is
            available.

    Returns:
        ``None`` when the session is cancelled, otherwise the confirmed editor
        result.

    Raises:
        KeyboardInterrupt: If the session is interrupted from the main thread.
    """
    from .server import EditorServer

    session = EditorSession(
        initial_spec=initial_spec,
        default_engine=default_engine,
        default_collection_format=default_collection_format,
        print_code=print_code,
        code_path=code_path,
        template_catalog_path=template_catalog_path,
        subnetwork_catalog_path=subnetwork_catalog_path,
        shared_subnetwork_catalog_path=shared_subnetwork_catalog_path,
    )
    LOGGER.info("[session=%s] Starting editor session", session.session_id)
    server = EditorServer(session=session, host=host, port=port)
    previous_sigint_handler: SignalHandler | int | None = None
    server_started = False

    try:
        if threading.current_thread() is threading.main_thread():
            previous_sigint_handler = signal.getsignal(signal.SIGINT)

            def _handle_sigint(_signum: int, _frame: FrameType | None) -> None:
                """Cancel the session before re-raising Ctrl+C as KeyboardInterrupt."""
                session.cancel()
                raise KeyboardInterrupt

            signal.signal(signal.SIGINT, _handle_sigint)

        server.start()
        server_started = True
        if _on_server_ready is not None:
            _on_server_ready(server.base_url)
        should_print_editor_url = not open_browser
        if open_browser:
            LOGGER.info(
                "[session=%s] Opening browser at %s",
                session.session_id,
                server.base_url,
            )
            try:
                opened = webbrowser.open(server.base_url)
            except Exception:  # pragma: no cover - platform dependent browser errors
                LOGGER.exception(
                    "[session=%s] Failed to open the system browser for the editor.",
                    session.session_id,
                )
                should_print_editor_url = True
            else:
                if not opened:
                    LOGGER.warning(
                        "[session=%s] Browser open request was not acknowledged.",
                        session.session_id,
                    )
                    should_print_editor_url = True
        if should_print_editor_url:
            _print_editor_url(server.base_url)
        return wait_for_editor_result(session)
    except KeyboardInterrupt:
        LOGGER.info(
            "[session=%s] Editor session interrupted by keyboard input",
            session.session_id,
        )
        session.cancel()
        raise
    finally:
        if previous_sigint_handler is not None:
            signal.signal(signal.SIGINT, previous_sigint_handler)
        if server_started:
            server.stop()
