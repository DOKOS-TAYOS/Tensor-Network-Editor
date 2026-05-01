"""Session lifecycle helpers for the local browser editor."""

from __future__ import annotations

import logging
import signal
import threading
import webbrowser
from base64 import b64decode
from collections.abc import Callable, Mapping, Sequence
from importlib import import_module
from pathlib import Path
from types import FrameType
from typing import Any, Literal
from uuid import uuid4

from .._themes import DEFAULT_EDITOR_THEME, EditorThemeName, normalize_editor_theme
from ..codegen.registry import engine_name_to_text
from ..internal._logging import (
    DEFAULT_LOG_FILE_BACKUP_COUNT,
    DEFAULT_LOG_FILE_MAX_BYTES,
    build_frontend_logging_payload,
    get_active_logging_runtime,
    log_branch,
    log_operation,
    package_logging_scope,
    summarize_spec_counts,
)
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
from ._drafts import resolve_project_draft_path
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
SessionUiMode = Literal["browser", "pywebview", "server"]


def _print_editor_url(base_url: str) -> None:
    """Print the local editor URL for manual browser opening."""
    print(f"Open the editor at {base_url}", flush=True)


def _print_browser_open_fallback_message(base_url: str) -> None:
    """Explain that browser opening failed but the local server is still live."""
    print(
        "Could not open the browser automatically. "
        "The local editor server is still running.",
        flush=True,
    )
    _print_editor_url(base_url)


def _import_pywebview() -> Any:
    """Import the optional pywebview module on demand."""
    try:
        return import_module("webview")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pywebview mode requires the optional desktop extra. Install it with "
            'python -m pip install "tensor-network-editor[desktop]".'
        ) from exc


def _resolve_pywebview_icon_path() -> Path:
    """Return the packaged desktop icon used for the native pywebview window."""
    return Path(__file__).resolve().parent / "static" / "favicon.ico"


def _apply_pywebview_native_window_icon(window: Any) -> None:
    """Apply the packaged icon to the native pywebview window when supported."""
    icon_path = _resolve_pywebview_icon_path()
    if not icon_path.is_file():
        return
    native_window = getattr(window, "native", None)
    if native_window is None:
        return
    try:
        from System.Drawing import Icon as DrawingIcon  # type: ignore[import-not-found]
    except Exception:
        return
    try:
        native_window.Icon = DrawingIcon(str(icon_path))
        if hasattr(native_window, "ShowIcon"):
            native_window.ShowIcon = True
    except Exception as exc:
        log_branch(
            LOGGER,
            "Could not apply the native pywebview window icon",
            level=logging.WARNING,
            context={
                "icon_path": str(icon_path),
                "error": str(exc),
            },
        )


class _PywebviewExportApi:
    """Expose native save-file helpers to the embedded pywebview frontend."""

    def __init__(self, pywebview_module: Any) -> None:
        """Store the imported pywebview module for later dialog calls."""
        self._pywebview_module = pywebview_module
        self._window: Any | None = None

    def bind_window(self, window: Any) -> None:
        """Attach the created pywebview window once it exists."""
        self._window = window

    def save_text_file(
        self,
        filename: str,
        text: str,
        content_type: str = "text/plain;charset=utf-8",
    ) -> bool:
        """Prompt for a path and write one UTF-8 text file."""
        del content_type
        output_path = self._select_output_path(filename)
        if output_path is None:
            return False
        output_path.write_text(text, encoding="utf-8")
        return True

    def save_binary_file(
        self,
        filename: str,
        base64_payload: str,
        content_type: str = "application/octet-stream",
    ) -> bool:
        """Prompt for a path and write one decoded binary export file."""
        del content_type
        output_path = self._select_output_path(filename)
        if output_path is None:
            return False
        output_path.write_bytes(b64decode(base64_payload))
        return True

    def _select_output_path(self, filename: str) -> Path | None:
        """Ask pywebview for a target save path and normalize the response."""
        if self._window is None:
            raise RuntimeError("pywebview export API is not bound to a window.")
        dialog_result = self._window.create_file_dialog(
            self._pywebview_module.SAVE_DIALOG,
            save_filename=filename,
            file_types=self._build_file_types(filename),
        )
        if dialog_result is None:
            return None
        if isinstance(dialog_result, str):
            return Path(dialog_result)
        if isinstance(dialog_result, Sequence) and dialog_result:
            first_entry = dialog_result[0]
            if isinstance(first_entry, str) and first_entry:
                return Path(first_entry)
        return None

    def _build_file_types(self, filename: str) -> tuple[str, ...]:
        """Build a compact pywebview filter tuple from one filename."""
        suffix = Path(filename).suffix.lower()
        if not suffix:
            return ()
        label = {
            ".dot": "DOT",
            ".json": "JSON",
            ".mmd": "Mermaid",
            ".pdf": "PDF",
            ".png": "PNG",
            ".py": "Python",
            ".svg": "SVG",
            ".tex": "LaTeX",
        }.get(suffix, suffix.removeprefix(".").upper())
        return (f"{label} (*{suffix})",)


def _run_pywebview_session(
    session: EditorSession, base_url: str
) -> EditorResult | None:
    """Open the local editor in a pywebview window and wait for the result."""
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("pywebview mode must be launched from the main thread.")

    try:
        pywebview = _import_pywebview()
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pywebview mode requires the optional desktop extra. Install it with "
            'python -m pip install "tensor-network-editor[desktop]".'
        ) from exc
    pywebview_export_api = _PywebviewExportApi(pywebview)
    pywebview_window = pywebview.create_window(
        "Tensor Network Editor",
        base_url,
        maximized=True,
        js_api=pywebview_export_api,
    )
    pywebview_export_api.bind_window(pywebview_window)
    pywebview_window.events.before_show += lambda: _apply_pywebview_native_window_icon(
        pywebview_window
    )

    def _handle_window_closed(*_args: object) -> None:
        """Cancel the editor session when the native window is closed."""
        if not session.is_finished():
            session.cancel()

    def _wait_for_session_and_close_window(window: Any) -> None:
        """Close the native window after the editor session finishes."""
        wait_for_editor_result(session)
        try:
            window.destroy()
        except Exception:
            return None

    pywebview_window.events.closed += _handle_window_closed
    pywebview.start(_wait_for_session_and_close_window, pywebview_window)
    return wait_for_editor_result(session)


class EditorSession:
    """Mutable session state shared between the HTTP server and the caller."""

    def __init__(
        self,
        initial_spec: NetworkSpec | None = None,
        default_engine: EngineIdentifier = EngineName.TENSORKROWCH,
        default_collection_format: TensorCollectionFormat = TensorCollectionFormat.LIST,
        *,
        theme: EditorThemeName = DEFAULT_EDITOR_THEME,
        print_code: bool = False,
        code_path: StrPath | None = None,
        template_catalog_path: StrPath | None = None,
        subnetwork_catalog_path: StrPath | None = None,
        shared_subnetwork_catalog_path: StrPath | None = None,
        draft_path: StrPath | None = None,
    ) -> None:
        """Initialize one mutable editor session.

        Args:
            initial_spec: Optional network specification to preload.
            default_engine: Backend initially selected in the editor.
            default_collection_format: Initial tensor collection layout for
                generated code.
            theme: Visual theme selected for this editor session.
            print_code: Whether to print generated code after confirmation.
            code_path: Optional output path for generated code after
                confirmation.
            template_catalog_path: Optional per-project static template catalog
                path.
            subnetwork_catalog_path: Optional per-project reusable subnetwork
                catalog path.
            shared_subnetwork_catalog_path: Optional shared reusable subnetwork
                catalog path.
            draft_path: Optional path for the recoverable project draft.
        """
        self.initial_spec = initial_spec or build_blank_network_spec()
        self.session_id = uuid4().hex[:8]
        self.default_engine = default_engine
        self.default_collection_format = default_collection_format
        self.theme = normalize_editor_theme(theme)
        self.print_code = print_code
        self.code_path = code_path
        self.draft_path = resolve_project_draft_path(draft_path)
        self.frontend_logging_payload = build_frontend_logging_payload()
        self._catalog_state = SessionCatalogState.load(
            template_catalog_path=template_catalog_path,
            subnetwork_catalog_path=subnetwork_catalog_path,
            shared_subnetwork_catalog_path=shared_subnetwork_catalog_path,
        )
        self._finished_event = threading.Event()
        self._result: EditorResult | None = None
        self._lock = threading.Lock()
        log_branch(
            LOGGER,
            "Initialized editor session",
            context={
                "session": self.session_id,
                "engine": engine_name_to_text(self.default_engine),
                **summarize_spec_counts(self.initial_spec),
            },
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
        include_roundtrip_metadata: bool = True,
    ) -> CodegenResult:
        """Generate preview code without finalizing the session."""
        with log_operation(
            LOGGER,
            "Session preview generation",
            context={
                "session": self.session_id,
                "engine": engine_name_to_text(engine),
                "format": collection_format,
            },
        ):
            return generate_session_request(
                self,
                serialized_spec,
                engine,
                collection_format,
                include_roundtrip_metadata,
            )

    def complete(
        self,
        serialized_spec: Mapping[str, object],
        engine: EngineIdentifier,
        collection_format: TensorCollectionFormat | None = None,
        include_roundtrip_metadata: bool = True,
    ) -> EditorResult:
        """Finalize the session and store the resulting editor output."""
        with log_operation(
            LOGGER,
            "Session completion",
            start_level=logging.INFO,
            success_level=logging.INFO,
            context={
                "session": self.session_id,
                "engine": engine_name_to_text(engine),
                "format": collection_format,
            },
        ):
            with self._lock:
                if self._finished_event.is_set() and self._result is not None:
                    log_branch(LOGGER, "Duplicate completion request ignored")
                    return self._result
            result = complete_session_request(
                self,
                serialized_spec,
                engine,
                collection_format,
                include_roundtrip_metadata,
            )
            with self._lock:
                if self._finished_event.is_set() and self._result is not None:
                    log_branch(
                        LOGGER,
                        "Duplicate completion request returned existing result",
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
        with log_operation(
            LOGGER,
            "Session cancel",
            start_level=logging.DEBUG,
            success_level=logging.INFO,
            context={"session": self.session_id},
        ):
            with self._lock:
                if self._finished_event.is_set():
                    log_branch(LOGGER, "Cancel request ignored for finished session")
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

    def is_finished(self) -> bool:
        """Return whether the session has already completed or been cancelled."""
        return self._finished_event.is_set()


def launch_editor_session(
    initial_spec: NetworkSpec | None = None,
    *,
    default_engine: EngineIdentifier = EngineName.TENSORKROWCH,
    default_collection_format: TensorCollectionFormat = TensorCollectionFormat.LIST,
    theme: EditorThemeName = DEFAULT_EDITOR_THEME,
    ui_mode: SessionUiMode | None = None,
    open_browser: bool = True,
    host: str = "127.0.0.1",
    port: int = 0,
    print_code: bool = False,
    code_path: StrPath | None = None,
    log_file_path: StrPath | None = None,
    log_file_max_bytes: int = DEFAULT_LOG_FILE_MAX_BYTES,
    log_file_backup_count: int = DEFAULT_LOG_FILE_BACKUP_COUNT,
    template_catalog_path: StrPath | None = None,
    subnetwork_catalog_path: StrPath | None = None,
    shared_subnetwork_catalog_path: StrPath | None = None,
    draft_path: StrPath | None = None,
    _on_server_ready: Callable[[str], None] | None = None,
) -> EditorResult | None:
    """Create the local server, optionally open the browser, and wait.

    Args:
        initial_spec: Optional network specification to preload.
        default_engine: Backend initially selected in the editor UI.
        default_collection_format: Initial tensor collection layout for
            generated code.
        theme: Visual theme selected for this editor session.
        ui_mode: Explicit UI launch mode for the editor session.
        open_browser: Whether to ask the system browser to open the local URL.
        host: Local host interface to bind.
        port: Local port to bind. Use ``0`` for an ephemeral port.
        print_code: Whether to print generated code after confirmation.
        code_path: Optional output path for generated code after confirmation.
        log_file_path: Optional log file path used for this editor session.
        log_file_max_bytes: Maximum active log-file size before rotation.
        log_file_backup_count: Number of rotated log backups to retain.
        template_catalog_path: Optional per-project static template catalog
            path.
        subnetwork_catalog_path: Optional per-project reusable subnetwork
            catalog path.
        shared_subnetwork_catalog_path: Optional shared reusable subnetwork
            catalog path.
        draft_path: Optional path for the recoverable project draft.
        _on_server_ready: Internal callback used by tests once the local URL is
            available.

    Returns:
        ``None`` when the session is cancelled, otherwise the confirmed editor
        result.

    Raises:
        KeyboardInterrupt: If the session is interrupted from the main thread.
    """
    from ..editor import resolve_editor_ui_mode
    from .server import EditorServer

    active_logging_runtime = get_active_logging_runtime()
    logging_scope = (
        package_logging_scope(
            active_logging_runtime.level_name
            if active_logging_runtime is not None
            else None,
            log_file_path=log_file_path,
            log_file_max_bytes=log_file_max_bytes,
            log_file_backup_count=log_file_backup_count,
            enable_stderr=False,
        )
        if _should_open_session_logging_scope(log_file_path, active_logging_runtime)
        else _NullLoggingScope()
    )

    with logging_scope:
        session = EditorSession(
            initial_spec=initial_spec,
            default_engine=default_engine,
            default_collection_format=default_collection_format,
            theme=theme,
            print_code=print_code,
            code_path=code_path,
            template_catalog_path=template_catalog_path,
            subnetwork_catalog_path=subnetwork_catalog_path,
            shared_subnetwork_catalog_path=shared_subnetwork_catalog_path,
            draft_path=draft_path,
        )
        server = EditorServer(session=session, host=host, port=port)
        effective_ui_mode = resolve_editor_ui_mode(
            ui_mode=ui_mode,
            open_browser=open_browser,
        )
        previous_sigint_handler: SignalHandler | int | None = None
        server_started = False

        try:
            with log_operation(
                LOGGER,
                "Editor session launch",
                start_level=logging.INFO,
                success_level=logging.INFO,
                context={
                    "session": session.session_id,
                    "engine": engine_name_to_text(default_engine),
                    "mode": theme,
                    "ui_mode": effective_ui_mode,
                },
            ):
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
                if effective_ui_mode == "pywebview":
                    return _run_pywebview_session(session, server.base_url)
                should_print_editor_url = effective_ui_mode == "server"
                should_print_browser_fallback_message = False
                if effective_ui_mode == "browser":
                    try:
                        with log_operation(
                            LOGGER,
                            "Browser open attempt",
                            start_level=logging.INFO,
                            success_level=logging.INFO,
                            failure_level=logging.ERROR,
                            context={
                                "session": session.session_id,
                                "url": server.base_url,
                            },
                        ):
                            opened = webbrowser.open(server.base_url)
                    except (
                        Exception
                    ):  # pragma: no cover - platform dependent browser errors
                        should_print_editor_url = True
                        should_print_browser_fallback_message = True
                    else:
                        if not opened:
                            log_branch(
                                LOGGER,
                                "Browser open attempt was not acknowledged",
                                level=logging.WARNING,
                                context={
                                    "session": session.session_id,
                                    "url": server.base_url,
                                },
                            )
                            should_print_editor_url = True
                            should_print_browser_fallback_message = True
                if should_print_browser_fallback_message:
                    _print_browser_open_fallback_message(server.base_url)
                elif should_print_editor_url:
                    _print_editor_url(server.base_url)
                return wait_for_editor_result(session)
        except KeyboardInterrupt:
            log_branch(
                LOGGER,
                "Editor session interrupted by keyboard input",
                level=logging.INFO,
                context={"session": session.session_id},
            )
            session.cancel()
            raise
        finally:
            if previous_sigint_handler is not None:
                signal.signal(signal.SIGINT, previous_sigint_handler)
            if server_started:
                server.stop()


class _NullLoggingScope:
    """No-op context manager used when session logging is already active."""

    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> Literal[False]:
        del exc_type, exc, traceback
        return False


def _should_open_session_logging_scope(
    log_file_path: StrPath | None,
    active_logging_runtime: object,
) -> bool:
    """Return whether ``launch_editor_session`` should attach its own log scope."""
    if log_file_path is None:
        return False
    if active_logging_runtime is None:
        return True
    runtime_log_file_path = getattr(active_logging_runtime, "log_file_path", None)
    if runtime_log_file_path is None:
        return True
    return Path(log_file_path).resolve() != Path(runtime_log_file_path).resolve()
