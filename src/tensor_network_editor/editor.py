"""Public helpers for launching the local editor session."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ._themes import DEFAULT_EDITOR_THEME, EditorThemeName, normalize_editor_theme
from .app.session import launch_editor_session
from .internal._logging import (
    DEFAULT_LOG_FILE_BACKUP_COUNT,
    DEFAULT_LOG_FILE_MAX_BYTES,
    get_active_logging_runtime,
    log_operation,
    package_logging_scope,
    summarize_spec_counts,
    validate_positive_log_setting,
)
from .models import (
    EditorResult,
    EngineIdentifier,
    EngineName,
    NetworkSpec,
    TensorCollectionFormat,
)
from .types import StrPath

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class EditorLaunchOptions:
    """Public options for opening the local browser editor."""

    default_engine: EngineIdentifier = EngineName.TENSORKROWCH
    default_collection_format: TensorCollectionFormat = TensorCollectionFormat.LIST
    theme: EditorThemeName = DEFAULT_EDITOR_THEME
    open_browser: bool = True
    host: str = "127.0.0.1"
    port: int = 0
    print_code: bool = False
    code_path: StrPath | None = None
    log_file_path: StrPath | None = None
    log_file_max_bytes: int = DEFAULT_LOG_FILE_MAX_BYTES
    log_file_backup_count: int = DEFAULT_LOG_FILE_BACKUP_COUNT
    template_catalog_path: StrPath | None = None
    subnetwork_catalog_path: StrPath | None = None
    shared_subnetwork_catalog_path: StrPath | None = None
    draft_path: StrPath | None = None
    _on_server_ready: Callable[[str], None] | None = None

    def __post_init__(self) -> None:
        """Normalize and validate theme names passed at runtime."""
        object.__setattr__(self, "theme", normalize_editor_theme(self.theme))
        validate_positive_log_setting(
            self.log_file_max_bytes,
            name="log_file_max_bytes",
        )
        validate_positive_log_setting(
            self.log_file_backup_count,
            name="log_file_backup_count",
        )


def open_editor(
    spec: NetworkSpec | None = None,
    *,
    theme: EditorThemeName | None = None,
    options: EditorLaunchOptions | None = None,
) -> EditorResult | None:
    """Launch the local browser editor and wait for the final session result."""
    resolved_options = options or EditorLaunchOptions()
    resolved_theme = (
        normalize_editor_theme(theme) if theme is not None else resolved_options.theme
    )
    active_logging_runtime = get_active_logging_runtime()
    context: dict[str, object] = {
        "engine": resolved_options.default_engine,
        "mode": resolved_theme,
    }
    if spec is not None:
        spec_context = summarize_spec_counts(spec)
        spec_mode = spec_context.pop("mode", None)
        if spec_mode is not None:
            context["spec_mode"] = spec_mode
        context.update(spec_context)
    should_open_logging_scope = _should_open_editor_logging_scope(
        resolved_options.log_file_path,
        active_logging_runtime,
    )
    logging_scope = (
        package_logging_scope(
            active_logging_runtime.level_name
            if active_logging_runtime is not None
            else None,
            log_file_path=resolved_options.log_file_path,
            log_file_max_bytes=resolved_options.log_file_max_bytes,
            log_file_backup_count=resolved_options.log_file_backup_count,
            enable_stderr=False,
        )
        if should_open_logging_scope
        else _null_logging_scope()
    )
    with logging_scope:
        with log_operation(LOGGER, "Editor launch", context=context):
            return launch_editor_session(
                initial_spec=spec,
                default_engine=resolved_options.default_engine,
                default_collection_format=resolved_options.default_collection_format,
                theme=resolved_theme,
                open_browser=resolved_options.open_browser,
                host=resolved_options.host,
                port=resolved_options.port,
                print_code=resolved_options.print_code,
                code_path=resolved_options.code_path,
                log_file_path=resolved_options.log_file_path,
                log_file_max_bytes=resolved_options.log_file_max_bytes,
                log_file_backup_count=resolved_options.log_file_backup_count,
                template_catalog_path=resolved_options.template_catalog_path,
                subnetwork_catalog_path=resolved_options.subnetwork_catalog_path,
                shared_subnetwork_catalog_path=resolved_options.shared_subnetwork_catalog_path,
                draft_path=resolved_options.draft_path,
                _on_server_ready=resolved_options._on_server_ready,
            )


@dataclass(slots=True, frozen=True)
class _NullLoggingScope:
    """No-op context manager used when editor logging is already active."""

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


def _null_logging_scope() -> _NullLoggingScope:
    """Return a no-op logging context manager."""
    return _NullLoggingScope()


def _should_open_editor_logging_scope(
    log_file_path: StrPath | None,
    active_logging_runtime: object,
) -> bool:
    """Return whether ``open_editor`` should attach its own file handler scope."""
    if log_file_path is None:
        return False
    if active_logging_runtime is None:
        return True
    runtime_log_file_path = getattr(active_logging_runtime, "log_file_path", None)
    if runtime_log_file_path is None:
        return True
    return Path(log_file_path).resolve() != Path(runtime_log_file_path).resolve()


__all__ = ["EditorLaunchOptions", "EditorThemeName", "open_editor"]
