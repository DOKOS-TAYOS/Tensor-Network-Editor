"""Shared logging helpers for package-wide observability."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

from ..types import StrPath

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping


PACKAGE_LOGGER_NAME = "tensor_network_editor"
FRONTEND_LOGGER_NAME = f"{PACKAGE_LOGGER_NAME}.frontend"
FRONTEND_LOG_TRANSPORT_ENDPOINT = "/api/client-log"
DEFAULT_LOG_FILE_MAX_BYTES = 10_485_760
DEFAULT_LOG_FILE_BACKUP_COUNT = 5
LOG_LEVEL_NAMES: tuple[str, ...] = (
    "critical",
    "error",
    "warning",
    "info",
    "debug",
)
LOG_LEVEL_VALUES: dict[str, int] = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}
_STREAM_HANDLER_NAME = "tensor_network_editor_stream_handler"
_FILE_HANDLER_NAME_PREFIX = "tensor_network_editor_file_handler:"
_STREAM_FORMAT = "%(levelname)s %(name)s: %(message)s"
_FILE_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_LOG_CONTEXT: ContextVar[tuple[tuple[str, str], ...]] = ContextVar(
    "tensor_network_editor_log_context",
    default=(),
)
_ACTIVE_LOGGING_RUNTIME: ContextVar[LoggingRuntimeConfig | None] = ContextVar(
    "tensor_network_editor_active_logging_runtime",
    default=None,
)
_CONTEXT_FIELD_ORDER: tuple[str, ...] = (
    "command",
    "path",
    "max_bytes",
    "backup_count",
    "url",
    "before",
    "after",
    "engine",
    "format",
    "export_format",
    "output_path",
    "python_import_mode",
    "source_profile",
    "python_reconstruction_level",
    "memory_dtype",
    "mode",
    "spec_mode",
    "session",
    "route",
    "request_id",
    "refresh_reason",
    "cache_state",
    "analysis_status",
    "analysis_source",
    "planner_mode",
    "benchmark_position",
    "scheme_count",
    "template_name",
    "subnetwork_name",
    "selected_template",
    "selected_subnetwork",
    "tensor_id_count",
    "tag_count",
    "warning_count",
    "issue_count",
    "asset_count",
    "client_ts_ms",
    "status",
    "outcome",
    "elapsed_ms",
)


@dataclass(slots=True, frozen=True)
class LoggingRuntimeConfig:
    """Resolved runtime logging configuration for one scoped session."""

    level_name: str | None
    log_file_path: Path | None
    log_file_max_bytes: int
    log_file_backup_count: int
    stderr_enabled: bool
    frontend_enabled: bool
    frontend_level: str
    frontend_persist: bool
    frontend_transport_endpoint: str

    def frontend_payload(self) -> dict[str, object]:
        """Return the frontend runtime payload derived from this scope."""
        return {
            "enabled": self.frontend_enabled,
            "level": self.frontend_level,
            "persist": self.frontend_persist,
            "transport_endpoint": self.frontend_transport_endpoint,
        }


@contextmanager
def package_logging_scope(
    requested_level_name: str | None,
    *,
    log_file_path: StrPath | None = None,
    log_file_max_bytes: int = DEFAULT_LOG_FILE_MAX_BYTES,
    log_file_backup_count: int = DEFAULT_LOG_FILE_BACKUP_COUNT,
    enable_stderr: bool = True,
) -> Iterator[LoggingRuntimeConfig | None]:
    """Temporarily attach package logging handlers for one controlled scope."""
    resolved_log_file_path = (
        Path(log_file_path).resolve() if log_file_path is not None else None
    )
    runtime_config = build_logging_runtime_config(
        requested_level_name,
        log_file_path=resolved_log_file_path,
        log_file_max_bytes=log_file_max_bytes,
        log_file_backup_count=log_file_backup_count,
        stderr_enabled=enable_stderr,
    )
    if runtime_config.level_name is None and runtime_config.log_file_path is None:
        yield None
        return

    package_logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    original_level = package_logger.level
    original_propagate = package_logger.propagate
    added_handlers: list[logging.Handler] = []

    if runtime_config.level_name is not None:
        package_logger.setLevel(LOG_LEVEL_VALUES[runtime_config.level_name])
    package_logger.propagate = False

    if runtime_config.stderr_enabled and runtime_config.level_name is not None:
        stream_handler = _create_stream_handler(runtime_config.level_name)
        package_logger.addHandler(stream_handler)
        added_handlers.append(stream_handler)

    if (
        runtime_config.log_file_path is not None
        and runtime_config.level_name is not None
    ):
        file_handler = _create_file_handler(
            runtime_config.log_file_path,
            runtime_config.level_name,
            runtime_config.log_file_max_bytes,
            runtime_config.log_file_backup_count,
        )
        package_logger.addHandler(file_handler)
        added_handlers.append(file_handler)
        log_branch(
            package_logger,
            "Configured persistent file logging",
            level=logging.INFO,
            context={
                "path": runtime_config.log_file_path,
                "max_bytes": runtime_config.log_file_max_bytes,
                "backup_count": runtime_config.log_file_backup_count,
            },
        )

    token = _ACTIVE_LOGGING_RUNTIME.set(runtime_config)
    try:
        yield runtime_config
    finally:
        _ACTIVE_LOGGING_RUNTIME.reset(token)
        for handler in added_handlers:
            package_logger.removeHandler(handler)
            handler.close()
        package_logger.setLevel(original_level)
        package_logger.propagate = original_propagate


def build_logging_runtime_config(
    requested_level_name: str | None,
    *,
    log_file_path: Path | None,
    log_file_max_bytes: int,
    log_file_backup_count: int,
    stderr_enabled: bool,
) -> LoggingRuntimeConfig:
    """Resolve the effective logging behavior for one scope."""
    normalized_log_file_max_bytes = validate_positive_log_setting(
        log_file_max_bytes,
        name="log_file_max_bytes",
    )
    normalized_log_file_backup_count = validate_positive_log_setting(
        log_file_backup_count,
        name="log_file_backup_count",
    )
    level_name = _resolve_effective_log_level_name(
        requested_level_name,
        log_file_path=log_file_path,
    )
    frontend_level = _normalize_frontend_log_level(level_name)
    frontend_enabled = frontend_level != "off"
    return LoggingRuntimeConfig(
        level_name=level_name,
        log_file_path=log_file_path,
        log_file_max_bytes=normalized_log_file_max_bytes,
        log_file_backup_count=normalized_log_file_backup_count,
        stderr_enabled=stderr_enabled and requested_level_name is not None,
        frontend_enabled=frontend_enabled,
        frontend_level=frontend_level,
        frontend_persist=log_file_path is not None,
        frontend_transport_endpoint=FRONTEND_LOG_TRANSPORT_ENDPOINT,
    )


def get_active_logging_runtime() -> LoggingRuntimeConfig | None:
    """Return the active scoped logging configuration when one exists."""
    return _ACTIVE_LOGGING_RUNTIME.get()


def build_frontend_logging_payload(
    runtime_config: LoggingRuntimeConfig | None = None,
) -> dict[str, object]:
    """Return the frontend logging payload for the active or provided scope."""
    resolved_runtime = (
        runtime_config if runtime_config is not None else get_active_logging_runtime()
    )
    if resolved_runtime is not None:
        return resolved_runtime.frontend_payload()
    frontend_debug_enabled = logging.getLogger(PACKAGE_LOGGER_NAME).isEnabledFor(
        logging.DEBUG
    )
    return {
        "enabled": frontend_debug_enabled,
        "level": "debug" if frontend_debug_enabled else "off",
        "persist": False,
        "transport_endpoint": FRONTEND_LOG_TRANSPORT_ENDPOINT,
    }


@contextmanager
def bind_log_context(**fields: object) -> Iterator[None]:
    """Temporarily bind one or more context fields to nested logs."""
    normalized_fields = {
        key: normalized
        for key, value in fields.items()
        if (normalized := _normalize_context_value(value)) is not None
    }
    if not normalized_fields:
        yield
        return
    merged_fields: dict[str, str] = dict(_LOG_CONTEXT.get())
    merged_fields.update(normalized_fields)
    token = _LOG_CONTEXT.set(tuple(merged_fields.items()))
    try:
        yield
    finally:
        _LOG_CONTEXT.reset(token)


@contextmanager
def log_operation(
    logger: logging.Logger,
    operation_name: str,
    *,
    start_level: int = logging.DEBUG,
    success_level: int = logging.DEBUG,
    failure_level: int = logging.WARNING,
    context: Mapping[str, object] | None = None,
    emit_start: bool = True,
) -> Iterator[dict[str, object]]:
    """Log a start/finish/failure lifecycle around one operation."""
    operation_context = dict(context or {})
    success_context: dict[str, object] = {}
    with bind_log_context(**operation_context):
        if emit_start:
            logger.log(start_level, format_log_message(f"{operation_name} started"))
        start_time = perf_counter()
        try:
            yield success_context
        except Exception as exc:
            elapsed_ms = _elapsed_ms_text(start_time)
            with bind_log_context(outcome="error", elapsed_ms=elapsed_ms):
                logger.log(
                    failure_level,
                    format_log_message(f"{operation_name} failed: {exc}"),
                    exc_info=logger.isEnabledFor(logging.DEBUG),
                )
            raise
        else:
            elapsed_ms = _elapsed_ms_text(start_time)
            completed_context = dict(success_context)
            outcome = completed_context.pop("outcome", "success")
            with bind_log_context(
                outcome=outcome,
                elapsed_ms=elapsed_ms,
                **completed_context,
            ):
                logger.log(
                    success_level, format_log_message(f"{operation_name} finished")
                )


def log_branch(
    logger: logging.Logger,
    message: str,
    *,
    level: int = logging.DEBUG,
    context: Mapping[str, object] | None = None,
) -> None:
    """Log one branch or decision message with optional extra context."""
    with bind_log_context(**dict(context or {})):
        logger.log(level, format_log_message(message))


def format_log_message(
    message: str,
    *,
    context: Mapping[str, object] | None = None,
) -> str:
    """Return ``message`` with ordered context suffix fields when present."""
    merged: dict[str, str] = dict(_LOG_CONTEXT.get())
    for key, value in dict(context or {}).items():
        normalized = _normalize_context_value(value)
        if normalized is not None:
            merged[key] = normalized
    ordered: dict[str, str] = {}
    for key in _CONTEXT_FIELD_ORDER:
        value = merged.pop(key, None)
        if isinstance(value, str):
            ordered[key] = value
    for key in sorted(merged):
        ordered[key] = merged[key]
    if not ordered:
        return message
    suffix = " ".join(f"{key}={value}" for key, value in ordered.items())
    return f"{message} {suffix}"


def summarize_spec_counts(spec: object) -> dict[str, object]:
    """Return a compact entity summary for objects that look like ``NetworkSpec``."""
    tensors = getattr(spec, "tensors", ())
    edges = getattr(spec, "edges", ())
    hyperedges = getattr(spec, "hyperedges", ())
    groups = getattr(spec, "groups", ())
    notes = getattr(spec, "notes", ())
    return {
        "tensor_count": len(tensors),
        "edge_count": len(edges),
        "hyperedge_count": len(hyperedges),
        "group_count": len(groups),
        "note_count": len(notes),
        "mode": _spec_mode(spec),
    }


def summarize_contraction_analysis(result: object) -> dict[str, object]:
    """Return a compact summary for contraction-analysis style payloads."""
    manual = getattr(result, "manual", None)
    automatic_full = getattr(result, "automatic_full", None)
    automatic_future = getattr(result, "automatic_future", None)
    automatic_past = getattr(result, "automatic_past", None)
    warnings = getattr(result, "warnings", ())
    manual_steps = getattr(manual, "steps", ())
    return {
        "analysis_status": "ready",
        "warning_count": len(warnings),
        "manual_step_count": len(manual_steps),
        "manual_status": getattr(manual, "status", None),
        "automatic_full_status": getattr(automatic_full, "status", None),
        "automatic_future_status": getattr(automatic_future, "status", None),
        "automatic_past_status": getattr(automatic_past, "status", None),
    }


def validate_positive_log_setting(value: int, *, name: str) -> int:
    """Validate one positive integer used by persistent log rotation."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be > 0.")
    if value <= 0:
        raise ValueError(f"{name} must be > 0.")
    return value


def _resolve_effective_log_level_name(
    requested_level_name: str | None,
    *,
    log_file_path: Path | None,
) -> str | None:
    if requested_level_name is not None:
        return requested_level_name
    if log_file_path is not None:
        return "debug"
    return None


def _normalize_frontend_log_level(level_name: str | None) -> str:
    if level_name in {"debug", "info", "warning"}:
        return level_name
    if level_name in {"critical", "error"}:
        return "error"
    return "off"


def _create_stream_handler(level_name: str) -> logging.Handler:
    stream_handler = logging.StreamHandler()
    stream_handler.name = _STREAM_HANDLER_NAME
    stream_handler.setLevel(LOG_LEVEL_VALUES[level_name])
    stream_handler.setFormatter(logging.Formatter(_STREAM_FORMAT))
    return stream_handler


def _create_file_handler(
    path: Path,
    level_name: str,
    max_bytes: int,
    backup_count: int,
) -> logging.Handler:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.name = f"{_FILE_HANDLER_NAME_PREFIX}{path}"
    file_handler.setLevel(LOG_LEVEL_VALUES[level_name])
    file_handler.setFormatter(logging.Formatter(_FILE_FORMAT))
    return file_handler


def _normalize_context_value(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _elapsed_ms_text(start_time: float) -> str:
    return str(int(round((perf_counter() - start_time) * 1000)))


def _spec_mode(spec: object) -> str:
    if getattr(spec, "linear_periodic_chain", None) is not None:
        return "linear_periodic"
    if getattr(spec, "grid_periodic_grid", None) is not None:
        return "grid_periodic"
    if getattr(spec, "tree_periodic_tree", None) is not None:
        return "tree_periodic"
    return "normal"
