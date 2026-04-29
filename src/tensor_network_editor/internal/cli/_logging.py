"""Opt-in logging helpers and runtime diagnostics for the package."""

from __future__ import annotations

import importlib.metadata
import json
import logging
import os
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

from ..._version import __version__
from .._logging import (
    DEFAULT_LOG_FILE_BACKUP_COUNT,
    DEFAULT_LOG_FILE_MAX_BYTES,
    LOG_LEVEL_NAMES,
    LOG_LEVEL_VALUES,
    PACKAGE_LOGGER_NAME,
    LoggingRuntimeConfig,
    package_logging_scope,
)

ENV_LOG_LEVEL = "TNE_LOG_LEVEL"


@dataclass(slots=True, frozen=True)
class RuntimeDiagnostics:
    """Resolved runtime metadata that helps debug environment mismatches."""

    python_executable: Path
    cwd: Path
    package_path: Path
    version: str
    current_checkout_root: Path | None
    editable_install_root: Path | None

    def checkout_mismatch_message(self) -> str | None:
        """Return a warning when the current checkout and editable root differ."""
        current_checkout_root = self.current_checkout_root
        editable_install_root = self.editable_install_root
        if current_checkout_root is None or editable_install_root is None:
            return None
        if current_checkout_root == editable_install_root:
            return None
        return (
            "Active editable install points to "
            f"'{editable_install_root}' while the current checkout is "
            f"'{current_checkout_root}'."
        )


def resolve_log_level_name(
    cli_log_level: str | None,
    *,
    env: Mapping[str, str] | None = None,
) -> str | None:
    """Resolve the requested log level from CLI and environment inputs."""
    if cli_log_level is not None:
        return _normalize_log_level_name(cli_log_level)
    current_env = env if env is not None else os.environ
    env_log_level = current_env.get(ENV_LOG_LEVEL)
    if env_log_level is None:
        return None
    return _normalize_log_level_name(env_log_level)


@contextmanager
def configure_package_logging(
    cli_log_level: str | None,
    *,
    log_file_path: str | Path | None = None,
    log_file_max_bytes: int = DEFAULT_LOG_FILE_MAX_BYTES,
    log_file_backup_count: int = DEFAULT_LOG_FILE_BACKUP_COUNT,
    env: Mapping[str, str] | None = None,
) -> Iterator[LoggingRuntimeConfig | None]:
    """Configure package logging for one CLI execution scope."""
    resolved_level_name = resolve_log_level_name(cli_log_level, env=env)
    with package_logging_scope(
        resolved_level_name,
        log_file_path=log_file_path,
        log_file_max_bytes=log_file_max_bytes,
        log_file_backup_count=log_file_backup_count,
        enable_stderr=resolved_level_name is not None,
    ) as runtime_config:
        yield runtime_config


def collect_runtime_diagnostics() -> RuntimeDiagnostics:
    """Collect runtime details that help explain editable-install confusion."""
    package_path = Path(__file__).resolve().parents[2]
    return RuntimeDiagnostics(
        python_executable=Path(sys.executable).resolve(),
        cwd=Path.cwd().resolve(),
        package_path=package_path,
        version=__version__,
        current_checkout_root=find_checkout_root(Path.cwd()),
        editable_install_root=find_editable_install_root(),
    )


def emit_runtime_diagnostics(log_level_name: str | None) -> None:
    """Log a compact runtime diagnostic summary when logging is enabled."""
    if log_level_name is None:
        return

    diagnostics = collect_runtime_diagnostics()
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    logger.log(
        LOG_LEVEL_VALUES[log_level_name],
        (
            "Runtime diagnostics: python=%s cwd=%s package=%s version=%s "
            "current_checkout=%s editable_install=%s"
        ),
        diagnostics.python_executable,
        diagnostics.cwd,
        diagnostics.package_path,
        diagnostics.version,
        diagnostics.current_checkout_root,
        diagnostics.editable_install_root,
    )
    mismatch_message = diagnostics.checkout_mismatch_message()
    if mismatch_message is not None:
        logger.warning(mismatch_message)


def find_checkout_root(start: Path) -> Path | None:
    """Return the nearest checkout root that looks like this project."""
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (
            (candidate / "pyproject.toml").is_file()
            and (candidate / "src").is_dir()
            and (candidate / "tests").is_dir()
        ):
            return candidate
    return None


def find_editable_install_root() -> Path | None:
    """Return the editable install root when the current package was installed so."""
    try:
        distribution = importlib.metadata.distribution("tensor-network-editor")
    except importlib.metadata.PackageNotFoundError:
        return None

    direct_url_payload = distribution.read_text("direct_url.json")
    if direct_url_payload is None:
        return None

    try:
        payload = json.loads(direct_url_payload)
    except json.JSONDecodeError:
        return None

    dir_info = payload.get("dir_info")
    if not isinstance(dir_info, dict) or dir_info.get("editable") is not True:
        return None

    url = payload.get("url")
    if not isinstance(url, str):
        return None

    parsed_url = urlparse(url)
    if parsed_url.scheme != "file":
        return None

    local_path = Path(url2pathname(parsed_url.path)).resolve()
    return local_path


def _normalize_log_level_name(raw_value: str) -> str:
    normalized_value = raw_value.strip().lower()
    if normalized_value not in LOG_LEVEL_VALUES:
        supported_levels = ", ".join(LOG_LEVEL_NAMES)
        raise ValueError(
            f"Unsupported log level '{raw_value}'. Expected one of: {supported_levels}."
        )
    return normalized_value
