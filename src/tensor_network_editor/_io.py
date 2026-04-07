"""UTF-8 text I/O helpers that raise package-specific exceptions."""

from __future__ import annotations

import logging
from pathlib import Path

from .errors import PackageIOError
from .types import StrPath

LOGGER = logging.getLogger(__name__)


def read_utf8_text(path: StrPath, *, description: str) -> str:
    """Read a UTF-8 text file and wrap filesystem errors."""
    target_path = Path(path)
    LOGGER.info("Reading %s from %s", description, target_path)
    try:
        return target_path.read_text(encoding="utf-8")
    except OSError as exc:
        message = f"Could not read {description} from '{target_path}': {exc}"
        LOGGER.warning(message)
        raise PackageIOError(message) from exc


def write_utf8_text(path: StrPath, content: str, *, description: str) -> None:
    """Write UTF-8 text to disk and wrap filesystem errors."""
    target_path = Path(path)
    LOGGER.info("Writing %s to %s", description, target_path)
    try:
        target_path.write_text(content, encoding="utf-8")
    except OSError as exc:
        message = f"Could not write {description} to '{target_path}': {exc}"
        LOGGER.warning(message)
        raise PackageIOError(message) from exc
