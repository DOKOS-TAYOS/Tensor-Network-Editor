"""Runtime helpers shared by editor session entrypoints."""

from __future__ import annotations

import warnings
from typing import Protocol

from ..models import EditorResult, NetworkSpec

_DEPRECATED_POLL_INTERVAL_SENTINEL = object()
_POLL_INTERVAL_REMOVAL_DATE = "2026-10-01"
_EDITOR_WAIT_POLL_INTERVAL_SECONDS = 0.1


class SupportsWaitForResult(Protocol):
    """Protocol implemented by session-like objects that can wait for results."""

    def wait_for_result(self, timeout: float | None = None) -> EditorResult | None:
        """Wait for the final editor result or ``None`` on timeout."""
        ...

    def is_finished(self) -> bool:
        """Return whether the session has already finished."""
        ...


def build_blank_network_spec() -> NetworkSpec:
    """Build the default empty network shown in a new editor session."""
    return NetworkSpec(name="Untitled Network")


def wait_for_editor_result(
    session: SupportsWaitForResult,
    *,
    poll_interval: float | object = _DEPRECATED_POLL_INTERVAL_SENTINEL,
) -> EditorResult | None:
    """Wait for an editor session result using the session's blocking API."""
    if poll_interval is not _DEPRECATED_POLL_INTERVAL_SENTINEL:
        warnings.warn(
            "wait_for_editor_result(..., poll_interval=...) is deprecated and has "
            "no effect. Remove this argument; it will be removed on "
            f"{_POLL_INTERVAL_REMOVAL_DATE}.",
            DeprecationWarning,
            stacklevel=2,
        )
    while True:
        result = session.wait_for_result(timeout=_EDITOR_WAIT_POLL_INTERVAL_SECONDS)
        if result is not None:
            return result
        if session.is_finished():
            return None
