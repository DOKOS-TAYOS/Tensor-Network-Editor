from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from tensor_network_editor.cli import main
from tensor_network_editor.internal.cli import _logging as logging_support


@pytest.fixture(autouse=True)
def restore_package_logger() -> Iterator[None]:
    package_logger = logging.getLogger(logging_support.PACKAGE_LOGGER_NAME)
    original_handlers = list(package_logger.handlers)
    original_level = package_logger.level
    original_propagate = package_logger.propagate
    try:
        yield
    finally:
        package_logger.handlers.clear()
        for handler in original_handlers:
            package_logger.addHandler(handler)
        package_logger.setLevel(original_level)
        package_logger.propagate = original_propagate


def test_resolve_log_level_name_prefers_cli_over_environment() -> None:
    resolved_level = logging_support.resolve_log_level_name(
        "debug",
        env={logging_support.ENV_LOG_LEVEL: "error"},
    )

    assert resolved_level == "debug"


def test_resolve_log_level_name_uses_environment_when_cli_missing() -> None:
    resolved_level = logging_support.resolve_log_level_name(
        None,
        env={logging_support.ENV_LOG_LEVEL: "info"},
    )

    assert resolved_level == "info"


def test_resolve_log_level_name_returns_none_when_unset() -> None:
    resolved_level = logging_support.resolve_log_level_name(None, env={})

    assert resolved_level is None


def test_collect_runtime_diagnostics_reports_current_package_and_python() -> None:
    diagnostics = logging_support.collect_runtime_diagnostics()

    assert diagnostics.python_executable == Path(sys.executable).resolve()
    assert (
        diagnostics.package_path
        == (Path.cwd().resolve() / "src" / "tensor_network_editor").resolve()
    )


def test_runtime_diagnostics_detect_mismatched_editable_checkout() -> None:
    diagnostics = logging_support.RuntimeDiagnostics(
        python_executable=Path(sys.executable).resolve(),
        cwd=Path.cwd().resolve(),
        package_path=(Path.cwd().resolve() / "src" / "tensor_network_editor"),
        version="0.0.0",
        current_checkout_root=Path("C:/repo/.worktrees/current").resolve(),
        editable_install_root=Path("C:/repo/.worktrees/other").resolve(),
    )

    mismatch_message = diagnostics.checkout_mismatch_message()

    assert mismatch_message is not None
    assert "Active editable install points to" in mismatch_message
    assert "current checkout" in mismatch_message


def test_main_emits_runtime_diagnostics_when_log_level_is_requested(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["--log-level", "info", "template", "list", "--format", "json"])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Runtime diagnostics:" in captured.err


def test_main_uses_tne_log_level_when_flag_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(logging_support.ENV_LOG_LEVEL, "info")

    exit_code = main(["template", "list", "--format", "json"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Runtime diagnostics:" in captured.err


def test_main_cli_log_level_overrides_environment(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(logging_support.ENV_LOG_LEVEL, "error")

    exit_code = main(["--log-level", "debug", "template", "list", "--format", "json"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "DEBUG tensor_network_editor: Runtime diagnostics:" in captured.err


def test_main_stays_silent_without_explicit_logging(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["template", "list", "--format", "json"])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
