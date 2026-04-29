from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest

from tensor_network_editor import analyze_spec, generate_code
from tensor_network_editor.cli import main
from tensor_network_editor.errors import SerializationError
from tensor_network_editor.internal.cli import _cli_parser as cli_parser
from tensor_network_editor.internal.cli import _logging as logging_support
from tensor_network_editor.io import SCHEMA_VERSION, PythonLoadOptions, load_python_spec
from tensor_network_editor.models import EngineName
from tensor_network_editor.rendering import render_spec_tikz
from tests.factories import build_sample_spec, build_tree_periodic_tree_spec


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


def _read_rotated_log_family(log_file_path: Path) -> str:
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(log_file_path.parent.glob(f"{log_file_path.name}*"))
        if path.is_file()
    )


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


def test_main_debug_logs_cli_command_lifecycle(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["--log-level", "debug", "template", "list", "--format", "json"])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "CLI command started" in captured.err
    assert "CLI command finished" in captured.err
    assert "command=template.list" in captured.err
    assert "outcome=success" in captured.err
    assert "elapsed_ms=" in captured.err


def test_main_debug_logs_benchmark_command_story(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec_path = tmp_path / "benchmark-network.json"
    spec_path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "network": build_sample_spec().to_dict(),
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--log-level",
            "debug",
            "benchmark",
            str(spec_path),
            "--format",
            "json",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Benchmark command started" in captured.err
    assert "Built benchmark report" in captured.err
    assert "analysis_status=ready" in captured.err
    assert "scheme_count=4" in captured.err
    assert "export_format=json" in captured.err


def test_main_log_file_persists_debug_logs_without_stderr_noise(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    log_file_path = tmp_path / "tensor-network-editor.log"

    exit_code = main(
        [
            "--log-file",
            str(log_file_path),
            "template",
            "list",
            "--format",
            "json",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    assert log_file_path.exists()
    log_text = log_file_path.read_text(encoding="utf-8")
    assert "CLI command started" in log_text
    assert "Runtime diagnostics:" in log_text
    assert "command=template.list" in log_text
    assert "outcome=success" in log_text


def test_main_benchmark_log_file_captures_semantic_summary_without_stderr_noise(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec_path = tmp_path / "benchmark-network.json"
    log_file_path = tmp_path / "benchmark.log"
    spec_path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "network": build_sample_spec().to_dict(),
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--log-file",
            str(log_file_path),
            "benchmark",
            str(spec_path),
            "--format",
            "json",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    log_text = log_file_path.read_text(encoding="utf-8")
    assert "Benchmark command started" in log_text
    assert "Built benchmark report" in log_text
    assert "analysis_status=ready" in log_text
    assert "scheme_count=4" in log_text


def test_main_log_file_rotates_by_default_when_threshold_is_exceeded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    log_file_path = tmp_path / "rotating.log"
    monkeypatch.setattr(cli_parser, "DEFAULT_LOG_FILE_MAX_BYTES", 1024)
    monkeypatch.setattr(cli_parser, "DEFAULT_LOG_FILE_BACKUP_COUNT", 3)

    first_exit_code = main(
        [
            "--log-file",
            str(log_file_path),
            "template",
            "list",
            "--format",
            "json",
        ]
    )
    second_exit_code = main(
        [
            "--log-file",
            str(log_file_path),
            "template",
            "list",
            "--format",
            "json",
        ]
    )
    third_exit_code = main(
        [
            "--log-file",
            str(log_file_path),
            "template",
            "list",
            "--format",
            "json",
        ]
    )

    captured = capsys.readouterr()
    combined_log_text = _read_rotated_log_family(log_file_path)

    assert first_exit_code == 0
    assert second_exit_code == 0
    assert third_exit_code == 0
    assert captured.err == ""
    assert log_file_path.exists()
    assert (tmp_path / "rotating.log.1").exists()
    assert "Configured persistent file logging" in combined_log_text
    assert "Runtime diagnostics:" in combined_log_text
    assert "CLI command started" in combined_log_text


def test_main_log_file_rotation_respects_explicit_info_level(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    log_file_path = tmp_path / "rotating-info.log"

    first_exit_code = main(
        [
            "--log-level",
            "info",
            "--log-file",
            str(log_file_path),
            "--log-max-bytes",
            "256",
            "--log-backup-count",
            "2",
            "template",
            "list",
            "--format",
            "json",
        ]
    )
    second_exit_code = main(
        [
            "--log-level",
            "info",
            "--log-file",
            str(log_file_path),
            "--log-max-bytes",
            "256",
            "--log-backup-count",
            "2",
            "template",
            "list",
            "--format",
            "json",
        ]
    )

    captured = capsys.readouterr()
    combined_log_text = _read_rotated_log_family(log_file_path)

    assert first_exit_code == 0
    assert second_exit_code == 0
    assert "Runtime diagnostics:" in captured.err
    assert (tmp_path / "rotating-info.log.1").exists()
    assert "Runtime diagnostics:" in combined_log_text
    assert "CLI command started" not in combined_log_text
    assert "Resolved CLI command arguments" not in combined_log_text


def test_main_log_file_respects_explicit_info_level(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    log_file_path = tmp_path / "tensor-network-editor.log"

    exit_code = main(
        [
            "--log-level",
            "info",
            "--log-file",
            str(log_file_path),
            "template",
            "list",
            "--format",
            "json",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Runtime diagnostics:" in captured.err
    assert log_file_path.exists()
    log_text = log_file_path.read_text(encoding="utf-8")
    assert "Runtime diagnostics:" in log_text
    assert "CLI command started" not in log_text
    assert "Resolved CLI command arguments" not in log_text


def test_main_info_log_file_skips_low_level_io_success_logs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec_path = tmp_path / "network.json"
    log_file_path = tmp_path / "validate.log"
    spec_path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "network": build_sample_spec().to_dict(),
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--log-level",
            "info",
            "--log-file",
            str(log_file_path),
            "validate",
            str(spec_path),
        ]
    )

    captured = capsys.readouterr()
    log_text = log_file_path.read_text(encoding="utf-8")

    assert exit_code == 0
    assert "Runtime diagnostics:" in captured.err
    assert "Reading network specification JSON" not in log_text
    assert "Writing network specification JSON" not in log_text


def test_main_log_file_does_not_duplicate_lines_across_reconfiguration(
    tmp_path: Path,
) -> None:
    log_file_path = tmp_path / "tensor-network-editor.log"

    first_exit_code = main(
        [
            "--log-file",
            str(log_file_path),
            "template",
            "list",
            "--format",
            "json",
        ]
    )
    second_exit_code = main(
        [
            "--log-file",
            str(log_file_path),
            "template",
            "list",
            "--format",
            "json",
        ]
    )

    log_lines = log_file_path.read_text(encoding="utf-8").splitlines()
    cli_start_lines = [line for line in log_lines if "CLI command started" in line]

    assert first_exit_code == 0
    assert second_exit_code == 0
    assert len(cli_start_lines) == 2


@pytest.mark.parametrize(
    ("option_name", "raw_value"),
    [
        ("--log-max-bytes", "0"),
        ("--log-backup-count", "-1"),
    ],
)
def test_main_rejects_non_positive_log_rotation_values(
    option_name: str,
    raw_value: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "--log-file",
            "session.log",
            option_name,
            raw_value,
            "template",
            "list",
            "--format",
            "json",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 2
    assert option_name in captured.err
    assert "must be > 0" in captured.err


def test_main_debug_logs_handled_cli_errors_with_context(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["--log-level", "debug", "validate", "does_not_exist.json"])

    captured = capsys.readouterr()

    assert exit_code == 2
    assert "Could not read network specification JSON from 'does_not_exist.json'" in (
        captured.out
    )
    assert "CLI command failed" in captured.err
    assert "command=validate" in captured.err
    assert "path=does_not_exist.json" in captured.err
    assert "outcome=error" in captured.err
    assert "PackageIOError" in captured.err


def test_load_python_spec_logs_live_import_fallback_to_static_parser(
    caplog: pytest.LogCaptureFixture,
) -> None:
    generated = generate_code(build_sample_spec(), engine=EngineName.TENSORKROWCH)

    with patch(
        "tensor_network_editor.internal.io._serialization.import_live_python_source",
        side_effect=SerializationError("No module named 'torch'"),
    ):
        with caplog.at_level(logging.DEBUG, logger="tensor_network_editor"):
            spec = load_python_spec(
                generated.code,
                python=PythonLoadOptions(import_mode="live"),
            )

    assert spec.name == "Imported Python Network"
    assert "Python spec load started" in caplog.text
    assert "Live Python import fell back to the static parser" in caplog.text
    assert "python_import_mode=live" in caplog.text
    assert "source_profile=generated" in caplog.text
    assert "outcome=success" in caplog.text


def test_analyze_spec_logs_periodic_normalization_decision(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.DEBUG, logger="tensor_network_editor"):
        report = analyze_spec(build_tree_periodic_tree_spec())

    assert report.network.tensor_count == 5
    assert "Analyzing spec started" in caplog.text
    assert "Normalized spec for contraction analysis" in caplog.text
    assert "mode=tree_periodic" in caplog.text
    assert "outcome=success" in caplog.text


def test_render_spec_tikz_logs_selected_format_and_output_path(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    output_path = tmp_path / "figure.tex"

    with caplog.at_level(logging.DEBUG, logger="tensor_network_editor"):
        render_spec_tikz(build_sample_spec(), output_path=output_path)

    assert output_path.exists()
    assert "Render spec started" in caplog.text
    assert "format=tikz" in caplog.text
    assert f"output_path={output_path}" in caplog.text
    assert "Render spec finished" in caplog.text
