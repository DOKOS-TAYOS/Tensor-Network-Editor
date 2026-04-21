from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tensor_network_editor.api import save_spec
from tensor_network_editor.serialization import SCHEMA_VERSION
from tests.factories import build_sample_spec

pytestmark = pytest.mark.integration
REPO_ROOT = Path(__file__).resolve().parents[1]


def _checkout_python_env(cwd: Path) -> dict[str, str]:
    env = os.environ.copy()
    current_src_path = str((cwd / "src").resolve())
    current_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        current_src_path
        if not current_pythonpath
        else os.pathsep.join([current_src_path, current_pythonpath])
    )
    return env


def _run_cli(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "tensor_network_editor", *args],
        cwd=cwd,
        env=_checkout_python_env(cwd),
        capture_output=True,
        text=True,
        check=False,
    )


def _assert_cli_success(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 0, result.stdout + result.stderr


def test_cli_subprocess_prefers_current_checkout_src() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from pathlib import Path; "
                "import tensor_network_editor; "
                "print(Path(tensor_network_editor.__file__).resolve())"
            ),
        ],
        cwd=REPO_ROOT,
        env=_checkout_python_env(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    _assert_cli_success(result)
    assert (
        Path(result.stdout.strip()).resolve()
        == (REPO_ROOT / "src" / "tensor_network_editor" / "__init__.py").resolve()
    )


def test_headless_cli_commands_work_with_real_files(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    before_spec = build_sample_spec()
    after_spec = build_sample_spec()
    after_spec.tensors[0].name = "A prime"
    before_path = tmp_path / "before.json"
    after_path = tmp_path / "after.json"
    generated_path = tmp_path / "generated.py"
    canonical_path = tmp_path / "canonical.json"
    template_path = tmp_path / "template.json"
    save_spec(before_spec, before_path)
    save_spec(after_spec, after_path)

    validate_result = _run_cli(
        "validate", str(before_path), "--format", "json", cwd=repo_root
    )
    _assert_cli_success(validate_result)
    assert json.loads(validate_result.stdout)["issues"] == []

    lint_result = _run_cli("lint", str(before_path), "--format", "json", cwd=repo_root)
    _assert_cli_success(lint_result)
    lint_payload = json.loads(lint_result.stdout)
    assert {issue["code"] for issue in lint_payload["issues"]} >= {
        "suspicious-open-index"
    }

    analyze_result = _run_cli(
        "analyze", str(before_path), "--format", "json", cwd=repo_root
    )
    _assert_cli_success(analyze_result)
    assert json.loads(analyze_result.stdout)["network"]["tensor_count"] == 2

    export_result = _run_cli(
        "export",
        str(before_path),
        "--engine",
        "einsum_numpy",
        "--output",
        str(generated_path),
        cwd=repo_root,
    )
    _assert_cli_success(export_result)
    assert "np.einsum(" in generated_path.read_text(encoding="utf-8")

    canonicalize_result = _run_cli(
        "canonicalize",
        str(before_path),
        "--output",
        str(canonical_path),
        cwd=repo_root,
    )
    _assert_cli_success(canonicalize_result)
    assert (
        json.loads(canonical_path.read_text(encoding="utf-8"))["schema_version"]
        == SCHEMA_VERSION
    )

    diff_result = _run_cli(
        "diff",
        str(before_path),
        str(after_path),
        "--semantic",
        "--format",
        "json",
        cwd=repo_root,
    )
    _assert_cli_success(diff_result)
    assert json.loads(diff_result.stdout)["entries"]

    template_result = _run_cli(
        "template",
        "build",
        "mps",
        "--graph-size",
        "4",
        "--output",
        str(template_path),
        cwd=repo_root,
    )
    _assert_cli_success(template_result)
    template_payload = json.loads(template_path.read_text(encoding="utf-8"))
    assert template_payload["schema_version"] == SCHEMA_VERSION
    assert template_payload["network"]["name"] == "MPS"
    assert len(template_payload["network"]["tensors"]) == 4


def test_benchmark_cli_command_outputs_json_and_csv(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    spec = build_sample_spec()
    spec_path = tmp_path / "benchmark.json"
    csv_path = tmp_path / "benchmark.csv"
    save_spec(spec, spec_path)

    json_result = _run_cli(
        "benchmark",
        str(spec_path),
        "--format",
        "json",
        cwd=repo_root,
    )
    _assert_cli_success(json_result)
    json_payload = json.loads(json_result.stdout)
    assert json_payload["memory_dtype"] == "float64"
    assert [row["key"] for row in json_payload["rows"]] == [
        "manual",
        "auto_full",
        "auto_future",
        "auto_past",
    ]

    csv_result = _run_cli(
        "benchmark",
        str(spec_path),
        "--format",
        "csv",
        "--output",
        str(csv_path),
        cwd=repo_root,
    )
    _assert_cli_success(csv_result)
    csv_body = csv_path.read_text(encoding="utf-8")
    assert csv_body.startswith("Name,FLOP,MAC,Peak,Peak Memory")
    assert "Manual," in csv_body
    assert "Auto future," in csv_body
