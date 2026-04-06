from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


def seed_generated_artifacts(root: Path) -> None:
    directories_to_create = [
        root / "build",
        root / "dist",
        root / ".pytest_cache",
        root / ".mypy_cache",
        root / ".ruff_cache",
        root / ".test_output",
        root / "pytest-cache-files-demo",
        root / "package.egg-info",
        root / "__pycache__",
        root / "src" / "tensor_network_editor" / "__pycache__",
        root / "tests" / "__pycache__",
        root / "examples" / "__pycache__",
        root / "scripts" / "__pycache__",
        root / ".venv",
    ]
    for directory in directories_to_create:
        directory.mkdir(parents=True, exist_ok=True)

    files_to_create = [
        root / "build" / "artifact.txt",
        root / "dist" / "artifact.whl",
        root / ".pytest_cache" / "cache.txt",
        root / ".mypy_cache" / "cache.txt",
        root / ".ruff_cache" / "cache.txt",
        root / ".test_output" / "output.txt",
        root / "pytest-cache-files-demo" / "temp.txt",
        root / "package.egg-info" / "PKG-INFO",
        root / "__pycache__" / "root.pyc",
        root / "src" / "tensor_network_editor" / "__pycache__" / "module.pyc",
        root / "tests" / "__pycache__" / "module.pyc",
        root / "examples" / "__pycache__" / "module.pyc",
        root / "scripts" / "__pycache__" / "module.pyc",
        root / ".venv" / "marker.txt",
        root / ".coverage",
        root / ".coverage.unit",
        root / "coverage.xml",
    ]
    for file_path in files_to_create:
        file_path.write_text("temporary", encoding="utf-8")


def assert_cleanup_removed_artifacts(root: Path) -> None:
    removed_paths = [
        root / "build",
        root / "dist",
        root / ".pytest_cache",
        root / ".mypy_cache",
        root / ".ruff_cache",
        root / ".test_output",
        root / "pytest-cache-files-demo",
        root / "package.egg-info",
        root / "__pycache__",
        root / "src" / "tensor_network_editor" / "__pycache__",
        root / "tests" / "__pycache__",
        root / "examples" / "__pycache__",
        root / "scripts" / "__pycache__",
        root / ".coverage",
        root / ".coverage.unit",
        root / "coverage.xml",
    ]
    for path in removed_paths:
        assert not path.exists()

    assert (root / ".venv").exists()
    assert (root / ".venv" / "marker.txt").exists()


def prepare_script_workspace(tmp_path: Path, script_name: str) -> Path:
    workspace = tmp_path / script_name.replace(".", "_")
    scripts_dir = workspace / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path.cwd() / "scripts" / script_name, scripts_dir / script_name)
    seed_generated_artifacts(workspace)
    return workspace


@pytest.mark.skipif(os.name != "nt", reason="clean.bat is a Windows-only helper")
def test_clean_bat_removes_generated_artifacts_and_preserves_venv(
    tmp_path: Path,
) -> None:
    workspace = prepare_script_workspace(tmp_path, "clean.bat")

    for _ in range(2):
        result = subprocess.run(
            ["cmd", "/c", "scripts\\clean.bat"],
            cwd=workspace,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    assert_cleanup_removed_artifacts(workspace)


def test_clean_sh_removes_generated_artifacts_and_preserves_venv(
    tmp_path: Path,
) -> None:
    shell_path = shutil.which("sh")
    source_script = Path.cwd() / "scripts" / "clean.sh"

    if shell_path is None:
        script_text = source_script.read_text(encoding="utf-8")
        assert 'remove_dir ".pytest_cache"' in script_text
        assert 'remove_dir "build"' in script_text
        assert 'remove_named_dirs "./scripts" "__pycache__"' in script_text
        assert ".venv" not in script_text
        return

    workspace = prepare_script_workspace(tmp_path, "clean.sh")
    copied_script = workspace / "scripts" / "clean.sh"
    copied_script.chmod(0o755)

    for _ in range(2):
        result = subprocess.run(
            [shell_path, "scripts/clean.sh"],
            cwd=workspace,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    assert_cleanup_removed_artifacts(workspace)
