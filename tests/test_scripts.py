from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import types
from pathlib import Path

import pytest


def load_script_module(script_name: str) -> types.ModuleType:
    script_path = Path.cwd() / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(
        script_name.replace(".py", ""), script_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def create_checkout_layout(root: Path) -> None:
    (root / "src" / "tensor_network_editor").mkdir(parents=True, exist_ok=True)
    (root / "tests").mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text(
        "[project]\nname='tensor-network-editor'\n", encoding="utf-8"
    )


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


def test_run_pyright_locates_checkout_and_shared_venv_roots(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    worktree_root = repo_root / ".worktrees" / "feature"
    create_checkout_layout(repo_root)
    create_checkout_layout(worktree_root)
    python_path = (
        repo_root / ".venv" / "Scripts" / "python.exe"
        if os.name == "nt"
        else repo_root / ".venv" / "bin" / "python"
    )
    python_path.parent.mkdir(parents=True, exist_ok=True)
    python_path.write_text("", encoding="utf-8")
    module = load_script_module("run_pyright.py")

    assert module.find_checkout_root(worktree_root / "src") == worktree_root
    assert module.find_shared_venv_root(worktree_root) == repo_root
    assert module.python_executable(repo_root) == python_path


def test_run_pyright_builds_config_for_current_checkout(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    worktree_root = repo_root / ".worktrees" / "feature"
    create_checkout_layout(repo_root)
    create_checkout_layout(worktree_root)
    module = load_script_module("run_pyright.py")

    config = module.build_pyright_config(worktree_root, repo_root)

    assert config == {
        "venvPath": str(repo_root.resolve()),
        "venv": ".venv",
        "include": ["src", "tests"],
        "extraPaths": ["src", "."],
        "typeCheckingMode": "standard",
    }


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
