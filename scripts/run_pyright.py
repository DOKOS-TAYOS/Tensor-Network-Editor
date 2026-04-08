"""Run Pyright against the current checkout using the shared project virtualenv."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Final

DEFAULT_TARGETS: Final[tuple[str, ...]] = ("src", "tests")


def find_checkout_root(start: Path | None = None) -> Path:
    """Return the current repository checkout root from ``start`` or the CWD."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (
            (candidate / "pyproject.toml").is_file()
            and (candidate / "src").is_dir()
            and (candidate / "tests").is_dir()
        ):
            return candidate
    raise FileNotFoundError(
        "Could not locate the checkout root containing pyproject.toml, src, and tests."
    )


def find_shared_venv_root(checkout_root: Path) -> Path:
    """Return the nearest ancestor that provides the shared project ``.venv``."""
    for candidate in (checkout_root.resolve(), *checkout_root.resolve().parents):
        if (candidate / ".venv").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate a shared .venv for Pyright.")


def python_executable(shared_root: Path) -> Path:
    """Return the Python executable inside the shared virtual environment."""
    venv_root = shared_root / ".venv"
    candidates = (
        venv_root / "Scripts" / "python.exe",
        venv_root / "bin" / "python",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find the Python executable inside the shared .venv."
    )


def build_pyright_config(checkout_root: Path, shared_root: Path) -> dict[str, object]:
    """Build a temporary Pyright config for the current checkout."""
    del checkout_root
    return {
        "venvPath": str(shared_root.resolve()),
        "venv": ".venv",
        "include": list(DEFAULT_TARGETS),
        "extraPaths": ["src", "."],
        "typeCheckingMode": "standard",
    }


def resolve_targets(checkout_root: Path, args: Sequence[str]) -> list[str]:
    """Resolve CLI targets relative to the current checkout."""
    if not args:
        return [str((checkout_root / target).resolve()) for target in DEFAULT_TARGETS]
    return [str((checkout_root / argument).resolve()) for argument in args]


def run_pyright(
    checkout_root: Path,
    shared_root: Path,
    args: Sequence[str],
) -> int:
    """Execute Pyright for the current checkout and return its exit code."""
    config = build_pyright_config(checkout_root, shared_root)
    pyright_python = python_executable(shared_root)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            suffix=".json",
            encoding="utf-8",
            delete=False,
            dir=checkout_root,
            prefix=".pyright-",
        ) as handle:
            json.dump(config, handle, indent=2)
            handle.flush()
            temp_path = Path(handle.name)
        result = subprocess.run(
            [
                str(pyright_python),
                "-m",
                "pyright",
                "-p",
                str(temp_path),
                *resolve_targets(checkout_root, args),
            ],
            cwd=checkout_root,
            check=False,
        )
        return int(result.returncode)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Run Pyright against the current checkout root or a subpath of it."""
    checkout_root = find_checkout_root()
    shared_root = find_shared_venv_root(checkout_root)
    return run_pyright(checkout_root, shared_root, list(argv or sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
