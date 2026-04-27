from __future__ import annotations

import runpy
from pathlib import Path

import pytest

EXAMPLE_SCRIPTS = [
    "mps_template_codegen.py",
    "peps_academic_render.py",
    "hyperedge_network.py",
    "contraction_benchmark.py",
    "tensor_initializers.py",
]
REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("script_name", EXAMPLE_SCRIPTS)
def test_public_examples_run_without_optional_backends(
    script_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    script_path = REPO_ROOT / "examples" / script_name

    runpy.run_path(script_path, run_name="__main__")

    output = capsys.readouterr().out
    assert "OK:" in output
