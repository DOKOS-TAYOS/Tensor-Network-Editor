from __future__ import annotations

from pathlib import Path

import pytest

from tensor_network_editor._io import read_utf8_text, write_utf8_text
from tensor_network_editor.errors import PackageIOError


def test_read_utf8_text_reads_existing_files(tmp_path: Path) -> None:
    target_path = tmp_path / "sample.txt"
    target_path.write_text("hello", encoding="utf-8")

    assert read_utf8_text(target_path, description="sample file") == "hello"


def test_read_utf8_text_wraps_os_errors(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.txt"

    with pytest.raises(PackageIOError, match="Could not read sample file"):
        read_utf8_text(missing_path, description="sample file")


def test_write_utf8_text_writes_requested_content(tmp_path: Path) -> None:
    target_path = tmp_path / "written.txt"

    write_utf8_text(target_path, "content", description="sample file")

    assert target_path.read_text(encoding="utf-8") == "content"


def test_write_utf8_text_wraps_os_errors(tmp_path: Path) -> None:
    target_path = tmp_path / "missing" / "written.txt"

    with pytest.raises(PackageIOError, match="Could not write sample file"):
        write_utf8_text(target_path, "content", description="sample file")
