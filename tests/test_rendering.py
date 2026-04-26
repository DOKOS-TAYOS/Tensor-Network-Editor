from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from tensor_network_editor.rendering import SvgRenderOptions, render_spec_svg
from tests.factories import build_sample_spec, build_three_tensor_hyperedge_spec


def test_render_spec_svg_returns_standalone_svg_for_normal_network() -> None:
    svg = render_spec_svg(build_sample_spec())

    assert svg.startswith('<?xml version="1.0" encoding="UTF-8"?>')
    assert '<svg xmlns="http://www.w3.org/2000/svg"' in svg
    assert "demo" in svg
    assert "A" in svg
    assert "B" in svg
    assert "bond_x" in svg
    assert "Demo Group" in svg
    assert "Check the contraction order" in svg


def test_render_spec_svg_renders_hyperedge_hubs_and_spokes() -> None:
    spec = build_three_tensor_hyperedge_spec()
    spec.hyperedges[0].hub_offset.x = 16.0
    spec.hyperedges[0].hub_offset.y = -8.0

    svg = render_spec_svg(spec, options=SvgRenderOptions(show_index_labels=False))

    assert "shared_h" in svg
    assert 'class="hyperedge-spoke"' in svg
    assert 'class="hyperedge-hub"' in svg
    assert "tensor_a_h" not in svg


def test_render_spec_svg_writes_output_path(tmp_path: Path) -> None:
    output_path = tmp_path / "network.svg"

    svg = render_spec_svg(build_sample_spec(), output_path=output_path)

    assert output_path.read_text(encoding="utf-8") == svg


def test_render_spec_png_reports_missing_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    from tensor_network_editor.rendering import render_spec_png

    original_import = builtins.__import__

    def reject_pillow(
        name: str,
        globals_: Mapping[str, object] | None = None,
        locals_: Mapping[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "PIL" or name.startswith("PIL."):
            raise ImportError("Pillow disabled for test")
        return original_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", reject_pillow)

    with pytest.raises(RuntimeError, match=r"tensor-network-editor\[png\]"):
        render_spec_png(build_sample_spec())


def test_render_spec_png_returns_png_bytes_and_writes_output_path(
    tmp_path: Path,
) -> None:
    pytest.importorskip("PIL.Image")
    from tensor_network_editor.rendering import render_spec_png

    output_path = tmp_path / "network.png"

    png_bytes = render_spec_png(build_sample_spec(), output_path=output_path)

    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    assert output_path.read_bytes() == png_bytes
