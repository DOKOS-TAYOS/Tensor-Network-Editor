from __future__ import annotations

from pathlib import Path

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
