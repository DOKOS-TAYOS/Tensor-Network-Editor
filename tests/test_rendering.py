from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from tensor_network_editor.rendering import (
    DotRenderOptions,
    SvgRenderOptions,
    TikzRenderOptions,
    render_spec_dot,
    render_spec_svg,
    render_spec_tikz,
)
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


def test_render_spec_tikz_returns_tikzpicture_for_normal_network() -> None:
    tikz = render_spec_tikz(build_sample_spec())

    assert tikz.startswith(r"\begin{tikzpicture}")
    assert r"tne tensor/.style={draw, circle" in tikz
    assert r"\node[tne tensor," in tikz
    assert "(tensor_tensor_a)" in tikz
    assert r"\node[tne index] (index_tensor_a_i)" in tikz
    assert r"\draw[tne edge] (index_tensor_a_x) -- (index_tensor_b_x)" in tikz
    assert r"bond\_x" in tikz
    assert r"Demo Group" in tikz
    assert r"Check the contraction order" in tikz
    assert tikz.endswith(r"\end{tikzpicture}")


def test_render_spec_tikz_writes_output_path(tmp_path: Path) -> None:
    output_path = tmp_path / "network.tex"

    tikz = render_spec_tikz(build_sample_spec(), output_path=output_path)

    assert output_path.read_text(encoding="utf-8") == tikz


def test_render_spec_dot_returns_graphviz_graph_for_normal_network() -> None:
    dot = render_spec_dot(build_sample_spec())

    assert dot.startswith('graph "demo" {')
    assert '"tensor_a" [label="A", shape="circle"]' in dot
    assert '"open_tensor_a_i" [label="i (2)", shape="circle"]' in dot
    assert '"tensor_a" -- "tensor_b" [label="bond_x / x=3"]' in dot
    assert "subgraph cluster_group_demo" in dot
    assert '"note_demo" [label="Check the contraction order", shape="note"]' in dot
    assert dot.endswith("}")


def test_render_spec_dot_writes_output_path(tmp_path: Path) -> None:
    output_path = tmp_path / "network.dot"

    dot = render_spec_dot(build_sample_spec(), output_path=output_path)

    assert output_path.read_text(encoding="utf-8") == dot


def test_academic_renderers_include_hyperedges_and_open_indices() -> None:
    spec = build_three_tensor_hyperedge_spec()
    spec.hyperedges[0].hub_offset.x = 16.0
    spec.hyperedges[0].hub_offset.y = -8.0

    tikz = render_spec_tikz(spec, options=TikzRenderOptions(show_index_labels=False))
    dot = render_spec_dot(spec, options=DotRenderOptions(include_open_indices=True))

    assert r"\node[tne hyperedge] (hyperedge_hyperedge_h)" in tikz
    assert "shared\\_h" in tikz
    assert r"\node[tne index label]" not in tikz
    assert '"hyperedge_h" [label="shared_h", shape="point"]' in dot
    assert '"tensor_a" -- "hyperedge_h" [label="h=3"]' in dot
    assert '"open_tensor_a_i" [label="i (2)", shape="circle"]' in dot


def test_academic_renderers_can_hide_tensor_index_and_bond_labels() -> None:
    spec = build_sample_spec()

    tikz = render_spec_tikz(
        spec,
        options=TikzRenderOptions(
            show_tensor_labels=False,
            show_index_labels=False,
            show_edge_labels=False,
        ),
    )
    dot = render_spec_dot(
        spec,
        options=DotRenderOptions(
            show_tensor_labels=False,
            show_index_labels=False,
            show_edge_labels=False,
        ),
    )

    assert r"\node[tne tensor, minimum size=" in tikz
    assert r"\node[tne index label]" not in tikz
    assert r"bond\_x" not in tikz
    assert r"{A}" not in tikz
    assert r"{B}" not in tikz
    assert '"tensor_a" [label="", shape="circle"]' in dot
    assert '"open_tensor_a_i" [label="", shape="circle"]' in dot
    assert '"tensor_a" -- "tensor_b";' in dot
    assert "bond_x" not in dot
    assert "x=3" not in dot


def test_dot_renderer_keeps_index_and_bond_labels_separately_optional() -> None:
    spec = build_sample_spec()

    only_index_labels = render_spec_dot(
        spec,
        options=DotRenderOptions(show_edge_labels=False),
    )
    only_bond_labels = render_spec_dot(
        spec,
        options=DotRenderOptions(show_index_labels=False),
    )

    assert '"tensor_a" -- "tensor_b" [label="x=3"]' in only_index_labels
    assert "bond_x / x=3" not in only_index_labels
    assert '"tensor_a" -- "tensor_b" [label="bond_x"]' in only_bond_labels
    assert "bond_x / x=3" not in only_bond_labels


def test_academic_renderers_escape_labels_conservatively() -> None:
    spec = build_sample_spec()
    spec.name = 'demo_100% & "quote"\nline'
    spec.tensors[0].name = "A_%&"
    spec.tensors[0].indices[0].name = "i_%&"
    spec.edges[0].name = "bond_%&"
    spec.notes[0].text = 'note "quoted"\nsecond line'

    tikz = render_spec_tikz(spec)
    dot = render_spec_dot(spec)

    assert r"A\_\%\&" in tikz
    assert r"i\_\%\& 2" in tikz
    assert r"bond\_\%\&" in tikz
    assert "note ``quoted'' second line" in tikz
    assert 'graph "demo_100% & \\"quote\\"\\nline"' in dot
    assert '"tensor_a" [label="A_%&", shape="circle"]' in dot
    assert '"note_demo" [label="note \\"quoted\\"\\nsecond line", shape="note"]' in dot


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
