from __future__ import annotations

from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import pytest

from tensor_network_editor.models import NetworkSpec
from tensor_network_editor.rendering import (
    DotRenderOptions,
    SvgRenderOptions,
    TikzRenderOptions,
    _SvgRenderer,
    render_spec_dot,
    render_spec_pdf,
    render_spec_svg,
    render_spec_tikz,
)
from tests.factories import build_sample_spec, build_three_tensor_hyperedge_spec


def _build_colored_parallel_edge_spec() -> NetworkSpec:
    spec = build_sample_spec()
    spec.tensors[0].metadata["color"] = "#123456"
    spec.tensors[1].metadata["color"] = "#abcdef"
    spec.edges[0].metadata["color"] = "#ff00aa"
    spec.tensors[0].indices[0].offset.x = -58.0
    spec.tensors[0].indices[1].offset.x = 58.0
    spec.tensors[1].indices[0].offset.x = -58.0
    spec.tensors[1].indices[1].offset.x = 58.0
    spec.tensors[1].indices[1].dimension = 2
    spec.edges.append(
        type(spec.edges[0])(
            id="edge_parallel",
            name="bond_parallel",
            left=type(spec.edges[0].left)(
                tensor_id="tensor_a",
                index_id="tensor_a_i",
            ),
            right=type(spec.edges[0].right)(
                tensor_id="tensor_b",
                index_id="tensor_b_j",
            ),
            metadata={"color": "#00ffaa"},
        )
    )
    return spec


def _assign_demo_index_offsets() -> NetworkSpec:
    spec = build_sample_spec()
    spec.tensors[0].indices[0].offset.x = -58.0
    spec.tensors[0].indices[1].offset.x = 58.0
    spec.tensors[1].indices[0].offset.x = -58.0
    spec.tensors[1].indices[1].offset.x = 58.0
    return spec


def _build_three_parallel_edge_spec() -> NetworkSpec:
    spec = _assign_demo_index_offsets()
    spec.tensors[1].indices[1].dimension = 2
    spec.tensors[0].indices.append(
        type(spec.tensors[0].indices[0])(
            id="tensor_a_k",
            name="k",
            dimension=5,
        )
    )
    spec.tensors[1].indices.append(
        type(spec.tensors[1].indices[0])(
            id="tensor_b_k",
            name="k",
            dimension=5,
        )
    )
    spec.edges.append(
        type(spec.edges[0])(
            id="edge_parallel_left",
            name="bond_left",
            left=type(spec.edges[0].left)(
                tensor_id="tensor_a",
                index_id="tensor_a_i",
            ),
            right=type(spec.edges[0].right)(
                tensor_id="tensor_b",
                index_id="tensor_b_j",
            ),
        )
    )
    spec.edges.append(
        type(spec.edges[0])(
            id="edge_parallel_right",
            name="bond_right",
            left=type(spec.edges[0].left)(
                tensor_id="tensor_a",
                index_id="tensor_a_k",
            ),
            right=type(spec.edges[0].right)(
                tensor_id="tensor_b",
                index_id="tensor_b_k",
            ),
        )
    )
    return spec


def _svg_text_content(svg: str) -> list[str]:
    root = ET.fromstring(svg)
    text_nodes = root.findall(".//{http://www.w3.org/2000/svg}text")
    return [
        "".join(text_node.itertext()).strip()
        for text_node in text_nodes
        if "".join(text_node.itertext()).strip()
    ]


def test_render_spec_svg_returns_standalone_svg_for_normal_network() -> None:
    pytest.importorskip("matplotlib")
    svg = render_spec_svg(build_sample_spec())
    root = ET.fromstring(svg)
    text_content = _svg_text_content(svg)

    assert svg.startswith('<?xml version="1.0" encoding="UTF-8"?>')
    assert root.tag == "{http://www.w3.org/2000/svg}svg"
    assert "<text" in svg
    assert "A" in text_content
    assert "B" in text_content
    assert "bond_x" in text_content
    assert "Demo Group" in text_content
    assert any("Check the contraction order" in item for item in text_content)


def test_academic_svg_and_tikz_exports_use_tensor_circles_and_dangling_ports() -> None:
    spec = _assign_demo_index_offsets()

    pytest.importorskip("matplotlib")
    svg = render_spec_svg(spec)
    tikz = render_spec_tikz(spec)
    text_content = _svg_text_content(svg)

    assert "<path" in svg
    assert "i 2" in text_content
    assert "j 4" in text_content
    assert r"\node[tne index]" not in tikz
    assert r"\draw[tne open index]" in tikz


def test_academic_svg_tikz_and_dot_preserve_entity_colors_and_parallel_edges() -> None:
    spec = _build_colored_parallel_edge_spec()

    pytest.importorskip("matplotlib")
    svg = render_spec_svg(spec)
    tikz = render_spec_tikz(spec)
    dot = render_spec_dot(spec)

    assert "#123456" in svg
    assert "#ff00aa" in svg
    assert svg.count("<path") >= 2
    assert r"\definecolor{tneColor123456}{HTML}{123456}" in tikz
    assert r"\definecolor{tneColorff00aa}{HTML}{ff00aa}" in tikz
    assert r"draw=tneColorff00aa" in tikz
    assert ".. controls" in tikz
    assert 'fillcolor="#123456"' in dot
    assert 'color="#ff00aa"' in dot


def test_academic_parallel_edges_curve_far_enough_to_separate_three_bonds() -> None:
    spec = _build_three_parallel_edge_spec()
    edge_render_infos = _SvgRenderer(
        spec,
        SvgRenderOptions(
            include_groups=False,
            include_notes=False,
            show_index_labels=False,
            show_edge_labels=False,
        ),
    )._edge_render_infos()
    control_y_values = [
        edge_info.control.y
        for edge_info in edge_render_infos
        if edge_info.control is not None
    ]

    assert len(control_y_values) == 3
    assert max(control_y_values) - min(control_y_values) >= 80.0


def test_academic_edges_reach_tensor_centers_in_svg_and_tikz() -> None:
    spec = _assign_demo_index_offsets()

    edge_render_infos = _SvgRenderer(spec, SvgRenderOptions())._edge_render_infos()
    tikz = render_spec_tikz(spec)

    assert edge_render_infos[0].source == spec.tensors[0].position
    assert edge_render_infos[0].target == spec.tensors[1].position
    assert "(150, 116) -- (390, 116)" in tikz


def test_academic_svg_renderer_can_hide_tensor_index_and_bond_labels() -> None:
    spec = _assign_demo_index_offsets()

    pytest.importorskip("matplotlib")
    svg = render_spec_svg(
        spec,
        options=SvgRenderOptions(
            show_tensor_labels=False,
            show_index_labels=False,
            show_edge_labels=False,
        ),
    )
    text_content = _svg_text_content(svg)

    assert "A" not in text_content
    assert "B" not in text_content
    assert "bond_x" not in text_content
    assert "i 2" not in text_content
    assert "j 4" not in text_content


def test_render_spec_svg_renders_hyperedge_hubs_and_spokes() -> None:
    spec = build_three_tensor_hyperedge_spec()
    spec.hyperedges[0].hub_offset.x = 16.0
    spec.hyperedges[0].hub_offset.y = -8.0

    pytest.importorskip("matplotlib")
    svg = render_spec_svg(spec, options=SvgRenderOptions(show_index_labels=False))
    text_content = _svg_text_content(svg)

    assert "shared_h" in text_content
    assert svg.count("<path") >= 3
    assert "tensor_a_h" not in svg


def test_render_spec_svg_writes_output_path(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    output_path = tmp_path / "network.svg"

    svg = render_spec_svg(build_sample_spec(), output_path=output_path)

    assert output_path.read_text(encoding="utf-8") == svg


def test_render_spec_svg_reuses_edge_geometry_within_one_render(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tensor_network_editor.rendering as rendering_module

    pytest.importorskip("matplotlib")
    spec = _build_colored_parallel_edge_spec()
    edge_render_info_call_count = 0
    original_edge_render_infos = rendering_module._SvgRenderer._edge_render_infos

    def counting_edge_render_infos(self: Any) -> list[Any]:
        nonlocal edge_render_info_call_count
        edge_render_info_call_count += 1
        return original_edge_render_infos(self)

    monkeypatch.setattr(
        rendering_module._SvgRenderer,
        "_edge_render_infos",
        counting_edge_render_infos,
    )

    render_spec_svg(spec)

    assert edge_render_info_call_count == 1


def test_render_spec_svg_keeps_labels_as_svg_text_elements() -> None:
    pytest.importorskip("matplotlib")

    svg = render_spec_svg(build_sample_spec())

    assert "<text" in svg
    assert "<path" in svg


def test_render_spec_tikz_returns_tikzpicture_for_normal_network() -> None:
    tikz = render_spec_tikz(build_sample_spec())

    assert r"\def\tneGlobalWidth{\linewidth}" in tikz
    assert (
        r"\begin{tikzpicture}[x=\tneUnit, y=\tneUnit, line width=\tneLineWidth]" in tikz
    )
    assert tikz.splitlines()[0] == r"\def\tneGlobalWidth{\linewidth}"
    assert r"tne tensor/.style={draw, circle" in tikz
    assert (
        r"tne tensor/.style={draw, circle, fill=blue!12, align=center, inner sep=3*\tneUnit, font=\scriptsize}"
        in tikz
    )
    assert r"tne index label/.style={font=\fontsize{4}{4.8}\selectfont" in tikz
    assert r"tne edge label/.style={font=\fontsize{4}{4.8}\selectfont" in tikz
    assert r"\node[tne tensor," in tikz
    assert "(tensor_tensor_a)" in tikz
    assert r"\node[tne index]" not in tikz
    assert r"\draw[tne edge]" in tikz
    assert r"\draw[tne open index]" in tikz
    assert r"bond\_x" in tikz
    assert r"Demo Group" in tikz
    assert r"Check the contraction order" in tikz
    assert tikz.endswith(r"\end{tikzpicture}")


def test_render_spec_tikz_uses_single_global_width_control() -> None:
    tikz = render_spec_tikz(
        build_sample_spec(),
        options=TikzRenderOptions(global_width=r"0.8\textwidth"),
    )

    assert r"\def\tneGlobalWidth{0.8\textwidth}" in tikz
    assert r"\pgfmathsetlengthmacro{\tneUnit}{" in tikz
    assert r"minimum size=\tneHyperedgeSize" in tikz
    assert r"line width=\tneLineWidth" in tikz
    assert r"*\tneUnit" in tikz
    assert r"\pgfmathsetlengthmacro{\tneHyperedgeSpokeWidth}" not in tikz


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
    assert r"\node[tne index]" not in tikz
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


def test_render_spec_svg_png_and_pdf_report_missing_matplotlib_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tensor_network_editor.rendering as rendering_module

    def reject_matplotlib_modules() -> tuple[object, object, object, object]:
        raise RuntimeError(
            "PNG/SVG/PDF rendering requires Matplotlib. Reinstall the package or add Matplotlib to the current environment."
        )

    monkeypatch.setattr(
        rendering_module,
        "_load_matplotlib_modules",
        reject_matplotlib_modules,
    )

    with pytest.raises(RuntimeError, match=r"requires Matplotlib|add Matplotlib"):
        rendering_module.render_spec_svg(build_sample_spec())
    with pytest.raises(RuntimeError, match=r"requires Matplotlib|add Matplotlib"):
        rendering_module.render_spec_png(build_sample_spec())
    with pytest.raises(RuntimeError, match=r"requires Matplotlib|add Matplotlib"):
        rendering_module.render_spec_pdf(build_sample_spec())


def test_render_spec_png_returns_png_bytes_and_writes_output_path(
    tmp_path: Path,
) -> None:
    pytest.importorskip("matplotlib")
    from tensor_network_editor.rendering import render_spec_png

    output_path = tmp_path / "network.png"

    png_bytes = render_spec_png(build_sample_spec(), output_path=output_path)

    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    assert output_path.read_bytes() == png_bytes


def test_render_spec_pdf_returns_pdf_bytes_and_writes_output_path(
    tmp_path: Path,
) -> None:
    pytest.importorskip("matplotlib")

    output_path = tmp_path / "network.pdf"

    pdf_bytes = render_spec_pdf(_assign_demo_index_offsets(), output_path=output_path)

    assert pdf_bytes.startswith(b"%PDF")
    assert b"/Font" in pdf_bytes
    assert output_path.read_bytes() == pdf_bytes
