from __future__ import annotations

import re
from math import hypot
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import pytest

from tensor_network_editor.models import (
    CanvasPosition,
    EdgeEndpointRef,
    EdgeSpec,
    IndexSpec,
    NetworkSpec,
    TensorSpec,
)
from tensor_network_editor.rendering import (
    DotRenderOptions,
    SvgRenderOptions,
    TikzRenderOptions,
    _number,
    _SvgRenderer,
    render_spec_dot,
    render_spec_mermaid,
    render_spec_pdf,
    render_spec_svg,
    render_spec_tikz,
)
from tensor_network_editor.templates import TemplateParameters, build_template_spec
from tests.factories import (
    build_sample_spec,
    build_three_tensor_hyperedge_spec,
    build_three_tensor_spec,
)


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


def _build_cycle_spec() -> NetworkSpec:
    return NetworkSpec(
        id="network_cycle",
        name="cycle",
        tensors=[
            TensorSpec(
                id="tensor_a",
                name="A",
                position=CanvasPosition(x=120.0, y=120.0),
                indices=[
                    IndexSpec(id="tensor_a_free", name="fa", dimension=2),
                    IndexSpec(id="tensor_a_ab", name="ab", dimension=3),
                    IndexSpec(id="tensor_a_da", name="da", dimension=5),
                ],
            ),
            TensorSpec(
                id="tensor_b",
                name="B",
                position=CanvasPosition(x=280.0, y=120.0),
                indices=[
                    IndexSpec(id="tensor_b_free", name="fb", dimension=2),
                    IndexSpec(id="tensor_b_ab", name="ab", dimension=3),
                    IndexSpec(id="tensor_b_bc", name="bc", dimension=7),
                ],
            ),
            TensorSpec(
                id="tensor_c",
                name="C",
                position=CanvasPosition(x=280.0, y=280.0),
                indices=[
                    IndexSpec(id="tensor_c_free", name="fc", dimension=2),
                    IndexSpec(id="tensor_c_bc", name="bc", dimension=7),
                    IndexSpec(id="tensor_c_cd", name="cd", dimension=11),
                ],
            ),
            TensorSpec(
                id="tensor_d",
                name="D",
                position=CanvasPosition(x=120.0, y=280.0),
                indices=[
                    IndexSpec(id="tensor_d_free", name="fd", dimension=2),
                    IndexSpec(id="tensor_d_cd", name="cd", dimension=11),
                    IndexSpec(id="tensor_d_da", name="da", dimension=5),
                ],
            ),
        ],
        edges=[
            EdgeSpec(
                id="edge_ab",
                name="ab",
                left=EdgeEndpointRef(tensor_id="tensor_a", index_id="tensor_a_ab"),
                right=EdgeEndpointRef(tensor_id="tensor_b", index_id="tensor_b_ab"),
            ),
            EdgeSpec(
                id="edge_bc",
                name="bc",
                left=EdgeEndpointRef(tensor_id="tensor_b", index_id="tensor_b_bc"),
                right=EdgeEndpointRef(tensor_id="tensor_c", index_id="tensor_c_bc"),
            ),
            EdgeSpec(
                id="edge_cd",
                name="cd",
                left=EdgeEndpointRef(tensor_id="tensor_c", index_id="tensor_c_cd"),
                right=EdgeEndpointRef(tensor_id="tensor_d", index_id="tensor_d_cd"),
            ),
            EdgeSpec(
                id="edge_da",
                name="da",
                left=EdgeEndpointRef(tensor_id="tensor_d", index_id="tensor_d_da"),
                right=EdgeEndpointRef(tensor_id="tensor_a", index_id="tensor_a_da"),
            ),
        ],
    )


def _build_grid_export_spec() -> NetworkSpec:
    tensors: list[TensorSpec] = []
    edges: list[EdgeSpec] = []
    for row_index in range(3):
        for column_index in range(3):
            tensor_id = f"tensor_{row_index}_{column_index}"
            indices = [
                IndexSpec(
                    id=f"{tensor_id}_free",
                    name=f"f_{row_index}_{column_index}",
                    dimension=2,
                )
            ]
            if column_index < 2:
                indices.append(
                    IndexSpec(
                        id=f"{tensor_id}_right",
                        name=f"h_{row_index}_{column_index}",
                        dimension=3,
                    )
                )
            if column_index > 0:
                indices.append(
                    IndexSpec(
                        id=f"{tensor_id}_left",
                        name=f"h_{row_index}_{column_index - 1}",
                        dimension=3,
                    )
                )
            if row_index < 2:
                indices.append(
                    IndexSpec(
                        id=f"{tensor_id}_down",
                        name=f"v_{row_index}_{column_index}",
                        dimension=5,
                    )
                )
            if row_index > 0:
                indices.append(
                    IndexSpec(
                        id=f"{tensor_id}_up",
                        name=f"v_{row_index - 1}_{column_index}",
                        dimension=5,
                    )
                )
            tensors.append(
                TensorSpec(
                    id=tensor_id,
                    name=f"T{row_index}{column_index}",
                    position=CanvasPosition(
                        x=120.0 + 140.0 * column_index,
                        y=120.0 + 140.0 * row_index,
                    ),
                    indices=indices,
                )
            )
    for row_index in range(3):
        for column_index in range(2):
            left_tensor_id = f"tensor_{row_index}_{column_index}"
            right_tensor_id = f"tensor_{row_index}_{column_index + 1}"
            edge_name = f"h_{row_index}_{column_index}"
            edges.append(
                EdgeSpec(
                    id=f"edge_{edge_name}",
                    name=edge_name,
                    left=EdgeEndpointRef(
                        tensor_id=left_tensor_id,
                        index_id=f"{left_tensor_id}_right",
                    ),
                    right=EdgeEndpointRef(
                        tensor_id=right_tensor_id,
                        index_id=f"{right_tensor_id}_left",
                    ),
                )
            )
    for row_index in range(2):
        for column_index in range(3):
            top_tensor_id = f"tensor_{row_index}_{column_index}"
            bottom_tensor_id = f"tensor_{row_index + 1}_{column_index}"
            edge_name = f"v_{row_index}_{column_index}"
            edges.append(
                EdgeSpec(
                    id=f"edge_{edge_name}",
                    name=edge_name,
                    left=EdgeEndpointRef(
                        tensor_id=top_tensor_id,
                        index_id=f"{top_tensor_id}_down",
                    ),
                    right=EdgeEndpointRef(
                        tensor_id=bottom_tensor_id,
                        index_id=f"{bottom_tensor_id}_up",
                    ),
                )
            )
    return NetworkSpec(
        id="network_grid_export",
        name="grid-export",
        tensors=tensors,
        edges=edges,
    )


def _build_vertical_three_tensor_spec() -> NetworkSpec:
    spec = build_three_tensor_spec()
    spec.tensors[0].position = CanvasPosition(x=240.0, y=80.0)
    spec.tensors[1].position = CanvasPosition(x=240.0, y=240.0)
    spec.tensors[2].position = CanvasPosition(x=240.0, y=400.0)
    return spec


def _build_vertical_three_tensor_named_hint_spec() -> NetworkSpec:
    spec = _build_vertical_three_tensor_spec()
    spec.tensors[0].indices[0].name = "up"
    return spec


def _build_diagonal_three_tensor_spec() -> NetworkSpec:
    spec = build_three_tensor_spec()
    spec.tensors[0].position = CanvasPosition(x=80.0, y=80.0)
    spec.tensors[1].position = CanvasPosition(x=240.0, y=240.0)
    spec.tensors[2].position = CanvasPosition(x=400.0, y=400.0)
    return spec


def _build_rotated_grid_export_spec() -> NetworkSpec:
    spec = _build_grid_export_spec()
    center = CanvasPosition(x=240.0, y=240.0)
    column_step = CanvasPosition(x=100.0, y=100.0)
    row_step = CanvasPosition(x=-100.0, y=100.0)
    for tensor in spec.tensors:
        _, row_text, column_text = tensor.id.split("_")
        row_index = int(row_text)
        column_index = int(column_text)
        tensor.position = CanvasPosition(
            x=center.x
            + (column_index - 1) * column_step.x
            + (row_index - 1) * row_step.x,
            y=center.y
            + (column_index - 1) * column_step.y
            + (row_index - 1) * row_step.y,
        )
    return spec


def _build_vertical_mpo_export_spec() -> NetworkSpec:
    spec = build_template_spec(
        "mpo",
        TemplateParameters(
            graph_size=4,
            bond_dimension=3,
            physical_dimension=2,
            boundary_condition="open",
            j=1.0,
            h=1.0,
        ),
    )
    for tensor_index, tensor in enumerate(spec.tensors):
        tensor.position = CanvasPosition(x=240.0, y=80.0 + tensor_index * 160.0)
    return spec


def _build_generic_export_spec() -> NetworkSpec:
    return NetworkSpec(
        id="network_generic_export",
        name="generic-export",
        tensors=[
            TensorSpec(
                id="tensor_center",
                name="Center",
                position=CanvasPosition(x=220.0, y=200.0),
                indices=[
                    IndexSpec(id="tensor_center_free", name="free", dimension=2),
                    IndexSpec(id="tensor_center_right", name="r", dimension=3),
                    IndexSpec(id="tensor_center_down", name="d", dimension=5),
                ],
            ),
            TensorSpec(
                id="tensor_right",
                name="Right",
                position=CanvasPosition(x=360.0, y=180.0),
                indices=[
                    IndexSpec(id="tensor_right_left", name="r", dimension=3),
                ],
            ),
            TensorSpec(
                id="tensor_down",
                name="Down",
                position=CanvasPosition(x=260.0, y=340.0),
                indices=[
                    IndexSpec(id="tensor_down_up", name="d", dimension=5),
                ],
            ),
        ],
        edges=[
            EdgeSpec(
                id="edge_center_right",
                name="r",
                left=EdgeEndpointRef(
                    tensor_id="tensor_center", index_id="tensor_center_right"
                ),
                right=EdgeEndpointRef(
                    tensor_id="tensor_right", index_id="tensor_right_left"
                ),
            ),
            EdgeSpec(
                id="edge_center_down",
                name="d",
                left=EdgeEndpointRef(
                    tensor_id="tensor_center", index_id="tensor_center_down"
                ),
                right=EdgeEndpointRef(
                    tensor_id="tensor_down", index_id="tensor_down_up"
                ),
            ),
        ],
    )


def _dot(left: CanvasPosition, right: CanvasPosition) -> float:
    return left.x * right.x + left.y * right.y


def _normalize(vector: CanvasPosition) -> CanvasPosition:
    magnitude = hypot(vector.x, vector.y)
    assert magnitude > 1e-9
    return CanvasPosition(x=vector.x / magnitude, y=vector.y / magnitude)


def _svg_text_content(svg: str) -> list[str]:
    root = ET.fromstring(svg)
    text_nodes = root.findall(".//{http://www.w3.org/2000/svg}text")
    return [
        "".join(text_node.itertext()).strip()
        for text_node in text_nodes
        if "".join(text_node.itertext()).strip()
    ]


def _svg_text_fill_by_content(svg: str) -> dict[str, str]:
    root = ET.fromstring(svg)
    text_nodes = root.findall(".//{http://www.w3.org/2000/svg}text")
    fill_by_content: dict[str, str] = {}
    for text_node in text_nodes:
        text_content = "".join(text_node.itertext()).strip()
        if not text_content:
            continue
        style = text_node.attrib.get("style", "")
        fill_match = re.search(r"fill:\s*([^;]+)", style)
        fill = (
            fill_match.group(1).strip() if fill_match else text_node.attrib.get("fill")
        )
        if fill is not None:
            fill_by_content[text_content] = fill
    return fill_by_content


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


def test_export_geometry_prefers_perpendicular_free_index_directions_for_linear_chain() -> (
    None
):
    spec = build_three_tensor_spec()
    renderer = _SvgRenderer(spec, SvgRenderOptions())

    left_tensor = spec.tensors[0]
    left_index = left_tensor.indices[0]
    direction = renderer._index_direction(left_tensor, left_index)
    source = renderer.connection_point(left_tensor, left_index)
    target = renderer.open_index_endpoint(left_tensor, left_index)

    assert abs(direction.x) < 0.25
    assert abs(direction.y) > 0.9
    assert hypot(target.x - source.x, target.y - source.y) == pytest.approx(
        2.0 * renderer.tensor_radius(left_tensor)
    )


def test_export_geometry_respects_vertical_linear_chain_orientation() -> None:
    spec = _build_vertical_three_tensor_spec()
    renderer = _SvgRenderer(spec, SvgRenderOptions())

    first_tensor = spec.tensors[0]
    free_index = first_tensor.indices[0]
    direction = renderer._index_direction(first_tensor, free_index)

    assert abs(direction.x) > 0.9
    assert abs(direction.y) < 0.25


def test_export_geometry_prefers_linear_component_orientation_over_named_hints() -> (
    None
):
    spec = _build_vertical_three_tensor_named_hint_spec()
    renderer = _SvgRenderer(spec, SvgRenderOptions())

    first_tensor = spec.tensors[0]
    free_index = first_tensor.indices[0]
    direction = renderer._index_direction(first_tensor, free_index)

    assert abs(direction.x) > 0.9
    assert abs(direction.y) < 0.25


def test_export_geometry_respects_diagonal_linear_chain_orientation() -> None:
    spec = _build_diagonal_three_tensor_spec()
    renderer = _SvgRenderer(spec, SvgRenderOptions())

    first_tensor = spec.tensors[0]
    free_index = first_tensor.indices[0]
    direction = renderer._index_direction(first_tensor, free_index)
    chain_axis = _normalize(CanvasPosition(x=1.0, y=1.0))
    diagonal_perpendicular = _normalize(CanvasPosition(x=-1.0, y=1.0))

    assert abs(_dot(direction, chain_axis)) < 0.25
    assert abs(_dot(direction, diagonal_perpendicular)) > 0.9


def test_export_geometry_prefers_vertical_mpo_component_orientation_over_index_names() -> (
    None
):
    spec = _build_vertical_mpo_export_spec()
    renderer = _SvgRenderer(spec, SvgRenderOptions())
    first_tensor = spec.tensors[0]
    bra_index = next(index for index in first_tensor.indices if index.name == "bra")
    ket_index = next(index for index in first_tensor.indices if index.name == "ket")
    bra_direction = renderer._index_direction(first_tensor, bra_index)
    ket_direction = renderer._index_direction(first_tensor, ket_index)

    assert abs(bra_direction.x) > 0.9
    assert abs(ket_direction.x) > 0.9
    assert abs(bra_direction.y) < 0.25
    assert abs(ket_direction.y) < 0.25
    assert _dot(bra_direction, ket_direction) < -0.85


def test_export_geometry_points_cycle_free_indices_outward() -> None:
    spec = _build_cycle_spec()
    renderer = _SvgRenderer(spec, SvgRenderOptions())
    cycle_center = CanvasPosition(x=200.0, y=200.0)

    for tensor in spec.tensors:
        free_index = tensor.indices[0]
        direction = renderer._index_direction(tensor, free_index)
        radial = _normalize(
            CanvasPosition(
                x=tensor.position.x - cycle_center.x,
                y=tensor.position.y - cycle_center.y,
            )
        )
        assert _dot(direction, radial) > 0.85


def test_export_geometry_points_grid_boundary_free_indices_outward() -> None:
    spec = _build_grid_export_spec()
    renderer = _SvgRenderer(spec, SvgRenderOptions())
    expectations = {
        "tensor_0_1": CanvasPosition(x=0.0, y=-1.0),
        "tensor_1_0": CanvasPosition(x=-1.0, y=0.0),
        "tensor_1_2": CanvasPosition(x=1.0, y=0.0),
        "tensor_2_1": CanvasPosition(x=0.0, y=1.0),
    }

    for tensor_id, expected_direction in expectations.items():
        tensor = next(tensor for tensor in spec.tensors if tensor.id == tensor_id)
        free_index = tensor.indices[0]
        direction = renderer._index_direction(tensor, free_index)
        assert _dot(direction, expected_direction) > 0.85


def test_export_geometry_points_rotated_grid_boundary_free_indices_outward() -> None:
    spec = _build_rotated_grid_export_spec()
    renderer = _SvgRenderer(spec, SvgRenderOptions())
    expectations = {
        "tensor_0_1": _normalize(CanvasPosition(x=1.0, y=-1.0)),
        "tensor_1_0": _normalize(CanvasPosition(x=-1.0, y=-1.0)),
        "tensor_1_2": _normalize(CanvasPosition(x=1.0, y=1.0)),
        "tensor_2_1": _normalize(CanvasPosition(x=-1.0, y=1.0)),
    }

    for tensor_id, expected_direction in expectations.items():
        tensor = next(tensor for tensor in spec.tensors if tensor.id == tensor_id)
        free_index = tensor.indices[0]
        direction = renderer._index_direction(tensor, free_index)
        assert _dot(direction, expected_direction) > 0.85


def test_export_geometry_generic_free_indices_point_away_from_local_neighbors() -> None:
    spec = _build_generic_export_spec()
    renderer = _SvgRenderer(spec, SvgRenderOptions())
    center_tensor = spec.tensors[0]
    free_index = center_tensor.indices[0]

    direction = renderer._index_direction(center_tensor, free_index)
    away_from_neighbors = _normalize(CanvasPosition(x=-180.0, y=-120.0))

    assert _dot(direction, away_from_neighbors) > 0.75


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


def test_academic_exports_use_contrasting_tensor_label_colors_for_custom_tensor_fills() -> (
    None
):
    spec = _build_colored_parallel_edge_spec()

    pytest.importorskip("matplotlib")
    svg = render_spec_svg(spec)
    tikz = render_spec_tikz(spec)
    dot = render_spec_dot(spec)
    text_fill_by_content = _svg_text_fill_by_content(svg)

    assert text_fill_by_content["A"] == "#f5f9ff"
    assert text_fill_by_content["B"] == "#091018"
    assert r"\definecolor{tneColorf5f9ff}{HTML}{f5f9ff}" in tikz
    assert r"\definecolor{tneColor091018}{HTML}{091018}" in tikz
    assert (
        r"\node[tne tensor, minimum size=120*\tneUnit, fill=tneColor123456, "
        r"draw=tneColor385a7c, text=tneColorf5f9ff] (tensor_tensor_a)"
    ) in tikz
    assert (
        r"\node[tne tensor, minimum size=108*\tneUnit, fill=tneColorabcdef, "
        r"draw=tneColord1f3ff, text=tneColor091018] (tensor_tensor_b)"
    ) in tikz
    assert 'fontcolor="#f5f9ff"' in dot
    assert 'fontcolor="#091018"' in dot


def test_render_spec_mermaid_returns_flowchart_for_normal_network() -> None:
    mermaid = render_spec_mermaid(build_sample_spec())

    assert mermaid.startswith("flowchart LR\n")
    assert 'tensor_tensor_a["A"]' in mermaid
    assert 'tensor_tensor_b["B"]' in mermaid
    assert 'open_tensor_a_i((" "))' in mermaid
    assert "tensor_tensor_a ---|i (2)| open_tensor_a_i" in mermaid
    assert "tensor_tensor_a ---|bond_x / x=3| tensor_tensor_b" in mermaid


def test_render_spec_mermaid_can_hide_tensor_index_and_bond_labels() -> None:
    mermaid = render_spec_mermaid(
        build_sample_spec(),
        options=DotRenderOptions(
            show_tensor_labels=False,
            show_index_labels=False,
            show_edge_labels=False,
        ),
    )

    assert 'tensor_tensor_a["tensor_a"]' in mermaid
    assert 'tensor_tensor_b["tensor_b"]' in mermaid
    assert "bond_x" not in mermaid
    assert "x=3" not in mermaid
    assert 'open_tensor_a_i((" "))' in mermaid
    assert "i (2)" not in mermaid


def test_render_spec_mermaid_includes_hyperedges_groups_and_notes() -> None:
    mermaid = render_spec_mermaid(build_sample_spec())
    hyperedge_mermaid = render_spec_mermaid(build_three_tensor_hyperedge_spec())

    assert 'subgraph group_group_demo["Demo Group"]' in mermaid
    assert "%% Note: Check the contraction order" in mermaid
    assert 'hyperedge_hyperedge_h["shared_h"]' in hyperedge_mermaid
    assert "tensor_tensor_a ---|h=3| hyperedge_hyperedge_h" in hyperedge_mermaid


def test_render_spec_mermaid_writes_output_path(tmp_path: Path) -> None:
    output_path = tmp_path / "network.mmd"

    mermaid = render_spec_mermaid(build_sample_spec(), output_path=output_path)

    assert output_path.read_text(encoding="utf-8") == mermaid


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

    renderer = _SvgRenderer(spec, SvgRenderOptions())
    edge_render_infos = renderer._edge_render_infos()
    bounds = renderer._compute_bounds(edge_render_infos)
    tikz = render_spec_tikz(spec)

    assert edge_render_infos[0].source == spec.tensors[0].position
    assert edge_render_infos[0].target == spec.tensors[1].position
    expected_segment = (
        f"({_number(edge_render_infos[0].source.x - bounds.x1)}, "
        f"{_number(bounds.y2 - edge_render_infos[0].source.y)}) -- "
        f"({_number(edge_render_infos[0].target.x - bounds.x1)}, "
        f"{_number(bounds.y2 - edge_render_infos[0].target.y)})"
    )
    assert expected_segment in tikz


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


def test_validate_positive_render_scale_normalizes_and_rejects_invalid_values() -> None:
    import tensor_network_editor.rendering as rendering_module

    assert rendering_module._validate_positive_render_scale(
        2,
        description="PNG render scale",
    ) == pytest.approx(2.0)
    assert rendering_module._validate_positive_render_scale(
        1.5,
        description="TikZ render scale",
    ) == pytest.approx(1.5)

    for invalid_scale in (True, 0, -1, float("inf"), float("nan"), "2"):
        with pytest.raises(
            ValueError,
            match="PNG render scale must be a positive finite number.",
        ):
            rendering_module._validate_positive_render_scale(
                invalid_scale,
                description="PNG render scale",
            )


def test_render_spec_output_validates_renders_and_writes_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import tensor_network_editor.rendering as rendering_module

    spec = build_sample_spec()
    validated_spec = build_three_tensor_spec()
    output_path = tmp_path / "network.svg"
    calls: dict[str, Any] = {}

    def fake_validate(received_spec: NetworkSpec) -> NetworkSpec:
        calls["validate"] = received_spec
        return validated_spec

    def fake_render(
        received_spec: NetworkSpec,
        received_options: SvgRenderOptions,
    ) -> str:
        calls["render"] = (received_spec, received_options)
        return "<svg />"

    def fake_write(
        path: Path,
        content: str,
        *,
        description: str,
    ) -> None:
        calls["write"] = (path, content, description)

    monkeypatch.setattr(rendering_module, "ensure_valid_spec", fake_validate)
    options = SvgRenderOptions(show_tensor_labels=False)

    rendered = rendering_module._render_spec_output(
        spec,
        format_name="svg",
        options=options,
        output_path=output_path,
        description="SVG network rendering",
        render=fake_render,
        writer=fake_write,
    )

    assert rendered == "<svg />"
    assert calls["validate"] is spec
    assert calls["render"] == (validated_spec, options)
    assert calls["write"] == (output_path, "<svg />", "SVG network rendering")


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
