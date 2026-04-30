"""Grid and layer-oriented internal template builders."""

from __future__ import annotations

from collections.abc import Callable

from ...models import EdgeSpec, NetworkSpec, TensorSpec
from ._template_builders_common import (
    DOWN_OFFSET,
    GATE_LOWER_LEFT_OFFSET,
    GATE_LOWER_RIGHT_OFFSET,
    GATE_UPPER_LEFT_OFFSET,
    GATE_UPPER_RIGHT_OFFSET,
    HORIZONTAL_SPACING,
    LEFT_OFFSET,
    LOWER_LEFT_OFFSET,
    LOWER_PHYSICAL_OFFSET,
    LOWER_RIGHT_OFFSET,
    RIGHT_OFFSET,
    UP_OFFSET,
    UPPER_PHYSICAL_OFFSET,
    VERTICAL_SPACING,
    TemplateIndexConfig,
    _annotate_physics_1d_indices,
    _make_edge,
    _make_tensor,
    _resolve_graph_size,
)
from ._template_builders_linear import _build_mps_site_indices, _make_linear_chain_edges
from ._template_catalog import TEMPLATE_DEFINITIONS, TemplateParameters

GridSiteIndexBuilder = Callable[
    [int, int, int, TemplateParameters], list[TemplateIndexConfig]
]


def _build_peps_template(parameters: TemplateParameters) -> NetworkSpec:
    """Build the requested PEPS template variant."""
    if _resolve_graph_size(parameters) == 2:
        return _build_default_peps_template(parameters)
    return _build_generic_peps_template(parameters)


def _build_default_peps_template(parameters: TemplateParameters) -> NetworkSpec:
    """Build the default 2x2 PEPS layout."""
    tensors = [
        _make_tensor(
            "tensor_a",
            "A",
            0.0,
            0.0,
            [
                ("right", parameters.bond_dimension, RIGHT_OFFSET),
                ("down", parameters.bond_dimension, DOWN_OFFSET),
                ("phys", parameters.physical_dimension, LOWER_LEFT_OFFSET),
            ],
        ),
        _make_tensor(
            "tensor_b",
            "B",
            340.0,
            0.0,
            [
                ("left", parameters.bond_dimension, LEFT_OFFSET),
                ("down", parameters.bond_dimension, DOWN_OFFSET),
                ("phys", parameters.physical_dimension, LOWER_RIGHT_OFFSET),
            ],
        ),
        _make_tensor(
            "tensor_c",
            "C",
            0.0,
            VERTICAL_SPACING,
            [
                ("right", parameters.bond_dimension, RIGHT_OFFSET),
                ("up", parameters.bond_dimension, UP_OFFSET),
                ("phys", parameters.physical_dimension, LOWER_LEFT_OFFSET),
            ],
        ),
        _make_tensor(
            "tensor_d",
            "D",
            340.0,
            VERTICAL_SPACING,
            [
                ("left", parameters.bond_dimension, LEFT_OFFSET),
                ("up", parameters.bond_dimension, UP_OFFSET),
                ("phys", parameters.physical_dimension, LOWER_RIGHT_OFFSET),
            ],
        ),
    ]
    edges = [
        _make_edge("edge_ab", tensors[0], "right", tensors[1], "left"),
        _make_edge("edge_cd", tensors[2], "right", tensors[3], "left"),
        _make_edge("edge_ac", tensors[0], "down", tensors[2], "up"),
        _make_edge("edge_bd", tensors[1], "down", tensors[3], "up"),
    ]
    return NetworkSpec(
        id="template_peps_2",
        name="PEPS 2x2",
        tensors=tensors,
        edges=edges,
    )


def _build_generic_peps_template(parameters: TemplateParameters) -> NetworkSpec:
    """Build a square PEPS grid larger than the default 2x2 layout."""
    size = _resolve_graph_size(parameters)
    tensors, edges = _build_square_grid_tensors_and_edges(
        size=size,
        parameters=parameters,
        site_index_builder=_build_peps_grid_site_indices,
    )
    return NetworkSpec(
        id=f"template_peps_{size}",
        name=f"PEPS {size}x{size}",
        tensors=tensors,
        edges=edges,
    )


def _build_pepo_template(parameters: TemplateParameters) -> NetworkSpec:
    """Build a square PEPO operator grid with bra and ket physical legs."""
    size = _resolve_graph_size(parameters)
    tensors, edges = _build_square_grid_tensors_and_edges(
        size=size,
        parameters=parameters,
        site_index_builder=_build_pepo_grid_site_indices,
    )
    return NetworkSpec(
        id=f"template_pepo_{size}",
        name=f"PEPO {size}x{size}",
        tensors=tensors,
        edges=edges,
    )


def _build_square_grid_tensors_and_edges(
    *,
    size: int,
    parameters: TemplateParameters,
    site_index_builder: GridSiteIndexBuilder,
) -> tuple[list[TensorSpec], list[EdgeSpec]]:
    """Build the tensors and nearest-neighbor edges for one square grid."""
    tensors: list[TensorSpec] = []
    tensor_lookup: dict[tuple[int, int], TensorSpec] = {}
    for row_index in range(size):
        for column_index in range(size):
            tensor = _make_tensor(
                f"tensor_r{row_index + 1}_c{column_index + 1}",
                _grid_tensor_name(row_index, column_index),
                340.0 * column_index,
                VERTICAL_SPACING * row_index,
                site_index_builder(row_index, column_index, size, parameters),
            )
            tensors.append(tensor)
            tensor_lookup[(row_index, column_index)] = tensor
    edges: list[EdgeSpec] = []
    for row_index in range(size):
        for column_index in range(size):
            current_tensor = tensor_lookup[(row_index, column_index)]
            if column_index + 1 < size:
                edges.append(
                    _make_edge(
                        f"edge_r{row_index + 1}_c{column_index + 1}_right",
                        current_tensor,
                        "right",
                        tensor_lookup[(row_index, column_index + 1)],
                        "left",
                    )
                )
            if row_index + 1 < size:
                edges.append(
                    _make_edge(
                        f"edge_r{row_index + 1}_c{column_index + 1}_down",
                        current_tensor,
                        "down",
                        tensor_lookup[(row_index + 1, column_index)],
                        "up",
                    )
                )
    return tensors, edges


def _build_peps_grid_site_indices(
    row_index: int,
    column_index: int,
    size: int,
    parameters: TemplateParameters,
) -> list[TemplateIndexConfig]:
    """Return the index layout for one PEPS grid tensor."""
    tensor_indices = _build_grid_neighbor_indices(
        row_index=row_index,
        column_index=column_index,
        size=size,
        parameters=parameters,
    )
    tensor_indices.append(
        (
            "phys",
            parameters.physical_dimension,
            LOWER_LEFT_OFFSET if column_index % 2 == 0 else LOWER_RIGHT_OFFSET,
        )
    )
    return tensor_indices


def _build_pepo_grid_site_indices(
    row_index: int,
    column_index: int,
    size: int,
    parameters: TemplateParameters,
) -> list[TemplateIndexConfig]:
    """Return the index layout for one PEPO grid tensor."""
    tensor_indices = _build_grid_neighbor_indices(
        row_index=row_index,
        column_index=column_index,
        size=size,
        parameters=parameters,
    )
    tensor_indices.extend(
        [
            ("bra", parameters.physical_dimension, UPPER_PHYSICAL_OFFSET),
            ("ket", parameters.physical_dimension, LOWER_PHYSICAL_OFFSET),
        ]
    )
    return tensor_indices


def _build_grid_neighbor_indices(
    *,
    row_index: int,
    column_index: int,
    size: int,
    parameters: TemplateParameters,
) -> list[TemplateIndexConfig]:
    """Return the horizontal and vertical bond indices for one grid tensor."""
    tensor_indices: list[TemplateIndexConfig] = []
    if column_index > 0:
        tensor_indices.append(("left", parameters.bond_dimension, LEFT_OFFSET))
    if column_index < size - 1:
        tensor_indices.append(("right", parameters.bond_dimension, RIGHT_OFFSET))
    if row_index > 0:
        tensor_indices.append(("up", parameters.bond_dimension, UP_OFFSET))
    if row_index < size - 1:
        tensor_indices.append(("down", parameters.bond_dimension, DOWN_OFFSET))
    return tensor_indices


def _grid_tensor_name(row_index: int, column_index: int) -> str:
    """Return a readable tensor name for a square-grid position."""
    if row_index < 26:
        return f"{chr(ord('A') + row_index)}{column_index + 1}"
    return f"R{row_index + 1}C{column_index + 1}"


def _build_tebd_gate_layer_template(parameters: TemplateParameters) -> NetworkSpec:
    """Build an MPS chain with an even TEBD two-site gate layer."""
    site_count = _resolve_graph_size(parameters)
    site_tensors = [
        _make_tensor(
            f"tensor_site_{site_index + 1}",
            f"A{site_index + 1}",
            HORIZONTAL_SPACING * site_index,
            0.0,
            _build_mps_site_indices(site_index, site_count, parameters),
        )
        for site_index in range(site_count)
    ]
    gate_tensors = [
        _make_tensor(
            f"tensor_gate_{site_index + 1}_{site_index + 2}",
            f"G{site_index + 1}-{site_index + 2}",
            HORIZONTAL_SPACING * (site_index + 0.5),
            220.0,
            [
                ("out_left", parameters.physical_dimension, GATE_UPPER_LEFT_OFFSET),
                ("out_right", parameters.physical_dimension, GATE_UPPER_RIGHT_OFFSET),
                ("in_left", parameters.physical_dimension, GATE_LOWER_LEFT_OFFSET),
                ("in_right", parameters.physical_dimension, GATE_LOWER_RIGHT_OFFSET),
            ],
        )
        for site_index in range(0, site_count - 1, 2)
    ]
    edges = _make_linear_chain_edges(site_tensors)
    for gate_index, gate_tensor in enumerate(gate_tensors):
        left_site_index = gate_index * 2
        right_site_index = left_site_index + 1
        edges.extend(
            [
                _make_edge(
                    f"edge_gate_{left_site_index + 1}_{right_site_index + 1}_left",
                    site_tensors[left_site_index],
                    "phys",
                    gate_tensor,
                    "in_left",
                ),
                _make_edge(
                    f"edge_gate_{left_site_index + 1}_{right_site_index + 1}_right",
                    site_tensors[right_site_index],
                    "phys",
                    gate_tensor,
                    "in_right",
                ),
            ]
        )
    for tensor in site_tensors:
        tensor.metadata = {
            "role": "state",
            "state": "tebd_input",
            "symmetry": "z2",
            "tags": "tebd mps site",
        }
        _annotate_physics_1d_indices(tensor, symmetry="z2")
    for tensor in gate_tensors:
        tensor.metadata = {
            "role": "gate",
            "symmetry": "z2",
            "tags": "tebd even layer",
        }
        _annotate_physics_1d_indices(tensor, symmetry="z2")
    definition = TEMPLATE_DEFINITIONS["tebd_gate_layer"]
    spec_name = (
        definition.display_name
        if site_count == definition.defaults.graph_size
        else f"{definition.display_name} ({site_count} {definition.graph_size_label.lower()})"
    )
    return NetworkSpec(
        id=f"template_tebd_gate_layer_{site_count}",
        name=spec_name,
        tensors=[*site_tensors, *gate_tensors],
        edges=edges,
    )
