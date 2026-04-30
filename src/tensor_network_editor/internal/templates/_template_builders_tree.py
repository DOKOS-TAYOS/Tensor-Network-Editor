"""Tree-family internal template builders."""

from __future__ import annotations

from ...models import NetworkSpec, TensorSpec
from ._template_builders_common import (
    DOWN_OFFSET,
    HORIZONTAL_SPACING,
    LAYER_SPACING,
    LEFT_OFFSET,
    LOWER_LEFT_OFFSET,
    LOWER_RIGHT_OFFSET,
    RIGHT_OFFSET,
    TREE_LEAF_SPACING,
    UP_OFFSET,
    TemplateIndexConfig,
    _annotate_physics_1d_indices,
    _make_edge,
    _make_tensor,
    _resolve_graph_size,
    _resolve_ttn_depth,
)
from ._template_catalog import TEMPLATE_DEFINITIONS, TemplateParameters


def _build_mera_template(parameters: TemplateParameters) -> NetworkSpec:
    """Build the requested MERA template variant."""
    if (
        _resolve_graph_size(parameters)
        == TEMPLATE_DEFINITIONS["mera"].defaults.graph_size
    ):
        return _build_default_mera_template(parameters)
    return _build_generic_mera_template(parameters)


def _build_default_mera_template(parameters: TemplateParameters) -> NetworkSpec:
    """Build the default depth-3 MERA layout."""
    tensors = [
        _make_tensor(
            "tensor_top",
            "Top",
            320.0,
            0.0,
            [
                ("left", parameters.bond_dimension, LEFT_OFFSET),
                ("right", parameters.bond_dimension, RIGHT_OFFSET),
            ],
        ),
        _make_tensor(
            "tensor_mid_left",
            "Mid L",
            120.0,
            210.0,
            [
                ("up", parameters.bond_dimension, UP_OFFSET),
                ("left", parameters.bond_dimension, LEFT_OFFSET),
                ("down", parameters.bond_dimension, DOWN_OFFSET),
            ],
        ),
        _make_tensor(
            "tensor_mid_right",
            "Mid R",
            520.0,
            210.0,
            [
                ("up", parameters.bond_dimension, UP_OFFSET),
                ("down", parameters.bond_dimension, DOWN_OFFSET),
                ("right", parameters.bond_dimension, RIGHT_OFFSET),
            ],
        ),
        _make_tensor(
            "tensor_leaf_left",
            "Leaf L",
            0.0,
            420.0,
            [
                ("up", parameters.bond_dimension, UP_OFFSET),
                ("phys", parameters.physical_dimension, DOWN_OFFSET),
            ],
        ),
        _make_tensor(
            "tensor_leaf_mid",
            "Leaf M",
            320.0,
            420.0,
            [
                ("left", parameters.bond_dimension, LEFT_OFFSET),
                ("right", parameters.bond_dimension, RIGHT_OFFSET),
                ("phys", parameters.physical_dimension, DOWN_OFFSET),
            ],
        ),
        _make_tensor(
            "tensor_leaf_right",
            "Leaf R",
            640.0,
            420.0,
            [
                ("up", parameters.bond_dimension, UP_OFFSET),
                ("phys", parameters.physical_dimension, DOWN_OFFSET),
            ],
        ),
    ]
    edges = [
        _make_edge("edge_top_left", tensors[0], "left", tensors[1], "up"),
        _make_edge("edge_top_right", tensors[0], "right", tensors[2], "up"),
        _make_edge("edge_left_leaf", tensors[1], "left", tensors[3], "up"),
        _make_edge("edge_center_leaf", tensors[1], "down", tensors[4], "left"),
        _make_edge("edge_right_center", tensors[2], "down", tensors[4], "right"),
        _make_edge("edge_right_leaf", tensors[2], "right", tensors[5], "up"),
    ]
    return NetworkSpec(
        id="template_mera_3",
        name="MERA",
        tensors=tensors,
        edges=edges,
    )


def _build_generic_mera_template(parameters: TemplateParameters) -> NetworkSpec:
    """Build a generic MERA layout with the requested depth."""
    depth = _resolve_graph_size(parameters)
    levels: list[list[TensorSpec]] = []
    for level_index in range(depth):
        level_tensors: list[TensorSpec] = []
        for position_index in range(level_index + 1):
            tensor_indices: list[TemplateIndexConfig] = []
            if position_index > 0:
                tensor_indices.append(
                    ("up_left", parameters.bond_dimension, LEFT_OFFSET)
                )
            if position_index < level_index:
                tensor_indices.append(
                    ("up_right", parameters.bond_dimension, UP_OFFSET)
                )
            if level_index < depth - 1:
                tensor_indices.append(
                    ("down_left", parameters.bond_dimension, LOWER_LEFT_OFFSET)
                )
                tensor_indices.append(
                    ("down_right", parameters.bond_dimension, LOWER_RIGHT_OFFSET)
                )
            if level_index == depth - 1:
                tensor_indices.append(
                    ("phys", parameters.physical_dimension, DOWN_OFFSET)
                )
            tensor = _make_tensor(
                f"tensor_l{level_index + 1}_{position_index + 1}",
                f"L{level_index + 1}-{position_index + 1}",
                position_index * HORIZONTAL_SPACING
                + ((depth - level_index - 1) * HORIZONTAL_SPACING) / 2,
                level_index * LAYER_SPACING,
                tensor_indices,
            )
            level_tensors.append(tensor)
        levels.append(level_tensors)
    edges = []
    for level_index in range(depth - 1):
        for position_index, tensor in enumerate(levels[level_index]):
            left_child = levels[level_index + 1][position_index]
            right_child = levels[level_index + 1][position_index + 1]
            edges.append(
                _make_edge(
                    f"edge_l{level_index + 1}_{position_index + 1}_left",
                    tensor,
                    "down_left",
                    left_child,
                    "up_right",
                )
            )
            edges.append(
                _make_edge(
                    f"edge_l{level_index + 1}_{position_index + 1}_right",
                    tensor,
                    "down_right",
                    right_child,
                    "up_left",
                )
            )
    return NetworkSpec(
        id=f"template_mera_{depth}",
        name=f"MERA depth {depth}",
        tensors=[tensor for level in levels for tensor in level],
        edges=edges,
    )


def _build_ttn_template(parameters: TemplateParameters) -> NetworkSpec:
    """Build the canonical TTN layout."""
    depth = _resolve_ttn_depth(parameters)
    spec = _build_generic_ttn_template(parameters, depth=depth)
    spec.id = f"template_ttn_{depth}"
    spec.name = f"TTN depth {depth}"
    spec.metadata = {
        "template_name": "ttn",
        "depth": depth,
        "leaf_physical_legs": parameters.leaf_physical_legs,
        "root_open_leg": parameters.root_open_leg,
        "isometric": parameters.isometric,
    }
    for tensor in spec.tensors:
        tensor.metadata = {
            "role": "isometry" if parameters.isometric else "tensor",
            "family": "ttn",
            "isometric": parameters.isometric,
        }
        _annotate_physics_1d_indices(tensor, symmetry="none")
    return spec


def _build_generic_ttn_template(
    parameters: TemplateParameters,
    *,
    depth: int,
) -> NetworkSpec:
    """Build a generic TTN with the requested depth."""
    levels: list[list[TensorSpec]] = []
    for level_index in range(depth):
        level_tensors: list[TensorSpec] = []
        node_count = 2**level_index
        for position_index in range(node_count):
            tensor_indices: list[TemplateIndexConfig] = []
            if level_index > 0:
                tensor_indices.append(("up", parameters.bond_dimension, UP_OFFSET))
            if level_index < depth - 1:
                tensor_indices.append(("left", parameters.bond_dimension, LEFT_OFFSET))
                tensor_indices.append(
                    ("right", parameters.bond_dimension, RIGHT_OFFSET)
                )
            if level_index == 0 and parameters.root_open_leg:
                tensor_indices.append(("out", parameters.bond_dimension, UP_OFFSET))
            if level_index == depth - 1 and parameters.leaf_physical_legs:
                tensor_indices.append(
                    ("phys", parameters.physical_dimension, DOWN_OFFSET)
                )
            x_position = (
                ((2 * position_index + 1) * (2 ** (depth - level_index - 1)) - 1)
                * TREE_LEAF_SPACING
                / 2
            )
            tensor = _make_tensor(
                f"tensor_l{level_index + 1}_{position_index + 1}",
                f"L{level_index + 1}-{position_index + 1}",
                x_position,
                level_index * LAYER_SPACING,
                tensor_indices,
            )
            level_tensors.append(tensor)
        levels.append(level_tensors)
    edges = []
    for level_index in range(depth - 1):
        for position_index, tensor in enumerate(levels[level_index]):
            left_child = levels[level_index + 1][position_index * 2]
            right_child = levels[level_index + 1][position_index * 2 + 1]
            edges.append(
                _make_edge(
                    f"edge_l{level_index + 1}_{position_index + 1}_left",
                    tensor,
                    "left",
                    left_child,
                    "up",
                )
            )
            edges.append(
                _make_edge(
                    f"edge_l{level_index + 1}_{position_index + 1}_right",
                    tensor,
                    "right",
                    right_child,
                    "up",
                )
            )
    return NetworkSpec(
        id=f"template_ttn_{depth}",
        name=f"TTN depth {depth}",
        tensors=[tensor for level in levels for tensor in level],
        edges=edges,
    )
