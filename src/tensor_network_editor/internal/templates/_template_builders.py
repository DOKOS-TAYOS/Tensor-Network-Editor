"""Builders for the package's built-in tensor-network templates."""

from __future__ import annotations

from collections.abc import Callable

from ...models import (
    CanvasPosition,
    EdgeEndpointRef,
    EdgeSpec,
    IndexSpec,
    NetworkSpec,
    TensorDataMode,
    TensorDataSpec,
    TensorSpec,
)
from ...validation import ensure_valid_spec
from ..models._model_tensor_data import TensorNumericLiteral
from ._template_catalog import (
    TEMPLATE_DEFINITIONS,
    TemplateParameters,
    get_template_builder,
    get_template_definition,
    register_template,
    validate_template_parameters,
)

HORIZONTAL_SPACING = 320.0
VERTICAL_SPACING = 280.0
LAYER_SPACING = 210.0
TREE_LEAF_SPACING = 220.0
LEFT_OFFSET = (-58.0, 0.0)
RIGHT_OFFSET = (58.0, 0.0)
UP_OFFSET = (0.0, -28.0)
DOWN_OFFSET = (0.0, 28.0)
LOWER_LEFT_OFFSET = (-24.0, 34.0)
LOWER_RIGHT_OFFSET = (24.0, 34.0)
UPPER_PHYSICAL_OFFSET = (-26.0, -54.0)
LOWER_PHYSICAL_OFFSET = (26.0, 42.0)
GATE_UPPER_LEFT_OFFSET = (-36.0, -38.0)
GATE_UPPER_RIGHT_OFFSET = (36.0, -38.0)
GATE_LOWER_LEFT_OFFSET = (-36.0, 38.0)
GATE_LOWER_RIGHT_OFFSET = (36.0, 38.0)
TemplateIndexConfig = tuple[str, int | None, tuple[float, float]]
LinearChainSiteIndexBuilder = Callable[
    [int, int, TemplateParameters], list[TemplateIndexConfig]
]


def build_template(
    template_name: str, parameters: TemplateParameters | None = None
) -> NetworkSpec:
    """Build and validate the named built-in template."""
    definition = get_template_definition(template_name)
    resolved_parameters = (
        validate_template_parameters(
            template_name,
            parameters or definition.defaults,
        )
        if definition.supports_parameters
        else definition.defaults
    )
    builder = get_template_builder(template_name)
    return ensure_valid_spec(builder(resolved_parameters))


def _build_mps_template(parameters: TemplateParameters) -> NetworkSpec:
    """Build an MPS template with the requested site count and dimensions."""
    spec = _build_linear_chain_template(
        "mps",
        parameters,
        tensor_name_prefix="A",
        spacing=HORIZONTAL_SPACING,
        site_index_builder=_build_mps_site_indices,
        periodic=parameters.boundary_condition == "periodic",
    )
    return _apply_mps_template_configuration(spec, parameters)


def _build_mpo_template(parameters: TemplateParameters) -> NetworkSpec:
    """Build an MPO template with the requested site count and dimensions."""
    spec = _build_linear_chain_template(
        "mpo",
        parameters,
        tensor_name_prefix="W",
        spacing=330.0,
        site_index_builder=_build_mpo_site_indices,
        periodic=parameters.boundary_condition == "periodic",
    )
    return _apply_mpo_template_configuration(spec, parameters)


def _build_linear_chain_template(
    template_name: str,
    parameters: TemplateParameters,
    *,
    tensor_name_prefix: str,
    spacing: float,
    site_index_builder: LinearChainSiteIndexBuilder,
    periodic: bool = False,
) -> NetworkSpec:
    """Build one left-to-right chain template from a per-site index factory."""
    length = _resolve_graph_size(parameters)
    tensors = [
        _make_tensor(
            f"tensor_{site_index}",
            f"{tensor_name_prefix}{site_index + 1}",
            spacing * site_index,
            0.0,
            site_index_builder(site_index, length, parameters),
        )
        for site_index in range(length)
    ]
    definition = TEMPLATE_DEFINITIONS[template_name]
    spec_name = (
        definition.display_name
        if length == definition.defaults.graph_size
        else f"{definition.display_name} ({length} {definition.graph_size_label.lower()})"
    )
    return NetworkSpec(
        id=f"template_{template_name}_{length}",
        name=spec_name,
        tensors=tensors,
        edges=_make_linear_chain_edges(tensors, periodic=periodic),
    )


def _build_mps_site_indices(
    site_index: int,
    length: int,
    parameters: TemplateParameters,
) -> list[TemplateIndexConfig]:
    """Return the named index layout for one MPS site."""
    tensor_indices: list[TemplateIndexConfig] = []
    if parameters.boundary_condition == "periodic" or site_index > 0:
        tensor_indices.append(("left", parameters.bond_dimension, LEFT_OFFSET))
    if parameters.boundary_condition == "periodic" or site_index < length - 1:
        tensor_indices.append(("right", parameters.bond_dimension, RIGHT_OFFSET))
    tensor_indices.append(("phys", parameters.physical_dimension, DOWN_OFFSET))
    return tensor_indices


def _build_mpo_site_indices(
    site_index: int,
    length: int,
    parameters: TemplateParameters,
) -> list[TemplateIndexConfig]:
    """Return the named index layout for one MPO site."""
    tensor_indices: list[TemplateIndexConfig] = []
    if parameters.boundary_condition == "periodic" or site_index > 0:
        tensor_indices.append(("left", parameters.bond_dimension, LEFT_OFFSET))
    if parameters.boundary_condition == "periodic" or site_index < length - 1:
        tensor_indices.append(("right", parameters.bond_dimension, RIGHT_OFFSET))
    tensor_indices.extend(
        [
            ("bra", parameters.physical_dimension, UP_OFFSET),
            ("ket", parameters.physical_dimension, DOWN_OFFSET),
        ]
    )
    return tensor_indices


def _make_linear_chain_edges(
    tensors: list[TensorSpec],
    *,
    periodic: bool = False,
) -> list[EdgeSpec]:
    """Return the right-to-left bonds between adjacent chain tensors."""
    edges = [
        _make_edge(
            f"edge_{site_index}_{site_index + 1}",
            tensors[site_index],
            "right",
            tensors[site_index + 1],
            "left",
        )
        for site_index in range(len(tensors) - 1)
    ]
    if periodic and len(tensors) > 1:
        edges.append(
            _make_edge(
                f"edge_{len(tensors)}_1",
                tensors[-1],
                "right",
                tensors[0],
                "left",
            )
        )
    return edges


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
    tensors: list[TensorSpec] = []
    tensor_lookup: dict[tuple[int, int], TensorSpec] = {}
    for row_index in range(size):
        for column_index in range(size):
            tensor_indices: list[TemplateIndexConfig] = []
            if column_index > 0:
                tensor_indices.append(("left", parameters.bond_dimension, LEFT_OFFSET))
            if column_index < size - 1:
                tensor_indices.append(
                    ("right", parameters.bond_dimension, RIGHT_OFFSET)
                )
            if row_index > 0:
                tensor_indices.append(("up", parameters.bond_dimension, UP_OFFSET))
            if row_index < size - 1:
                tensor_indices.append(("down", parameters.bond_dimension, DOWN_OFFSET))
            tensor_indices.append(
                (
                    "phys",
                    parameters.physical_dimension,
                    LOWER_LEFT_OFFSET if column_index % 2 == 0 else LOWER_RIGHT_OFFSET,
                )
            )
            tensor = _make_tensor(
                f"tensor_r{row_index + 1}_c{column_index + 1}",
                _grid_tensor_name(row_index, column_index),
                340.0 * column_index,
                VERTICAL_SPACING * row_index,
                tensor_indices,
            )
            tensors.append(tensor)
            tensor_lookup[(row_index, column_index)] = tensor
    edges = []
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
    return NetworkSpec(
        id=f"template_peps_{size}",
        name=f"PEPS {size}x{size}",
        tensors=tensors,
        edges=edges,
    )


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
        level_tensors = []
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
        level_tensors = []
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


def _grid_tensor_name(row_index: int, column_index: int) -> str:
    """Return a readable tensor name for a PEPS grid position."""
    if row_index < 26:
        return f"{chr(ord('A') + row_index)}{column_index + 1}"
    return f"R{row_index + 1}C{column_index + 1}"


def _build_pepo_template(parameters: TemplateParameters) -> NetworkSpec:
    """Build a square PEPO operator grid with bra and ket physical legs."""
    size = _resolve_graph_size(parameters)
    tensors: list[TensorSpec] = []
    tensor_lookup: dict[tuple[int, int], TensorSpec] = {}
    for row_index in range(size):
        for column_index in range(size):
            tensor_indices: list[TemplateIndexConfig] = []
            if column_index > 0:
                tensor_indices.append(("left", parameters.bond_dimension, LEFT_OFFSET))
            if column_index < size - 1:
                tensor_indices.append(
                    ("right", parameters.bond_dimension, RIGHT_OFFSET)
                )
            if row_index > 0:
                tensor_indices.append(("up", parameters.bond_dimension, UP_OFFSET))
            if row_index < size - 1:
                tensor_indices.append(("down", parameters.bond_dimension, DOWN_OFFSET))
            tensor_indices.extend(
                [
                    ("bra", parameters.physical_dimension, UPPER_PHYSICAL_OFFSET),
                    ("ket", parameters.physical_dimension, LOWER_PHYSICAL_OFFSET),
                ]
            )
            tensor = _make_tensor(
                f"tensor_r{row_index + 1}_c{column_index + 1}",
                _grid_tensor_name(row_index, column_index),
                340.0 * column_index,
                VERTICAL_SPACING * row_index,
                tensor_indices,
            )
            tensors.append(tensor)
            tensor_lookup[(row_index, column_index)] = tensor
    edges = []
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
    return NetworkSpec(
        id=f"template_pepo_{size}",
        name=f"PEPO {size}x{size}",
        tensors=tensors,
        edges=edges,
    )


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
                (
                    "out_left",
                    parameters.physical_dimension,
                    GATE_UPPER_LEFT_OFFSET,
                ),
                (
                    "out_right",
                    parameters.physical_dimension,
                    GATE_UPPER_RIGHT_OFFSET,
                ),
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


def _annotate_physics_1d_indices(tensor: TensorSpec, *, symmetry: str) -> None:
    """Add guided metadata to standard 1D physics template indices."""
    for index in tensor.indices:
        if index.name in {"left", "right"}:
            index.metadata = {"leg_kind": "bond", "symmetry": symmetry}
        else:
            index.metadata = {"leg_kind": "physical", "symmetry": symmetry}


def _apply_mps_template_configuration(
    spec: NetworkSpec,
    parameters: TemplateParameters,
) -> NetworkSpec:
    """Attach metadata and tensor initialization presets to the built MPS."""
    spec.metadata = {
        "template_name": "mps",
        "role": "state",
        "boundary_condition": parameters.boundary_condition,
        "symmetry": parameters.symmetry,
        "initial_state": parameters.initial_state,
    }
    for tensor_index, tensor in enumerate(spec.tensors):
        tensor.metadata = {
            "role": "state",
            "family": "mps",
            "symmetry": parameters.symmetry,
            "initial_state": parameters.initial_state,
        }
        _annotate_physics_1d_indices(tensor, symmetry=parameters.symmetry)
        tensor.tensor_data = _build_mps_tensor_data(
            tensor,
            tensor_index=tensor_index,
            parameters=parameters,
        )
    return spec


def _apply_mpo_template_configuration(
    spec: NetworkSpec,
    parameters: TemplateParameters,
) -> NetworkSpec:
    """Attach semantic MPO metadata to the built operator chain."""
    spec.metadata = {
        "template_name": "mpo",
        "role": "operator",
        "boundary_condition": parameters.boundary_condition,
        "j": parameters.j,
        "h": parameters.h,
    }
    for tensor in spec.tensors:
        tensor.metadata = {
            "role": "operator",
            "family": "mpo",
            "boundary_condition": parameters.boundary_condition,
            "j": parameters.j,
            "h": parameters.h,
        }
        _annotate_physics_1d_indices(tensor, symmetry="none")
    return spec


def _build_mps_tensor_data(
    tensor: TensorSpec,
    *,
    tensor_index: int,
    parameters: TemplateParameters,
) -> TensorDataSpec:
    """Return the tensor-data initializer matching the chosen MPS preset."""
    if parameters.initial_state == "zeros":
        return TensorDataSpec(mode=TensorDataMode.ZEROS)
    if parameters.initial_state == "random":
        return TensorDataSpec(
            mode=TensorDataMode.RANDOM,
            seed=tensor_index,
        )
    if parameters.initial_state == "all_up":
        return _build_mps_literal_state_tensor_data(tensor, basis_index=0)
    if parameters.initial_state == "all_down":
        return _build_mps_literal_state_tensor_data(tensor, basis_index=1)
    return _build_mps_literal_state_tensor_data(
        tensor,
        basis_index=tensor_index % 2,
    )


def _build_mps_literal_state_tensor_data(
    tensor: TensorSpec,
    *,
    basis_index: int,
) -> TensorDataSpec:
    """Build one explicit basis-state tensor embedded into the current shape."""
    values = _build_zero_literal(list(tensor.shape))
    _set_nested_literal_value(
        values,
        [0] * (len(tensor.shape) - 1) + [basis_index],
        1.0,
    )
    return TensorDataSpec(
        mode=TensorDataMode.LITERAL,
        values=values,
    )


def _build_zero_literal(shape: list[int]) -> TensorNumericLiteral:
    """Build one nested zero-filled literal matching the provided shape."""
    if not shape:
        return 0.0
    return [_build_zero_literal(shape[1:]) for _ in range(shape[0])]


def _set_nested_literal_value(
    values: TensorNumericLiteral,
    index_path: list[int],
    value: float,
) -> None:
    """Assign one scalar inside a nested tensor literal structure."""
    current: TensorNumericLiteral = values
    for index in index_path[:-1]:
        current = current[index]  # type: ignore[index]
    current[index_path[-1]] = value  # type: ignore[index]


def _make_tensor(
    tensor_id: str,
    name: str,
    x: float,
    y: float,
    indices: list[TemplateIndexConfig],
) -> TensorSpec:
    """Create one template tensor with named indices and canvas placement."""
    return TensorSpec(
        id=tensor_id,
        name=name,
        position=CanvasPosition(x=x, y=y),
        indices=[
            _make_named_index(tensor_id, suffix, dimension, offset)
            for suffix, dimension, offset in indices
        ],
    )


def _make_named_index(
    tensor_id: str,
    suffix: str,
    dimension: int | None,
    offset: tuple[float, float],
) -> IndexSpec:
    """Create one named index for a template tensor."""
    return IndexSpec(
        id=f"{tensor_id}_{suffix}",
        name=suffix,
        dimension=_resolve_required_dimension(dimension),
        offset=CanvasPosition(x=offset[0], y=offset[1]),
    )


def _resolve_required_dimension(dimension: int | None) -> int:
    """Return one validated template index dimension."""
    if dimension is None:
        raise ValueError("Template index dimensions must be resolved before building.")
    return dimension


def _make_edge(
    edge_id: str,
    left_tensor: TensorSpec,
    left_index_suffix: str,
    right_tensor: TensorSpec,
    right_index_suffix: str,
) -> EdgeSpec:
    """Create one template edge between two named tensor indices."""
    return EdgeSpec(
        id=edge_id,
        name=edge_id.replace("_", "-"),
        left=EdgeEndpointRef(
            tensor_id=left_tensor.id,
            index_id=f"{left_tensor.id}_{left_index_suffix}",
        ),
        right=EdgeEndpointRef(
            tensor_id=right_tensor.id,
            index_id=f"{right_tensor.id}_{right_index_suffix}",
        ),
    )


def _resolve_graph_size(parameters: TemplateParameters) -> int:
    """Return the validated graph-size parameter for size-based templates."""
    if parameters.graph_size is None:
        raise ValueError("Template parameter 'graph_size' is required.")
    return parameters.graph_size


def _resolve_ttn_depth(parameters: TemplateParameters) -> int:
    """Return the validated depth parameter for the TTN template."""
    if parameters.depth is None:
        raise ValueError("Template parameter 'depth' is required.")
    return parameters.depth


def register_builtin_templates() -> None:
    """Register the built-in templates in their stable display order."""
    register_template(
        "mps",
        TEMPLATE_DEFINITIONS["mps"],
        _build_mps_template,
        overwrite=True,
    )
    register_template(
        "mpo",
        TEMPLATE_DEFINITIONS["mpo"],
        _build_mpo_template,
        overwrite=True,
    )
    register_template(
        "peps_2x2",
        TEMPLATE_DEFINITIONS["peps_2x2"],
        _build_peps_template,
        overwrite=True,
    )
    register_template(
        "mera",
        TEMPLATE_DEFINITIONS["mera"],
        _build_mera_template,
        overwrite=True,
    )
    register_template(
        "ttn",
        TEMPLATE_DEFINITIONS["ttn"],
        _build_ttn_template,
        overwrite=True,
    )
    register_template(
        "pepo",
        TEMPLATE_DEFINITIONS["pepo"],
        _build_pepo_template,
        overwrite=True,
    )
    register_template(
        "tebd_gate_layer",
        TEMPLATE_DEFINITIONS["tebd_gate_layer"],
        _build_tebd_gate_layer_template,
        overwrite=True,
    )
