from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .models import (
    CanvasPosition,
    EdgeEndpointRef,
    EdgeSpec,
    IndexSpec,
    NetworkSpec,
    TensorSpec,
)
from .validation import ensure_valid_spec

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
TemplateIndexConfig = tuple[str, int, tuple[float, float]]


@dataclass(frozen=True)
class TemplateParameters:
    graph_size: int
    bond_dimension: int
    physical_dimension: int


@dataclass(frozen=True)
class TemplateDefinition:
    name: str
    display_name: str
    graph_size_label: str
    defaults: TemplateParameters
    minimum_graph_size: int = 2
    minimum_bond_dimension: int = 1
    minimum_physical_dimension: int = 1

    def to_dict(self) -> dict[str, object]:
        return {
            "display_name": self.display_name,
            "graph_size_label": self.graph_size_label,
            "defaults": {
                "graph_size": self.defaults.graph_size,
                "bond_dimension": self.defaults.bond_dimension,
                "physical_dimension": self.defaults.physical_dimension,
            },
            "minimums": {
                "graph_size": self.minimum_graph_size,
                "bond_dimension": self.minimum_bond_dimension,
                "physical_dimension": self.minimum_physical_dimension,
            },
        }


TEMPLATE_DEFINITIONS: dict[str, TemplateDefinition] = {
    "mps": TemplateDefinition(
        name="mps",
        display_name="MPS",
        graph_size_label="Sites",
        defaults=TemplateParameters(
            graph_size=4,
            bond_dimension=3,
            physical_dimension=2,
        ),
    ),
    "mpo": TemplateDefinition(
        name="mpo",
        display_name="MPO",
        graph_size_label="Sites",
        defaults=TemplateParameters(
            graph_size=4,
            bond_dimension=3,
            physical_dimension=2,
        ),
    ),
    "peps_2x2": TemplateDefinition(
        name="peps_2x2",
        display_name="PEPS",
        graph_size_label="Side length",
        defaults=TemplateParameters(
            graph_size=2,
            bond_dimension=3,
            physical_dimension=2,
        ),
    ),
    "mera": TemplateDefinition(
        name="mera",
        display_name="MERA",
        graph_size_label="Depth",
        defaults=TemplateParameters(
            graph_size=3,
            bond_dimension=3,
            physical_dimension=2,
        ),
    ),
    "binary_tree": TemplateDefinition(
        name="binary_tree",
        display_name="Binary Tree",
        graph_size_label="Depth",
        defaults=TemplateParameters(
            graph_size=3,
            bond_dimension=3,
            physical_dimension=2,
        ),
    ),
}
TEMPLATE_NAMES = list(TEMPLATE_DEFINITIONS)


def list_template_names() -> list[str]:
    return list(TEMPLATE_NAMES)


def serialize_template_definitions() -> dict[str, dict[str, object]]:
    return {
        template_name: definition.to_dict()
        for template_name, definition in TEMPLATE_DEFINITIONS.items()
    }


def parse_template_parameters(
    template_name: str, raw_parameters: object | None = None
) -> TemplateParameters:
    definition = _get_template_definition(template_name)
    defaults = definition.defaults
    if raw_parameters is None:
        return defaults
    if not isinstance(raw_parameters, dict):
        raise ValueError("Template 'parameters' payload must be an object.")
    return _validate_template_parameters(
        template_name,
        TemplateParameters(
            graph_size=_parse_template_integer(
                raw_parameters.get("graph_size"),
                field_name="graph_size",
                default=defaults.graph_size,
                minimum=definition.minimum_graph_size,
            ),
            bond_dimension=_parse_template_integer(
                raw_parameters.get("bond_dimension"),
                field_name="bond_dimension",
                default=defaults.bond_dimension,
                minimum=definition.minimum_bond_dimension,
            ),
            physical_dimension=_parse_template_integer(
                raw_parameters.get("physical_dimension"),
                field_name="physical_dimension",
                default=defaults.physical_dimension,
                minimum=definition.minimum_physical_dimension,
            ),
        ),
    )


def build_template_spec(
    template_name: str, parameters: TemplateParameters | None = None
) -> NetworkSpec:
    builders: dict[str, Callable[[TemplateParameters], NetworkSpec]] = {
        "mps": _build_mps_template,
        "mpo": _build_mpo_template,
        "peps_2x2": _build_peps_template,
        "mera": _build_mera_template,
        "binary_tree": _build_binary_tree_template,
    }
    resolved_parameters = _validate_template_parameters(
        template_name,
        parameters or _get_template_definition(template_name).defaults,
    )
    try:
        builder = builders[template_name]
    except KeyError as exc:
        raise ValueError(f"Unknown template '{template_name}'.") from exc
    return ensure_valid_spec(builder(resolved_parameters))


def _get_template_definition(template_name: str) -> TemplateDefinition:
    try:
        return TEMPLATE_DEFINITIONS[template_name]
    except KeyError as exc:
        raise ValueError(f"Unknown template '{template_name}'.") from exc


def _validate_template_parameters(
    template_name: str, parameters: TemplateParameters
) -> TemplateParameters:
    definition = _get_template_definition(template_name)
    return TemplateParameters(
        graph_size=_parse_template_integer(
            parameters.graph_size,
            field_name="graph_size",
            default=definition.defaults.graph_size,
            minimum=definition.minimum_graph_size,
        ),
        bond_dimension=_parse_template_integer(
            parameters.bond_dimension,
            field_name="bond_dimension",
            default=definition.defaults.bond_dimension,
            minimum=definition.minimum_bond_dimension,
        ),
        physical_dimension=_parse_template_integer(
            parameters.physical_dimension,
            field_name="physical_dimension",
            default=definition.defaults.physical_dimension,
            minimum=definition.minimum_physical_dimension,
        ),
    )


def _parse_template_integer(
    value: object, *, field_name: str, default: int, minimum: int
) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Template parameter '{field_name}' must be an integer.")
    if value < minimum:
        raise ValueError(
            f"Template parameter '{field_name}' must be greater than or equal to {minimum}."
        )
    return value


def _build_mps_template(parameters: TemplateParameters) -> NetworkSpec:
    length = parameters.graph_size
    tensors = []
    for site_index in range(length):
        tensor_indices: list[TemplateIndexConfig] = []
        if site_index > 0:
            tensor_indices.append(("left", parameters.bond_dimension, LEFT_OFFSET))
        if site_index < length - 1:
            tensor_indices.append(("right", parameters.bond_dimension, RIGHT_OFFSET))
        tensor_indices.append(("phys", parameters.physical_dimension, DOWN_OFFSET))
        tensors.append(
            _make_tensor(
                f"tensor_{site_index}",
                f"A{site_index + 1}",
                HORIZONTAL_SPACING * site_index,
                0.0,
                tensor_indices,
            )
        )
    edges = [
        _make_edge(
            f"edge_{site_index}_{site_index + 1}",
            tensors[site_index],
            "right",
            tensors[site_index + 1],
            "left",
        )
        for site_index in range(length - 1)
    ]
    spec_name = (
        "MPS"
        if length == TEMPLATE_DEFINITIONS["mps"].defaults.graph_size
        else f"MPS ({length} sites)"
    )
    return NetworkSpec(
        id=f"template_mps_{length}",
        name=spec_name,
        tensors=tensors,
        edges=edges,
    )


def _build_mpo_template(parameters: TemplateParameters) -> NetworkSpec:
    length = parameters.graph_size
    tensors = []
    for site_index in range(length):
        tensor_indices: list[TemplateIndexConfig] = []
        if site_index > 0:
            tensor_indices.append(("left", parameters.bond_dimension, LEFT_OFFSET))
        if site_index < length - 1:
            tensor_indices.append(("right", parameters.bond_dimension, RIGHT_OFFSET))
        tensor_indices.extend(
            [
                ("bra", parameters.physical_dimension, UP_OFFSET),
                ("ket", parameters.physical_dimension, DOWN_OFFSET),
            ]
        )
        tensors.append(
            _make_tensor(
                f"tensor_{site_index}",
                f"W{site_index + 1}",
                330.0 * site_index,
                0.0,
                tensor_indices,
            )
        )
    edges = [
        _make_edge(
            f"edge_{site_index}_{site_index + 1}",
            tensors[site_index],
            "right",
            tensors[site_index + 1],
            "left",
        )
        for site_index in range(length - 1)
    ]
    spec_name = (
        "MPO"
        if length == TEMPLATE_DEFINITIONS["mpo"].defaults.graph_size
        else f"MPO ({length} sites)"
    )
    return NetworkSpec(
        id=f"template_mpo_{length}",
        name=spec_name,
        tensors=tensors,
        edges=edges,
    )


def _build_peps_template(parameters: TemplateParameters) -> NetworkSpec:
    if parameters.graph_size == TEMPLATE_DEFINITIONS["peps_2x2"].defaults.graph_size:
        return _build_default_peps_template(parameters)
    return _build_generic_peps_template(parameters)


def _build_default_peps_template(parameters: TemplateParameters) -> NetworkSpec:
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
    size = parameters.graph_size
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
    if parameters.graph_size == TEMPLATE_DEFINITIONS["mera"].defaults.graph_size:
        return _build_default_mera_template(parameters)
    return _build_generic_mera_template(parameters)


def _build_default_mera_template(parameters: TemplateParameters) -> NetworkSpec:
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
    depth = parameters.graph_size
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


def _build_binary_tree_template(parameters: TemplateParameters) -> NetworkSpec:
    if parameters.graph_size == TEMPLATE_DEFINITIONS["binary_tree"].defaults.graph_size:
        return _build_default_binary_tree_template(parameters)
    return _build_generic_binary_tree_template(parameters)


def _build_default_binary_tree_template(parameters: TemplateParameters) -> NetworkSpec:
    tensors = [
        _make_tensor(
            "tensor_root",
            "Root",
            320.0,
            0.0,
            [
                ("left", parameters.bond_dimension, LEFT_OFFSET),
                ("right", parameters.bond_dimension, RIGHT_OFFSET),
            ],
        ),
        _make_tensor(
            "tensor_left",
            "Left",
            110.0,
            210.0,
            [
                ("up", parameters.bond_dimension, UP_OFFSET),
                ("left", parameters.bond_dimension, LEFT_OFFSET),
                ("right", parameters.bond_dimension, RIGHT_OFFSET),
            ],
        ),
        _make_tensor(
            "tensor_right",
            "Right",
            530.0,
            210.0,
            [
                ("up", parameters.bond_dimension, UP_OFFSET),
                ("left", parameters.bond_dimension, LEFT_OFFSET),
                ("right", parameters.bond_dimension, RIGHT_OFFSET),
            ],
        ),
        _make_tensor(
            "tensor_ll",
            "LL",
            0.0,
            420.0,
            [
                ("up", parameters.bond_dimension, UP_OFFSET),
                ("phys", parameters.physical_dimension, DOWN_OFFSET),
            ],
        ),
        _make_tensor(
            "tensor_lr",
            "LR",
            220.0,
            420.0,
            [
                ("up", parameters.bond_dimension, UP_OFFSET),
                ("phys", parameters.physical_dimension, DOWN_OFFSET),
            ],
        ),
        _make_tensor(
            "tensor_rl",
            "RL",
            420.0,
            420.0,
            [
                ("up", parameters.bond_dimension, UP_OFFSET),
                ("phys", parameters.physical_dimension, DOWN_OFFSET),
            ],
        ),
        _make_tensor(
            "tensor_rr",
            "RR",
            640.0,
            420.0,
            [
                ("up", parameters.bond_dimension, UP_OFFSET),
                ("phys", parameters.physical_dimension, DOWN_OFFSET),
            ],
        ),
    ]
    edges = [
        _make_edge("edge_root_left", tensors[0], "left", tensors[1], "up"),
        _make_edge("edge_root_right", tensors[0], "right", tensors[2], "up"),
        _make_edge("edge_left_ll", tensors[1], "left", tensors[3], "up"),
        _make_edge("edge_left_lr", tensors[1], "right", tensors[4], "up"),
        _make_edge("edge_right_rl", tensors[2], "left", tensors[5], "up"),
        _make_edge("edge_right_rr", tensors[2], "right", tensors[6], "up"),
    ]
    return NetworkSpec(
        id="template_binary_tree_3",
        name="Binary Tree",
        tensors=tensors,
        edges=edges,
    )


def _build_generic_binary_tree_template(parameters: TemplateParameters) -> NetworkSpec:
    depth = parameters.graph_size
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
            if level_index == depth - 1:
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
        id=f"template_binary_tree_{depth}",
        name=f"Binary Tree depth {depth}",
        tensors=[tensor for level in levels for tensor in level],
        edges=edges,
    )


def _grid_tensor_name(row_index: int, column_index: int) -> str:
    if row_index < 26:
        return f"{chr(ord('A') + row_index)}{column_index + 1}"
    return f"R{row_index + 1}C{column_index + 1}"


def _make_tensor(
    tensor_id: str,
    name: str,
    x: float,
    y: float,
    indices: list[TemplateIndexConfig],
) -> TensorSpec:
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
    dimension: int,
    offset: tuple[float, float],
) -> IndexSpec:
    return IndexSpec(
        id=f"{tensor_id}_{suffix}",
        name=suffix,
        dimension=dimension,
        offset=CanvasPosition(x=offset[0], y=offset[1]),
    )


def _make_edge(
    edge_id: str,
    left_tensor: TensorSpec,
    left_index_suffix: str,
    right_tensor: TensorSpec,
    right_index_suffix: str,
) -> EdgeSpec:
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
