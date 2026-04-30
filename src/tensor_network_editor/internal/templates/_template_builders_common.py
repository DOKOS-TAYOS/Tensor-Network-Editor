"""Shared primitives and constants for internal template builders."""

from __future__ import annotations

from typing import cast

from ...models import CanvasPosition, EdgeEndpointRef, EdgeSpec, IndexSpec, TensorSpec
from ..models._model_tensor_data import TensorNumericLiteral
from ._template_catalog import TemplateParameters

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


def _annotate_physics_1d_indices(tensor: TensorSpec, *, symmetry: str) -> None:
    """Add guided metadata to standard 1D physics template indices."""
    for index in tensor.indices:
        if index.name in {"left", "right"}:
            index.metadata = {"leg_kind": "bond", "symmetry": symmetry}
        else:
            index.metadata = {"leg_kind": "physical", "symmetry": symmetry}


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
    current_values = cast(list[TensorNumericLiteral], values)
    for index in index_path[:-1]:
        current_values = cast(list[TensorNumericLiteral], current_values[index])
    current_values[index_path[-1]] = value


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
