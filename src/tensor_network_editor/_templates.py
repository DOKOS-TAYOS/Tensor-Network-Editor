from __future__ import annotations

from collections.abc import Callable

from .models import (
    CanvasPosition,
    EdgeEndpointRef,
    EdgeSpec,
    IndexSpec,
    NetworkSpec,
    TensorSpec,
)
from .validation import ensure_valid_spec

TEMPLATE_NAMES = ["mps", "mpo", "peps_2x2", "mera", "binary_tree"]
HORIZONTAL_SPACING = 320.0
VERTICAL_SPACING = 280.0


def list_template_names() -> list[str]:
    return list(TEMPLATE_NAMES)


def build_template_spec(template_name: str) -> NetworkSpec:
    builders: dict[str, Callable[[], NetworkSpec]] = {
        "mps": _build_mps_template,
        "mpo": _build_mpo_template,
        "peps_2x2": _build_peps_template,
        "mera": _build_mera_template,
        "binary_tree": _build_binary_tree_template,
    }
    try:
        builder = builders[template_name]
    except KeyError as exc:
        raise ValueError(f"Unknown template '{template_name}'.") from exc
    return ensure_valid_spec(builder())


def _build_mps_template() -> NetworkSpec:
    tensors = [
        _make_tensor(
            f"tensor_{index}",
            f"A{index + 1}",
            index * HORIZONTAL_SPACING,
            0.0,
        )
        for index in range(4)
    ]
    edges = [
        _make_edge("edge_01", tensors[0], "right", tensors[1], "left", 3),
        _make_edge("edge_12", tensors[1], "right", tensors[2], "left", 3),
        _make_edge("edge_23", tensors[2], "right", tensors[3], "left", 3),
    ]
    for tensor in tensors:
        tensor.indices.append(_make_open_index(tensor.id, "phys", 2))
    return NetworkSpec(id="template_mps", name="MPS", tensors=tensors, edges=edges)


def _build_mpo_template() -> NetworkSpec:
    tensors = [
        _make_tensor(
            f"tensor_{index}",
            f"W{index + 1}",
            index * 330.0,
            0.0,
        )
        for index in range(4)
    ]
    edges = [
        _make_edge("edge_01", tensors[0], "right", tensors[1], "left", 3),
        _make_edge("edge_12", tensors[1], "right", tensors[2], "left", 3),
        _make_edge("edge_23", tensors[2], "right", tensors[3], "left", 3),
    ]
    for tensor in tensors:
        tensor.indices.extend(
            [
                _make_named_index(tensor.id, "bra", 2),
                _make_named_index(tensor.id, "ket", 2),
            ]
        )
    return NetworkSpec(id="template_mpo", name="MPO", tensors=tensors, edges=edges)


def _build_peps_template() -> NetworkSpec:
    tensors = [
        _make_tensor("tensor_a", "A", 0.0, 0.0),
        _make_tensor("tensor_b", "B", 340.0, 0.0),
        _make_tensor("tensor_c", "C", 0.0, VERTICAL_SPACING),
        _make_tensor("tensor_d", "D", 340.0, VERTICAL_SPACING),
    ]
    edges = [
        _make_edge("edge_ab", tensors[0], "right", tensors[1], "left", 3),
        _make_edge("edge_cd", tensors[2], "right", tensors[3], "left", 3),
        _make_edge("edge_ac", tensors[0], "down", tensors[2], "up", 3),
        _make_edge("edge_bd", tensors[1], "down", tensors[3], "up", 3),
    ]
    for tensor in tensors:
        tensor.indices.append(_make_open_index(tensor.id, "phys", 2))
    return NetworkSpec(id="template_peps", name="PEPS 2x2", tensors=tensors, edges=edges)


def _build_mera_template() -> NetworkSpec:
    tensors = [
        _make_tensor("tensor_top", "Top", 320.0, 0.0),
        _make_tensor("tensor_mid_left", "Mid L", 120.0, 210.0),
        _make_tensor("tensor_mid_right", "Mid R", 520.0, 210.0),
        _make_tensor("tensor_leaf_left", "Leaf L", 0.0, 420.0),
        _make_tensor("tensor_leaf_mid", "Leaf M", 320.0, 420.0),
        _make_tensor("tensor_leaf_right", "Leaf R", 640.0, 420.0),
    ]
    edges = [
        _make_edge("edge_top_left", tensors[0], "left", tensors[1], "up", 3),
        _make_edge("edge_top_right", tensors[0], "right", tensors[2], "up", 3),
        _make_edge("edge_left_leaf", tensors[1], "left", tensors[3], "up", 3),
        _make_edge("edge_center_leaf", tensors[1], "down", tensors[4], "left", 3),
        _make_edge("edge_right_center", tensors[2], "down", tensors[4], "right", 3),
        _make_edge("edge_right_leaf", tensors[2], "right", tensors[5], "up", 3),
    ]
    for tensor in tensors[3:]:
        tensor.indices.append(_make_open_index(tensor.id, "phys", 2))
    return NetworkSpec(id="template_mera", name="MERA", tensors=tensors, edges=edges)


def _build_binary_tree_template() -> NetworkSpec:
    tensors = [
        _make_tensor("tensor_root", "Root", 320.0, 0.0),
        _make_tensor("tensor_left", "Left", 110.0, 210.0),
        _make_tensor("tensor_right", "Right", 530.0, 210.0),
        _make_tensor("tensor_ll", "LL", 0.0, 420.0),
        _make_tensor("tensor_lr", "LR", 220.0, 420.0),
        _make_tensor("tensor_rl", "RL", 420.0, 420.0),
        _make_tensor("tensor_rr", "RR", 640.0, 420.0),
    ]
    edges = [
        _make_edge("edge_root_left", tensors[0], "left", tensors[1], "up", 3),
        _make_edge("edge_root_right", tensors[0], "right", tensors[2], "up", 3),
        _make_edge("edge_left_ll", tensors[1], "left", tensors[3], "up", 3),
        _make_edge("edge_left_lr", tensors[1], "right", tensors[4], "up", 3),
        _make_edge("edge_right_rl", tensors[2], "left", tensors[5], "up", 3),
        _make_edge("edge_right_rr", tensors[2], "right", tensors[6], "up", 3),
    ]
    for tensor in tensors[3:]:
        tensor.indices.append(_make_open_index(tensor.id, "phys", 2))
    return NetworkSpec(
        id="template_binary_tree",
        name="Binary Tree",
        tensors=tensors,
        edges=edges,
    )


def _make_tensor(tensor_id: str, name: str, x: float, y: float) -> TensorSpec:
    return TensorSpec(
        id=tensor_id,
        name=name,
        position=CanvasPosition(x=x, y=y),
        indices=[
            _make_named_index(tensor_id, "left", 3),
            _make_named_index(tensor_id, "right", 3),
            _make_named_index(tensor_id, "up", 3),
            _make_named_index(tensor_id, "down", 3),
        ],
    )


def _make_named_index(tensor_id: str, suffix: str, dimension: int) -> IndexSpec:
    return IndexSpec(
        id=f"{tensor_id}_{suffix}",
        name=suffix,
        dimension=dimension,
    )


def _make_open_index(tensor_id: str, suffix: str, dimension: int) -> IndexSpec:
    return IndexSpec(
        id=f"{tensor_id}_{suffix}",
        name=suffix,
        dimension=dimension,
        offset=CanvasPosition(x=0.0, y=0.0),
    )


def _make_edge(
    edge_id: str,
    left_tensor: TensorSpec,
    left_index_suffix: str,
    right_tensor: TensorSpec,
    right_index_suffix: str,
    dimension: int,
) -> EdgeSpec:
    _set_tensor_index_dimension(left_tensor, left_index_suffix, dimension)
    _set_tensor_index_dimension(right_tensor, right_index_suffix, dimension)
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


def _set_tensor_index_dimension(
    tensor: TensorSpec, index_suffix: str, dimension: int
) -> None:
    for index in tensor.indices:
        if index.id == f"{tensor.id}_{index_suffix}":
            index.dimension = dimension
            return
    raise ValueError(f"Tensor '{tensor.id}' does not contain index '{index_suffix}'.")
