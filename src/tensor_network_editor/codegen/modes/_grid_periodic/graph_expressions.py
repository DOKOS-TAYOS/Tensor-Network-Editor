"""Graph expression helpers for periodic-grid code generation."""

from __future__ import annotations

from ....models import EngineName, TensorCollectionFormat
from ...shared._linear_periodic_expressions import _axis_name_for_engine
from ...shared.common import (
    PreparedNetwork,
    tensor_collection_reference_by_id,
)


def _tensor_value_expression(
    *,
    prepared: PreparedNetwork,
    tensor_id: str,
    engine: EngineName,
) -> str:
    """Render the backend-specific tensor constructor for one tensor id."""
    tensor = next(item for item in prepared.tensors if item.spec.id == tensor_id)
    if engine is EngineName.TENSORNETWORK:
        return (
            f"tn.Node(np.zeros({tensor.spec.shape!r}, dtype=float), "
            f"name={tensor.spec.name!r}, "
            f"axis_names={[index.spec.name for index in tensor.indices]!r})"
        )
    return (
        f"tk.Node(tensor=torch.zeros({tensor.spec.shape!r}, dtype=torch.float32), "
        f"axes_names={tuple(index.spec.name for index in tensor.indices)!r}, "
        f"name={tensor.spec.name!r}, "
        "network=network)"
    )


def _render_network_connection_lines(
    *,
    prepared: PreparedNetwork,
    engine: EngineName,
    collection_format: TensorCollectionFormat,
    collection_name: str,
) -> list[str]:
    """Render the internal edge construction section for one graph backend."""
    if not prepared.edges:
        return []
    lines = ["edges_list = []"]
    connect_prefix = (
        "tn.connect" if engine is EngineName.TENSORNETWORK else "tk.connect"
    )
    for edge in prepared.edges:
        left_tensor = tensor_collection_reference_by_id(
            prepared,
            edge.spec.left.tensor_id,
            collection_format,
            collection_name,
        )
        right_tensor = tensor_collection_reference_by_id(
            prepared,
            edge.spec.right.tensor_id,
            collection_format,
            collection_name,
        )
        if engine is EngineName.TENSORNETWORK:
            lines.append(
                "edges_list.append(tn.connect("
                f"{left_tensor}[{edge.left.spec.name!r}], "
                f"{right_tensor}[{edge.right.spec.name!r}], "
                f"name={edge.spec.name!r}))"
            )
            continue
        left_axis_name = _axis_name_for_engine(
            EngineName.TENSORKROWCH, edge.left.spec.name
        )
        right_axis_name = _axis_name_for_engine(
            EngineName.TENSORKROWCH,
            edge.right.spec.name,
        )
        lines.append(
            "edges_list.append(("
            f"{edge.spec.name!r}, "
            f"{connect_prefix}({left_tensor}[{left_axis_name!r}], {right_tensor}[{right_axis_name!r}])"
            "))"
        )
    return lines


def _build_label_expression_map(
    *,
    prepared: PreparedNetwork,
    engine: EngineName,
    collection_format: TensorCollectionFormat,
    collection_name: str,
) -> dict[str, str]:
    """Resolve every open label to the generated Python expression."""
    label_expression_by_label: dict[str, str] = {}
    for tensor in prepared.tensors:
        tensor_reference = tensor_collection_reference_by_id(
            prepared,
            tensor.spec.id,
            collection_format,
            collection_name,
        )
        for index in tensor.indices:
            axis_name = (
                index.spec.name
                if engine is EngineName.TENSORNETWORK
                else _axis_name_for_engine(engine, index.spec.name)
            )
            label_expression_by_label[index.label] = (
                f"{tensor_reference}[{axis_name!r}]"
            )
    return label_expression_by_label
