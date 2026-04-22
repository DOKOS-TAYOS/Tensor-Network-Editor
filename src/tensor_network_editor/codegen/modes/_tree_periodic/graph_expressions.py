"""Graph expression helpers for tree-periodic code generation."""

from __future__ import annotations

from ....internal.modes._tree_periodic import TreePeriodicInterfacePort
from ....models import EngineName, TensorCollectionFormat
from ...shared._linear_periodic_expressions import _axis_names_for_engine
from ...shared.common import PreparedNetwork, tensor_collection_reference


def _build_edge_expression_by_index_id(
    *,
    prepared: PreparedNetwork,
    engine: EngineName,
    collection_format: TensorCollectionFormat,
    collection_name: str,
) -> dict[str, str]:
    """Resolve each prepared internal index to its runtime edge expression."""
    edge_expression_by_index_id: dict[str, str] = {}
    for tensor in prepared.tensors:
        tensor_reference = tensor_collection_reference(
            tensor,
            collection_format,
            collection_name,
        )
        runtime_axis_names = _axis_names_for_engine(
            engine,
            tuple(index.spec.name for index in tensor.indices),
        )
        for index, axis_name in zip(
            tensor.indices,
            runtime_axis_names,
            strict=True,
        ):
            edge_expression_by_index_id[index.spec.id] = (
                f"{tensor_reference}[{axis_name!r}]"
            )
    return edge_expression_by_index_id


def _render_child_interface_lines(
    *,
    child_ports_by_index: dict[int, tuple[TreePeriodicInterfacePort, ...]],
    edge_expression_by_index_id: dict[str, str],
) -> list[str]:
    """Render the resolved child-interface payloads for graph backends."""
    lines = ["child_interfaces = []"]
    for child_index in sorted(child_ports_by_index):
        child_interface_expression = _render_python_list_expression(
            [
                edge_expression_by_index_id[port.internal_index_id]
                for port in child_ports_by_index[child_index]
                if port.internal_index_id in edge_expression_by_index_id
            ]
        )
        lines.append(f"child_interfaces.append({child_interface_expression})")
    return lines


def _render_python_list_expression(values: list[str]) -> str:
    """Render a Python list literal from already-rendered expressions."""
    return "[" + ", ".join(values) + "]"
