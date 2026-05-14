"""Complexity limits for local editor API payloads."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from ..internal.models._model_periodic import LinearPeriodicCellSpec
from ..internal.templates._template_catalog import TemplateParameters
from ..models import NetworkSpec, TensorSpec

MAX_API_TENSORS = 512
MAX_API_INDICES = 4096
MAX_API_CONNECTIONS = 4096
MAX_API_TENSOR_RANK = 64
MAX_API_INDEX_DIMENSION = 1_000_000
MAX_API_TENSOR_CARDINALITY = 10_000_000
MAX_API_TEMPLATE_LINEAR_GRAPH_SIZE = 512
MAX_API_TEMPLATE_GRID_SIDE_LENGTH = 32
MAX_API_TEMPLATE_TREE_DEPTH = 10
MAX_API_TEMPLATE_DIMENSION = 4096
_GRID_TEMPLATE_NAMES = frozenset({"peps_2x2", "pepo"})
_TREE_TEMPLATE_NAMES = frozenset({"mera", "ttn"})


@dataclass(slots=True)
class _SpecComplexity:
    """Accumulated size information for one editor API payload."""

    tensor_count: int = 0
    index_count: int = 0
    connection_count: int = 0


def enforce_spec_api_limits(spec: NetworkSpec) -> None:
    """Reject a network spec that is too expensive for the local HTTP API."""
    complexity = _SpecComplexity()
    for tensors, edge_count in _iter_spec_parts(spec):
        complexity.tensor_count += len(tensors)
        complexity.connection_count += edge_count
        for tensor in tensors:
            _enforce_tensor_api_limits(tensor)
            complexity.index_count += len(tensor.indices)

    complexity.connection_count += sum(
        len(hyperedge.endpoints) for hyperedge in spec.hyperedges
    )
    _enforce_count_limit(
        name="tensors",
        count=complexity.tensor_count,
        limit=MAX_API_TENSORS,
    )
    _enforce_count_limit(
        name="indices",
        count=complexity.index_count,
        limit=MAX_API_INDICES,
    )
    _enforce_count_limit(
        name="connections",
        count=complexity.connection_count,
        limit=MAX_API_CONNECTIONS,
    )


def enforce_template_api_limits(
    template_name: str,
    parameters: TemplateParameters | None,
) -> None:
    """Reject built-in template parameters that would create huge payloads."""
    if parameters is None:
        return
    graph_size_limit = _template_graph_size_limit(template_name)
    if parameters.graph_size is not None and parameters.graph_size > graph_size_limit:
        raise ValueError(
            "Template parameter 'graph_size' "
            f"is {parameters.graph_size}, above the API limit of {graph_size_limit}."
        )
    _enforce_optional_template_dimension(
        parameters.bond_dimension,
        field_name="bond_dimension",
    )
    _enforce_optional_template_dimension(
        parameters.physical_dimension,
        field_name="physical_dimension",
    )


def _iter_spec_parts(spec: NetworkSpec) -> Iterator[tuple[list[TensorSpec], int]]:
    """Yield tensor and edge collections stored in a spec payload."""
    yield spec.tensors, len(spec.edges)
    if spec.linear_periodic_chain is not None:
        for cell in (
            spec.linear_periodic_chain.initial_cell,
            spec.linear_periodic_chain.periodic_cell,
            spec.linear_periodic_chain.final_cell,
        ):
            yield from _iter_cell_parts(cell)
    if spec.grid_periodic_grid is not None:
        for cell in (
            spec.grid_periodic_grid.top_left_cell,
            spec.grid_periodic_grid.top_cell,
            spec.grid_periodic_grid.top_right_cell,
            spec.grid_periodic_grid.left_cell,
            spec.grid_periodic_grid.center_cell,
            spec.grid_periodic_grid.right_cell,
            spec.grid_periodic_grid.bottom_left_cell,
            spec.grid_periodic_grid.bottom_cell,
            spec.grid_periodic_grid.bottom_right_cell,
        ):
            yield from _iter_cell_parts(cell)
    if spec.tree_periodic_tree is not None:
        for cell in (
            spec.tree_periodic_tree.root_cell,
            spec.tree_periodic_tree.branch_cell,
            spec.tree_periodic_tree.leaf_cell,
        ):
            yield from _iter_cell_parts(cell)


def _iter_cell_parts(
    cell: LinearPeriodicCellSpec,
) -> Iterator[tuple[list[TensorSpec], int]]:
    """Yield tensor and edge collections stored in one periodic cell."""
    yield cell.tensors, len(cell.edges)


def _enforce_tensor_api_limits(tensor: TensorSpec) -> None:
    """Reject one tensor whose local shape is too expensive."""
    rank = len(tensor.indices)
    if rank > MAX_API_TENSOR_RANK:
        raise ValueError(
            f"Tensor '{tensor.name}' has rank {rank}, "
            f"above the API limit of {MAX_API_TENSOR_RANK}."
        )
    cardinality = 1
    for index in tensor.indices:
        if index.dimension > MAX_API_INDEX_DIMENSION:
            raise ValueError(
                f"Index '{index.name}' on tensor '{tensor.name}' has dimension "
                f"{index.dimension}, above the API limit of {MAX_API_INDEX_DIMENSION}."
            )
        if index.dimension > 0:
            cardinality *= index.dimension
        if cardinality > MAX_API_TENSOR_CARDINALITY:
            raise ValueError(
                f"Tensor '{tensor.name}' spans {cardinality} elements, "
                f"above the API limit of {MAX_API_TENSOR_CARDINALITY}."
            )


def _enforce_count_limit(*, name: str, count: int, limit: int) -> None:
    """Reject one aggregate count when it exceeds its API limit."""
    if count <= limit:
        return
    raise ValueError(
        f"Network contains {count} {name}, above the API limit of {limit}."
    )


def _enforce_optional_template_dimension(
    value: int | None,
    *,
    field_name: str,
) -> None:
    """Reject template dimensions that would produce very large tensors."""
    if value is None or value <= MAX_API_TEMPLATE_DIMENSION:
        return
    raise ValueError(
        f"Template parameter '{field_name}' is {value}, "
        f"above the API limit of {MAX_API_TEMPLATE_DIMENSION}."
    )


def _template_graph_size_limit(template_name: str) -> int:
    """Return the graph-size limit appropriate for one template family."""
    if template_name in _GRID_TEMPLATE_NAMES:
        return MAX_API_TEMPLATE_GRID_SIDE_LENGTH
    if template_name in _TREE_TEMPLATE_NAMES:
        return MAX_API_TEMPLATE_TREE_DEPTH
    return MAX_API_TEMPLATE_LINEAR_GRAPH_SIZE
