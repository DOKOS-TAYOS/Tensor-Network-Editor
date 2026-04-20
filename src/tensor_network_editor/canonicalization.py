"""Helpers for canonicalizing tensor-network specifications."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from .models import (
    CanvasNoteSpec,
    ContractionPlanSpec,
    EdgeEndpointRef,
    EdgeSpec,
    GridPeriodicGridSpec,
    GroupSpec,
    IndexSpec,
    LinearPeriodicChainSpec,
    NetworkSpec,
    TensorSpec,
    TreePeriodicTreeSpec,
)
from .types import JSONValue, MetadataDict


@dataclass(slots=True)
class _GraphSection:
    """Mutable view over one graph section that can be canonicalized."""

    tensors: list[TensorSpec]
    groups: list[GroupSpec]
    edges: list[EdgeSpec]
    notes: list[CanvasNoteSpec]
    contraction_plan: ContractionPlanSpec | None


def canonicalize_spec(
    spec: NetworkSpec,
    *,
    deterministic_ids: bool = False,
) -> NetworkSpec:
    """Return a canonicalized copy of ``spec``.

    The returned specification preserves behavior while normalizing metadata,
    sorting top-level entities deterministically, and optionally rewriting ids.
    """
    canonical = NetworkSpec.from_dict(spec.to_dict())
    canonical.metadata = _canonicalize_metadata(canonical.metadata)
    _canonicalize_graph_section(
        _GraphSection(
            tensors=canonical.tensors,
            groups=canonical.groups,
            edges=canonical.edges,
            notes=canonical.notes,
            contraction_plan=canonical.contraction_plan,
        )
    )
    if canonical.linear_periodic_chain is not None:
        _canonicalize_linear_periodic_chain(canonical.linear_periodic_chain)
    if canonical.grid_periodic_grid is not None:
        _canonicalize_grid_periodic_grid(canonical.grid_periodic_grid)
    if canonical.tree_periodic_tree is not None:
        _canonicalize_tree_periodic_tree(canonical.tree_periodic_tree)
    if deterministic_ids:
        canonical.id = "network_001"
        _rewrite_graph_section_ids(
            _GraphSection(
                tensors=canonical.tensors,
                groups=canonical.groups,
                edges=canonical.edges,
                notes=canonical.notes,
                contraction_plan=canonical.contraction_plan,
            )
        )
        if canonical.linear_periodic_chain is not None:
            _rewrite_linear_periodic_chain_ids(canonical.linear_periodic_chain)
        if canonical.grid_periodic_grid is not None:
            _rewrite_grid_periodic_grid_ids(canonical.grid_periodic_grid)
        if canonical.tree_periodic_tree is not None:
            _rewrite_tree_periodic_tree_ids(canonical.tree_periodic_tree)
    return canonical


def _canonicalize_linear_periodic_chain(chain: LinearPeriodicChainSpec) -> None:
    """Canonicalize metadata and cell-local graph entities for ``chain``."""
    chain.metadata = _canonicalize_metadata(chain.metadata)
    for cell in (
        chain.initial_cell,
        chain.periodic_cell,
        chain.final_cell,
    ):
        cell.metadata = _canonicalize_metadata(cell.metadata)
        _canonicalize_graph_section(
            _GraphSection(
                tensors=cell.tensors,
                groups=cell.groups,
                edges=cell.edges,
                notes=cell.notes,
                contraction_plan=cell.contraction_plan,
            )
        )


def _rewrite_linear_periodic_chain_ids(chain: LinearPeriodicChainSpec) -> None:
    """Rewrite cell-local ids with stable prefixes for ``chain``."""
    _rewrite_graph_section_ids(
        _GraphSection(
            tensors=chain.initial_cell.tensors,
            groups=chain.initial_cell.groups,
            edges=chain.initial_cell.edges,
            notes=chain.initial_cell.notes,
            contraction_plan=chain.initial_cell.contraction_plan,
        ),
        prefix="initial",
    )
    _rewrite_graph_section_ids(
        _GraphSection(
            tensors=chain.periodic_cell.tensors,
            groups=chain.periodic_cell.groups,
            edges=chain.periodic_cell.edges,
            notes=chain.periodic_cell.notes,
            contraction_plan=chain.periodic_cell.contraction_plan,
        ),
        prefix="periodic",
    )
    _rewrite_graph_section_ids(
        _GraphSection(
            tensors=chain.final_cell.tensors,
            groups=chain.final_cell.groups,
            edges=chain.final_cell.edges,
            notes=chain.final_cell.notes,
            contraction_plan=chain.final_cell.contraction_plan,
        ),
        prefix="final",
    )


def _canonicalize_grid_periodic_grid(grid: GridPeriodicGridSpec) -> None:
    """Canonicalize metadata and cell-local graph entities for ``grid``."""
    grid.metadata = _canonicalize_metadata(grid.metadata)
    for cell in (
        grid.top_left_cell,
        grid.top_cell,
        grid.top_right_cell,
        grid.left_cell,
        grid.center_cell,
        grid.right_cell,
        grid.bottom_left_cell,
        grid.bottom_cell,
        grid.bottom_right_cell,
    ):
        cell.metadata = _canonicalize_metadata(cell.metadata)
        _canonicalize_graph_section(
            _GraphSection(
                tensors=cell.tensors,
                groups=cell.groups,
                edges=cell.edges,
                notes=cell.notes,
                contraction_plan=cell.contraction_plan,
            )
        )


def _rewrite_grid_periodic_grid_ids(grid: GridPeriodicGridSpec) -> None:
    """Rewrite cell-local ids with stable prefixes for ``grid``."""
    _rewrite_graph_section_ids(
        _GraphSection(
            tensors=grid.top_left_cell.tensors,
            groups=grid.top_left_cell.groups,
            edges=grid.top_left_cell.edges,
            notes=grid.top_left_cell.notes,
            contraction_plan=grid.top_left_cell.contraction_plan,
        ),
        prefix="top_left",
    )
    _rewrite_graph_section_ids(
        _GraphSection(
            tensors=grid.top_cell.tensors,
            groups=grid.top_cell.groups,
            edges=grid.top_cell.edges,
            notes=grid.top_cell.notes,
            contraction_plan=grid.top_cell.contraction_plan,
        ),
        prefix="top",
    )
    _rewrite_graph_section_ids(
        _GraphSection(
            tensors=grid.top_right_cell.tensors,
            groups=grid.top_right_cell.groups,
            edges=grid.top_right_cell.edges,
            notes=grid.top_right_cell.notes,
            contraction_plan=grid.top_right_cell.contraction_plan,
        ),
        prefix="top_right",
    )
    _rewrite_graph_section_ids(
        _GraphSection(
            tensors=grid.left_cell.tensors,
            groups=grid.left_cell.groups,
            edges=grid.left_cell.edges,
            notes=grid.left_cell.notes,
            contraction_plan=grid.left_cell.contraction_plan,
        ),
        prefix="left",
    )
    _rewrite_graph_section_ids(
        _GraphSection(
            tensors=grid.center_cell.tensors,
            groups=grid.center_cell.groups,
            edges=grid.center_cell.edges,
            notes=grid.center_cell.notes,
            contraction_plan=grid.center_cell.contraction_plan,
        ),
        prefix="center",
    )
    _rewrite_graph_section_ids(
        _GraphSection(
            tensors=grid.right_cell.tensors,
            groups=grid.right_cell.groups,
            edges=grid.right_cell.edges,
            notes=grid.right_cell.notes,
            contraction_plan=grid.right_cell.contraction_plan,
        ),
        prefix="right",
    )
    _rewrite_graph_section_ids(
        _GraphSection(
            tensors=grid.bottom_left_cell.tensors,
            groups=grid.bottom_left_cell.groups,
            edges=grid.bottom_left_cell.edges,
            notes=grid.bottom_left_cell.notes,
            contraction_plan=grid.bottom_left_cell.contraction_plan,
        ),
        prefix="bottom_left",
    )
    _rewrite_graph_section_ids(
        _GraphSection(
            tensors=grid.bottom_cell.tensors,
            groups=grid.bottom_cell.groups,
            edges=grid.bottom_cell.edges,
            notes=grid.bottom_cell.notes,
            contraction_plan=grid.bottom_cell.contraction_plan,
        ),
        prefix="bottom",
    )
    _rewrite_graph_section_ids(
        _GraphSection(
            tensors=grid.bottom_right_cell.tensors,
            groups=grid.bottom_right_cell.groups,
            edges=grid.bottom_right_cell.edges,
            notes=grid.bottom_right_cell.notes,
            contraction_plan=grid.bottom_right_cell.contraction_plan,
        ),
        prefix="bottom_right",
    )


def _canonicalize_tree_periodic_tree(tree: TreePeriodicTreeSpec) -> None:
    """Canonicalize metadata and cell-local graph entities for ``tree``."""
    tree.metadata = _canonicalize_metadata(tree.metadata)
    for cell in (tree.root_cell, tree.branch_cell, tree.leaf_cell):
        cell.metadata = _canonicalize_metadata(cell.metadata)
        _canonicalize_graph_section(
            _GraphSection(
                tensors=cell.tensors,
                groups=cell.groups,
                edges=cell.edges,
                notes=cell.notes,
                contraction_plan=cell.contraction_plan,
            )
        )


def _rewrite_tree_periodic_tree_ids(tree: TreePeriodicTreeSpec) -> None:
    """Rewrite cell-local ids with stable prefixes for ``tree``."""
    _rewrite_graph_section_ids(
        _GraphSection(
            tensors=tree.root_cell.tensors,
            groups=tree.root_cell.groups,
            edges=tree.root_cell.edges,
            notes=tree.root_cell.notes,
            contraction_plan=tree.root_cell.contraction_plan,
        ),
        prefix="root",
    )
    _rewrite_graph_section_ids(
        _GraphSection(
            tensors=tree.branch_cell.tensors,
            groups=tree.branch_cell.groups,
            edges=tree.branch_cell.edges,
            notes=tree.branch_cell.notes,
            contraction_plan=tree.branch_cell.contraction_plan,
        ),
        prefix="branch",
    )
    _rewrite_graph_section_ids(
        _GraphSection(
            tensors=tree.leaf_cell.tensors,
            groups=tree.leaf_cell.groups,
            edges=tree.leaf_cell.edges,
            notes=tree.leaf_cell.notes,
            contraction_plan=tree.leaf_cell.contraction_plan,
        ),
        prefix="leaf",
    )


def _canonicalize_graph_section(section: _GraphSection) -> None:
    """Canonicalize one graph section in place."""
    for tensor in section.tensors:
        tensor.metadata = _canonicalize_metadata(tensor.metadata)
        for index in tensor.indices:
            index.metadata = _canonicalize_metadata(index.metadata)
        tensor.indices.sort(key=_index_sort_key)
    for edge in section.edges:
        edge.metadata = _canonicalize_metadata(edge.metadata)
    for group in section.groups:
        group.metadata = _canonicalize_metadata(group.metadata)
        group.tensor_ids = sorted(group.tensor_ids)
    for note in section.notes:
        note.metadata = _canonicalize_metadata(note.metadata)
    if section.contraction_plan is not None:
        _canonicalize_contraction_plan(section.contraction_plan)

    section.tensors.sort(key=_tensor_sort_key)
    section.edges.sort(key=_edge_sort_key)
    section.groups.sort(key=_group_sort_key)
    section.notes.sort(key=_note_sort_key)


def _canonicalize_contraction_plan(plan: ContractionPlanSpec) -> None:
    """Canonicalize metadata and stable snapshot ordering for ``plan``."""
    plan.metadata = _canonicalize_metadata(plan.metadata)
    for step in plan.steps:
        step.metadata = _canonicalize_metadata(step.metadata)
    plan.view_snapshots.sort(key=lambda snapshot: snapshot.applied_step_count)
    for snapshot in plan.view_snapshots:
        snapshot.operand_layouts.sort(key=lambda layout: layout.operand_id)


def _rewrite_graph_section_ids(
    section: _GraphSection, prefix: str | None = None
) -> None:
    """Rewrite ids inside one graph section while preserving references."""
    tensor_id_map: dict[str, str] = {}
    index_id_map: dict[str, str] = {}
    step_id_map: dict[str, str] = {}

    for tensor_index, tensor in enumerate(section.tensors, start=1):
        new_tensor_id = _format_canonical_id("tensor", tensor_index, prefix)
        tensor_id_map[tensor.id] = new_tensor_id
        tensor.id = new_tensor_id
    index_counter = 1
    for tensor in section.tensors:
        for index in tensor.indices:
            new_index_id = _format_canonical_id("index", index_counter, prefix)
            index_id_map[index.id] = new_index_id
            index.id = new_index_id
            index_counter += 1
    for edge_index, edge in enumerate(section.edges, start=1):
        edge.id = _format_canonical_id("edge", edge_index, prefix)
        edge.left = EdgeEndpointRef(
            tensor_id=tensor_id_map[edge.left.tensor_id],
            index_id=index_id_map[edge.left.index_id],
        )
        edge.right = EdgeEndpointRef(
            tensor_id=tensor_id_map[edge.right.tensor_id],
            index_id=index_id_map[edge.right.index_id],
        )
    for group_index, group in enumerate(section.groups, start=1):
        group.id = _format_canonical_id("group", group_index, prefix)
        group.tensor_ids = sorted(
            tensor_id_map[tensor_id] for tensor_id in group.tensor_ids
        )
    for note_index, note in enumerate(section.notes, start=1):
        note.id = _format_canonical_id("note", note_index, prefix)
    if section.contraction_plan is not None:
        section.contraction_plan.id = _format_canonical_id("plan", 1, prefix)
        for step_index, step in enumerate(section.contraction_plan.steps, start=1):
            step_id_map[step.id] = _format_canonical_id("step", step_index, prefix)
        for step in section.contraction_plan.steps:
            step.id = step_id_map[step.id]
            step.left_operand_id = _rewrite_operand_id(
                step.left_operand_id,
                tensor_id_map=tensor_id_map,
                step_id_map=step_id_map,
            )
            step.right_operand_id = _rewrite_operand_id(
                step.right_operand_id,
                tensor_id_map=tensor_id_map,
                step_id_map=step_id_map,
            )
        for snapshot in section.contraction_plan.view_snapshots:
            for layout in snapshot.operand_layouts:
                layout.operand_id = _rewrite_operand_id(
                    layout.operand_id,
                    tensor_id_map=tensor_id_map,
                    step_id_map=step_id_map,
                )


def _rewrite_operand_id(
    operand_id: str,
    *,
    tensor_id_map: dict[str, str],
    step_id_map: dict[str, str],
) -> str:
    """Rewrite one operand id using known tensor and step mappings."""
    if operand_id in tensor_id_map:
        return tensor_id_map[operand_id]
    if operand_id in step_id_map:
        return step_id_map[operand_id]
    return operand_id


def _format_canonical_id(kind: str, number: int, prefix: str | None) -> str:
    """Format one canonical identifier."""
    base = f"{kind}_{number:03d}"
    if prefix:
        return f"{prefix}_{base}"
    return base


def _tensor_sort_key(tensor: TensorSpec) -> tuple[object, ...]:
    """Return a stable sort key for tensors."""
    return (
        tensor.name.casefold(),
        round(tensor.position.y, 6),
        round(tensor.position.x, 6),
        tuple(index.name.casefold() for index in tensor.indices),
        tensor.id,
    )


def _index_sort_key(index: IndexSpec) -> tuple[object, ...]:
    """Return a stable sort key for indices."""
    return (
        index.name.casefold(),
        index.dimension,
        round(index.offset.x, 6),
        round(index.offset.y, 6),
        index.id,
    )


def _edge_sort_key(edge: EdgeSpec) -> tuple[object, ...]:
    """Return a stable sort key for edges."""
    endpoints = sorted(
        (
            (edge.left.tensor_id, edge.left.index_id),
            (edge.right.tensor_id, edge.right.index_id),
        )
    )
    return (
        edge.name.casefold(),
        endpoints[0],
        endpoints[1],
        edge.id,
    )


def _group_sort_key(group: GroupSpec) -> tuple[object, ...]:
    """Return a stable sort key for groups."""
    return (group.name.casefold(), tuple(sorted(group.tensor_ids)), group.id)


def _note_sort_key(note: CanvasNoteSpec) -> tuple[object, ...]:
    """Return a stable sort key for notes."""
    return (
        round(note.position.y, 6),
        round(note.position.x, 6),
        note.text.casefold(),
        note.id,
    )


def _canonicalize_metadata(metadata: MetadataDict) -> MetadataDict:
    """Return a recursively normalized metadata mapping."""
    normalized: MetadataDict = {}
    for key in sorted(metadata.keys()):
        value = metadata[key]
        if key == "tags":
            normalized[key] = _canonicalize_tags(value)
        else:
            normalized[key] = _canonicalize_json_value(value)
    return normalized


def _canonicalize_json_value(value: JSONValue) -> JSONValue:
    """Return a recursively normalized JSON value."""
    if isinstance(value, dict):
        normalized_mapping: dict[str, JSONValue] = {}
        for key in sorted(value.keys()):
            child_value = value[key]
            if key == "tags":
                normalized_mapping[key] = _canonicalize_tags(child_value)
            else:
                normalized_mapping[key] = _canonicalize_json_value(child_value)
        return normalized_mapping
    if isinstance(value, list):
        return [_canonicalize_json_value(item) for item in value]
    return value


def _canonicalize_tags(value: JSONValue) -> JSONValue:
    """Normalize a tags value when it is a string list."""
    if isinstance(value, list):
        string_items = [item for item in value if isinstance(item, str)]
        if len(string_items) != len(value):
            return _canonicalize_json_value(value)
        stripped_items = [item.strip() for item in string_items]
        return cast(JSONValue, sorted({item for item in stripped_items if item}))
    return _canonicalize_json_value(value)
