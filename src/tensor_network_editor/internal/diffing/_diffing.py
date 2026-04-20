"""Public helpers for comparing two tensor-network specifications."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Protocol, cast

from ...models import (
    CanvasNoteSpec,
    ContractionPlanSpec,
    ContractionStepSpec,
    EdgeSpec,
    GroupSpec,
    IndexSpec,
    NetworkSpec,
    TensorSpec,
)
from ...types import JSONValue
from ..canonicalization._canonicalization import canonicalize_spec
from ..models._headless_models import (
    DiffEntityChanges,
    SemanticDiffEntry,
    SemanticFieldChange,
    SemanticSpecDiffResult,
    SpecDiffResult,
)


class _DiffableEntity(Protocol):
    """Protocol for entities that can be compared by id and serialized payload."""

    id: str

    def to_dict(self) -> dict[str, JSONValue]:
        """Return the serialized payload used for equality checks."""
        ...


@dataclass(frozen=True)
class _SemanticEntityConfig:
    """Describe one standard semantic-diff entity family."""

    entity_type: str
    entities: Callable[[NetworkSpec], Iterable[_DiffableEntity]]
    payload_getter: Callable[[_DiffableEntity], dict[str, JSONValue]]


_ENTITY_LABELS: dict[str, str] = {
    "tensor": "Tensor",
    "index": "Index",
    "edge": "Edge",
    "group": "Group",
    "note": "Note",
    "plan": "Contraction plan",
    "step": "Contraction step",
    "linear_periodic_chain": "Linear periodic chain",
    "grid_periodic_grid": "Grid periodic grid",
    "tree_periodic_tree": "Tree periodic tree",
}

_ENTITY_SORT_ORDER: dict[str, int] = {
    "tensor": 0,
    "index": 1,
    "edge": 2,
    "group": 3,
    "note": 4,
    "plan": 5,
    "step": 6,
    "linear_periodic_chain": 7,
    "grid_periodic_grid": 8,
    "tree_periodic_tree": 9,
}


def diff_specs(before: NetworkSpec, after: NetworkSpec) -> SpecDiffResult:
    """Return a structured diff between two specs based on stable ids.

    Args:
        before: Earlier specification snapshot.
        after: Later specification snapshot.

    Returns:
        Entity-level additions, removals, and payload changes grouped by type.
    """
    return SpecDiffResult(
        tensor=_diff_named_entities(before.tensors, after.tensors),
        edge=_diff_named_entities(before.edges, after.edges),
        group=_diff_named_entities(before.groups, after.groups),
        note=_diff_named_entities(before.notes, after.notes),
        plan=_diff_plan(before, after),
    )


def semantic_diff_specs(
    before: NetworkSpec,
    after: NetworkSpec,
) -> SemanticSpecDiffResult:
    """Return field-level semantic changes grouped by stable entity ids."""
    normalized_before = canonicalize_spec(before)
    normalized_after = canonicalize_spec(after)
    entries = _semantic_diff_standard_entities(normalized_before, normalized_after)
    entries.extend(
        _semantic_diff_plan(
            normalized_before.contraction_plan,
            normalized_after.contraction_plan,
        )
    )
    entries.extend(
        _semantic_diff_linear_periodic_chain(normalized_before, normalized_after)
    )
    entries.extend(
        _semantic_diff_grid_periodic_grid(normalized_before, normalized_after)
    )
    entries.extend(
        _semantic_diff_tree_periodic_tree(normalized_before, normalized_after)
    )
    return SemanticSpecDiffResult(entries=_sort_semantic_entries(entries))


def _semantic_diff_standard_entities(
    before: NetworkSpec,
    after: NetworkSpec,
) -> list[SemanticDiffEntry]:
    """Build semantic diff entries for the standard entity families."""
    entries: list[SemanticDiffEntry] = []
    for config in _STANDARD_SEMANTIC_ENTITY_CONFIGS:
        entries.extend(
            _semantic_diff_named_entities(
                config.entities(before),
                config.entities(after),
                entity_type=config.entity_type,
                payload_getter=config.payload_getter,
            )
        )
    return entries


def _diff_named_entities(
    before: Iterable[_DiffableEntity], after: Iterable[_DiffableEntity]
) -> DiffEntityChanges:
    """Diff two entity collections that expose ``id`` and ``to_dict``."""
    before_by_id = {item.id: item for item in before}
    after_by_id = {item.id: item for item in after}
    shared_ids = sorted(before_by_id.keys() & after_by_id.keys())
    return DiffEntityChanges(
        added=sorted(after_by_id.keys() - before_by_id.keys()),
        removed=sorted(before_by_id.keys() - after_by_id.keys()),
        changed=[
            entity_id
            for entity_id in shared_ids
            if before_by_id[entity_id].to_dict() != after_by_id[entity_id].to_dict()
        ],
    )


def _diff_plan(before: NetworkSpec, after: NetworkSpec) -> DiffEntityChanges:
    """Diff the optional contraction plan by id and serialized payload."""
    if before.contraction_plan is None and after.contraction_plan is None:
        return DiffEntityChanges()
    if before.contraction_plan is None and after.contraction_plan is not None:
        return DiffEntityChanges(added=[after.contraction_plan.id])
    if before.contraction_plan is not None and after.contraction_plan is None:
        return DiffEntityChanges(removed=[before.contraction_plan.id])
    assert before.contraction_plan is not None
    assert after.contraction_plan is not None
    if before.contraction_plan.id != after.contraction_plan.id:
        return DiffEntityChanges(
            added=[after.contraction_plan.id],
            removed=[before.contraction_plan.id],
        )
    if before.contraction_plan.to_dict() != after.contraction_plan.to_dict():
        return DiffEntityChanges(changed=[before.contraction_plan.id])
    return DiffEntityChanges()


def _semantic_diff_named_entities(
    before: Iterable[_DiffableEntity],
    after: Iterable[_DiffableEntity],
    *,
    entity_type: str,
    payload_getter: Callable[[_DiffableEntity], dict[str, JSONValue]],
) -> list[SemanticDiffEntry]:
    """Build semantic diff entries for a family of named entities."""
    entries: list[SemanticDiffEntry] = []
    before_by_id = {item.id: item for item in before}
    after_by_id = {item.id: item for item in after}
    for entity_id in sorted(before_by_id.keys() - after_by_id.keys()):
        entries.append(_build_simple_semantic_entry(entity_type, entity_id, "removed"))
    for entity_id in sorted(after_by_id.keys() - before_by_id.keys()):
        entries.append(_build_simple_semantic_entry(entity_type, entity_id, "added"))
    for entity_id in sorted(before_by_id.keys() & after_by_id.keys()):
        field_changes = _diff_json_fields(
            payload_getter(before_by_id[entity_id]),
            payload_getter(after_by_id[entity_id]),
        )
        if field_changes:
            entries.append(
                _build_changed_semantic_entry(
                    entity_type,
                    entity_id,
                    field_changes,
                )
            )
    return entries


def _semantic_diff_plan(
    before: ContractionPlanSpec | None,
    after: ContractionPlanSpec | None,
) -> list[SemanticDiffEntry]:
    """Build semantic diff entries for the optional contraction plan."""
    if before is None and after is None:
        return []
    if before is None and after is not None:
        return [
            _build_simple_semantic_entry("plan", after.id, "added"),
            *_plan_step_add_remove_entries(after.steps, change_type="added"),
        ]
    if before is not None and after is None:
        return [
            _build_simple_semantic_entry("plan", before.id, "removed"),
            *_plan_step_add_remove_entries(before.steps, change_type="removed"),
        ]

    assert before is not None
    assert after is not None
    entries: list[SemanticDiffEntry] = []
    if before.id != after.id:
        entries.extend(
            [
                _build_simple_semantic_entry("plan", before.id, "removed"),
                _build_simple_semantic_entry("plan", after.id, "added"),
            ]
        )
    else:
        field_changes = _diff_json_fields(_plan_payload(before), _plan_payload(after))
        if field_changes:
            entries.append(
                _build_changed_semantic_entry("plan", after.id, field_changes)
            )

    entries.extend(
        _semantic_diff_named_entities(
            before.steps,
            after.steps,
            entity_type="step",
            payload_getter=_plan_step_payload,
        )
    )
    before_step_order = [step.id for step in before.steps]
    after_step_order = [step.id for step in after.steps]
    if before_step_order != after_step_order:
        entries.append(
            SemanticDiffEntry(
                entity_type="plan",
                entity_id=after.id,
                change_type="reordered",
                summary=_summary_for_entity("plan", "reordered"),
                field_changes=[
                    SemanticFieldChange(
                        path="steps.order",
                        before=cast(JSONValue, before_step_order),
                        after=cast(JSONValue, after_step_order),
                    )
                ],
            )
        )
    return entries


def _semantic_diff_linear_periodic_chain(
    before: NetworkSpec,
    after: NetworkSpec,
) -> list[SemanticDiffEntry]:
    """Report linear periodic-chain changes as one opaque field in v1."""
    before_payload = (
        before.linear_periodic_chain.to_dict()
        if before.linear_periodic_chain is not None
        else None
    )
    after_payload = (
        after.linear_periodic_chain.to_dict()
        if after.linear_periodic_chain is not None
        else None
    )
    if before_payload == after_payload:
        return []
    if before_payload is None and after_payload is not None:
        return [
            _build_simple_semantic_entry(
                "linear_periodic_chain",
                "linear_periodic_chain",
                "added",
            )
        ]
    if before_payload is not None and after_payload is None:
        return [
            _build_simple_semantic_entry(
                "linear_periodic_chain",
                "linear_periodic_chain",
                "removed",
            )
        ]
    return [
        SemanticDiffEntry(
            entity_type="linear_periodic_chain",
            entity_id="linear_periodic_chain",
            change_type="changed",
            summary=_summary_for_entity("linear_periodic_chain", "changed"),
            field_changes=[
                SemanticFieldChange(
                    path="linear_periodic_chain",
                    before=cast(JSONValue, before_payload),
                    after=cast(JSONValue, after_payload),
                )
            ],
        )
    ]


def _semantic_diff_grid_periodic_grid(
    before: NetworkSpec,
    after: NetworkSpec,
) -> list[SemanticDiffEntry]:
    """Report grid periodic-grid changes as one opaque field in v1."""
    before_payload = (
        before.grid_periodic_grid.to_dict()
        if before.grid_periodic_grid is not None
        else None
    )
    after_payload = (
        after.grid_periodic_grid.to_dict()
        if after.grid_periodic_grid is not None
        else None
    )
    if before_payload == after_payload:
        return []
    if before_payload is None and after_payload is not None:
        return [
            _build_simple_semantic_entry(
                "grid_periodic_grid",
                "grid_periodic_grid",
                "added",
            )
        ]
    if before_payload is not None and after_payload is None:
        return [
            _build_simple_semantic_entry(
                "grid_periodic_grid",
                "grid_periodic_grid",
                "removed",
            )
        ]
    return [
        SemanticDiffEntry(
            entity_type="grid_periodic_grid",
            entity_id="grid_periodic_grid",
            change_type="changed",
            summary=_summary_for_entity("grid_periodic_grid", "changed"),
            field_changes=[
                SemanticFieldChange(
                    path="grid_periodic_grid",
                    before=cast(JSONValue, before_payload),
                    after=cast(JSONValue, after_payload),
                )
            ],
        )
    ]


def _semantic_diff_tree_periodic_tree(
    before: NetworkSpec,
    after: NetworkSpec,
) -> list[SemanticDiffEntry]:
    """Report tree periodic-tree changes as one opaque field in v1."""
    before_payload = (
        before.tree_periodic_tree.to_dict()
        if before.tree_periodic_tree is not None
        else None
    )
    after_payload = (
        after.tree_periodic_tree.to_dict()
        if after.tree_periodic_tree is not None
        else None
    )
    if before_payload == after_payload:
        return []
    if before_payload is None and after_payload is not None:
        return [
            _build_simple_semantic_entry(
                "tree_periodic_tree",
                "tree_periodic_tree",
                "added",
            )
        ]
    if before_payload is not None and after_payload is None:
        return [
            _build_simple_semantic_entry(
                "tree_periodic_tree",
                "tree_periodic_tree",
                "removed",
            )
        ]
    return [
        SemanticDiffEntry(
            entity_type="tree_periodic_tree",
            entity_id="tree_periodic_tree",
            change_type="changed",
            summary=_summary_for_entity("tree_periodic_tree", "changed"),
            field_changes=[
                SemanticFieldChange(
                    path="tree_periodic_tree",
                    before=cast(JSONValue, before_payload),
                    after=cast(JSONValue, after_payload),
                )
            ],
        )
    ]


def _index_entities(tensors: Iterable[TensorSpec]) -> list[IndexSpec]:
    """Flatten tensor indices to a top-level list keyed by stable index ids."""
    return [index for tensor in tensors for index in tensor.indices]


def _plan_step_add_remove_entries(
    steps: Iterable[ContractionStepSpec],
    *,
    change_type: str,
) -> list[SemanticDiffEntry]:
    """Build step addition or removal entries in step order."""
    return [
        _build_simple_semantic_entry("step", step.id, change_type) for step in steps
    ]


def _tensor_payload(entity: _DiffableEntity) -> dict[str, JSONValue]:
    """Return a tensor payload without nested index entities."""
    tensor = cast(TensorSpec, entity)
    return {
        "name": tensor.name,
        "position": tensor.position.to_dict(),
        "size": tensor.size.to_dict(),
        "linear_periodic_role": (
            tensor.linear_periodic_role.value
            if tensor.linear_periodic_role is not None
            else None
        ),
        "grid_periodic_role": (
            tensor.grid_periodic_role.value
            if tensor.grid_periodic_role is not None
            else None
        ),
        "tree_periodic_role": (
            tensor.tree_periodic_role.value
            if tensor.tree_periodic_role is not None
            else None
        ),
        "tree_periodic_child_index": tensor.tree_periodic_child_index,
        "metadata": tensor.metadata,
    }


def _index_payload(entity: _DiffableEntity) -> dict[str, JSONValue]:
    """Return the semantic payload for one index."""
    index = cast(IndexSpec, entity)
    return {
        "name": index.name,
        "dimension": index.dimension,
        "offset": index.offset.to_dict(),
        "metadata": index.metadata,
    }


def _edge_payload(entity: _DiffableEntity) -> dict[str, JSONValue]:
    """Return the semantic payload for one edge."""
    edge = cast(EdgeSpec, entity)
    return {
        "name": edge.name,
        "left": edge.left.to_dict(),
        "right": edge.right.to_dict(),
        "metadata": edge.metadata,
    }


def _group_payload(entity: _DiffableEntity) -> dict[str, JSONValue]:
    """Return the semantic payload for one group."""
    group = cast(GroupSpec, entity)
    return {
        "name": group.name,
        "tensor_ids": cast(JSONValue, list(group.tensor_ids)),
        "metadata": group.metadata,
    }


def _note_payload(entity: _DiffableEntity) -> dict[str, JSONValue]:
    """Return the semantic payload for one canvas note."""
    note = cast(CanvasNoteSpec, entity)
    return {
        "text": note.text,
        "position": note.position.to_dict(),
        "metadata": note.metadata,
    }


def _plan_payload(plan: ContractionPlanSpec) -> dict[str, JSONValue]:
    """Return the root contraction-plan payload excluding step bodies."""
    return {
        "name": plan.name,
        "view_snapshots": cast(
            JSONValue, [snapshot.to_dict() for snapshot in plan.view_snapshots]
        ),
        "metadata": plan.metadata,
    }


def _plan_step_payload(entity: _DiffableEntity) -> dict[str, JSONValue]:
    """Return the semantic payload for one contraction step."""
    step = cast(ContractionStepSpec, entity)
    return {
        "left_operand_id": step.left_operand_id,
        "right_operand_id": step.right_operand_id,
        "metadata": step.metadata,
    }


def _diff_json_fields(
    before: JSONValue,
    after: JSONValue,
    *,
    prefix: str = "",
) -> list[SemanticFieldChange]:
    """Recursively diff JSON-like payloads into field-level changes."""
    if before == after:
        return []
    if isinstance(before, dict) and isinstance(after, dict):
        field_changes: list[SemanticFieldChange] = []
        for key in sorted(set(before.keys()) | set(after.keys())):
            child_path = _join_field_path(prefix, key)
            if key not in before:
                field_changes.append(
                    SemanticFieldChange(
                        path=child_path,
                        before=None,
                        after=after[key],
                    )
                )
                continue
            if key not in after:
                field_changes.append(
                    SemanticFieldChange(
                        path=child_path,
                        before=before[key],
                        after=None,
                    )
                )
                continue
            field_changes.extend(
                _diff_json_fields(before[key], after[key], prefix=child_path)
            )
        return field_changes
    if not prefix:
        raise ValueError("Semantic field diffs require a non-empty field path.")
    return [SemanticFieldChange(path=prefix, before=before, after=after)]


def _join_field_path(prefix: str, field_name: str) -> str:
    """Join a field name onto an existing dotted path."""
    if not prefix:
        return field_name
    return f"{prefix}.{field_name}"


def _sort_semantic_entries(
    entries: list[SemanticDiffEntry],
) -> list[SemanticDiffEntry]:
    """Return semantic entries in a stable user-facing order."""
    change_order = {
        "removed": 0,
        "added": 1,
        "changed": 2,
        "reordered": 3,
    }
    return sorted(
        entries,
        key=lambda entry: (
            _ENTITY_SORT_ORDER.get(entry.entity_type, 999),
            entry.entity_id,
            change_order.get(entry.change_type, 999),
        ),
    )


def _entity_label(entity_type: str) -> str:
    """Return the user-facing label for one semantic diff entity type."""
    return _ENTITY_LABELS.get(entity_type, entity_type.replace("_", " ").title())


def _summary_for_entity(entity_type: str, change_type: str) -> str:
    """Return a compact user-facing summary for one semantic diff entry."""
    if entity_type == "plan" and change_type == "reordered":
        return "Contraction step order changed."
    return f"{_entity_label(entity_type)} {change_type}."


def _summary_for_changed_entry(
    entity_type: str,
    field_changes: list[SemanticFieldChange],
) -> str:
    """Return a changed-entry summary that names the affected fields."""
    field_names = ", ".join(
        _unique_field_paths(field_change.path for field_change in field_changes)
    )
    return f"{_entity_label(entity_type)} fields changed: {field_names}."


def _build_simple_semantic_entry(
    entity_type: str,
    entity_id: str,
    change_type: str,
) -> SemanticDiffEntry:
    """Build one semantic diff entry without field-level payload changes."""
    return SemanticDiffEntry(
        entity_type=entity_type,
        entity_id=entity_id,
        change_type=change_type,
        summary=_summary_for_entity(entity_type, change_type),
    )


def _build_changed_semantic_entry(
    entity_type: str,
    entity_id: str,
    field_changes: list[SemanticFieldChange],
) -> SemanticDiffEntry:
    """Build one semantic diff entry with field-level payload changes."""
    return SemanticDiffEntry(
        entity_type=entity_type,
        entity_id=entity_id,
        change_type="changed",
        summary=_summary_for_changed_entry(entity_type, field_changes),
        field_changes=field_changes,
    )


def _unique_field_paths(paths: Iterable[str]) -> list[str]:
    """Return field paths in first-seen order without duplicates."""
    ordered_paths: list[str] = []
    seen_paths: set[str] = set()
    for path in paths:
        if path in seen_paths:
            continue
        seen_paths.add(path)
        ordered_paths.append(path)
    return ordered_paths


_STANDARD_SEMANTIC_ENTITY_CONFIGS: tuple[_SemanticEntityConfig, ...] = (
    _SemanticEntityConfig(
        entity_type="tensor",
        entities=lambda spec: spec.tensors,
        payload_getter=_tensor_payload,
    ),
    _SemanticEntityConfig(
        entity_type="index",
        entities=lambda spec: _index_entities(spec.tensors),
        payload_getter=_index_payload,
    ),
    _SemanticEntityConfig(
        entity_type="edge",
        entities=lambda spec: spec.edges,
        payload_getter=_edge_payload,
    ),
    _SemanticEntityConfig(
        entity_type="group",
        entities=lambda spec: spec.groups,
        payload_getter=_group_payload,
    ),
    _SemanticEntityConfig(
        entity_type="note",
        entities=lambda spec: spec.notes,
        payload_getter=_note_payload,
    ),
)


__all__ = [
    "DiffEntityChanges",
    "SemanticDiffEntry",
    "SemanticFieldChange",
    "SemanticSpecDiffResult",
    "SpecDiffResult",
    "diff_specs",
    "semantic_diff_specs",
]
