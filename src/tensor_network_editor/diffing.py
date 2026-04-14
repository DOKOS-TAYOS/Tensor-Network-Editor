"""Public helpers for comparing two tensor-network specifications."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Protocol, cast

from ._headless_models import (
    DiffEntityChanges,
    SemanticDiffEntry,
    SemanticFieldChange,
    SemanticSpecDiffResult,
    SpecDiffResult,
)
from .canonicalization import canonicalize_spec
from .models import (
    CanvasNoteSpec,
    ContractionPlanSpec,
    ContractionStepSpec,
    EdgeSpec,
    GroupSpec,
    IndexSpec,
    NetworkSpec,
    TensorSpec,
)
from .types import JSONValue


class _DiffableEntity(Protocol):
    """Protocol for entities that can be compared by id and serialized payload."""

    id: str

    def to_dict(self) -> dict[str, JSONValue]:
        """Return the serialized payload used for equality checks."""
        ...


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
    entries: list[SemanticDiffEntry] = []
    entries.extend(
        _semantic_diff_named_entities(
            normalized_before.tensors,
            normalized_after.tensors,
            entity_type="tensor",
            payload_getter=_tensor_payload,
        )
    )
    entries.extend(
        _semantic_diff_named_entities(
            _index_entities(normalized_before.tensors),
            _index_entities(normalized_after.tensors),
            entity_type="index",
            payload_getter=_index_payload,
        )
    )
    entries.extend(
        _semantic_diff_named_entities(
            normalized_before.edges,
            normalized_after.edges,
            entity_type="edge",
            payload_getter=_edge_payload,
        )
    )
    entries.extend(
        _semantic_diff_named_entities(
            normalized_before.groups,
            normalized_after.groups,
            entity_type="group",
            payload_getter=_group_payload,
        )
    )
    entries.extend(
        _semantic_diff_named_entities(
            normalized_before.notes,
            normalized_after.notes,
            entity_type="note",
            payload_getter=_note_payload,
        )
    )
    entries.extend(
        _semantic_diff_plan(
            normalized_before.contraction_plan,
            normalized_after.contraction_plan,
        )
    )
    entries.extend(
        _semantic_diff_linear_periodic_chain(normalized_before, normalized_after)
    )
    return SemanticSpecDiffResult(entries=_sort_semantic_entries(entries))


def _diff_named_entities(
    before: Iterable[_DiffableEntity], after: Iterable[_DiffableEntity]
) -> DiffEntityChanges:
    """Diff two entity collections that expose ``id`` and ``to_dict``."""
    before_by_id = {_entity_id(item): item for item in before}
    after_by_id = {_entity_id(item): item for item in after}
    shared_ids = sorted(before_by_id.keys() & after_by_id.keys())
    return DiffEntityChanges(
        added=sorted(after_by_id.keys() - before_by_id.keys()),
        removed=sorted(before_by_id.keys() - after_by_id.keys()),
        changed=[
            entity_id
            for entity_id in shared_ids
            if _entity_payload(before_by_id[entity_id])
            != _entity_payload(after_by_id[entity_id])
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
    before_by_id = {_entity_id(item): item for item in before}
    after_by_id = {_entity_id(item): item for item in after}
    for entity_id in sorted(before_by_id.keys() - after_by_id.keys()):
        entries.append(
            SemanticDiffEntry(
                entity_type=entity_type,
                entity_id=entity_id,
                change_type="removed",
                summary=_summary_for_entity(entity_type, "removed"),
            )
        )
    for entity_id in sorted(after_by_id.keys() - before_by_id.keys()):
        entries.append(
            SemanticDiffEntry(
                entity_type=entity_type,
                entity_id=entity_id,
                change_type="added",
                summary=_summary_for_entity(entity_type, "added"),
            )
        )
    for entity_id in sorted(before_by_id.keys() & after_by_id.keys()):
        field_changes = _diff_json_fields(
            payload_getter(before_by_id[entity_id]),
            payload_getter(after_by_id[entity_id]),
        )
        if field_changes:
            entries.append(
                SemanticDiffEntry(
                    entity_type=entity_type,
                    entity_id=entity_id,
                    change_type="changed",
                    summary=_summary_for_entity(entity_type, "changed"),
                    field_changes=field_changes,
                )
            )
    return entries


def _semantic_diff_plan(
    before: ContractionPlanSpec | None,
    after: ContractionPlanSpec | None,
) -> list[SemanticDiffEntry]:
    """Build semantic diff entries for the optional contraction plan."""
    entries: list[SemanticDiffEntry] = []
    if before is None and after is None:
        return entries
    if before is None and after is not None:
        entries.append(
            SemanticDiffEntry(
                entity_type="plan",
                entity_id=after.id,
                change_type="added",
                summary=_summary_for_entity("plan", "added"),
            )
        )
        entries.extend(_plan_step_add_remove_entries(after.steps, change_type="added"))
        return entries
    if before is not None and after is None:
        entries.append(
            SemanticDiffEntry(
                entity_type="plan",
                entity_id=before.id,
                change_type="removed",
                summary=_summary_for_entity("plan", "removed"),
            )
        )
        entries.extend(
            _plan_step_add_remove_entries(before.steps, change_type="removed")
        )
        return entries

    assert before is not None
    assert after is not None
    if before.id != after.id:
        entries.append(
            SemanticDiffEntry(
                entity_type="plan",
                entity_id=before.id,
                change_type="removed",
                summary=_summary_for_entity("plan", "removed"),
            )
        )
        entries.append(
            SemanticDiffEntry(
                entity_type="plan",
                entity_id=after.id,
                change_type="added",
                summary=_summary_for_entity("plan", "added"),
            )
        )
    else:
        field_changes = _diff_json_fields(_plan_payload(before), _plan_payload(after))
        if field_changes:
            entries.append(
                SemanticDiffEntry(
                    entity_type="plan",
                    entity_id=after.id,
                    change_type="changed",
                    summary=_summary_for_entity("plan", "changed"),
                    field_changes=field_changes,
                )
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
            SemanticDiffEntry(
                entity_type="linear_periodic_chain",
                entity_id="linear_periodic_chain",
                change_type="added",
                summary=_summary_for_entity("linear_periodic_chain", "added"),
            )
        ]
    if before_payload is not None and after_payload is None:
        return [
            SemanticDiffEntry(
                entity_type="linear_periodic_chain",
                entity_id="linear_periodic_chain",
                change_type="removed",
                summary=_summary_for_entity("linear_periodic_chain", "removed"),
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
        SemanticDiffEntry(
            entity_type="step",
            entity_id=step.id,
            change_type=change_type,
            summary=_summary_for_entity("step", change_type),
        )
        for step in steps
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
    entity_order = {
        "tensor": 0,
        "index": 1,
        "edge": 2,
        "group": 3,
        "note": 4,
        "plan": 5,
        "step": 6,
        "linear_periodic_chain": 7,
    }
    change_order = {
        "removed": 0,
        "added": 1,
        "changed": 2,
        "reordered": 3,
    }
    return sorted(
        entries,
        key=lambda entry: (
            entity_order.get(entry.entity_type, 999),
            entry.entity_id,
            change_order.get(entry.change_type, 999),
        ),
    )


def _summary_for_entity(entity_type: str, change_type: str) -> str:
    """Return a compact user-facing summary for one semantic diff entry."""
    if entity_type == "plan" and change_type == "reordered":
        return "Contraction step order changed."
    label = {
        "tensor": "Tensor",
        "index": "Index",
        "edge": "Edge",
        "group": "Group",
        "note": "Note",
        "plan": "Contraction plan",
        "step": "Contraction step",
        "linear_periodic_chain": "Linear periodic chain",
    }.get(entity_type, entity_type.replace("_", " ").title())
    return f"{label} {change_type}."


def _entity_id(entity: _DiffableEntity) -> str:
    """Read the ``id`` attribute from one serializable entity."""
    return entity.id


def _entity_payload(entity: _DiffableEntity) -> dict[str, JSONValue]:
    """Serialize one entity for diff comparison."""
    return entity.to_dict()


__all__ = [
    "DiffEntityChanges",
    "SemanticDiffEntry",
    "SemanticFieldChange",
    "SemanticSpecDiffResult",
    "SpecDiffResult",
    "diff_specs",
    "semantic_diff_specs",
]
