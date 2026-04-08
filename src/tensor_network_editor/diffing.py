"""Public helpers for comparing two tensor-network specifications."""

from __future__ import annotations

from collections.abc import Iterable

from ._headless_models import DiffEntityChanges, SpecDiffResult
from .models import NetworkSpec


def diff_specs(before: NetworkSpec, after: NetworkSpec) -> SpecDiffResult:
    """Return a structured diff between two specs based on stable ids."""
    return SpecDiffResult(
        tensor=_diff_named_entities(before.tensors, after.tensors),
        edge=_diff_named_entities(before.edges, after.edges),
        group=_diff_named_entities(before.groups, after.groups),
        note=_diff_named_entities(before.notes, after.notes),
        plan=_diff_plan(before, after),
    )


def _diff_named_entities(
    before: Iterable[object], after: Iterable[object]
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


def _entity_id(entity: object) -> str:
    """Read the ``id`` attribute from one serializable entity."""
    entity_id = getattr(entity, "id", None)
    if not isinstance(entity_id, str):
        raise TypeError("Expected an entity with a string 'id' attribute.")
    return entity_id


def _entity_payload(entity: object) -> object:
    """Serialize one entity for diff comparison."""
    to_dict = getattr(entity, "to_dict", None)
    if not callable(to_dict):
        raise TypeError("Expected an entity with a 'to_dict()' method.")
    return to_dict()


__all__ = ["DiffEntityChanges", "SpecDiffResult", "diff_specs"]
