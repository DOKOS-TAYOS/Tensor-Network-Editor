from __future__ import annotations

from typing import Any

from tensor_network_editor import semantic_diff_specs
from tensor_network_editor.internal.diffing._diffing import _entity_label
from tensor_network_editor.models import (
    SemanticDiffEntry,
    SemanticSpecDiffResult,
)
from tests.factories import (
    build_linear_periodic_chain_spec,
    build_sample_spec,
    build_three_tensor_complete_plan_spec,
    build_three_tensor_spec,
    build_tree_periodic_tree_spec,
)


def _entries_by_key(
    result: SemanticSpecDiffResult,
) -> dict[tuple[str, str, str], SemanticDiffEntry]:
    return {
        (entry.entity_type, entry.entity_id, entry.change_type): entry
        for entry in result.entries
    }


def _field_changes_by_path(entry: SemanticDiffEntry) -> dict[str, tuple[Any, Any]]:
    return {
        change.path: (change.before, change.after) for change in entry.field_changes
    }


def test_entity_label_internal_helper_covers_known_and_fallback_types() -> None:
    assert _entity_label("tensor") == "Tensor"
    assert _entity_label("linear_periodic_chain") == "Linear periodic chain"
    assert _entity_label("tree_periodic_tree") == "Tree periodic tree"
    assert _entity_label("manual_subtree") == "Manual Subtree"


def test_semantic_diff_specs_reports_tensor_and_index_field_changes() -> None:
    before = build_sample_spec()
    after = build_sample_spec()
    after.tensors[0].name = "A prime"
    after.tensors[0].indices[0].dimension = 5
    after.tensors[0].indices[0].metadata = {"tags": [" beta ", "alpha", "beta"]}

    result = semantic_diff_specs(before, after)
    entries = _entries_by_key(result)

    tensor_entry = entries[("tensor", "tensor_a", "changed")]
    assert tensor_entry.summary == "Tensor fields changed: name."
    assert _field_changes_by_path(tensor_entry) == {
        "name": ("A", "A prime"),
    }

    index_entry = entries[("index", "tensor_a_i", "changed")]
    assert index_entry.summary == "Index fields changed: dimension, metadata.tags."
    assert _field_changes_by_path(index_entry) == {
        "dimension": (2, 5),
        "metadata.tags": (None, ["alpha", "beta"]),
    }


def test_semantic_diff_specs_reports_edge_and_step_changes() -> None:
    before = build_three_tensor_spec()
    after = build_three_tensor_spec()
    after.edges[0].left = after.edges[0].left.__class__(
        tensor_id="tensor_a",
        index_id="tensor_a_i",
    )
    assert after.contraction_plan is not None
    after.contraction_plan.steps[0].right_operand_id = "tensor_c"
    after.contraction_plan.steps[0].metadata = {"tags": [" future ", "debug", "future"]}

    result = semantic_diff_specs(before, after)
    entries = _entries_by_key(result)

    edge_entry = entries[("edge", "edge_x", "changed")]
    assert edge_entry.summary == "Edge fields changed: left.index_id."
    assert _field_changes_by_path(edge_entry) == {
        "left.index_id": ("tensor_a_x", "tensor_a_i"),
    }

    step_entry = entries[("step", "step_ab", "changed")]
    assert (
        step_entry.summary
        == "Contraction step fields changed: metadata.tags, right_operand_id."
    )
    assert _field_changes_by_path(step_entry) == {
        "right_operand_id": ("tensor_b", "tensor_c"),
        "metadata.tags": (None, ["debug", "future"]),
    }


def test_semantic_diff_specs_reports_step_additions_and_plan_reordering() -> None:
    before = build_three_tensor_spec()
    after = build_three_tensor_complete_plan_spec()
    assert after.contraction_plan is not None
    after.contraction_plan.id = "plan_chain"
    after.contraction_plan.steps = [
        after.contraction_plan.steps[1],
        after.contraction_plan.steps[0],
    ]

    result = semantic_diff_specs(before, after)
    entries = _entries_by_key(result)

    changed_plan_entry = entries[("plan", "plan_chain", "changed")]
    assert changed_plan_entry.summary == "Contraction plan fields changed: name."
    assert _field_changes_by_path(changed_plan_entry) == {
        "name": ("Chain path", "Complete chain path"),
    }

    added_step_entry = entries[("step", "step_abc", "added")]
    assert added_step_entry.summary == "Contraction step added."
    assert added_step_entry.field_changes == []

    reorder_entry = entries[("plan", "plan_chain", "reordered")]
    assert reorder_entry.summary == "Contraction step order changed."
    assert _field_changes_by_path(reorder_entry) == {
        "steps.order": (["step_ab"], ["step_abc", "step_ab"]),
    }


def test_semantic_diff_specs_reports_linear_periodic_chain_as_opaque_change() -> None:
    before = build_linear_periodic_chain_spec()
    after = build_linear_periodic_chain_spec()
    assert after.linear_periodic_chain is not None
    after.linear_periodic_chain.active_cell = (
        after.linear_periodic_chain.active_cell.__class__.FINAL
    )

    result = semantic_diff_specs(before, after)
    entries = _entries_by_key(result)

    chain_entry = entries[("linear_periodic_chain", "linear_periodic_chain", "changed")]
    assert chain_entry.summary == "Linear periodic chain changed."
    assert list(_field_changes_by_path(chain_entry).keys()) == ["linear_periodic_chain"]


def test_semantic_diff_specs_reports_tree_periodic_tree_as_opaque_change() -> None:
    before = build_tree_periodic_tree_spec()
    after = build_tree_periodic_tree_spec()
    assert after.tree_periodic_tree is not None
    after.tree_periodic_tree.branching_factor = 4

    result = semantic_diff_specs(before, after)
    entries = _entries_by_key(result)

    tree_entry = entries[("tree_periodic_tree", "tree_periodic_tree", "changed")]
    assert tree_entry.summary == "Tree periodic tree changed."
    assert list(_field_changes_by_path(tree_entry).keys()) == ["tree_periodic_tree"]
