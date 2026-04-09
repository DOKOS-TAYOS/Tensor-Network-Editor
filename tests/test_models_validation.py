from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import pytest

from tensor_network_editor.errors import SpecValidationError
from tensor_network_editor.models import (
    CanvasNoteSpec,
    CanvasPosition,
    ContractionOperandLayoutSpec,
    ContractionPlanSpec,
    ContractionStepSpec,
    ContractionViewSnapshotSpec,
    EdgeEndpointRef,
    EdgeSpec,
    GroupSpec,
    IndexSpec,
    LinearPeriodicTensorRole,
    NetworkSpec,
    TensorSize,
    TensorSpec,
    ValidationIssue,
)
from tensor_network_editor.validation import ensure_valid_spec, validate_spec
from tests.factories import (
    build_linear_periodic_carry_chain_spec,
    build_linear_periodic_chain_spec,
    build_linear_periodic_partial_carry_chain_spec,
)


def build_valid_spec() -> NetworkSpec:
    return NetworkSpec(
        id="network_validation",
        name="validation-demo",
        tensors=[
            TensorSpec(
                id="tensor_left",
                name="Left",
                position=CanvasPosition(x=40.0, y=80.0),
                size=TensorSize(width=196.0, height=118.0),
                indices=[
                    IndexSpec(id="tensor_left_open", name="left_open", dimension=2),
                    IndexSpec(id="tensor_left_bond", name="shared", dimension=5),
                ],
            ),
            TensorSpec(
                id="tensor_right",
                name="Right",
                position=CanvasPosition(x=220.0, y=80.0),
                indices=[
                    IndexSpec(id="tensor_right_bond", name="shared", dimension=5),
                    IndexSpec(id="tensor_right_open", name="right_open", dimension=7),
                ],
            ),
        ],
        groups=[
            GroupSpec(
                id="group_pair",
                name="Pair",
                tensor_ids=["tensor_left", "tensor_right"],
            )
        ],
        edges=[
            EdgeSpec(
                id="edge_shared",
                name="shared",
                left=EdgeEndpointRef(
                    tensor_id="tensor_left", index_id="tensor_left_bond"
                ),
                right=EdgeEndpointRef(
                    tensor_id="tensor_right", index_id="tensor_right_bond"
                ),
            )
        ],
    )


def find_issue(issues: list[ValidationIssue], code: str) -> ValidationIssue:
    return next(issue for issue in issues if issue.code == code)


def duplicate_index_connection(spec: NetworkSpec) -> None:
    spec.edges.append(
        EdgeSpec(
            id="edge_duplicate",
            name="duplicate",
            left=EdgeEndpointRef(tensor_id="tensor_left", index_id="tensor_left_bond"),
            right=EdgeEndpointRef(
                tensor_id="tensor_right", index_id="tensor_right_open"
            ),
        )
    )


def dimension_mismatch(spec: NetworkSpec) -> None:
    spec.tensors[1].indices[0] = IndexSpec(
        id="tensor_right_bond",
        name="shared",
        dimension=9,
    )


def duplicate_index_name(spec: NetworkSpec) -> None:
    spec.tensors[0].indices[1] = IndexSpec(
        id="tensor_left_bond",
        name="left_open",
        dimension=5,
    )


def invalid_size(spec: NetworkSpec) -> None:
    spec.tensors[0].size = TensorSize(width=0.0, height=118.0)


def missing_group_tensor(spec: NetworkSpec) -> None:
    spec.groups[0] = GroupSpec(
        id="group_pair",
        name="Pair",
        tensor_ids=["tensor_left", "tensor_missing"],
    )


def invalid_note_text(spec: NetworkSpec) -> None:
    spec.notes = [
        CanvasNoteSpec(
            id="note_empty",
            text="   ",
            position=CanvasPosition(x=3.0, y=7.0),
        )
    ]


def reused_contraction_operand(spec: NetworkSpec) -> None:
    spec.tensors.append(
        TensorSpec(
            id="tensor_extra",
            name="Extra",
            position=CanvasPosition(x=360.0, y=80.0),
            indices=[IndexSpec(id="tensor_extra_open", name="free", dimension=11)],
        )
    )
    spec.contraction_plan = ContractionPlanSpec(
        id="plan_invalid",
        name="Invalid path",
        steps=[
            ContractionStepSpec(
                id="step_pair",
                left_operand_id="tensor_left",
                right_operand_id="tensor_right",
            ),
            ContractionStepSpec(
                id="step_reuse",
                left_operand_id="tensor_left",
                right_operand_id="tensor_extra",
            ),
        ],
    )


def mismatched_edge_owner(spec: NetworkSpec) -> None:
    spec.edges[0] = EdgeSpec(
        id="edge_shared",
        name="shared",
        left=EdgeEndpointRef(
            tensor_id="tensor_right",
            index_id="tensor_left_bond",
        ),
        right=EdgeEndpointRef(
            tensor_id="tensor_right",
            index_id="tensor_right_bond",
        ),
    )


def non_serializable_metadata(spec: NetworkSpec) -> None:
    spec.metadata = cast(Any, {"bad": {1, 2, 3}})


def mismatched_linear_periodic_boundary(spec: NetworkSpec) -> None:
    assert spec.linear_periodic_chain is not None
    final_previous_boundary = spec.linear_periodic_chain.final_cell.tensors[1]
    final_previous_boundary.indices[0].dimension = 11


def test_canvas_note_round_trip_is_serializable() -> None:
    note = CanvasNoteSpec(
        id="note_canvas",
        text="Review this subnet",
        position=CanvasPosition(x=12.0, y=-4.0),
    )

    payload = note.to_dict()
    restored = CanvasNoteSpec.from_dict(cast(dict[str, object], payload))

    assert restored.text == "Review this subnet"
    assert restored.position.x == 12.0
    assert restored.position.y == -4.0


def test_contraction_plan_round_trip_is_serializable() -> None:
    plan = ContractionPlanSpec(
        id="plan_manual",
        name="Manual path",
        steps=[
            ContractionStepSpec(
                id="step_one",
                left_operand_id="tensor_left",
                right_operand_id="tensor_right",
            )
        ],
    )

    payload = plan.to_dict()
    restored = ContractionPlanSpec.from_dict(cast(dict[str, object], payload))

    assert restored.name == "Manual path"
    assert restored.steps[0].id == "step_one"
    assert restored.steps[0].left_operand_id == "tensor_left"


def test_contraction_plan_round_trip_preserves_view_snapshots() -> None:
    plan = ContractionPlanSpec(
        id="plan_manual",
        name="Manual path",
        steps=[
            ContractionStepSpec(
                id="step_one",
                left_operand_id="tensor_left",
                right_operand_id="tensor_right",
            )
        ],
        view_snapshots=[
            ContractionViewSnapshotSpec(
                applied_step_count=0,
                operand_layouts=[
                    ContractionOperandLayoutSpec(
                        operand_id="tensor_left",
                        position=CanvasPosition(x=20.0, y=40.0),
                        size=TensorSize(width=180.0, height=108.0),
                    ),
                    ContractionOperandLayoutSpec(
                        operand_id="tensor_right",
                        position=CanvasPosition(x=220.0, y=40.0),
                        size=TensorSize(width=190.0, height=118.0),
                    ),
                ],
            )
        ],
    )

    payload = plan.to_dict()
    restored = ContractionPlanSpec.from_dict(cast(dict[str, object], payload))

    assert len(restored.view_snapshots) == 1
    assert restored.view_snapshots[0].applied_step_count == 0
    assert restored.view_snapshots[0].operand_layouts[0].operand_id == "tensor_left"
    assert restored.view_snapshots[0].operand_layouts[1].size.width == 190.0


def test_index_offset_round_trip_is_serializable() -> None:
    index = IndexSpec(
        id="index_with_offset",
        name="offset_index",
        dimension=3,
        offset=CanvasPosition(x=34.0, y=-18.0),
    )

    payload = index.to_dict()
    restored = IndexSpec.from_dict(cast(dict[str, object], payload))

    assert restored.offset.x == 34.0
    assert restored.offset.y == -18.0


def test_tensor_size_round_trip_is_serializable() -> None:
    tensor = TensorSpec(
        id="tensor_with_size",
        name="Sized",
        size=TensorSize(width=212.0, height=132.0),
    )

    payload = tensor.to_dict()
    restored = TensorSpec.from_dict(cast(dict[str, object], payload))

    assert restored.size.width == 212.0
    assert restored.size.height == 132.0


def test_tensor_shape_uses_index_order() -> None:
    spec = build_valid_spec()

    assert spec.tensors[0].shape == (2, 5)
    assert spec.tensors[1].shape == (5, 7)


def test_tensor_round_trip_preserves_linear_periodic_role() -> None:
    tensor = TensorSpec(
        id="boundary_tensor",
        name="Boundary",
        linear_periodic_role=LinearPeriodicTensorRole.NEXT,
        indices=[IndexSpec(id="slot_1", name="slot_1", dimension=3)],
    )

    payload = tensor.to_dict()
    restored = TensorSpec.from_dict(cast(dict[str, object], payload))

    assert restored.linear_periodic_role is LinearPeriodicTensorRole.NEXT


def test_open_indices_are_derived_from_unconnected_ports() -> None:
    spec = build_valid_spec()

    assert [index.name for _, index in spec.open_indices()] == [
        "left_open",
        "right_open",
    ]


def test_validate_spec_accepts_valid_network() -> None:
    assert validate_spec(build_valid_spec()) == []


def test_validate_spec_accepts_valid_network_with_notes_and_plan() -> None:
    spec = build_valid_spec()
    spec.notes = [
        CanvasNoteSpec(
            id="note_plan",
            text="Contract from left to right",
            position=CanvasPosition(x=18.0, y=24.0),
        )
    ]
    spec.contraction_plan = ContractionPlanSpec(
        id="plan_pair",
        name="Pair path",
        steps=[
            ContractionStepSpec(
                id="step_pair",
                left_operand_id="tensor_left",
                right_operand_id="tensor_right",
            )
        ],
        view_snapshots=[
            ContractionViewSnapshotSpec(
                applied_step_count=0,
                operand_layouts=[
                    ContractionOperandLayoutSpec(
                        operand_id="tensor_left",
                        position=CanvasPosition(x=40.0, y=80.0),
                        size=TensorSize(width=196.0, height=118.0),
                    ),
                    ContractionOperandLayoutSpec(
                        operand_id="tensor_right",
                        position=CanvasPosition(x=220.0, y=80.0),
                        size=TensorSize(width=180.0, height=108.0),
                    ),
                    ContractionOperandLayoutSpec(
                        operand_id="unknown_stale_operand",
                        position=CanvasPosition(x=320.0, y=80.0),
                        size=TensorSize(width=180.0, height=108.0),
                    ),
                ],
            )
        ],
    )

    assert validate_spec(spec) == []


def test_validate_spec_accepts_valid_linear_periodic_chain() -> None:
    assert validate_spec(build_linear_periodic_chain_spec()) == []


def test_validate_spec_accepts_valid_linear_periodic_carry_chain() -> None:
    assert validate_spec(build_linear_periodic_carry_chain_spec()) == []


def test_validate_spec_accepts_linear_periodic_partial_carry_chain() -> None:
    assert validate_spec(build_linear_periodic_partial_carry_chain_spec()) == []


def test_validate_spec_rejects_linear_periodic_next_that_is_not_last() -> None:
    spec = build_linear_periodic_carry_chain_spec()
    assert spec.linear_periodic_chain is not None
    spec.linear_periodic_chain.initial_cell.tensors.append(
        TensorSpec(
            id="initial_extra_tensor",
            name="InitialExtra",
            position=CanvasPosition(x=220.0, y=240.0),
            indices=[IndexSpec(id="initial_extra_open", name="free", dimension=11)],
        )
    )
    assert spec.linear_periodic_chain.initial_cell.contraction_plan is not None
    spec.linear_periodic_chain.initial_cell.contraction_plan.steps.append(
        ContractionStepSpec(
            id="initial_after_carry",
            left_operand_id="initial_carry",
            right_operand_id="initial_extra_tensor",
        )
    )

    issue = find_issue(validate_spec(spec), "linear-periodic-carry-order")

    assert issue.path == (
        "linear_periodic_chain.initial_cell.contraction_plan.steps.initial_after_carry"
    )


def test_validate_spec_rejects_malformed_contraction_view_snapshot() -> None:
    spec = build_valid_spec()
    spec.contraction_plan = ContractionPlanSpec(
        id="plan_pair",
        name="Pair path",
        steps=[
            ContractionStepSpec(
                id="step_pair",
                left_operand_id="tensor_left",
                right_operand_id="tensor_right",
            )
        ],
        view_snapshots=[
            ContractionViewSnapshotSpec(
                applied_step_count=-1,
                operand_layouts=[
                    ContractionOperandLayoutSpec(
                        operand_id="tensor_left",
                        position=CanvasPosition(x=40.0, y=80.0),
                        size=TensorSize(width=0.0, height=118.0),
                    )
                ],
            )
        ],
    )

    issues = validate_spec(spec)

    assert {issue.code for issue in issues} >= {
        "invalid-contraction-view-snapshot",
        "invalid-size",
    }


def test_validate_spec_rejects_duplicate_operand_ids_in_contraction_view_snapshot() -> (
    None
):
    spec = build_valid_spec()
    spec.contraction_plan = ContractionPlanSpec(
        id="plan_pair",
        name="Pair path",
        steps=[
            ContractionStepSpec(
                id="step_pair",
                left_operand_id="tensor_left",
                right_operand_id="tensor_right",
            )
        ],
        view_snapshots=[
            ContractionViewSnapshotSpec(
                applied_step_count=0,
                operand_layouts=[
                    ContractionOperandLayoutSpec(
                        operand_id="tensor_left",
                        position=CanvasPosition(x=40.0, y=80.0),
                        size=TensorSize(width=196.0, height=118.0),
                    ),
                    ContractionOperandLayoutSpec(
                        operand_id="tensor_left",
                        position=CanvasPosition(x=220.0, y=80.0),
                        size=TensorSize(width=180.0, height=108.0),
                    ),
                ],
            )
        ],
    )

    issues = validate_spec(spec)

    assert find_issue(issues, "invalid-contraction-view-snapshot").path == (
        "contraction_plan.view_snapshots.0.operand_layouts.tensor_left.operand_id"
    )


@pytest.mark.parametrize(
    ("mutate", "expected_code", "expected_path"),
    [
        (
            duplicate_index_connection,
            "index-already-connected",
            "edges.edge_duplicate.left",
        ),
        (dimension_mismatch, "dimension-mismatch", "edges.edge_shared"),
        (duplicate_index_name, "duplicate-index-name", "tensors.tensor_left.indices"),
        (invalid_size, "invalid-size", "tensors.tensor_left.size"),
        (
            missing_group_tensor,
            "missing-group-tensor",
            "groups.group_pair.tensor_ids",
        ),
        (invalid_note_text, "invalid-note-text", "notes.note_empty.text"),
        (
            reused_contraction_operand,
            "contraction-operand-reused",
            "contraction_plan.steps.step_reuse.left_operand_id",
        ),
        (mismatched_edge_owner, "endpoint-tensor-mismatch", "edges.edge_shared.left"),
        (non_serializable_metadata, "metadata-not-serializable", "metadata"),
        (
            mismatched_linear_periodic_boundary,
            "linear-periodic-interface-mismatch",
            "linear_periodic_chain.periodic_cell.next_interface",
        ),
    ],
)
def test_validate_spec_reports_targeted_issue_codes_and_paths(
    mutate: Callable[[NetworkSpec], None],
    expected_code: str,
    expected_path: str,
) -> None:
    spec = (
        build_linear_periodic_chain_spec()
        if expected_code == "linear-periodic-interface-mismatch"
        else build_valid_spec()
    )

    mutate(spec)
    issue = find_issue(validate_spec(spec), expected_code)

    assert issue.path == expected_path


def test_validate_spec_accepts_multi_step_contraction_plan() -> None:
    spec = build_valid_spec()
    spec.tensors.append(
        TensorSpec(
            id="tensor_extra",
            name="Extra",
            position=CanvasPosition(x=360.0, y=80.0),
            indices=[IndexSpec(id="tensor_extra_open", name="free", dimension=11)],
        )
    )
    spec.contraction_plan = ContractionPlanSpec(
        id="plan_valid",
        name="Valid path",
        steps=[
            ContractionStepSpec(
                id="step_pair",
                left_operand_id="tensor_left",
                right_operand_id="tensor_right",
            ),
            ContractionStepSpec(
                id="step_total",
                left_operand_id="step_pair",
                right_operand_id="tensor_extra",
            ),
        ],
    )

    assert validate_spec(spec) == []


def test_ensure_valid_spec_raises_spec_validation_error() -> None:
    spec = build_valid_spec()
    spec.tensors[0].indices[0] = IndexSpec(id="tensor_left_open", name="", dimension=2)

    with pytest.raises(SpecValidationError, match="invalid"):
        ensure_valid_spec(spec)
