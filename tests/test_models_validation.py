from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import pytest

from tensor_network_editor._validation_linear_periodic import (
    _build_carry_validation_context,
)
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
    GridPeriodicTensorRole,
    GroupSpec,
    IndexSpec,
    LinearPeriodicCellName,
    LinearPeriodicTensorRole,
    NetworkSpec,
    TensorSize,
    TensorSpec,
    ValidationIssue,
)
from tensor_network_editor.validation import ensure_valid_spec, validate_spec
from tests.factories import (
    build_grid_periodic_grid_spec,
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


def find_issue_paths(issues: list[ValidationIssue], code: str) -> list[str]:
    return [issue.path for issue in issues if issue.code == code]


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


def circular_metadata(spec: NetworkSpec) -> None:
    recursive_list: list[object] = []
    recursive_list.append(recursive_list)
    spec.metadata = cast(Any, {"loop": recursive_list})


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


def test_tensor_round_trip_preserves_grid_periodic_role() -> None:
    tensor = TensorSpec(
        id="grid_boundary_tensor",
        name="GridBoundary",
        grid_periodic_role=GridPeriodicTensorRole.DOWN,
        indices=[IndexSpec(id="slot_1", name="slot_1", dimension=3)],
    )

    payload = tensor.to_dict()
    restored = TensorSpec.from_dict(cast(dict[str, object], payload))

    assert restored.grid_periodic_role is GridPeriodicTensorRole.DOWN


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


def test_validate_spec_accepts_valid_grid_periodic_grid() -> None:
    assert validate_spec(build_grid_periodic_grid_spec()) == []


def test_validate_spec_accepts_valid_linear_periodic_carry_chain() -> None:
    assert validate_spec(build_linear_periodic_carry_chain_spec()) == []


def test_validate_spec_accepts_linear_periodic_partial_carry_chain() -> None:
    assert validate_spec(build_linear_periodic_partial_carry_chain_spec()) == []


def test_build_carry_validation_context_internal_helper_collects_interface_state() -> (
    None
):
    spec = build_linear_periodic_carry_chain_spec()
    assert spec.linear_periodic_chain is not None

    context = _build_carry_validation_context(
        LinearPeriodicCellName.PERIODIC,
        spec.linear_periodic_chain.periodic_cell,
        previous_expected=1,
        next_expected=1,
    )

    assert context.cell_prefix == "linear_periodic_chain.periodic_cell"
    assert context.incoming_labels == ("periodicleft_left",)
    assert context.outgoing_labels == ("periodicright_right",)
    assert context.label_by_index_id["periodic_left_in"] == "periodicleft_left"
    assert context.dimension_by_label["periodicleft_left"] == 3
    assert "__linear_previous__" in context.operand_state_by_id
    assert "__linear_next__" in context.operand_state_by_id


def test_validate_spec_rejects_linear_periodic_step_using_previous_and_next_together() -> (
    None
):
    spec = build_linear_periodic_carry_chain_spec()
    assert spec.linear_periodic_chain is not None
    periodic_cell = spec.linear_periodic_chain.periodic_cell
    assert periodic_cell.contraction_plan is not None
    periodic_cell.contraction_plan.steps[0] = ContractionStepSpec(
        id="periodic_prev_next",
        left_operand_id="__linear_previous__",
        right_operand_id="__linear_next__",
    )
    periodic_cell.contraction_plan.steps[2] = ContractionStepSpec(
        id="periodic_after",
        left_operand_id="periodic_left_tensor",
        right_operand_id="periodic_right_tensor",
    )

    issue_paths = find_issue_paths(
        validate_spec(spec),
        "linear-periodic-carry-boundary",
    )

    assert (
        "linear_periodic_chain.periodic_cell.contraction_plan.steps.periodic_prev_next"
        in issue_paths
    )


def test_validate_spec_rejects_linear_periodic_next_step_without_outgoing_labels() -> (
    None
):
    spec = build_linear_periodic_carry_chain_spec()
    assert spec.linear_periodic_chain is not None
    initial_cell = spec.linear_periodic_chain.initial_cell
    initial_cell.tensors.append(
        TensorSpec(
            id="initial_extra_tensor",
            name="InitialExtra",
            position=CanvasPosition(x=220.0, y=240.0),
            indices=[IndexSpec(id="initial_extra_open", name="free", dimension=11)],
        )
    )
    assert initial_cell.contraction_plan is not None
    initial_cell.contraction_plan.steps[0] = ContractionStepSpec(
        id="initial_carry",
        left_operand_id="initial_extra_tensor",
        right_operand_id="__linear_next__",
    )

    issue_paths = find_issue_paths(
        validate_spec(spec),
        "linear-periodic-carry-operand",
    )

    assert (
        "linear_periodic_chain.initial_cell.contraction_plan.steps.initial_carry"
        in issue_paths
    )


def test_validate_spec_rejects_linear_periodic_previous_step_without_incoming_labels() -> (
    None
):
    spec = build_linear_periodic_carry_chain_spec()
    assert spec.linear_periodic_chain is not None
    final_cell = spec.linear_periodic_chain.final_cell
    final_cell.tensors.append(
        TensorSpec(
            id="final_extra_tensor",
            name="FinalExtra",
            position=CanvasPosition(x=280.0, y=260.0),
            indices=[IndexSpec(id="final_extra_open", name="free", dimension=11)],
        )
    )
    assert final_cell.contraction_plan is not None
    final_cell.contraction_plan.steps[0] = ContractionStepSpec(
        id="final_contract",
        left_operand_id="__linear_previous__",
        right_operand_id="final_extra_tensor",
    )

    issue_paths = find_issue_paths(
        validate_spec(spec),
        "linear-periodic-carry-operand",
    )

    assert (
        "linear_periodic_chain.final_cell.contraction_plan.steps.final_contract"
        in issue_paths
    )


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


def test_validate_spec_rejects_mixed_linear_and_grid_periodic_modes() -> None:
    spec = build_grid_periodic_grid_spec()
    spec.linear_periodic_chain = (
        build_linear_periodic_chain_spec().linear_periodic_chain
    )

    issue = find_issue(validate_spec(spec), "periodic-mode-conflict")

    assert issue.path == "grid_periodic_grid"


def test_validate_spec_rejects_grid_periodic_missing_boundary_tensor() -> None:
    spec = build_grid_periodic_grid_spec()
    assert spec.grid_periodic_grid is not None
    spec.grid_periodic_grid.center_cell.tensors = [
        tensor
        for tensor in spec.grid_periodic_grid.center_cell.tensors
        if tensor.grid_periodic_role is not GridPeriodicTensorRole.LEFT
    ]

    issue = find_issue(validate_spec(spec), "grid-periodic-boundary-role")

    assert issue.path == "grid_periodic_grid.center_cell.left_boundary"


def test_validate_spec_rejects_grid_periodic_interface_mismatch() -> None:
    spec = build_grid_periodic_grid_spec()
    assert spec.grid_periodic_grid is not None
    left_boundary = next(
        tensor
        for tensor in spec.grid_periodic_grid.center_cell.tensors
        if tensor.grid_periodic_role is GridPeriodicTensorRole.LEFT
    )
    left_boundary.indices[0].dimension = 6

    issue = find_issue(validate_spec(spec), "grid-periodic-interface-mismatch")

    assert issue.path == "grid_periodic_grid.horizontal_interfaces.middle_row"


def test_validate_spec_rejects_grid_periodic_contraction_plan() -> None:
    spec = build_grid_periodic_grid_spec()
    assert spec.grid_periodic_grid is not None
    spec.grid_periodic_grid.center_cell.contraction_plan = ContractionPlanSpec(
        id="grid_plan",
        name="Grid plan",
        steps=[
            ContractionStepSpec(
                id="grid_step",
                left_operand_id="center_tensor",
                right_operand_id="center_tensor",
            )
        ],
    )

    issue = find_issue(validate_spec(spec), "grid-periodic-contraction-plan")

    assert issue.path == "grid_periodic_grid.center_cell.contraction_plan"


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
        (circular_metadata, "metadata-not-serializable", "metadata"),
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
