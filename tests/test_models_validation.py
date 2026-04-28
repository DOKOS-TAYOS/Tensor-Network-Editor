from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any, cast

import pytest

from tensor_network_editor.errors import SpecValidationError
from tensor_network_editor.internal.validation._validation_common import (
    prefix_validation_issues,
)
from tensor_network_editor.internal.validation._validation_linear_periodic import (
    _build_carry_validation_context,
)
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
    HyperedgeSpec,
    IndexSpec,
    LinearPeriodicCellName,
    LinearPeriodicTensorRole,
    NetworkSpec,
    TensorDataMode,
    TensorDataSpec,
    TensorSize,
    TensorSpec,
    TreePeriodicTensorRole,
    ValidationIssue,
)
from tensor_network_editor.validation import ensure_valid_spec, validate_spec
from tests.factories import (
    build_grid_periodic_grid_spec,
    build_linear_periodic_carry_chain_spec,
    build_linear_periodic_chain_spec,
    build_linear_periodic_partial_carry_chain_spec,
    build_tree_periodic_tree_spec,
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


def build_valid_hyperedge_spec() -> NetworkSpec:
    return NetworkSpec(
        id="network_hyperedge_validation",
        name="hyperedge-demo",
        tensors=[
            TensorSpec(
                id="tensor_a",
                name="A",
                position=CanvasPosition(x=40.0, y=80.0),
                indices=[
                    IndexSpec(id="tensor_a_open", name="a_open", dimension=2),
                    IndexSpec(id="tensor_a_h", name="h", dimension=3),
                ],
            ),
            TensorSpec(
                id="tensor_b",
                name="B",
                position=CanvasPosition(x=220.0, y=80.0),
                indices=[
                    IndexSpec(id="tensor_b_h", name="h", dimension=3),
                    IndexSpec(id="tensor_b_open", name="b_open", dimension=5),
                ],
            ),
            TensorSpec(
                id="tensor_c",
                name="C",
                position=CanvasPosition(x=400.0, y=80.0),
                indices=[
                    IndexSpec(id="tensor_c_h", name="h", dimension=3),
                    IndexSpec(id="tensor_c_open", name="c_open", dimension=7),
                ],
            ),
        ],
        hyperedges=[
            HyperedgeSpec(
                id="hyperedge_shared",
                name="shared_h",
                endpoints=[
                    EdgeEndpointRef(tensor_id="tensor_a", index_id="tensor_a_h"),
                    EdgeEndpointRef(tensor_id="tensor_b", index_id="tensor_b_h"),
                    EdgeEndpointRef(tensor_id="tensor_c", index_id="tensor_c_h"),
                ],
            )
        ],
    )


def find_issue(issues: list[ValidationIssue], code: str) -> ValidationIssue:
    return next(issue for issue in issues if issue.code == code)


def find_issue_paths(issues: list[ValidationIssue], code: str) -> list[str]:
    return [issue.path for issue in issues if issue.code == code]


def test_prefix_validation_issues_nests_existing_paths() -> None:
    prefixed = prefix_validation_issues(
        "grid_periodic_grid.center_cell",
        [
            ValidationIssue(
                code="invalid-size",
                message="bad size",
                path="tensors.tensor_a.size",
            ),
            ValidationIssue(
                code="missing-note",
                message="missing note",
                path="",
            ),
        ],
    )

    assert prefixed == [
        ValidationIssue(
            code="invalid-size",
            message="bad size",
            path="grid_periodic_grid.center_cell.tensors.tensor_a.size",
        ),
        ValidationIssue(
            code="missing-note",
            message="missing note",
            path="grid_periodic_grid.center_cell",
        ),
    ]


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


def test_internal_model_graph_compatibility_module_is_removed() -> None:
    with pytest.raises(ModuleNotFoundError):
        import_module("tensor_network_editor.internal.models._model_graph")


def test_validation_module_reuses_internal_spec_validation_helpers() -> None:
    validation_module = import_module("tensor_network_editor.validation")
    internal_module = import_module(
        "tensor_network_editor.internal.validation._validation_spec"
    )

    assert validation_module.validate_spec_with_analysis is (
        internal_module.validate_spec_with_analysis
    )
    assert not hasattr(validation_module, "_validate_spec_with_analysis")


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


def test_hyperedge_hub_offset_round_trip_is_serializable() -> None:
    hyperedge = HyperedgeSpec(
        id="hyperedge_with_offset",
        name="shared_h",
        endpoints=[
            EdgeEndpointRef(tensor_id="tensor_a", index_id="tensor_a_h"),
            EdgeEndpointRef(tensor_id="tensor_b", index_id="tensor_b_h"),
            EdgeEndpointRef(tensor_id="tensor_c", index_id="tensor_c_h"),
        ],
        hub_offset=CanvasPosition(x=18.0, y=-12.0),
    )

    payload = hyperedge.to_dict()
    restored = HyperedgeSpec.from_dict(cast(dict[str, object], payload))

    assert restored.hub_offset.x == 18.0
    assert restored.hub_offset.y == -12.0


def test_hyperedge_without_hub_offset_defaults_to_origin_on_load() -> None:
    restored = HyperedgeSpec.from_dict(
        {
            "id": "hyperedge_legacy",
            "name": "legacy_h",
            "endpoints": [
                {"tensor_id": "tensor_a", "index_id": "tensor_a_h"},
                {"tensor_id": "tensor_b", "index_id": "tensor_b_h"},
                {"tensor_id": "tensor_c", "index_id": "tensor_c_h"},
            ],
            "metadata": {},
        }
    )

    assert restored.hub_offset.x == 0.0
    assert restored.hub_offset.y == 0.0


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


def test_open_indices_and_connected_index_ids_include_hyperedges() -> None:
    spec = build_valid_hyperedge_spec()

    assert spec.connected_index_ids() == {"tensor_a_h", "tensor_b_h", "tensor_c_h"}
    assert [index.name for _, index in spec.open_indices()] == [
        "a_open",
        "b_open",
        "c_open",
    ]


def test_validate_spec_accepts_valid_network() -> None:
    assert validate_spec(build_valid_spec()) == []


def test_validate_spec_accepts_valid_hyperedge() -> None:
    assert validate_spec(build_valid_hyperedge_spec()) == []


def test_validate_spec_accepts_tensor_literal_data_matching_shape() -> None:
    spec = build_valid_spec()
    spec.tensors[0].tensor_data = TensorDataSpec(
        mode=TensorDataMode.LITERAL,
        values=[
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0, 9.0, 10.0],
        ],
    )

    assert validate_spec(spec) == []


def test_validate_spec_accepts_identity_and_copy_tensor_initializers() -> None:
    spec = NetworkSpec(
        id="network_initializers",
        name="initializer-demo",
        tensors=[
            TensorSpec(
                id="tensor_identity",
                name="Identity",
                position=CanvasPosition(x=40.0, y=80.0),
                indices=[
                    IndexSpec(id="identity_left", name="left", dimension=3),
                    IndexSpec(id="identity_right", name="right", dimension=3),
                ],
                tensor_data=TensorDataSpec(mode=TensorDataMode.IDENTITY),
            ),
            TensorSpec(
                id="tensor_copy",
                name="Copy",
                position=CanvasPosition(x=220.0, y=80.0),
                indices=[
                    IndexSpec(id="copy_a", name="a", dimension=3),
                    IndexSpec(id="copy_b", name="b", dimension=3),
                    IndexSpec(id="copy_c", name="c", dimension=3),
                ],
                tensor_data=TensorDataSpec(mode=TensorDataMode.COPY),
            ),
        ],
    )

    assert validate_spec(spec) == []


def test_validate_spec_accepts_external_tensor_initializer() -> None:
    spec = build_valid_spec()
    spec.tensors[0].tensor_data = TensorDataSpec(
        mode=TensorDataMode.EXTERNAL,
        file_path="data/left.npy",
    )

    assert validate_spec(spec) == []


def test_validate_spec_accepts_pt_external_tensor_initializer_without_array_key() -> (
    None
):
    spec = build_valid_spec()
    spec.tensors[0].tensor_data = TensorDataSpec(
        mode=TensorDataMode.EXTERNAL,
        file_path="data/left.pt",
    )

    assert validate_spec(spec) == []


def test_validate_spec_rejects_external_tensor_initializer_without_path() -> None:
    spec = build_valid_spec()
    spec.tensors[0].tensor_data = TensorDataSpec(
        mode=TensorDataMode.EXTERNAL,
        file_path="",
    )

    issue = find_issue(validate_spec(spec), "invalid-tensor-data")

    assert issue.path == "tensors.tensor_left.tensor_data"


def test_validate_spec_rejects_external_tensor_initializer_for_unsupported_suffix() -> (
    None
):
    spec = build_valid_spec()
    spec.tensors[0].tensor_data = TensorDataSpec(
        mode=TensorDataMode.EXTERNAL,
        file_path="data/left.csv",
    )

    issue = find_issue(validate_spec(spec), "invalid-tensor-data")

    assert issue.path == "tensors.tensor_left.tensor_data"


def test_validate_spec_rejects_npz_external_tensor_initializer_without_array_key() -> (
    None
):
    spec = build_valid_spec()
    spec.tensors[0].tensor_data = TensorDataSpec(
        mode=TensorDataMode.EXTERNAL,
        file_path="data/left.npz",
    )

    issue = find_issue(validate_spec(spec), "invalid-tensor-data")

    assert issue.path == "tensors.tensor_left.tensor_data"


def test_validate_spec_rejects_identity_tensor_initializer_for_non_square_shape() -> (
    None
):
    spec = build_valid_spec()
    spec.tensors[0].tensor_data = TensorDataSpec(mode=TensorDataMode.IDENTITY)

    issue = find_issue(validate_spec(spec), "tensor-data-shape-mismatch")

    assert issue.path == "tensors.tensor_left.tensor_data"


def test_validate_spec_rejects_copy_tensor_initializer_for_mismatched_axes() -> None:
    spec = NetworkSpec(
        id="network_bad_copy",
        name="bad-copy",
        tensors=[
            TensorSpec(
                id="tensor_bad_copy",
                name="BadCopy",
                position=CanvasPosition(x=40.0, y=80.0),
                indices=[
                    IndexSpec(id="copy_a", name="a", dimension=3),
                    IndexSpec(id="copy_b", name="b", dimension=3),
                    IndexSpec(id="copy_c", name="c", dimension=4),
                ],
                tensor_data=TensorDataSpec(mode=TensorDataMode.COPY),
            )
        ],
    )

    issue = find_issue(validate_spec(spec), "tensor-data-shape-mismatch")

    assert issue.path == "tensors.tensor_bad_copy.tensor_data"


def test_validate_spec_rejects_tensor_literal_data_shape_mismatch() -> None:
    spec = build_valid_spec()
    spec.tensors[0].tensor_data = TensorDataSpec(
        mode=TensorDataMode.LITERAL,
        values=[1.0, 2.0],
    )

    issue = find_issue(validate_spec(spec), "tensor-data-shape-mismatch")

    assert issue.path == "tensors.tensor_left.tensor_data"


def test_validate_spec_rejects_ragged_tensor_literal_data() -> None:
    spec = build_valid_spec()
    spec.tensors[0].tensor_data = TensorDataSpec(
        mode=TensorDataMode.LITERAL,
        values=[
            [1.0, 2.0, 3.0],
            [4.0, 5.0],
        ],
    )

    issue = find_issue(validate_spec(spec), "invalid-tensor-data")

    assert issue.path == "tensors.tensor_left.tensor_data"


def test_validate_spec_rejects_non_numeric_tensor_literal_values() -> None:
    spec = build_valid_spec()
    spec.tensors[0].tensor_data = TensorDataSpec(
        mode=TensorDataMode.LITERAL,
        values=cast(Any, [[1.0, True, 3.0], [4.0, 5.0, 6.0]]),
    )

    issue = find_issue(validate_spec(spec), "invalid-tensor-data")

    assert issue.path == "tensors.tensor_left.tensor_data"


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


def test_validate_spec_accepts_valid_tree_periodic_tree() -> None:
    assert validate_spec(build_tree_periodic_tree_spec()) == []


def test_validate_spec_accepts_valid_linear_periodic_carry_chain() -> None:
    assert validate_spec(build_linear_periodic_carry_chain_spec()) == []


def test_validate_spec_rejects_hyperedge_with_duplicate_endpoints() -> None:
    spec = build_valid_hyperedge_spec()
    spec.hyperedges[0] = HyperedgeSpec(
        id="hyperedge_shared",
        name="shared_h",
        endpoints=[
            EdgeEndpointRef(tensor_id="tensor_a", index_id="tensor_a_h"),
            EdgeEndpointRef(tensor_id="tensor_a", index_id="tensor_a_h"),
            EdgeEndpointRef(tensor_id="tensor_c", index_id="tensor_c_h"),
        ],
    )

    issue = find_issue(validate_spec(spec), "duplicate-hyperedge-endpoint")

    assert issue.path == "hyperedges.hyperedge_shared.endpoints"


def test_validate_spec_rejects_hyperedge_with_dimension_mismatch() -> None:
    spec = build_valid_hyperedge_spec()
    spec.tensors[2].indices[0].dimension = 9

    issue = find_issue(validate_spec(spec), "dimension-mismatch")

    assert issue.path == "hyperedges.hyperedge_shared"


def test_validate_spec_rejects_hyperedge_with_too_few_endpoints() -> None:
    spec = build_valid_hyperedge_spec()
    spec.hyperedges[0] = HyperedgeSpec(
        id="hyperedge_shared",
        name="shared_h",
        endpoints=[
            EdgeEndpointRef(tensor_id="tensor_a", index_id="tensor_a_h"),
            EdgeEndpointRef(tensor_id="tensor_b", index_id="tensor_b_h"),
        ],
    )

    issue = find_issue(validate_spec(spec), "invalid-hyperedge")

    assert issue.path == "hyperedges.hyperedge_shared.endpoints"


def test_validate_spec_rejects_index_reused_between_edge_and_hyperedge() -> None:
    spec = build_valid_hyperedge_spec()
    spec.edges.append(
        EdgeSpec(
            id="edge_duplicate",
            name="duplicate",
            left=EdgeEndpointRef(tensor_id="tensor_a", index_id="tensor_a_h"),
            right=EdgeEndpointRef(tensor_id="tensor_b", index_id="tensor_b_open"),
        )
    )

    issue = find_issue(validate_spec(spec), "index-already-connected")

    assert issue.path in {
        "edges.edge_duplicate.left",
        "hyperedges.hyperedge_shared.endpoints",
    }


@pytest.mark.parametrize(
    "spec_factory",
    [
        build_linear_periodic_chain_spec,
        build_grid_periodic_grid_spec,
        build_tree_periodic_tree_spec,
    ],
)
def test_validate_spec_rejects_hyperedges_in_for_modes(
    spec_factory: Callable[[], NetworkSpec],
) -> None:
    spec = spec_factory()
    spec.hyperedges = build_valid_hyperedge_spec().hyperedges

    issue = find_issue(validate_spec(spec), "hyperedges-not-supported-in-for-mode")

    assert issue.path == "hyperedges"


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


def test_validate_spec_rejects_mixed_tree_and_grid_periodic_modes() -> None:
    spec = build_grid_periodic_grid_spec()
    spec.tree_periodic_tree = build_tree_periodic_tree_spec().tree_periodic_tree

    issue = find_issue(validate_spec(spec), "periodic-mode-conflict")

    assert issue.path == "tree_periodic_tree"


def test_validate_spec_rejects_tree_periodic_invalid_branching_factor() -> None:
    spec = build_tree_periodic_tree_spec()
    assert spec.tree_periodic_tree is not None
    spec.tree_periodic_tree.branching_factor = 1

    issue = find_issue(validate_spec(spec), "tree-periodic-branching-factor")

    assert issue.path == "tree_periodic_tree.branching_factor"


def test_validate_spec_rejects_tree_periodic_missing_branch_child_boundary() -> None:
    spec = build_tree_periodic_tree_spec()
    assert spec.tree_periodic_tree is not None
    spec.tree_periodic_tree.branch_cell.tensors = [
        tensor
        for tensor in spec.tree_periodic_tree.branch_cell.tensors
        if not (
            tensor.tree_periodic_role is TreePeriodicTensorRole.CHILD
            and tensor.tree_periodic_child_index == 2
        )
    ]

    issue = find_issue(validate_spec(spec), "tree-periodic-boundary-role")

    assert issue.path == "tree_periodic_tree.branch_cell.child_boundary_2"


def test_validate_spec_rejects_tree_periodic_duplicate_child_boundary_index() -> None:
    spec = build_tree_periodic_tree_spec()
    assert spec.tree_periodic_tree is not None
    duplicate_boundary = next(
        tensor
        for tensor in spec.tree_periodic_tree.root_cell.tensors
        if tensor.tree_periodic_role is TreePeriodicTensorRole.CHILD
        and tensor.tree_periodic_child_index == 2
    )
    duplicate_boundary.tree_periodic_child_index = 1

    issue = find_issue(validate_spec(spec), "tree-periodic-child-index")

    assert issue.path == "tree_periodic_tree.root_cell.child_boundaries"


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


def test_validate_spec_accepts_grid_periodic_reserved_border_contraction_plan() -> None:
    spec = build_grid_periodic_grid_spec()
    assert spec.grid_periodic_grid is not None
    spec.grid_periodic_grid.center_cell.contraction_plan = ContractionPlanSpec(
        id="grid_plan",
        name="Grid plan",
        steps=[
            ContractionStepSpec(
                id="grid_step",
                left_operand_id="__grid_left__",
                right_operand_id="center_tensor",
            )
        ],
    )

    issues = validate_spec(spec)

    assert not [issue for issue in issues if issue.code.endswith("contraction-plan")]
    assert not [
        issue for issue in issues if issue.code == "invalid-contraction-operand"
    ]


def test_validate_spec_rejects_grid_periodic_unknown_border_operand() -> None:
    spec = build_grid_periodic_grid_spec()
    assert spec.grid_periodic_grid is not None
    spec.grid_periodic_grid.center_cell.contraction_plan = ContractionPlanSpec(
        id="grid_plan",
        name="Grid plan",
        steps=[
            ContractionStepSpec(
                id="grid_step",
                left_operand_id="__grid_diagonal__",
                right_operand_id="center_tensor",
            )
        ],
    )

    issue = find_issue(validate_spec(spec), "invalid-contraction-operand")

    assert (
        issue.path
        == "grid_periodic_grid.center_cell.contraction_plan.steps.grid_step.left_operand_id"
    )


def test_validate_spec_accepts_tree_periodic_reserved_border_contraction_plan() -> None:
    spec = build_tree_periodic_tree_spec()
    assert spec.tree_periodic_tree is not None
    spec.tree_periodic_tree.branch_cell.contraction_plan = ContractionPlanSpec(
        id="tree_plan",
        name="Tree plan",
        steps=[
            ContractionStepSpec(
                id="tree_step",
                left_operand_id="__tree_parent__",
                right_operand_id="branch_tensor",
            )
        ],
    )

    issues = validate_spec(spec)

    assert not [issue for issue in issues if issue.code.endswith("contraction-plan")]
    assert not [
        issue for issue in issues if issue.code == "invalid-contraction-operand"
    ]


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


def test_validate_spec_skips_json_dump_for_simple_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = build_valid_spec()
    spec.metadata = {"tags": ["mps", {"boundary": "open"}], "verified": True}
    spec.tensors[0].metadata = {"role": "site", "weight": 1}
    dump_call_count = 0

    def counting_json_dumps(value: object) -> str:
        nonlocal dump_call_count
        del value
        dump_call_count += 1
        return "{}"

    monkeypatch.setattr(
        "tensor_network_editor.internal.validation._validation_common.json.dumps",
        counting_json_dumps,
    )

    assert validate_spec(spec) == []
    assert dump_call_count == 0


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
