from __future__ import annotations

import unittest
from typing import cast

from tensor_network_editor.models import (
    CanvasNoteSpec,
    CanvasPosition,
    ContractionPlanSpec,
    ContractionStepSpec,
    EdgeEndpointRef,
    EdgeSpec,
    GroupSpec,
    IndexSpec,
    NetworkSpec,
    TensorSize,
    TensorSpec,
)
from tensor_network_editor.validation import ensure_valid_spec, validate_spec


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


class ModelAndValidationTests(unittest.TestCase):
    def test_canvas_note_round_trip_is_serializable(self) -> None:
        note = CanvasNoteSpec(
            id="note_canvas",
            text="Review this subnet",
            position=CanvasPosition(x=12.0, y=-4.0),
        )

        payload = note.to_dict()
        restored = CanvasNoteSpec.from_dict(cast(dict[str, object], payload))

        self.assertEqual(restored.text, "Review this subnet")
        self.assertEqual(restored.position.x, 12.0)
        self.assertEqual(restored.position.y, -4.0)

    def test_contraction_plan_round_trip_is_serializable(self) -> None:
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

        self.assertEqual(restored.name, "Manual path")
        self.assertEqual(restored.steps[0].id, "step_one")
        self.assertEqual(restored.steps[0].left_operand_id, "tensor_left")

    def test_index_offset_round_trip_is_serializable(self) -> None:
        index = IndexSpec(
            id="index_with_offset",
            name="offset_index",
            dimension=3,
            offset=CanvasPosition(x=34.0, y=-18.0),
        )

        payload = index.to_dict()
        restored = IndexSpec.from_dict(cast(dict[str, object], payload))

        self.assertEqual(restored.offset.x, 34.0)
        self.assertEqual(restored.offset.y, -18.0)

    def test_tensor_size_round_trip_is_serializable(self) -> None:
        tensor = TensorSpec(
            id="tensor_with_size",
            name="Sized",
            size=TensorSize(width=212.0, height=132.0),
        )

        payload = tensor.to_dict()
        restored = TensorSpec.from_dict(cast(dict[str, object], payload))

        self.assertEqual(restored.size.width, 212.0)
        self.assertEqual(restored.size.height, 132.0)

    def test_tensor_shape_uses_index_order(self) -> None:
        spec = build_valid_spec()

        self.assertEqual(spec.tensors[0].shape, (2, 5))
        self.assertEqual(spec.tensors[1].shape, (5, 7))

    def test_open_indices_are_derived_from_unconnected_ports(self) -> None:
        spec = build_valid_spec()

        open_indices = [index.name for _, index in spec.open_indices()]

        self.assertEqual(open_indices, ["left_open", "right_open"])

    def test_validate_spec_accepts_valid_network(self) -> None:
        issues = validate_spec(build_valid_spec())

        self.assertEqual(issues, [])

    def test_validate_spec_accepts_valid_network_with_notes_and_plan(self) -> None:
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
        )

        issues = validate_spec(spec)

        self.assertEqual(issues, [])

    def test_validate_spec_rejects_duplicate_index_connection(self) -> None:
        spec = build_valid_spec()
        spec.edges.append(
            EdgeSpec(
                id="edge_duplicate",
                name="duplicate",
                left=EdgeEndpointRef(
                    tensor_id="tensor_left", index_id="tensor_left_bond"
                ),
                right=EdgeEndpointRef(
                    tensor_id="tensor_right", index_id="tensor_right_open"
                ),
            )
        )

        issues = validate_spec(spec)

        self.assertIn("index-already-connected", [issue.code for issue in issues])

    def test_validate_spec_rejects_dimension_mismatch(self) -> None:
        spec = build_valid_spec()
        spec.tensors[1].indices[0] = IndexSpec(
            id="tensor_right_bond",
            name="shared",
            dimension=9,
        )

        issues = validate_spec(spec)

        self.assertIn("dimension-mismatch", [issue.code for issue in issues])

    def test_validate_spec_rejects_duplicate_index_name_within_tensor(self) -> None:
        spec = build_valid_spec()
        spec.tensors[0].indices[1] = IndexSpec(
            id="tensor_left_bond",
            name="left_open",
            dimension=5,
        )

        issues = validate_spec(spec)

        self.assertIn("duplicate-index-name", [issue.code for issue in issues])

    def test_validate_spec_rejects_non_positive_tensor_size(self) -> None:
        spec = build_valid_spec()
        spec.tensors[0].size = TensorSize(width=0.0, height=118.0)

        issues = validate_spec(spec)

        self.assertIn("invalid-size", [issue.code for issue in issues])

    def test_validate_spec_rejects_groups_with_missing_tensor_ids(self) -> None:
        spec = build_valid_spec()
        spec.groups[0] = GroupSpec(
            id="group_pair",
            name="Pair",
            tensor_ids=["tensor_left", "tensor_missing"],
        )

        issues = validate_spec(spec)

        self.assertIn("missing-group-tensor", [issue.code for issue in issues])

    def test_validate_spec_rejects_empty_note_text(self) -> None:
        spec = build_valid_spec()
        spec.notes = [
            CanvasNoteSpec(
                id="note_empty",
                text="   ",
                position=CanvasPosition(x=3.0, y=7.0),
            )
        ]

        issues = validate_spec(spec)

        self.assertIn("invalid-note-text", [issue.code for issue in issues])

    def test_validate_spec_rejects_reused_contraction_operand(self) -> None:
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

        issues = validate_spec(spec)

        self.assertIn("contraction-operand-reused", [issue.code for issue in issues])

    def test_validate_spec_accepts_multi_step_contraction_plan(self) -> None:
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

        issues = validate_spec(spec)

        self.assertEqual(issues, [])

    def test_ensure_valid_spec_raises_clear_error(self) -> None:
        spec = build_valid_spec()
        spec.tensors[0].indices[0] = IndexSpec(
            id="tensor_left_open", name="", dimension=2
        )

        with self.assertRaisesRegex(Exception, "invalid"):
            ensure_valid_spec(spec)


if __name__ == "__main__":
    unittest.main()
