from __future__ import annotations

from typing import cast

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
    NetworkSpec,
    TensorSize,
    TensorSpec,
)
from tensor_network_editor.serialization import serialize_spec


def build_sample_spec_without_plan() -> NetworkSpec:
    spec = build_sample_spec()
    spec.contraction_plan = None
    return spec


def build_sample_spec() -> NetworkSpec:
    return NetworkSpec(
        id="network_demo",
        name="demo",
        tensors=[
            TensorSpec(
                id="tensor_a",
                name="A",
                position=CanvasPosition(x=120.0, y=160.0),
                size=TensorSize(width=200.0, height=120.0),
                indices=[
                    IndexSpec(id="tensor_a_i", name="i", dimension=2),
                    IndexSpec(id="tensor_a_x", name="x", dimension=3),
                ],
            ),
            TensorSpec(
                id="tensor_b",
                name="B",
                position=CanvasPosition(x=360.0, y=160.0),
                indices=[
                    IndexSpec(id="tensor_b_x", name="x", dimension=3),
                    IndexSpec(id="tensor_b_j", name="j", dimension=4),
                ],
            ),
        ],
        groups=[
            GroupSpec(
                id="group_demo",
                name="Demo Group",
                tensor_ids=["tensor_a", "tensor_b"],
            )
        ],
        edges=[
            EdgeSpec(
                id="edge_x",
                name="bond_x",
                left=EdgeEndpointRef(tensor_id="tensor_a", index_id="tensor_a_x"),
                right=EdgeEndpointRef(tensor_id="tensor_b", index_id="tensor_b_x"),
            )
        ],
        notes=[
            CanvasNoteSpec(
                id="note_demo",
                text="Check the contraction order",
                position=CanvasPosition(x=80.0, y=60.0),
            )
        ],
        contraction_plan=ContractionPlanSpec(
            id="plan_demo",
            name="Manual path",
            steps=[
                ContractionStepSpec(
                    id="step_contract_ab",
                    left_operand_id="tensor_a",
                    right_operand_id="tensor_b",
                )
            ],
        ),
    )


def build_sample_spec_with_view_snapshots() -> NetworkSpec:
    spec = build_sample_spec()
    assert spec.contraction_plan is not None
    spec.contraction_plan.view_snapshots = [
        ContractionViewSnapshotSpec(
            applied_step_count=0,
            operand_layouts=[
                ContractionOperandLayoutSpec(
                    operand_id="tensor_a",
                    position=CanvasPosition(x=120.0, y=160.0),
                    size=TensorSize(width=200.0, height=120.0),
                ),
                ContractionOperandLayoutSpec(
                    operand_id="tensor_b",
                    position=CanvasPosition(x=360.0, y=160.0),
                    size=TensorSize(width=180.0, height=120.0),
                ),
            ],
        ),
        ContractionViewSnapshotSpec(
            applied_step_count=1,
            operand_layouts=[
                ContractionOperandLayoutSpec(
                    operand_id="step_contract_ab",
                    position=CanvasPosition(x=180.0, y=200.0),
                    size=TensorSize(width=230.0, height=140.0),
                )
            ],
        ),
    ]
    return spec


def build_three_tensor_spec_without_plan() -> NetworkSpec:
    spec = build_three_tensor_spec()
    spec.contraction_plan = None
    return spec


def build_three_tensor_spec() -> NetworkSpec:
    return NetworkSpec(
        id="network_chain",
        name="chain",
        tensors=[
            TensorSpec(
                id="tensor_a",
                name="A",
                position=CanvasPosition(x=80.0, y=120.0),
                indices=[
                    IndexSpec(id="tensor_a_i", name="i", dimension=2),
                    IndexSpec(id="tensor_a_x", name="x", dimension=3),
                ],
            ),
            TensorSpec(
                id="tensor_b",
                name="B",
                position=CanvasPosition(x=240.0, y=120.0),
                indices=[
                    IndexSpec(id="tensor_b_x", name="x", dimension=3),
                    IndexSpec(id="tensor_b_y", name="y", dimension=5),
                ],
            ),
            TensorSpec(
                id="tensor_c",
                name="C",
                position=CanvasPosition(x=400.0, y=120.0),
                indices=[
                    IndexSpec(id="tensor_c_y", name="y", dimension=5),
                    IndexSpec(id="tensor_c_j", name="j", dimension=7),
                ],
            ),
        ],
        edges=[
            EdgeSpec(
                id="edge_x",
                name="bond_x",
                left=EdgeEndpointRef(tensor_id="tensor_a", index_id="tensor_a_x"),
                right=EdgeEndpointRef(tensor_id="tensor_b", index_id="tensor_b_x"),
            ),
            EdgeSpec(
                id="edge_y",
                name="bond_y",
                left=EdgeEndpointRef(tensor_id="tensor_b", index_id="tensor_b_y"),
                right=EdgeEndpointRef(tensor_id="tensor_c", index_id="tensor_c_y"),
            ),
        ],
        contraction_plan=ContractionPlanSpec(
            id="plan_chain",
            name="Chain path",
            steps=[
                ContractionStepSpec(
                    id="step_ab",
                    left_operand_id="tensor_a",
                    right_operand_id="tensor_b",
                )
            ],
        ),
    )


def build_three_tensor_complete_plan_spec() -> NetworkSpec:
    spec = build_three_tensor_spec()
    spec.contraction_plan = ContractionPlanSpec(
        id="plan_chain_complete",
        name="Complete chain path",
        steps=[
            ContractionStepSpec(
                id="step_ab",
                left_operand_id="tensor_a",
                right_operand_id="tensor_b",
            ),
            ContractionStepSpec(
                id="step_abc",
                left_operand_id="step_ab",
                right_operand_id="tensor_c",
            ),
        ],
    )
    return spec


def build_outer_product_plan_spec() -> NetworkSpec:
    return NetworkSpec(
        id="network_outer_product",
        name="outer-product",
        tensors=[
            TensorSpec(
                id="tensor_a",
                name="A",
                position=CanvasPosition(x=120.0, y=120.0),
                indices=[
                    IndexSpec(id="tensor_a_i", name="i", dimension=2),
                    IndexSpec(id="tensor_a_x", name="x", dimension=3),
                ],
            ),
            TensorSpec(
                id="tensor_b",
                name="B",
                position=CanvasPosition(x=360.0, y=120.0),
                indices=[
                    IndexSpec(id="tensor_b_y", name="y", dimension=5),
                    IndexSpec(id="tensor_b_j", name="j", dimension=7),
                ],
            ),
        ],
        contraction_plan=ContractionPlanSpec(
            id="plan_outer_product",
            name="Outer product path",
            steps=[
                ContractionStepSpec(
                    id="step_outer",
                    left_operand_id="tensor_a",
                    right_operand_id="tensor_b",
                )
            ],
        ),
    )


def serialize_spec_payload(spec: NetworkSpec) -> dict[str, object]:
    return cast(dict[str, object], serialize_spec(spec))
