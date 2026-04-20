from __future__ import annotations

from tensor_network_editor.app._protocol import JsonDict
from tensor_network_editor.models import (
    CanvasNoteSpec,
    CanvasPosition,
    ContractionOperandLayoutSpec,
    ContractionPlanSpec,
    ContractionStepSpec,
    ContractionViewSnapshotSpec,
    EdgeEndpointRef,
    EdgeSpec,
    GridPeriodicCellName,
    GridPeriodicGridSpec,
    GridPeriodicTensorRole,
    GroupSpec,
    IndexSpec,
    LinearPeriodicCellName,
    LinearPeriodicCellSpec,
    LinearPeriodicChainSpec,
    LinearPeriodicTensorRole,
    NetworkSpec,
    TensorSize,
    TensorSpec,
    TreePeriodicCellName,
    TreePeriodicTensorRole,
    TreePeriodicTreeSpec,
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


def build_linear_periodic_chain_spec() -> NetworkSpec:
    initial_cell = LinearPeriodicCellSpec(
        tensors=[
            TensorSpec(
                id="initial_tensor",
                name="Initial",
                position=CanvasPosition(x=100.0, y=140.0),
                indices=[
                    IndexSpec(id="initial_phys", name="phys", dimension=2),
                    IndexSpec(id="initial_bond", name="bond", dimension=3),
                ],
            ),
            TensorSpec(
                id="initial_next_boundary",
                name="Next cell",
                position=CanvasPosition(x=320.0, y=140.0),
                linear_periodic_role=LinearPeriodicTensorRole.NEXT,
                indices=[
                    IndexSpec(id="initial_next_slot", name="slot_1", dimension=3),
                ],
            ),
        ],
        edges=[
            EdgeSpec(
                id="initial_edge_to_next",
                name="initial_to_next",
                left=EdgeEndpointRef(
                    tensor_id="initial_tensor", index_id="initial_bond"
                ),
                right=EdgeEndpointRef(
                    tensor_id="initial_next_boundary", index_id="initial_next_slot"
                ),
            )
        ],
    )
    periodic_cell = LinearPeriodicCellSpec(
        tensors=[
            TensorSpec(
                id="periodic_left_tensor",
                name="PeriodicLeft",
                position=CanvasPosition(x=140.0, y=120.0),
                indices=[
                    IndexSpec(id="periodic_left_in", name="left", dimension=3),
                    IndexSpec(id="periodic_left_phys", name="phys_l", dimension=2),
                    IndexSpec(id="periodic_left_inner", name="inner", dimension=5),
                ],
            ),
            TensorSpec(
                id="periodic_right_tensor",
                name="PeriodicRight",
                position=CanvasPosition(x=320.0, y=120.0),
                indices=[
                    IndexSpec(id="periodic_right_inner", name="inner", dimension=5),
                    IndexSpec(id="periodic_right_phys", name="phys_r", dimension=2),
                    IndexSpec(id="periodic_right_out", name="right", dimension=3),
                ],
            ),
            TensorSpec(
                id="periodic_previous_boundary",
                name="Previous cell",
                position=CanvasPosition(x=20.0, y=120.0),
                linear_periodic_role=LinearPeriodicTensorRole.PREVIOUS,
                indices=[
                    IndexSpec(id="periodic_previous_slot", name="slot_1", dimension=3),
                ],
            ),
            TensorSpec(
                id="periodic_next_boundary",
                name="Next cell",
                position=CanvasPosition(x=460.0, y=120.0),
                linear_periodic_role=LinearPeriodicTensorRole.NEXT,
                indices=[
                    IndexSpec(id="periodic_next_slot", name="slot_1", dimension=3),
                ],
            ),
        ],
        edges=[
            EdgeSpec(
                id="periodic_edge_from_previous",
                name="from_previous",
                left=EdgeEndpointRef(
                    tensor_id="periodic_previous_boundary",
                    index_id="periodic_previous_slot",
                ),
                right=EdgeEndpointRef(
                    tensor_id="periodic_left_tensor", index_id="periodic_left_in"
                ),
            ),
            EdgeSpec(
                id="periodic_edge_inner",
                name="inner",
                left=EdgeEndpointRef(
                    tensor_id="periodic_left_tensor", index_id="periodic_left_inner"
                ),
                right=EdgeEndpointRef(
                    tensor_id="periodic_right_tensor",
                    index_id="periodic_right_inner",
                ),
            ),
            EdgeSpec(
                id="periodic_edge_to_next",
                name="to_next",
                left=EdgeEndpointRef(
                    tensor_id="periodic_right_tensor", index_id="periodic_right_out"
                ),
                right=EdgeEndpointRef(
                    tensor_id="periodic_next_boundary", index_id="periodic_next_slot"
                ),
            ),
        ],
        contraction_plan=ContractionPlanSpec(
            id="periodic_plan",
            name="Periodic plan",
            steps=[
                ContractionStepSpec(
                    id="periodic_contract_internal",
                    left_operand_id="periodic_left_tensor",
                    right_operand_id="periodic_right_tensor",
                )
            ],
        ),
    )
    final_cell = LinearPeriodicCellSpec(
        tensors=[
            TensorSpec(
                id="final_tensor",
                name="Final",
                position=CanvasPosition(x=260.0, y=140.0),
                indices=[
                    IndexSpec(id="final_bond", name="bond", dimension=3),
                    IndexSpec(id="final_phys", name="phys", dimension=7),
                ],
            ),
            TensorSpec(
                id="final_previous_boundary",
                name="Previous cell",
                position=CanvasPosition(x=60.0, y=140.0),
                linear_periodic_role=LinearPeriodicTensorRole.PREVIOUS,
                indices=[
                    IndexSpec(id="final_previous_slot", name="slot_1", dimension=3),
                ],
            ),
        ],
        edges=[
            EdgeSpec(
                id="final_edge_from_previous",
                name="from_previous",
                left=EdgeEndpointRef(
                    tensor_id="final_previous_boundary",
                    index_id="final_previous_slot",
                ),
                right=EdgeEndpointRef(tensor_id="final_tensor", index_id="final_bond"),
            )
        ],
    )
    return NetworkSpec(
        id="network_linear_periodic",
        name="linear-periodic-chain",
        linear_periodic_chain=LinearPeriodicChainSpec(
            active_cell=LinearPeriodicCellName.PERIODIC,
            initial_cell=initial_cell,
            periodic_cell=periodic_cell,
            final_cell=final_cell,
        ),
    )


def build_linear_periodic_carry_chain_spec() -> NetworkSpec:
    linear_previous_operand_id = "__linear_previous__"
    linear_next_operand_id = "__linear_next__"

    initial_cell = LinearPeriodicCellSpec(
        tensors=[
            TensorSpec(
                id="initial_tensor",
                name="Initial",
                position=CanvasPosition(x=100.0, y=140.0),
                indices=[
                    IndexSpec(id="initial_phys", name="phys", dimension=2),
                    IndexSpec(id="initial_bond", name="bond", dimension=3),
                ],
            ),
            TensorSpec(
                id="initial_next_boundary",
                name="Next cell",
                position=CanvasPosition(x=320.0, y=140.0),
                linear_periodic_role=LinearPeriodicTensorRole.NEXT,
                indices=[
                    IndexSpec(id="initial_next_slot", name="slot_1", dimension=3),
                ],
            ),
        ],
        edges=[
            EdgeSpec(
                id="initial_edge_to_next",
                name="initial_to_next",
                left=EdgeEndpointRef(
                    tensor_id="initial_tensor", index_id="initial_bond"
                ),
                right=EdgeEndpointRef(
                    tensor_id="initial_next_boundary", index_id="initial_next_slot"
                ),
            )
        ],
        contraction_plan=ContractionPlanSpec(
            id="initial_plan",
            name="Initial carry plan",
            steps=[
                ContractionStepSpec(
                    id="initial_carry",
                    left_operand_id="initial_tensor",
                    right_operand_id=linear_next_operand_id,
                )
            ],
        ),
    )
    periodic_cell = LinearPeriodicCellSpec(
        tensors=[
            TensorSpec(
                id="periodic_left_tensor",
                name="PeriodicLeft",
                position=CanvasPosition(x=140.0, y=120.0),
                indices=[
                    IndexSpec(id="periodic_left_in", name="left", dimension=3),
                    IndexSpec(id="periodic_left_phys", name="phys_l", dimension=2),
                    IndexSpec(id="periodic_left_inner", name="inner", dimension=5),
                ],
            ),
            TensorSpec(
                id="periodic_right_tensor",
                name="PeriodicRight",
                position=CanvasPosition(x=320.0, y=120.0),
                indices=[
                    IndexSpec(id="periodic_right_inner", name="inner", dimension=5),
                    IndexSpec(id="periodic_right_phys", name="phys_r", dimension=2),
                    IndexSpec(id="periodic_right_out", name="right", dimension=3),
                ],
            ),
            TensorSpec(
                id="periodic_previous_boundary",
                name="Previous cell",
                position=CanvasPosition(x=20.0, y=120.0),
                linear_periodic_role=LinearPeriodicTensorRole.PREVIOUS,
                indices=[
                    IndexSpec(id="periodic_previous_slot", name="slot_1", dimension=3),
                ],
            ),
            TensorSpec(
                id="periodic_next_boundary",
                name="Next cell",
                position=CanvasPosition(x=460.0, y=120.0),
                linear_periodic_role=LinearPeriodicTensorRole.NEXT,
                indices=[
                    IndexSpec(id="periodic_next_slot", name="slot_1", dimension=3),
                ],
            ),
        ],
        edges=[
            EdgeSpec(
                id="periodic_edge_from_previous",
                name="from_previous",
                left=EdgeEndpointRef(
                    tensor_id="periodic_previous_boundary",
                    index_id="periodic_previous_slot",
                ),
                right=EdgeEndpointRef(
                    tensor_id="periodic_left_tensor", index_id="periodic_left_in"
                ),
            ),
            EdgeSpec(
                id="periodic_edge_inner",
                name="inner",
                left=EdgeEndpointRef(
                    tensor_id="periodic_left_tensor", index_id="periodic_left_inner"
                ),
                right=EdgeEndpointRef(
                    tensor_id="periodic_right_tensor",
                    index_id="periodic_right_inner",
                ),
            ),
            EdgeSpec(
                id="periodic_edge_to_next",
                name="to_next",
                left=EdgeEndpointRef(
                    tensor_id="periodic_right_tensor", index_id="periodic_right_out"
                ),
                right=EdgeEndpointRef(
                    tensor_id="periodic_next_boundary", index_id="periodic_next_slot"
                ),
            ),
        ],
        contraction_plan=ContractionPlanSpec(
            id="periodic_carry_plan",
            name="Periodic carry plan",
            steps=[
                ContractionStepSpec(
                    id="periodic_from_previous",
                    left_operand_id=linear_previous_operand_id,
                    right_operand_id="periodic_left_tensor",
                ),
                ContractionStepSpec(
                    id="periodic_contract_full",
                    left_operand_id="periodic_from_previous",
                    right_operand_id="periodic_right_tensor",
                ),
                ContractionStepSpec(
                    id="periodic_carry",
                    left_operand_id="periodic_contract_full",
                    right_operand_id=linear_next_operand_id,
                ),
            ],
        ),
    )
    final_cell = LinearPeriodicCellSpec(
        tensors=[
            TensorSpec(
                id="final_tensor",
                name="Final",
                position=CanvasPosition(x=260.0, y=140.0),
                indices=[
                    IndexSpec(id="final_bond", name="bond", dimension=3),
                    IndexSpec(id="final_phys", name="phys", dimension=7),
                ],
            ),
            TensorSpec(
                id="final_previous_boundary",
                name="Previous cell",
                position=CanvasPosition(x=60.0, y=140.0),
                linear_periodic_role=LinearPeriodicTensorRole.PREVIOUS,
                indices=[
                    IndexSpec(id="final_previous_slot", name="slot_1", dimension=3),
                ],
            ),
        ],
        edges=[
            EdgeSpec(
                id="final_edge_from_previous",
                name="from_previous",
                left=EdgeEndpointRef(
                    tensor_id="final_previous_boundary",
                    index_id="final_previous_slot",
                ),
                right=EdgeEndpointRef(tensor_id="final_tensor", index_id="final_bond"),
            )
        ],
        contraction_plan=ContractionPlanSpec(
            id="final_plan",
            name="Final carry plan",
            steps=[
                ContractionStepSpec(
                    id="final_contract",
                    left_operand_id=linear_previous_operand_id,
                    right_operand_id="final_tensor",
                )
            ],
        ),
    )
    return NetworkSpec(
        id="network_linear_periodic_carry",
        name="linear-periodic-carry-chain",
        linear_periodic_chain=LinearPeriodicChainSpec(
            active_cell=LinearPeriodicCellName.PERIODIC,
            initial_cell=initial_cell,
            periodic_cell=periodic_cell,
            final_cell=final_cell,
        ),
    )


def build_linear_periodic_partial_carry_chain_spec() -> NetworkSpec:
    linear_previous_operand_id = "__linear_previous__"
    linear_next_operand_id = "__linear_next__"

    initial_cell = LinearPeriodicCellSpec(
        tensors=[
            TensorSpec(
                id="initial_left_tensor",
                name="InitialLeft",
                position=CanvasPosition(x=80.0, y=120.0),
                indices=[
                    IndexSpec(id="initial_left_phys", name="phys_l", dimension=2),
                    IndexSpec(
                        id="initial_left_to_next",
                        name="to_next_1",
                        dimension=3,
                    ),
                ],
            ),
            TensorSpec(
                id="initial_right_tensor",
                name="InitialRight",
                position=CanvasPosition(x=260.0, y=120.0),
                indices=[
                    IndexSpec(id="initial_right_phys", name="phys_r", dimension=5),
                    IndexSpec(
                        id="initial_right_to_next",
                        name="to_next_2",
                        dimension=7,
                    ),
                ],
            ),
            TensorSpec(
                id="initial_next_boundary",
                name="Next cell",
                position=CanvasPosition(x=440.0, y=120.0),
                linear_periodic_role=LinearPeriodicTensorRole.NEXT,
                indices=[
                    IndexSpec(id="initial_next_slot_1", name="slot_1", dimension=3),
                    IndexSpec(id="initial_next_slot_2", name="slot_2", dimension=7),
                ],
            ),
        ],
        edges=[
            EdgeSpec(
                id="initial_edge_to_next_1",
                name="initial_to_next_1",
                left=EdgeEndpointRef(
                    tensor_id="initial_left_tensor",
                    index_id="initial_left_to_next",
                ),
                right=EdgeEndpointRef(
                    tensor_id="initial_next_boundary",
                    index_id="initial_next_slot_1",
                ),
            ),
            EdgeSpec(
                id="initial_edge_to_next_2",
                name="initial_to_next_2",
                left=EdgeEndpointRef(
                    tensor_id="initial_right_tensor",
                    index_id="initial_right_to_next",
                ),
                right=EdgeEndpointRef(
                    tensor_id="initial_next_boundary",
                    index_id="initial_next_slot_2",
                ),
            ),
        ],
        contraction_plan=ContractionPlanSpec(
            id="initial_partial_plan",
            name="Initial partial carry plan",
            steps=[
                ContractionStepSpec(
                    id="initial_partial_carry",
                    left_operand_id="initial_right_tensor",
                    right_operand_id=linear_next_operand_id,
                )
            ],
        ),
    )
    periodic_cell = LinearPeriodicCellSpec(
        tensors=[
            TensorSpec(
                id="periodic_previous_left_tensor",
                name="PeriodicPreviousLeft",
                position=CanvasPosition(x=120.0, y=80.0),
                indices=[
                    IndexSpec(
                        id="periodic_previous_left_input",
                        name="from_previous_1",
                        dimension=3,
                    ),
                    IndexSpec(
                        id="periodic_previous_left_phys",
                        name="phys_prev_1",
                        dimension=11,
                    ),
                ],
            ),
            TensorSpec(
                id="periodic_previous_right_tensor",
                name="PeriodicPreviousRight",
                position=CanvasPosition(x=120.0, y=220.0),
                indices=[
                    IndexSpec(
                        id="periodic_previous_right_input",
                        name="from_previous_2",
                        dimension=7,
                    ),
                    IndexSpec(
                        id="periodic_previous_right_phys",
                        name="phys_prev_2",
                        dimension=13,
                    ),
                ],
            ),
            TensorSpec(
                id="periodic_next_left_tensor",
                name="PeriodicNextLeft",
                position=CanvasPosition(x=340.0, y=80.0),
                indices=[
                    IndexSpec(
                        id="periodic_next_left_output",
                        name="to_next_1",
                        dimension=3,
                    ),
                    IndexSpec(
                        id="periodic_next_left_phys",
                        name="phys_next_1",
                        dimension=19,
                    ),
                ],
            ),
            TensorSpec(
                id="periodic_next_right_tensor",
                name="PeriodicNextRight",
                position=CanvasPosition(x=340.0, y=220.0),
                indices=[
                    IndexSpec(
                        id="periodic_next_right_output",
                        name="to_next_2",
                        dimension=7,
                    ),
                    IndexSpec(
                        id="periodic_next_right_phys",
                        name="phys_next_2",
                        dimension=29,
                    ),
                ],
            ),
            TensorSpec(
                id="periodic_previous_boundary",
                name="Previous cell",
                position=CanvasPosition(x=-40.0, y=150.0),
                linear_periodic_role=LinearPeriodicTensorRole.PREVIOUS,
                indices=[
                    IndexSpec(
                        id="periodic_previous_slot_1", name="slot_1", dimension=3
                    ),
                    IndexSpec(
                        id="periodic_previous_slot_2", name="slot_2", dimension=7
                    ),
                ],
            ),
            TensorSpec(
                id="periodic_next_boundary",
                name="Next cell",
                position=CanvasPosition(x=520.0, y=150.0),
                linear_periodic_role=LinearPeriodicTensorRole.NEXT,
                indices=[
                    IndexSpec(id="periodic_next_slot_1", name="slot_1", dimension=3),
                    IndexSpec(id="periodic_next_slot_2", name="slot_2", dimension=7),
                ],
            ),
        ],
        edges=[
            EdgeSpec(
                id="periodic_edge_from_previous_1",
                name="from_previous_1",
                left=EdgeEndpointRef(
                    tensor_id="periodic_previous_boundary",
                    index_id="periodic_previous_slot_1",
                ),
                right=EdgeEndpointRef(
                    tensor_id="periodic_previous_left_tensor",
                    index_id="periodic_previous_left_input",
                ),
            ),
            EdgeSpec(
                id="periodic_edge_from_previous_2",
                name="from_previous_2",
                left=EdgeEndpointRef(
                    tensor_id="periodic_previous_boundary",
                    index_id="periodic_previous_slot_2",
                ),
                right=EdgeEndpointRef(
                    tensor_id="periodic_previous_right_tensor",
                    index_id="periodic_previous_right_input",
                ),
            ),
            EdgeSpec(
                id="periodic_edge_to_next_1",
                name="to_next_1",
                left=EdgeEndpointRef(
                    tensor_id="periodic_next_left_tensor",
                    index_id="periodic_next_left_output",
                ),
                right=EdgeEndpointRef(
                    tensor_id="periodic_next_boundary",
                    index_id="periodic_next_slot_1",
                ),
            ),
            EdgeSpec(
                id="periodic_edge_to_next_2",
                name="to_next_2",
                left=EdgeEndpointRef(
                    tensor_id="periodic_next_right_tensor",
                    index_id="periodic_next_right_output",
                ),
                right=EdgeEndpointRef(
                    tensor_id="periodic_next_boundary",
                    index_id="periodic_next_slot_2",
                ),
            ),
        ],
        contraction_plan=ContractionPlanSpec(
            id="periodic_partial_plan",
            name="Periodic partial carry plan",
            steps=[
                ContractionStepSpec(
                    id="periodic_from_previous_partial",
                    left_operand_id=linear_previous_operand_id,
                    right_operand_id="periodic_previous_left_tensor",
                ),
                ContractionStepSpec(
                    id="periodic_partial_carry",
                    left_operand_id="periodic_next_left_tensor",
                    right_operand_id=linear_next_operand_id,
                ),
            ],
        ),
    )
    final_cell = LinearPeriodicCellSpec(
        tensors=[
            TensorSpec(
                id="final_previous_left_tensor",
                name="FinalPreviousLeft",
                position=CanvasPosition(x=160.0, y=80.0),
                indices=[
                    IndexSpec(
                        id="final_previous_left_input",
                        name="from_previous_1",
                        dimension=3,
                    ),
                    IndexSpec(
                        id="final_previous_left_phys",
                        name="phys_final_1",
                        dimension=31,
                    ),
                ],
            ),
            TensorSpec(
                id="final_previous_right_tensor",
                name="FinalPreviousRight",
                position=CanvasPosition(x=160.0, y=220.0),
                indices=[
                    IndexSpec(
                        id="final_previous_right_input",
                        name="from_previous_2",
                        dimension=7,
                    ),
                    IndexSpec(
                        id="final_previous_right_phys",
                        name="phys_final_2",
                        dimension=37,
                    ),
                ],
            ),
            TensorSpec(
                id="final_previous_boundary",
                name="Previous cell",
                position=CanvasPosition(x=-40.0, y=150.0),
                linear_periodic_role=LinearPeriodicTensorRole.PREVIOUS,
                indices=[
                    IndexSpec(id="final_previous_slot_1", name="slot_1", dimension=3),
                    IndexSpec(id="final_previous_slot_2", name="slot_2", dimension=7),
                ],
            ),
        ],
        edges=[
            EdgeSpec(
                id="final_edge_from_previous_1",
                name="from_previous_1",
                left=EdgeEndpointRef(
                    tensor_id="final_previous_boundary",
                    index_id="final_previous_slot_1",
                ),
                right=EdgeEndpointRef(
                    tensor_id="final_previous_left_tensor",
                    index_id="final_previous_left_input",
                ),
            ),
            EdgeSpec(
                id="final_edge_from_previous_2",
                name="from_previous_2",
                left=EdgeEndpointRef(
                    tensor_id="final_previous_boundary",
                    index_id="final_previous_slot_2",
                ),
                right=EdgeEndpointRef(
                    tensor_id="final_previous_right_tensor",
                    index_id="final_previous_right_input",
                ),
            ),
        ],
        contraction_plan=ContractionPlanSpec(
            id="final_partial_plan",
            name="Final partial carry plan",
            steps=[
                ContractionStepSpec(
                    id="final_from_previous_partial",
                    left_operand_id=linear_previous_operand_id,
                    right_operand_id="final_previous_left_tensor",
                )
            ],
        ),
    )
    return NetworkSpec(
        id="network_linear_periodic_partial_carry",
        name="linear-periodic-partial-carry-chain",
        linear_periodic_chain=LinearPeriodicChainSpec(
            active_cell=LinearPeriodicCellName.PERIODIC,
            initial_cell=initial_cell,
            periodic_cell=periodic_cell,
            final_cell=final_cell,
        ),
    )


def build_grid_periodic_grid_spec() -> NetworkSpec:
    def build_grid_cell(
        *,
        cell_prefix: str,
        boundary_roles: tuple[GridPeriodicTensorRole, ...],
        boundary_dimensions: dict[GridPeriodicTensorRole, int],
    ) -> LinearPeriodicCellSpec:
        boundary_index_specs = [
            IndexSpec(
                id=f"{cell_prefix}_{role.value}_slot",
                name=f"{role.value}_slot",
                dimension=boundary_dimensions[role],
            )
            for role in boundary_roles
        ]
        tensor_indices = [
            IndexSpec(
                id=f"{cell_prefix}_{role.value}_bond",
                name=f"{role.value}_bond",
                dimension=boundary_dimensions[role],
            )
            for role in boundary_roles
        ] + [
            IndexSpec(
                id=f"{cell_prefix}_phys",
                name="phys",
                dimension=2,
            )
        ]
        cell_tensors = [
            TensorSpec(
                id=f"{cell_prefix}_tensor",
                name=cell_prefix.title().replace("_", " "),
                position=CanvasPosition(x=220.0, y=160.0),
                indices=tensor_indices,
            ),
            *[
                TensorSpec(
                    id=f"{cell_prefix}_{role.value}_boundary",
                    name=f"{role.value.title()} boundary",
                    position=CanvasPosition(x=0.0, y=0.0),
                    grid_periodic_role=role,
                    indices=[boundary_index_spec],
                )
                for role, boundary_index_spec in zip(
                    boundary_roles,
                    boundary_index_specs,
                    strict=True,
                )
            ],
        ]
        cell_edges = [
            EdgeSpec(
                id=f"{cell_prefix}_{role.value}_edge",
                name=f"{role.value}_edge",
                left=EdgeEndpointRef(
                    tensor_id=f"{cell_prefix}_tensor",
                    index_id=f"{cell_prefix}_{role.value}_bond",
                ),
                right=EdgeEndpointRef(
                    tensor_id=f"{cell_prefix}_{role.value}_boundary",
                    index_id=f"{cell_prefix}_{role.value}_slot",
                ),
            )
            for role in boundary_roles
        ]
        return LinearPeriodicCellSpec(tensors=cell_tensors, edges=cell_edges)

    top_left_cell = build_grid_cell(
        cell_prefix="top_left",
        boundary_roles=(GridPeriodicTensorRole.RIGHT, GridPeriodicTensorRole.DOWN),
        boundary_dimensions={
            GridPeriodicTensorRole.RIGHT: 2,
            GridPeriodicTensorRole.DOWN: 2,
        },
    )
    top_cell = build_grid_cell(
        cell_prefix="top",
        boundary_roles=(
            GridPeriodicTensorRole.LEFT,
            GridPeriodicTensorRole.RIGHT,
            GridPeriodicTensorRole.DOWN,
        ),
        boundary_dimensions={
            GridPeriodicTensorRole.LEFT: 2,
            GridPeriodicTensorRole.RIGHT: 2,
            GridPeriodicTensorRole.DOWN: 2,
        },
    )
    top_right_cell = build_grid_cell(
        cell_prefix="top_right",
        boundary_roles=(GridPeriodicTensorRole.LEFT, GridPeriodicTensorRole.DOWN),
        boundary_dimensions={
            GridPeriodicTensorRole.LEFT: 2,
            GridPeriodicTensorRole.DOWN: 2,
        },
    )
    left_cell = build_grid_cell(
        cell_prefix="left",
        boundary_roles=(
            GridPeriodicTensorRole.UP,
            GridPeriodicTensorRole.RIGHT,
            GridPeriodicTensorRole.DOWN,
        ),
        boundary_dimensions={
            GridPeriodicTensorRole.UP: 2,
            GridPeriodicTensorRole.RIGHT: 2,
            GridPeriodicTensorRole.DOWN: 2,
        },
    )
    center_cell = build_grid_cell(
        cell_prefix="center",
        boundary_roles=(
            GridPeriodicTensorRole.UP,
            GridPeriodicTensorRole.RIGHT,
            GridPeriodicTensorRole.DOWN,
            GridPeriodicTensorRole.LEFT,
        ),
        boundary_dimensions={
            GridPeriodicTensorRole.UP: 2,
            GridPeriodicTensorRole.RIGHT: 2,
            GridPeriodicTensorRole.DOWN: 2,
            GridPeriodicTensorRole.LEFT: 2,
        },
    )
    right_cell = build_grid_cell(
        cell_prefix="right",
        boundary_roles=(
            GridPeriodicTensorRole.UP,
            GridPeriodicTensorRole.DOWN,
            GridPeriodicTensorRole.LEFT,
        ),
        boundary_dimensions={
            GridPeriodicTensorRole.UP: 2,
            GridPeriodicTensorRole.DOWN: 2,
            GridPeriodicTensorRole.LEFT: 2,
        },
    )
    bottom_left_cell = build_grid_cell(
        cell_prefix="bottom_left",
        boundary_roles=(GridPeriodicTensorRole.UP, GridPeriodicTensorRole.RIGHT),
        boundary_dimensions={
            GridPeriodicTensorRole.UP: 2,
            GridPeriodicTensorRole.RIGHT: 2,
        },
    )
    bottom_cell = build_grid_cell(
        cell_prefix="bottom",
        boundary_roles=(
            GridPeriodicTensorRole.UP,
            GridPeriodicTensorRole.LEFT,
            GridPeriodicTensorRole.RIGHT,
        ),
        boundary_dimensions={
            GridPeriodicTensorRole.UP: 2,
            GridPeriodicTensorRole.LEFT: 2,
            GridPeriodicTensorRole.RIGHT: 2,
        },
    )
    bottom_right_cell = build_grid_cell(
        cell_prefix="bottom_right",
        boundary_roles=(GridPeriodicTensorRole.UP, GridPeriodicTensorRole.LEFT),
        boundary_dimensions={
            GridPeriodicTensorRole.UP: 2,
            GridPeriodicTensorRole.LEFT: 2,
        },
    )
    return NetworkSpec(
        id="network_grid_periodic",
        name="grid-periodic-grid",
        grid_periodic_grid=GridPeriodicGridSpec(
            active_cell=GridPeriodicCellName.CENTER,
            top_left_cell=top_left_cell,
            top_cell=top_cell,
            top_right_cell=top_right_cell,
            left_cell=left_cell,
            center_cell=center_cell,
            right_cell=right_cell,
            bottom_left_cell=bottom_left_cell,
            bottom_cell=bottom_cell,
            bottom_right_cell=bottom_right_cell,
        ),
    )


def build_tree_periodic_tree_spec() -> NetworkSpec:
    def build_tree_child_boundary_tensor(
        *,
        cell_prefix: str,
        child_index: int,
        x: float,
        y: float,
    ) -> TensorSpec:
        return TensorSpec(
            id=f"{cell_prefix}_child_boundary_{child_index}",
            name=f"Child {child_index}",
            position=CanvasPosition(x=x, y=y),
            tree_periodic_role=TreePeriodicTensorRole.CHILD,
            tree_periodic_child_index=child_index,
            indices=[
                IndexSpec(
                    id=f"{cell_prefix}_child_slot_{child_index}",
                    name=f"slot_{child_index}",
                    dimension=2,
                )
            ],
        )

    root_cell = LinearPeriodicCellSpec(
        tensors=[
            TensorSpec(
                id="root_tensor",
                name="Root",
                position=CanvasPosition(x=220.0, y=120.0),
                indices=[
                    IndexSpec(id="root_child_0", name="child_0", dimension=2),
                    IndexSpec(id="root_child_1", name="child_1", dimension=2),
                    IndexSpec(id="root_child_2", name="child_2", dimension=2),
                ],
            ),
            build_tree_child_boundary_tensor(
                cell_prefix="root", child_index=0, x=120.0, y=320.0
            ),
            build_tree_child_boundary_tensor(
                cell_prefix="root", child_index=1, x=220.0, y=320.0
            ),
            build_tree_child_boundary_tensor(
                cell_prefix="root", child_index=2, x=320.0, y=320.0
            ),
        ],
        edges=[
            EdgeSpec(
                id="root_edge_0",
                name="root_edge_0",
                left=EdgeEndpointRef(
                    tensor_id="root_tensor",
                    index_id="root_child_0",
                ),
                right=EdgeEndpointRef(
                    tensor_id="root_child_boundary_0",
                    index_id="root_child_slot_0",
                ),
            ),
            EdgeSpec(
                id="root_edge_1",
                name="root_edge_1",
                left=EdgeEndpointRef(
                    tensor_id="root_tensor",
                    index_id="root_child_1",
                ),
                right=EdgeEndpointRef(
                    tensor_id="root_child_boundary_1",
                    index_id="root_child_slot_1",
                ),
            ),
            EdgeSpec(
                id="root_edge_2",
                name="root_edge_2",
                left=EdgeEndpointRef(
                    tensor_id="root_tensor",
                    index_id="root_child_2",
                ),
                right=EdgeEndpointRef(
                    tensor_id="root_child_boundary_2",
                    index_id="root_child_slot_2",
                ),
            ),
        ],
    )
    branch_cell = LinearPeriodicCellSpec(
        tensors=[
            TensorSpec(
                id="branch_tensor",
                name="Branch",
                position=CanvasPosition(x=220.0, y=220.0),
                indices=[
                    IndexSpec(id="branch_parent", name="parent", dimension=2),
                    IndexSpec(id="branch_child_0", name="child_0", dimension=2),
                    IndexSpec(id="branch_child_1", name="child_1", dimension=2),
                    IndexSpec(id="branch_child_2", name="child_2", dimension=2),
                ],
            ),
            TensorSpec(
                id="branch_parent_boundary",
                name="Parent",
                position=CanvasPosition(x=220.0, y=40.0),
                tree_periodic_role=TreePeriodicTensorRole.PARENT,
                indices=[
                    IndexSpec(id="branch_parent_slot", name="parent_slot", dimension=2)
                ],
            ),
            build_tree_child_boundary_tensor(
                cell_prefix="branch", child_index=0, x=120.0, y=400.0
            ),
            build_tree_child_boundary_tensor(
                cell_prefix="branch", child_index=1, x=220.0, y=400.0
            ),
            build_tree_child_boundary_tensor(
                cell_prefix="branch", child_index=2, x=320.0, y=400.0
            ),
        ],
        edges=[
            EdgeSpec(
                id="branch_edge_parent",
                name="branch_edge_parent",
                left=EdgeEndpointRef(
                    tensor_id="branch_parent_boundary",
                    index_id="branch_parent_slot",
                ),
                right=EdgeEndpointRef(
                    tensor_id="branch_tensor",
                    index_id="branch_parent",
                ),
            ),
            EdgeSpec(
                id="branch_edge_0",
                name="branch_edge_0",
                left=EdgeEndpointRef(
                    tensor_id="branch_tensor",
                    index_id="branch_child_0",
                ),
                right=EdgeEndpointRef(
                    tensor_id="branch_child_boundary_0",
                    index_id="branch_child_slot_0",
                ),
            ),
            EdgeSpec(
                id="branch_edge_1",
                name="branch_edge_1",
                left=EdgeEndpointRef(
                    tensor_id="branch_tensor",
                    index_id="branch_child_1",
                ),
                right=EdgeEndpointRef(
                    tensor_id="branch_child_boundary_1",
                    index_id="branch_child_slot_1",
                ),
            ),
            EdgeSpec(
                id="branch_edge_2",
                name="branch_edge_2",
                left=EdgeEndpointRef(
                    tensor_id="branch_tensor",
                    index_id="branch_child_2",
                ),
                right=EdgeEndpointRef(
                    tensor_id="branch_child_boundary_2",
                    index_id="branch_child_slot_2",
                ),
            ),
        ],
    )
    leaf_cell = LinearPeriodicCellSpec(
        tensors=[
            TensorSpec(
                id="leaf_tensor",
                name="Leaf",
                position=CanvasPosition(x=220.0, y=220.0),
                indices=[
                    IndexSpec(id="leaf_parent", name="parent", dimension=2),
                    IndexSpec(id="leaf_phys", name="phys", dimension=3),
                ],
            ),
            TensorSpec(
                id="leaf_parent_boundary",
                name="Parent",
                position=CanvasPosition(x=220.0, y=40.0),
                tree_periodic_role=TreePeriodicTensorRole.PARENT,
                indices=[
                    IndexSpec(id="leaf_parent_slot", name="parent_slot", dimension=2)
                ],
            ),
        ],
        edges=[
            EdgeSpec(
                id="leaf_edge_parent",
                name="leaf_edge_parent",
                left=EdgeEndpointRef(
                    tensor_id="leaf_parent_boundary",
                    index_id="leaf_parent_slot",
                ),
                right=EdgeEndpointRef(
                    tensor_id="leaf_tensor",
                    index_id="leaf_parent",
                ),
            )
        ],
    )
    return NetworkSpec(
        id="network_tree_periodic",
        name="tree-periodic-tree",
        tree_periodic_tree=TreePeriodicTreeSpec(
            active_cell=TreePeriodicCellName.BRANCH,
            branching_factor=3,
            root_cell=root_cell,
            branch_cell=branch_cell,
            leaf_cell=leaf_cell,
        ),
    )


def serialize_spec_payload(spec: NetworkSpec) -> JsonDict:
    return serialize_spec(spec)
