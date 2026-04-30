from __future__ import annotations

import pytest

from tensor_network_editor.errors import CodeGenerationError
from tensor_network_editor.models import (
    CanvasPosition,
    ContractionPlanSpec,
    ContractionStepSpec,
    EdgeEndpointRef,
    EdgeSpec,
    EngineName,
    IndexSpec,
    LinearPeriodicCellName,
    LinearPeriodicCellSpec,
    LinearPeriodicTensorRole,
    TensorSpec,
)
from tests.factories import build_linear_periodic_carry_chain_spec


def test_linear_periodic_expression_helpers_live_in_internal_module() -> None:
    from tensor_network_editor.codegen.shared._linear_periodic_expressions import (
        _axis_name_for_engine,
        _axis_names_for_engine,
    )

    assert _axis_names_for_engine(EngineName.TENSORNETWORK, ("bond", "bond")) == (
        "bond_0",
        "bond_1",
    )
    assert _axis_names_for_engine(
        EngineName.TENSORKROWCH,
        ("bond_0", "bond_1", "open"),
    ) == ("bond_0", "bond_1", "open")
    assert _axis_name_for_engine(EngineName.TENSORKROWCH, "carry_0") == "carry"


def test_linear_periodic_carry_module_builds_carry_simulation_map() -> None:
    from tensor_network_editor.codegen.modes._linear_periodic.carry import (
        _build_carry_simulation_map,
    )

    chain = build_linear_periodic_carry_chain_spec().linear_periodic_chain
    assert chain is not None

    simulation_by_cell = _build_carry_simulation_map(
        chain=chain,
        engine=EngineName.TENSORNETWORK,
    )

    assert tuple(simulation_by_cell) == (
        LinearPeriodicCellName.INITIAL,
        LinearPeriodicCellName.PERIODIC,
        LinearPeriodicCellName.FINAL,
    )
    assert simulation_by_cell[LinearPeriodicCellName.INITIAL].carry_operand_id == (
        "initial_tensor"
    )
    assert simulation_by_cell[
        LinearPeriodicCellName.PERIODIC
    ].outgoing_interface_operand_ids == ("periodic_contract_full",)
    assert simulation_by_cell[LinearPeriodicCellName.FINAL].incoming_labels == (
        "final_bond",
    )


def test_simulate_carry_cell_initial_handoff_keeps_next_step_out_of_real_steps() -> (
    None
):
    from tensor_network_editor.codegen.modes._linear_periodic.carry import (
        _simulate_carry_cell,
    )

    chain = build_linear_periodic_carry_chain_spec().linear_periodic_chain
    assert chain is not None

    simulation = _simulate_carry_cell(
        cell=chain.initial_cell,
        cell_name=LinearPeriodicCellName.INITIAL,
        previous_payload_state=None,
        engine=EngineName.TENSORNETWORK,
    )

    assert simulation.real_steps == []
    assert simulation.carry_operand_id == "initial_tensor"
    assert simulation.outgoing_interface_operand_ids == ("initial_tensor",)


def test_simulate_carry_cell_periodic_records_only_simulated_steps() -> None:
    from tensor_network_editor.codegen.modes._linear_periodic.carry import (
        _build_carry_payload_state,
        _simulate_carry_cell,
    )

    chain = build_linear_periodic_carry_chain_spec().linear_periodic_chain
    assert chain is not None

    initial_simulation = _simulate_carry_cell(
        cell=chain.initial_cell,
        cell_name=LinearPeriodicCellName.INITIAL,
        previous_payload_state=None,
        engine=EngineName.TENSORNETWORK,
    )
    periodic_simulation = _simulate_carry_cell(
        cell=chain.periodic_cell,
        cell_name=LinearPeriodicCellName.PERIODIC,
        previous_payload_state=_build_carry_payload_state(initial_simulation),
        engine=EngineName.TENSORNETWORK,
    )

    assert [step.step_id for step in periodic_simulation.real_steps] == [
        "periodic_from_previous",
        "periodic_contract_full",
    ]
    assert periodic_simulation.outgoing_interface_operand_ids == (
        "periodic_contract_full",
    )


def test_simulate_carry_cell_rejects_step_using_previous_and_next_together() -> None:
    from tensor_network_editor.codegen.modes._linear_periodic.carry import (
        _build_carry_payload_state,
        _simulate_carry_cell,
    )

    chain = build_linear_periodic_carry_chain_spec().linear_periodic_chain
    assert chain is not None
    assert chain.periodic_cell.contraction_plan is not None
    chain.periodic_cell.contraction_plan.steps[0] = ContractionStepSpec(
        id="periodic_bad_boundary_step",
        left_operand_id="__linear_previous__",
        right_operand_id="__linear_next__",
    )

    initial_simulation = _simulate_carry_cell(
        cell=chain.initial_cell,
        cell_name=LinearPeriodicCellName.INITIAL,
        previous_payload_state=None,
        engine=EngineName.TENSORNETWORK,
    )

    with pytest.raises(
        CodeGenerationError,
        match=(
            "Carry step 'periodic_bad_boundary_step' in cell 'periodic' "
            "cannot use previous and next together."
        ),
    ):
        _simulate_carry_cell(
            cell=chain.periodic_cell,
            cell_name=LinearPeriodicCellName.PERIODIC,
            previous_payload_state=_build_carry_payload_state(initial_simulation),
            engine=EngineName.TENSORNETWORK,
        )


def test_simulate_carry_cell_rejects_next_step_that_is_not_final() -> None:
    from tensor_network_editor.codegen.modes._linear_periodic.carry import (
        _simulate_carry_cell,
    )

    chain = build_linear_periodic_carry_chain_spec().linear_periodic_chain
    assert chain is not None
    assert chain.initial_cell.contraction_plan is not None
    chain.initial_cell.contraction_plan.steps.append(
        ContractionStepSpec(
            id="initial_after_carry",
            left_operand_id="initial_tensor",
            right_operand_id="initial_tensor",
        )
    )

    with pytest.raises(
        CodeGenerationError,
        match=("Carry step 'initial_carry' in cell 'initial' must be the final step."),
    ):
        _simulate_carry_cell(
            cell=chain.initial_cell,
            cell_name=LinearPeriodicCellName.INITIAL,
            previous_payload_state=None,
            engine=EngineName.TENSORNETWORK,
        )


def test_simulate_carry_cell_accepts_previous_payload_labels_that_only_collide_by_name() -> (
    None
):
    from tensor_network_editor.codegen.modes._linear_periodic.carry import (
        _CarryOperandState,
        _CarryPayloadState,
        _simulate_carry_cell,
    )

    periodic_cell = LinearPeriodicCellSpec(
        tensors=[
            TensorSpec(
                id="periodic_previous_boundary",
                name="Previous cell",
                position=CanvasPosition(x=-100.0, y=140.0),
                linear_periodic_role=LinearPeriodicTensorRole.PREVIOUS,
                indices=[
                    IndexSpec(
                        id="periodic_previous_slot_1", name="slot_1", dimension=2
                    ),
                    IndexSpec(
                        id="periodic_previous_slot_2", name="slot_2", dimension=2
                    ),
                ],
            ),
            TensorSpec(
                id="periodic_next_boundary",
                name="Next cell",
                position=CanvasPosition(x=540.0, y=140.0),
                linear_periodic_role=LinearPeriodicTensorRole.NEXT,
                indices=[
                    IndexSpec(id="periodic_next_slot_1", name="slot_1", dimension=2),
                    IndexSpec(id="periodic_next_slot_2", name="slot_2", dimension=2),
                ],
            ),
            TensorSpec(
                id="tensor_a1",
                name="A1",
                position=CanvasPosition(x=-255.0, y=363.0),
                indices=[
                    IndexSpec(id="a1_right", name="right", dimension=3),
                    IndexSpec(id="a1_phys", name="phys", dimension=2),
                ],
            ),
            TensorSpec(
                id="tensor_a2",
                name="A2",
                position=CanvasPosition(x=65.0, y=363.0),
                indices=[
                    IndexSpec(id="a2_left", name="left", dimension=3),
                    IndexSpec(id="a2_right", name="right", dimension=3),
                    IndexSpec(id="a2_phys", name="phys", dimension=2),
                ],
            ),
            TensorSpec(
                id="tensor_a3",
                name="A3",
                position=CanvasPosition(x=385.0, y=363.0),
                indices=[
                    IndexSpec(id="a3_left", name="left", dimension=3),
                    IndexSpec(id="a3_right", name="right", dimension=3),
                    IndexSpec(id="a3_phys", name="phys", dimension=2),
                ],
            ),
            TensorSpec(
                id="tensor_a4",
                name="A4",
                position=CanvasPosition(x=705.0, y=363.0),
                indices=[
                    IndexSpec(id="a4_left", name="left", dimension=3),
                    IndexSpec(id="a4_phys", name="phys", dimension=2),
                ],
            ),
        ],
        edges=[
            EdgeSpec(
                id="edge_a1_a2",
                name="edge-0-1",
                left=EdgeEndpointRef(tensor_id="tensor_a1", index_id="a1_right"),
                right=EdgeEndpointRef(tensor_id="tensor_a2", index_id="a2_left"),
            ),
            EdgeSpec(
                id="edge_a2_a3",
                name="edge-1-2",
                left=EdgeEndpointRef(tensor_id="tensor_a2", index_id="a2_right"),
                right=EdgeEndpointRef(tensor_id="tensor_a3", index_id="a3_left"),
            ),
            EdgeSpec(
                id="edge_a3_a4",
                name="edge-2-3",
                left=EdgeEndpointRef(tensor_id="tensor_a3", index_id="a3_right"),
                right=EdgeEndpointRef(tensor_id="tensor_a4", index_id="a4_left"),
            ),
            EdgeSpec(
                id="edge_previous_a1",
                name="bond1",
                left=EdgeEndpointRef(
                    tensor_id="tensor_a1",
                    index_id="a1_phys",
                ),
                right=EdgeEndpointRef(
                    tensor_id="periodic_previous_boundary",
                    index_id="periodic_previous_slot_1",
                ),
            ),
            EdgeSpec(
                id="edge_previous_a2",
                name="bond2",
                left=EdgeEndpointRef(
                    tensor_id="periodic_previous_boundary",
                    index_id="periodic_previous_slot_2",
                ),
                right=EdgeEndpointRef(
                    tensor_id="tensor_a2",
                    index_id="a2_phys",
                ),
            ),
            EdgeSpec(
                id="edge_a3_next",
                name="bond3",
                left=EdgeEndpointRef(tensor_id="tensor_a3", index_id="a3_phys"),
                right=EdgeEndpointRef(
                    tensor_id="periodic_next_boundary",
                    index_id="periodic_next_slot_1",
                ),
            ),
            EdgeSpec(
                id="edge_a4_next",
                name="bond4",
                left=EdgeEndpointRef(
                    tensor_id="periodic_next_boundary",
                    index_id="periodic_next_slot_2",
                ),
                right=EdgeEndpointRef(tensor_id="tensor_a4", index_id="a4_phys"),
            ),
        ],
        contraction_plan=ContractionPlanSpec(
            id="periodic_plan",
            name="Manual path",
            steps=[
                ContractionStepSpec(
                    id="step_contract_right",
                    left_operand_id="tensor_a4",
                    right_operand_id="tensor_a3",
                ),
                ContractionStepSpec(
                    id="step_from_previous",
                    left_operand_id="__linear_previous__",
                    right_operand_id="tensor_a2",
                ),
                ContractionStepSpec(
                    id="step_merge",
                    left_operand_id="step_from_previous",
                    right_operand_id="step_contract_right",
                ),
                ContractionStepSpec(
                    id="step_to_next",
                    left_operand_id="step_merge",
                    right_operand_id="__linear_next__",
                ),
            ],
        ),
    )
    previous_payload_state = _CarryPayloadState(
        interface_operand_ids=("payload_left", "payload_right"),
        interface_labels=("a1_phys", "a2_phys"),
        operand_states={
            "payload_left": _CarryOperandState(
                labels=("payload_edge", "a1_phys"),
                axis_names=("left_payload", "slot_1"),
                dimensions=(3, 2),
            ),
            "payload_right": _CarryOperandState(
                labels=("a4_phys", "a3_phys", "payload_edge", "a2_phys"),
                axis_names=("carry_0", "carry_1", "bridge", "slot_2"),
                dimensions=(2, 2, 3, 2),
            ),
        },
    )

    simulation = _simulate_carry_cell(
        cell=periodic_cell,
        cell_name=LinearPeriodicCellName.PERIODIC,
        previous_payload_state=previous_payload_state,
        engine=EngineName.TENSORKROWCH,
    )

    assert simulation.carry_operand_id == "step_merge"
    assert simulation.outgoing_interface_operand_ids == ("step_merge", "step_merge")
    assert (
        simulation.remaining_operand_states["step_merge"].labels.count("a3_phys") == 1
    )
    assert (
        simulation.remaining_operand_states["step_merge"].labels.count("a4_phys") == 1
    )


def test_build_carry_simulation_context_collects_interface_state() -> None:
    from tensor_network_editor.codegen.modes._linear_periodic.carry import (
        _build_carry_simulation_context,
    )

    chain = build_linear_periodic_carry_chain_spec().linear_periodic_chain
    assert chain is not None

    context = _build_carry_simulation_context(
        cell=chain.initial_cell,
        cell_name=LinearPeriodicCellName.INITIAL,
        previous_payload_state=None,
        engine=EngineName.TENSORNETWORK,
    )

    assert context.cell_name is LinearPeriodicCellName.INITIAL
    assert context.previous_payload_state is None
    assert context.prepared.spec.name == "linear_periodic_internal_initial"
    assert context.incoming_labels == ()
    assert context.outgoing_labels == ("initial_bond",)
    assert tuple(port.internal_index_id for port in context.outgoing_ports) == (
        "initial_bond",
    )
    assert context.dimension_by_label["initial_bond"] == 3
