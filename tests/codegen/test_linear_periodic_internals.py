from __future__ import annotations

import pytest

from tensor_network_editor.errors import CodeGenerationError
from tensor_network_editor.models import (
    ContractionStepSpec,
    EngineName,
    LinearPeriodicCellName,
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
