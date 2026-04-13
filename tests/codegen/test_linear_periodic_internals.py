from __future__ import annotations

from tensor_network_editor.models import EngineName, LinearPeriodicCellName
from tests.factories import build_linear_periodic_carry_chain_spec


def test_linear_periodic_expression_helpers_live_in_internal_module() -> None:
    from tensor_network_editor.codegen._linear_periodic_expressions import (
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


def test_linear_periodic_shared_module_builds_carry_simulation_map() -> None:
    from tensor_network_editor.codegen._linear_periodic_shared import (
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
