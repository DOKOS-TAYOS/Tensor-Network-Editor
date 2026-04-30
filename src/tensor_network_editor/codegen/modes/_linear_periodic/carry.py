"""Carry-plan simulation helpers for linear-periodic codegen."""

from __future__ import annotations

from dataclasses import dataclass, replace

from ....errors import CodeGenerationError
from ....internal.analysis._contraction_plan import (
    SimulatedContractionStep,
    build_dimension_by_label,
)
from ....internal.modes._linear_periodic import (
    LINEAR_PERIODIC_NEXT_OPERAND_ID,
    LINEAR_PERIODIC_PREVIOUS_OPERAND_ID,
    LinearPeriodicInterfacePort,
    LinearPeriodicTensorRole,
    build_internal_linear_periodic_cell_network,
    build_linear_periodic_interface_axis_names,
    build_linear_periodic_interface_dimension_by_label,
    build_linear_periodic_interface_labels,
    build_linear_periodic_interface_ports,
)
from ....internal.modes._linear_periodic_carry import (
    LinearPeriodicCarryOperandState,
    linear_periodic_carry_partner_operand_id,
    linear_periodic_step_uses_reserved_operand,
    simulate_linear_periodic_carry_step,
)
from ....models import (
    ContractionStepSpec,
    EngineName,
    LinearPeriodicCellName,
    LinearPeriodicCellSpec,
    LinearPeriodicChainSpec,
)
from ...shared._linear_periodic_expressions import _axis_names_for_engine
from ...shared.common import (
    PreparedNetwork,
    prepare_network,
)
from .common import _cell_from_chain


@dataclass(slots=True, frozen=True)
class _CarryOperandState(LinearPeriodicCarryOperandState):
    """Track the current labels and axis names of one carry operand."""

    dimensions: tuple[int, ...]


@dataclass(slots=True)
class _CarryPayloadState:
    """Carry payload metadata passed from one cell to the next."""

    interface_operand_ids: tuple[str, ...]
    interface_labels: tuple[str, ...]
    operand_states: dict[str, _CarryOperandState]


@dataclass(slots=True)
class _CarryPlanSimulation:
    """Prepared rendering state for one carry-mode cell helper."""

    prepared: PreparedNetwork
    real_steps: list[SimulatedContractionStep]
    result_index_by_step_id: dict[str, int]
    remaining_operand_ids: tuple[str, ...]
    remaining_operand_states: dict[str, _CarryOperandState]
    carry_operand_id: str | None
    outgoing_interface_operand_ids: tuple[str, ...]
    previous_operand_interface_index: int | None
    local_open_labels: tuple[str, ...]
    incoming_labels: tuple[str, ...]
    outgoing_labels: tuple[str, ...]
    incoming_ports: tuple[LinearPeriodicInterfacePort, ...]
    outgoing_ports: tuple[LinearPeriodicInterfacePort, ...]


@dataclass(slots=True, frozen=True)
class _CarrySimulationContext:
    """Immutable carry-cell inputs shared across all step handlers."""

    cell_name: LinearPeriodicCellName
    previous_payload_state: _CarryPayloadState | None
    engine: EngineName
    prepared: PreparedNetwork
    incoming_ports: tuple[LinearPeriodicInterfacePort, ...]
    outgoing_ports: tuple[LinearPeriodicInterfacePort, ...]
    interface_index_ids: frozenset[str]
    incoming_labels: tuple[str, ...]
    outgoing_labels: tuple[str, ...]
    dimension_by_label: dict[str, int]


@dataclass(slots=True)
class _CarrySimulationState:
    """Mutable carry-cell simulation state updated by each plan step."""

    remaining_operand_ids: list[str]
    remaining_operand_states: dict[str, _CarryOperandState]
    real_steps: list[SimulatedContractionStep]
    result_index_by_step_id: dict[str, int]
    previous_operand_interface_index: int | None
    dimension_by_label: dict[str, int]


def _simulate_carry_step(
    *,
    step: ContractionStepSpec,
    left_state: _CarryOperandState,
    right_state: _CarryOperandState,
    dimension_by_label: dict[str, int],
    engine: EngineName,
) -> tuple[SimulatedContractionStep, _CarryOperandState]:
    """Simulate one carry-mode contraction while preserving axis names."""
    simulation, result_axis_names = simulate_linear_periodic_carry_step(
        step=step,
        left_state=left_state,
        right_state=right_state,
        dimension_by_label=dimension_by_label,
    )
    result_axis_names = _axis_names_for_engine(engine, result_axis_names)
    result_state = _CarryOperandState(
        labels=simulation.surviving_labels,
        axis_names=result_axis_names,
        dimensions=tuple(
            dimension_by_label[label] for label in simulation.surviving_labels
        ),
    )
    return replace(simulation, result_axis_names=result_axis_names), result_state


def _simulate_carry_cell(
    *,
    cell: LinearPeriodicCellSpec,
    cell_name: LinearPeriodicCellName,
    previous_payload_state: _CarryPayloadState | None,
    engine: EngineName,
) -> _CarryPlanSimulation:
    """Simulate one carry-mode cell with real ``previous``/``next`` steps."""
    if cell.contraction_plan is None or not cell.contraction_plan.steps:
        raise CodeGenerationError(
            f"Carry mode in cell '{cell_name.value}' requires a contraction plan."
        )

    context = _build_carry_simulation_context(
        cell=cell,
        cell_name=cell_name,
        previous_payload_state=previous_payload_state,
        engine=engine,
    )
    state = _build_carry_simulation_state(context)
    _simulate_carry_plan_steps(
        steps=cell.contraction_plan.steps,
        context=context,
        state=state,
    )
    return _build_carry_plan_simulation(context=context, state=state)


def _build_carry_simulation_context(
    *,
    cell: LinearPeriodicCellSpec,
    cell_name: LinearPeriodicCellName,
    previous_payload_state: _CarryPayloadState | None,
    engine: EngineName,
) -> _CarrySimulationContext:
    """Prepare the immutable carry-cell context shared by all step handlers."""
    prepared = prepare_network(
        build_internal_linear_periodic_cell_network(
            cell,
            cell_name=cell_name,
            include_contraction_plan=False,
        ),
        validate=False,
    )
    label_by_index_id = {
        index.spec.id: index.label
        for tensor in prepared.tensors
        for index in tensor.indices
    }
    incoming_ports = build_linear_periodic_interface_ports(
        cell,
        cell_name=cell_name,
        role=LinearPeriodicTensorRole.PREVIOUS,
    )
    outgoing_ports = build_linear_periodic_interface_ports(
        cell,
        cell_name=cell_name,
        role=LinearPeriodicTensorRole.NEXT,
    )
    interface_index_ids = {
        port.internal_index_id for port in (*incoming_ports, *outgoing_ports)
    }
    incoming_labels = build_linear_periodic_interface_labels(
        ports=incoming_ports,
        label_by_index_id=label_by_index_id,
    )
    outgoing_labels = build_linear_periodic_interface_labels(
        ports=outgoing_ports,
        label_by_index_id=label_by_index_id,
    )
    initial_operand_ids = tuple(tensor.spec.id for tensor in prepared.tensors)
    remaining_operand_states = {
        tensor.spec.id: _CarryOperandState(
            labels=tuple(index.label for index in tensor.indices),
            axis_names=_axis_names_for_engine(
                engine,
                tuple(index.spec.name for index in tensor.indices),
            ),
            dimensions=tuple(index.spec.dimension for index in tensor.indices),
        )
        for tensor in prepared.tensors
    }
    remaining_operand_ids = list(initial_operand_ids)
    if incoming_labels:
        remaining_operand_ids.insert(0, LINEAR_PERIODIC_PREVIOUS_OPERAND_ID)
    if outgoing_labels:
        remaining_operand_states[LINEAR_PERIODIC_NEXT_OPERAND_ID] = _CarryOperandState(
            labels=outgoing_labels,
            axis_names=_axis_names_for_engine(
                engine,
                build_linear_periodic_interface_axis_names(ports=outgoing_ports),
            ),
            dimensions=tuple(port.dimension for port in outgoing_ports),
        )
        remaining_operand_ids.append(LINEAR_PERIODIC_NEXT_OPERAND_ID)

    dimension_by_label = {
        **build_dimension_by_label(prepared),
        **build_linear_periodic_interface_dimension_by_label(
            ports=incoming_ports,
            label_by_index_id=label_by_index_id,
        ),
        **build_linear_periodic_interface_dimension_by_label(
            ports=outgoing_ports,
            label_by_index_id=label_by_index_id,
        ),
    }
    return _CarrySimulationContext(
        cell_name=cell_name,
        previous_payload_state=previous_payload_state,
        engine=engine,
        prepared=prepared,
        incoming_ports=incoming_ports,
        outgoing_ports=outgoing_ports,
        interface_index_ids=frozenset(interface_index_ids),
        incoming_labels=incoming_labels,
        outgoing_labels=outgoing_labels,
        dimension_by_label=dimension_by_label,
    )


def _build_carry_simulation_state(
    context: _CarrySimulationContext,
) -> _CarrySimulationState:
    """Build the mutable carry-cell state before simulating any plan steps."""
    remaining_operand_states = {
        tensor.spec.id: _CarryOperandState(
            labels=tuple(index.label for index in tensor.indices),
            axis_names=_axis_names_for_engine(
                context.engine,
                tuple(index.spec.name for index in tensor.indices),
            ),
            dimensions=tuple(index.spec.dimension for index in tensor.indices),
        )
        for tensor in context.prepared.tensors
    }
    remaining_operand_ids = [tensor.spec.id for tensor in context.prepared.tensors]
    if context.incoming_labels:
        remaining_operand_ids.insert(0, LINEAR_PERIODIC_PREVIOUS_OPERAND_ID)
    if context.outgoing_labels:
        remaining_operand_states[LINEAR_PERIODIC_NEXT_OPERAND_ID] = _CarryOperandState(
            labels=context.outgoing_labels,
            axis_names=_axis_names_for_engine(
                context.engine,
                build_linear_periodic_interface_axis_names(
                    ports=context.outgoing_ports
                ),
            ),
            dimensions=tuple(port.dimension for port in context.outgoing_ports),
        )
        remaining_operand_ids.append(LINEAR_PERIODIC_NEXT_OPERAND_ID)
    return _CarrySimulationState(
        remaining_operand_ids=remaining_operand_ids,
        remaining_operand_states=remaining_operand_states,
        real_steps=[],
        result_index_by_step_id={},
        previous_operand_interface_index=None,
        dimension_by_label=dict(context.dimension_by_label),
    )


def _simulate_carry_plan_steps(
    *,
    steps: list[ContractionStepSpec],
    context: _CarrySimulationContext,
    state: _CarrySimulationState,
) -> None:
    """Run all carry-mode plan steps in order for one cell."""
    for step_index, step in enumerate(steps):
        uses_previous = linear_periodic_step_uses_reserved_operand(
            step,
            LINEAR_PERIODIC_PREVIOUS_OPERAND_ID,
        )
        uses_next = linear_periodic_step_uses_reserved_operand(
            step,
            LINEAR_PERIODIC_NEXT_OPERAND_ID,
        )
        if uses_previous and uses_next:
            raise CodeGenerationError(
                f"Carry step '{step.id}' in cell '{context.cell_name.value}' cannot use previous and next together."
            )
        if uses_next:
            _simulate_carry_next_step(
                step=step,
                step_index=step_index,
                step_count=len(steps),
                context=context,
                state=state,
            )
            break
        if uses_previous:
            _simulate_carry_previous_step(
                step=step,
                context=context,
                state=state,
            )
            continue
        _simulate_standard_carry_step(
            step=step,
            context=context,
            state=state,
        )


def _simulate_carry_next_step(
    *,
    step: ContractionStepSpec,
    step_index: int,
    step_count: int,
    context: _CarrySimulationContext,
    state: _CarrySimulationState,
) -> None:
    """Validate and finalize a ``next`` handoff step without simulating it."""
    partner_operand_id = linear_periodic_carry_partner_operand_id(
        step,
        LINEAR_PERIODIC_NEXT_OPERAND_ID,
    )
    partner_state = state.remaining_operand_states.get(partner_operand_id)
    if partner_state is None:
        raise CodeGenerationError(
            f"Carry step '{step.id}' in cell '{context.cell_name.value}' references an unavailable operand."
        )
    if not set(partner_state.labels).intersection(context.outgoing_labels):
        raise CodeGenerationError(
            f"Carry step '{step.id}' in cell '{context.cell_name.value}' must carry an outgoing interface label."
        )
    if step_index < step_count - 1:
        raise CodeGenerationError(
            f"Carry step '{step.id}' in cell '{context.cell_name.value}' must be the final step."
        )
    state.remaining_operand_states.pop(LINEAR_PERIODIC_NEXT_OPERAND_ID, None)
    state.remaining_operand_ids = [
        operand_id
        for operand_id in state.remaining_operand_ids
        if operand_id != LINEAR_PERIODIC_NEXT_OPERAND_ID
    ]


def _simulate_carry_previous_step(
    *,
    step: ContractionStepSpec,
    context: _CarrySimulationContext,
    state: _CarrySimulationState,
) -> None:
    """Simulate one carry step that consumes the previous payload operand."""
    partner_operand_id = linear_periodic_carry_partner_operand_id(
        step,
        LINEAR_PERIODIC_PREVIOUS_OPERAND_ID,
    )
    partner_state = state.remaining_operand_states.pop(partner_operand_id, None)
    if partner_state is None:
        raise CodeGenerationError(
            f"Carry step '{step.id}' in cell '{context.cell_name.value}' references an unavailable operand."
        )
    previous_state, selected_interface_index = _resolve_previous_payload_operand_state(
        previous_payload_state=context.previous_payload_state,
        incoming_labels=context.incoming_labels,
        partner_state=partner_state,
        cell_name=context.cell_name,
        step_id=step.id,
    )
    state.previous_operand_interface_index = selected_interface_index
    previous_dimensions = dict(
        zip(
            previous_state.labels,
            previous_state.dimensions,
            strict=True,
        )
    )
    state.dimension_by_label.update(previous_dimensions)
    step_dimension_by_label = {
        **state.dimension_by_label,
        **previous_dimensions,
    }
    left_state = (
        previous_state
        if step.left_operand_id == LINEAR_PERIODIC_PREVIOUS_OPERAND_ID
        else partner_state
    )
    right_state = (
        partner_state
        if step.left_operand_id == LINEAR_PERIODIC_PREVIOUS_OPERAND_ID
        else previous_state
    )
    simulated_step, result_state = _simulate_carry_step(
        step=step,
        left_state=left_state,
        right_state=right_state,
        dimension_by_label=step_dimension_by_label,
        engine=context.engine,
    )
    _store_simulated_carry_step(
        step=step,
        simulated_step=simulated_step,
        result_state=result_state,
        consumed_operand_ids={
            step.left_operand_id,
            step.right_operand_id,
            LINEAR_PERIODIC_PREVIOUS_OPERAND_ID,
        },
        state=state,
    )


def _simulate_standard_carry_step(
    *,
    step: ContractionStepSpec,
    context: _CarrySimulationContext,
    state: _CarrySimulationState,
) -> None:
    """Simulate one standard carry step using only local remaining operands."""
    try:
        left_state = state.remaining_operand_states.pop(step.left_operand_id)
        right_state = state.remaining_operand_states.pop(step.right_operand_id)
    except KeyError as exc:
        raise CodeGenerationError(
            f"Carry step '{step.id}' in cell '{context.cell_name.value}' references an unavailable operand."
        ) from exc
    simulated_step, result_state = _simulate_carry_step(
        step=step,
        left_state=left_state,
        right_state=right_state,
        dimension_by_label=state.dimension_by_label,
        engine=context.engine,
    )
    _store_simulated_carry_step(
        step=step,
        simulated_step=simulated_step,
        result_state=result_state,
        consumed_operand_ids={step.left_operand_id, step.right_operand_id},
        state=state,
    )


def _store_simulated_carry_step(
    *,
    step: ContractionStepSpec,
    simulated_step: SimulatedContractionStep,
    result_state: _CarryOperandState,
    consumed_operand_ids: set[str],
    state: _CarrySimulationState,
) -> None:
    """Record one simulated carry step and update remaining operand state."""
    state.remaining_operand_states = {
        step.id: result_state,
        **state.remaining_operand_states,
    }
    state.dimension_by_label.update(
        dict(zip(result_state.labels, result_state.dimensions, strict=True))
    )
    state.remaining_operand_ids = [
        step.id,
        *[
            operand_id
            for operand_id in state.remaining_operand_ids
            if operand_id not in consumed_operand_ids
        ],
    ]
    state.result_index_by_step_id[step.id] = len(state.real_steps)
    state.real_steps.append(simulated_step)


def _build_carry_plan_simulation(
    *,
    context: _CarrySimulationContext,
    state: _CarrySimulationState,
) -> _CarryPlanSimulation:
    """Build the final carry simulation payload consumed by code generators."""
    outgoing_interface_operand_ids = tuple(
        _find_remaining_operand_id_for_label(
            label=label,
            remaining_operand_ids=tuple(state.remaining_operand_ids),
            remaining_operand_states=state.remaining_operand_states,
            cell_name=context.cell_name,
        )
        for label in context.outgoing_labels
    )
    carry_operand_id = (
        outgoing_interface_operand_ids[0] if outgoing_interface_operand_ids else None
    )
    local_open_labels = tuple(
        index.label
        for index in context.prepared.open_indices
        if index.spec.id not in context.interface_index_ids
    )
    return _CarryPlanSimulation(
        prepared=context.prepared,
        real_steps=state.real_steps,
        result_index_by_step_id=state.result_index_by_step_id,
        remaining_operand_ids=tuple(state.remaining_operand_ids),
        remaining_operand_states=state.remaining_operand_states,
        carry_operand_id=carry_operand_id,
        outgoing_interface_operand_ids=outgoing_interface_operand_ids,
        previous_operand_interface_index=state.previous_operand_interface_index,
        local_open_labels=local_open_labels,
        incoming_labels=context.incoming_labels,
        outgoing_labels=context.outgoing_labels,
        incoming_ports=context.incoming_ports,
        outgoing_ports=context.outgoing_ports,
    )


def _resolve_previous_payload_operand_state(
    *,
    previous_payload_state: _CarryPayloadState | None,
    incoming_labels: tuple[str, ...],
    partner_state: _CarryOperandState,
    cell_name: LinearPeriodicCellName,
    step_id: str,
) -> tuple[_CarryOperandState, int]:
    """Resolve which carried operand owns the incoming interface for a step."""
    if previous_payload_state is None:
        raise CodeGenerationError(
            f"Carry step '{step_id}' in cell '{cell_name.value}' needs a previous payload."
        )
    if len(previous_payload_state.interface_operand_ids) != len(incoming_labels):
        raise CodeGenerationError(
            f"Cell '{cell_name.value}' received a previous payload with a mismatched interface."
        )

    partner_labels = set(partner_state.labels)
    used_interface_indexes = [
        index for index, label in enumerate(incoming_labels) if label in partner_labels
    ]
    if not used_interface_indexes:
        used_interface_indexes = [0]

    owner_ids = {
        previous_payload_state.interface_operand_ids[index]
        for index in used_interface_indexes
    }
    if len(owner_ids) != 1:
        raise CodeGenerationError(
            f"Carry step '{step_id}' in cell '{cell_name.value}' needs one previous carry operand per step."
        )

    selected_index = used_interface_indexes[0]
    selected_operand_id = previous_payload_state.interface_operand_ids[selected_index]
    selected_state = previous_payload_state.operand_states.get(selected_operand_id)
    if selected_state is None:
        raise CodeGenerationError(
            f"Carry step '{step_id}' in cell '{cell_name.value}' references an unavailable previous operand."
        )

    incoming_label_by_payload_label = {
        previous_payload_state.interface_labels[index]: incoming_labels[index]
        for index, operand_id in enumerate(previous_payload_state.interface_operand_ids)
        if operand_id == selected_operand_id
    }
    renamed_labels = tuple(
        incoming_label_by_payload_label.get(
            label,
            _previous_payload_local_label(
                operand_id=selected_operand_id,
                axis_index=axis_index,
            ),
        )
        for axis_index, label in enumerate(selected_state.labels)
    )
    return (
        _CarryOperandState(
            labels=renamed_labels,
            axis_names=selected_state.axis_names,
            dimensions=selected_state.dimensions,
        ),
        selected_index,
    )


def _previous_payload_local_label(*, operand_id: str, axis_index: int) -> str:
    """Return a collision-free simulation label for one carried local axis."""
    return f"__previous_payload_{operand_id}_{axis_index}"


def _find_remaining_operand_id_for_label(
    *,
    label: str,
    remaining_operand_ids: tuple[str, ...],
    remaining_operand_states: dict[str, _CarryOperandState],
    cell_name: LinearPeriodicCellName,
) -> str:
    """Return the surviving operand that owns an outgoing interface label."""
    for operand_id in remaining_operand_ids:
        operand_state = remaining_operand_states.get(operand_id)
        if operand_state is not None and label in operand_state.labels:
            return operand_id
    raise CodeGenerationError(
        f"Cell '{cell_name.value}' could not expose outgoing carry label '{label}'."
    )


def _build_carry_payload_state(
    simulation: _CarryPlanSimulation,
) -> _CarryPayloadState | None:
    """Build the compile-time payload state passed to the next cell helper."""
    if not simulation.outgoing_interface_operand_ids:
        return None

    operand_states: dict[str, _CarryOperandState] = {}
    for operand_id in simulation.outgoing_interface_operand_ids:
        operand_state = simulation.remaining_operand_states.get(operand_id)
        if operand_state is None:
            raise CodeGenerationError(
                f"Cell '{simulation.prepared.spec.name}' has an unavailable outgoing carry operand."
            )
        operand_states[operand_id] = operand_state

    return _CarryPayloadState(
        interface_operand_ids=simulation.outgoing_interface_operand_ids,
        interface_labels=simulation.outgoing_labels,
        operand_states=operand_states,
    )


def _build_carry_simulation_map(
    *,
    chain: LinearPeriodicChainSpec,
    engine: EngineName,
) -> dict[LinearPeriodicCellName, _CarryPlanSimulation]:
    """Build the compile-time carry simulations for one backend."""
    carry_simulation_by_cell_name: dict[
        LinearPeriodicCellName, _CarryPlanSimulation
    ] = {}
    previous_payload_state: _CarryPayloadState | None = None
    for cell_name in (
        LinearPeriodicCellName.INITIAL,
        LinearPeriodicCellName.PERIODIC,
        LinearPeriodicCellName.FINAL,
    ):
        simulation = _simulate_carry_cell(
            cell=_cell_from_chain(chain, cell_name),
            cell_name=cell_name,
            previous_payload_state=previous_payload_state,
            engine=engine,
        )
        carry_simulation_by_cell_name[cell_name] = simulation
        previous_payload_state = _build_carry_payload_state(simulation)
    return carry_simulation_by_cell_name
