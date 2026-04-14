"""Shared types and carry-plan simulation helpers for linear periodic codegen."""

from __future__ import annotations

from dataclasses import dataclass, replace

from .._contraction_plan import (
    SimulatedContractionStep,
    build_dimension_by_label,
    simulate_contraction_step,
)
from .._linear_periodic import (
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
from ..errors import CodeGenerationError
from ..models import (
    ContractionStepSpec,
    EngineName,
    LinearPeriodicCellName,
    LinearPeriodicCellSpec,
    LinearPeriodicChainSpec,
)
from ._linear_periodic_expressions import _axis_names_for_engine
from .common import (
    CodeSection,
    PreparedNetwork,
    prepare_network,
    render_code_section_lines,
    render_code_sections,
)


@dataclass(slots=True)
class _RenderedCellHelper:
    """Generated helper function together with interface expressions."""

    lines: list[str]


@dataclass(slots=True, frozen=True)
class _CarryOperandState:
    """Track the current labels and axis names of one carry operand."""

    labels: tuple[str, ...]
    axis_names: tuple[str, ...]
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


_LINEAR_PERIODIC_CHAIN_LENGTH_ERROR = (
    "n must be at least 2 for a linear periodic chain."
)


def render_linear_periodic_shared_helpers(*, extra_lines: list[str]) -> list[str]:
    """Render shared top-level helpers plus backend-specific extras."""
    return [
        "def validate_chain_length(n: int) -> None:",
        "    if n < 2:",
        f"        raise ValueError({_LINEAR_PERIODIC_CHAIN_LENGTH_ERROR!r})",
        "",
        *extra_lines,
    ]


def render_linear_periodic_helper(
    *,
    helper_name: str,
    helper_signature: str,
    return_annotation: str,
    sections: list[CodeSection],
) -> _RenderedCellHelper:
    """Render one generated helper function with titled body sections."""
    helper_lines = [f"def {helper_name}({helper_signature}) -> {return_annotation}:"]
    body_lines = render_code_section_lines(*sections)
    helper_lines.extend([f"    {line}" if line else "" for line in body_lines])
    return _RenderedCellHelper(lines=helper_lines)


def render_linear_periodic_script(
    *,
    import_lines: list[str],
    shared_helper_lines: list[str],
    initial_cell_lines: list[str],
    periodic_cell_lines: list[str],
    final_cell_lines: list[str],
    main_loop_lines: list[str],
    output_lines: list[str],
) -> str:
    """Render one linear-periodic script with a fixed top-level section order."""
    return render_code_sections(
        CodeSection(title=None, lines=import_lines),
        CodeSection(title="Shared helpers", lines=shared_helper_lines),
        CodeSection(title="Initial cell", lines=initial_cell_lines),
        CodeSection(title="Periodic cell", lines=periodic_cell_lines),
        CodeSection(title="Final cell", lines=final_cell_lines),
        CodeSection(title="Main loop", lines=main_loop_lines),
        CodeSection(title="Outputs", lines=output_lines),
    )


def _simulate_carry_step(
    *,
    step: ContractionStepSpec,
    left_state: _CarryOperandState,
    right_state: _CarryOperandState,
    dimension_by_label: dict[str, int],
    engine: EngineName,
) -> tuple[SimulatedContractionStep, _CarryOperandState]:
    """Simulate one carry-mode contraction while preserving axis names."""
    simulation = simulate_contraction_step(
        step=step,
        left_labels=left_state.labels,
        right_labels=right_state.labels,
        left_axis_names=left_state.axis_names,
        right_axis_names=right_state.axis_names,
        dimension_by_label=dimension_by_label,
    )
    axis_name_by_label: dict[str, str] = {}
    for label, axis_name in zip(
        left_state.labels,
        left_state.axis_names,
        strict=True,
    ):
        axis_name_by_label[label] = axis_name
    for label, axis_name in zip(
        right_state.labels,
        right_state.axis_names,
        strict=True,
    ):
        axis_name_by_label.setdefault(label, axis_name)
    result_axis_names = tuple(
        axis_name_by_label[label] for label in simulation.surviving_labels
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
    real_steps: list[SimulatedContractionStep] = []
    result_index_by_step_id: dict[str, int] = {}
    carry_operand_id: str | None = None
    previous_operand_interface_index: int | None = None

    for step_index, step in enumerate(cell.contraction_plan.steps):
        uses_previous = LINEAR_PERIODIC_PREVIOUS_OPERAND_ID in {
            step.left_operand_id,
            step.right_operand_id,
        }
        uses_next = LINEAR_PERIODIC_NEXT_OPERAND_ID in {
            step.left_operand_id,
            step.right_operand_id,
        }
        if uses_previous and uses_next:
            raise CodeGenerationError(
                f"Carry step '{step.id}' in cell '{cell_name.value}' cannot use previous and next together."
            )
        if uses_next:
            partner_operand_id = (
                step.right_operand_id
                if step.left_operand_id == LINEAR_PERIODIC_NEXT_OPERAND_ID
                else step.left_operand_id
            )
            partner_state = remaining_operand_states.get(partner_operand_id)
            if partner_state is None:
                raise CodeGenerationError(
                    f"Carry step '{step.id}' in cell '{cell_name.value}' references an unavailable operand."
                )
            if not set(partner_state.labels).intersection(outgoing_labels):
                raise CodeGenerationError(
                    f"Carry step '{step.id}' in cell '{cell_name.value}' must carry an outgoing interface label."
                )
            if step_index < len(cell.contraction_plan.steps) - 1:
                raise CodeGenerationError(
                    f"Carry step '{step.id}' in cell '{cell_name.value}' must be the final step."
                )
            remaining_operand_states.pop(LINEAR_PERIODIC_NEXT_OPERAND_ID, None)
            remaining_operand_ids = [
                operand_id
                for operand_id in remaining_operand_ids
                if operand_id != LINEAR_PERIODIC_NEXT_OPERAND_ID
            ]
            break

        if uses_previous:
            partner_operand_id = (
                step.right_operand_id
                if step.left_operand_id == LINEAR_PERIODIC_PREVIOUS_OPERAND_ID
                else step.left_operand_id
            )
            partner_state = remaining_operand_states.pop(partner_operand_id, None)
            if partner_state is None:
                raise CodeGenerationError(
                    f"Carry step '{step.id}' in cell '{cell_name.value}' references an unavailable operand."
                )
            previous_state, selected_interface_index = (
                _resolve_previous_payload_operand_state(
                    previous_payload_state=previous_payload_state,
                    incoming_labels=incoming_labels,
                    partner_state=partner_state,
                    cell_name=cell_name,
                    step_id=step.id,
                )
            )
            previous_operand_interface_index = selected_interface_index
            dimension_by_label.update(
                dict(
                    zip(
                        previous_state.labels,
                        previous_state.dimensions,
                        strict=True,
                    )
                )
            )
            step_dimension_by_label = {
                **dimension_by_label,
                **dict(
                    zip(
                        previous_state.labels,
                        previous_state.dimensions,
                        strict=True,
                    )
                ),
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
                engine=engine,
            )
            remaining_operand_states = {
                step.id: result_state,
                **remaining_operand_states,
            }
            dimension_by_label.update(
                dict(zip(result_state.labels, result_state.dimensions, strict=True))
            )
            remaining_operand_ids = [
                step.id,
                *[
                    operand_id
                    for operand_id in remaining_operand_ids
                    if operand_id
                    not in {
                        step.left_operand_id,
                        step.right_operand_id,
                        LINEAR_PERIODIC_PREVIOUS_OPERAND_ID,
                    }
                ],
            ]
            result_index_by_step_id[step.id] = len(real_steps)
            real_steps.append(simulated_step)
            continue

        try:
            left_state = remaining_operand_states.pop(step.left_operand_id)
            right_state = remaining_operand_states.pop(step.right_operand_id)
        except KeyError as exc:
            raise CodeGenerationError(
                f"Carry step '{step.id}' in cell '{cell_name.value}' references an unavailable operand."
            ) from exc
        simulated_step, result_state = _simulate_carry_step(
            step=step,
            left_state=left_state,
            right_state=right_state,
            dimension_by_label=dimension_by_label,
            engine=engine,
        )
        remaining_operand_states = {
            step.id: result_state,
            **remaining_operand_states,
        }
        dimension_by_label.update(
            dict(zip(result_state.labels, result_state.dimensions, strict=True))
        )
        remaining_operand_ids = [
            step.id,
            *[
                operand_id
                for operand_id in remaining_operand_ids
                if operand_id not in {step.left_operand_id, step.right_operand_id}
            ],
        ]
        result_index_by_step_id[step.id] = len(real_steps)
        real_steps.append(simulated_step)

    outgoing_interface_operand_ids = tuple(
        _find_remaining_operand_id_for_label(
            label=label,
            remaining_operand_ids=tuple(remaining_operand_ids),
            remaining_operand_states=remaining_operand_states,
            cell_name=cell_name,
        )
        for label in outgoing_labels
    )
    if outgoing_interface_operand_ids:
        carry_operand_id = outgoing_interface_operand_ids[0]

    local_open_labels = tuple(
        index.label
        for index in prepared.open_indices
        if index.spec.id not in interface_index_ids
    )
    return _CarryPlanSimulation(
        prepared=prepared,
        real_steps=real_steps,
        result_index_by_step_id=result_index_by_step_id,
        remaining_operand_ids=tuple(remaining_operand_ids),
        remaining_operand_states=remaining_operand_states,
        carry_operand_id=carry_operand_id,
        outgoing_interface_operand_ids=outgoing_interface_operand_ids,
        previous_operand_interface_index=previous_operand_interface_index,
        local_open_labels=local_open_labels,
        incoming_labels=incoming_labels,
        outgoing_labels=outgoing_labels,
        incoming_ports=incoming_ports,
        outgoing_ports=outgoing_ports,
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
    return (
        _CarryOperandState(
            labels=tuple(
                incoming_label_by_payload_label.get(label, label)
                for label in selected_state.labels
            ),
            axis_names=selected_state.axis_names,
            dimensions=selected_state.dimensions,
        ),
        selected_index,
    )


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


def _cell_from_chain(
    chain: LinearPeriodicChainSpec,
    cell_name: LinearPeriodicCellName,
) -> LinearPeriodicCellSpec:
    """Return the matching cell from ``chain``."""
    if cell_name is LinearPeriodicCellName.INITIAL:
        return chain.initial_cell
    if cell_name is LinearPeriodicCellName.PERIODIC:
        return chain.periodic_cell
    return chain.final_cell
