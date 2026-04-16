"""Validation helpers for the typed linear periodic-chain mode."""

from __future__ import annotations

from dataclasses import dataclass, replace

from ._analysis import analyze_network
from ._contraction_plan import (
    PreparedContractionInputs,
    SimulatedContractionStep,
    prepare_contraction_inputs,
    simulate_contraction_step,
)
from ._linear_periodic import (
    LINEAR_PERIODIC_NEXT_OPERAND_ID,
    LINEAR_PERIODIC_PREVIOUS_OPERAND_ID,
    LINEAR_PERIODIC_RESERVED_OPERAND_IDS,
    LinearPeriodicInterfacePort,
    build_internal_linear_periodic_cell_network,
    build_linear_periodic_interface_axis_names,
    build_linear_periodic_interface_dimension_by_label,
    build_linear_periodic_interface_labels,
    build_linear_periodic_interface_ports,
    iter_linear_periodic_cells,
    linear_periodic_boundary_tensors,
    linear_periodic_cell_as_network,
    linear_periodic_cell_uses_carry_mode,
    linear_periodic_chain_uses_carry_mode,
)
from ._validation_common import append_issue, validate_metadata
from ._validation_contraction import validate_contraction_plan
from ._validation_edges import validate_edge
from ._validation_entities import (
    validate_group,
    validate_network,
    validate_note,
    validate_tensor,
)
from .codegen.common import prepare_network
from .models import (
    ContractionStepSpec,
    LinearPeriodicCellName,
    LinearPeriodicCellSpec,
    LinearPeriodicChainSpec,
    LinearPeriodicTensorRole,
    ValidationIssue,
)

_EXPECTED_BOUNDARY_ROLES: dict[
    LinearPeriodicCellName, tuple[LinearPeriodicTensorRole, ...]
] = {
    LinearPeriodicCellName.INITIAL: (LinearPeriodicTensorRole.NEXT,),
    LinearPeriodicCellName.PERIODIC: (
        LinearPeriodicTensorRole.PREVIOUS,
        LinearPeriodicTensorRole.NEXT,
    ),
    LinearPeriodicCellName.FINAL: (LinearPeriodicTensorRole.PREVIOUS,),
}

_EXPECTED_CARRY_COUNTS: dict[LinearPeriodicCellName, tuple[int, int]] = {
    LinearPeriodicCellName.INITIAL: (0, 1),
    LinearPeriodicCellName.PERIODIC: (1, 1),
    LinearPeriodicCellName.FINAL: (1, 0),
}


@dataclass(slots=True)
class _CarryOperandState:
    """Track labels for one operand while validating carry-mode plans."""

    labels: tuple[str, ...]
    axis_names: tuple[str, ...]


@dataclass(slots=True)
class _CarryValidationContext:
    """Gather the mutable carry-validation state for one cell."""

    cell_name: LinearPeriodicCellName
    cell_prefix: str
    label_by_index_id: dict[str, str]
    previous_ports: tuple[LinearPeriodicInterfacePort, ...]
    next_ports: tuple[LinearPeriodicInterfacePort, ...]
    incoming_labels: tuple[str, ...]
    outgoing_labels: tuple[str, ...]
    dimension_by_label: dict[str, int]
    operand_state_by_id: dict[str, _CarryOperandState]


def _build_carry_operand_state_map(
    contraction_inputs: PreparedContractionInputs,
) -> dict[str, _CarryOperandState]:
    """Build the starting carry-operand states from prepared contraction inputs."""
    return {
        operand_id: _CarryOperandState(
            labels=contraction_inputs.initial_operands[operand_id],
            axis_names=contraction_inputs.initial_axis_names[operand_id],
        )
        for operand_id in contraction_inputs.initial_operand_ids
    }


def _simulate_carry_step(
    *,
    step: ContractionStepSpec,
    left_state: _CarryOperandState,
    right_state: _CarryOperandState,
    dimension_by_label: dict[str, int],
) -> tuple[SimulatedContractionStep, tuple[str, ...]]:
    """Simulate one carry step while preserving explicit axis-name metadata."""
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
    return replace(simulation, result_axis_names=result_axis_names), result_axis_names


def _linear_periodic_cell_prefix(cell_name: LinearPeriodicCellName) -> str:
    """Return the validation path prefix for one linear periodic cell."""
    return f"linear_periodic_chain.{cell_name.value}_cell"


def _build_carry_validation_context(
    cell_name: LinearPeriodicCellName,
    cell: LinearPeriodicCellSpec,
    *,
    previous_expected: int,
    next_expected: int,
) -> _CarryValidationContext:
    """Build the mutable carry-validation context for one cell."""
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
    previous_ports = tuple(
        build_linear_periodic_interface_ports(
            cell,
            cell_name=cell_name,
            role=LinearPeriodicTensorRole.PREVIOUS,
        )
    )
    next_ports = tuple(
        build_linear_periodic_interface_ports(
            cell,
            cell_name=cell_name,
            role=LinearPeriodicTensorRole.NEXT,
        )
    )
    incoming_labels = build_linear_periodic_interface_labels(
        ports=previous_ports,
        label_by_index_id=label_by_index_id,
    )
    outgoing_labels = build_linear_periodic_interface_labels(
        ports=next_ports,
        label_by_index_id=label_by_index_id,
    )
    contraction_inputs = prepare_contraction_inputs(prepared)
    dimension_by_label = dict(contraction_inputs.dimension_by_label)
    dimension_by_label.update(
        build_linear_periodic_interface_dimension_by_label(
            ports=previous_ports,
            label_by_index_id=label_by_index_id,
        )
    )
    dimension_by_label.update(
        build_linear_periodic_interface_dimension_by_label(
            ports=next_ports,
            label_by_index_id=label_by_index_id,
        )
    )
    operand_state_by_id = _build_carry_operand_state_map(contraction_inputs)
    if previous_expected:
        operand_state_by_id[LINEAR_PERIODIC_PREVIOUS_OPERAND_ID] = _CarryOperandState(
            labels=incoming_labels,
            axis_names=build_linear_periodic_interface_axis_names(ports=previous_ports),
        )
    if next_expected:
        operand_state_by_id[LINEAR_PERIODIC_NEXT_OPERAND_ID] = _CarryOperandState(
            labels=outgoing_labels,
            axis_names=build_linear_periodic_interface_axis_names(ports=next_ports),
        )
    return _CarryValidationContext(
        cell_name=cell_name,
        cell_prefix=_linear_periodic_cell_prefix(cell_name),
        label_by_index_id=label_by_index_id,
        previous_ports=previous_ports,
        next_ports=next_ports,
        incoming_labels=incoming_labels,
        outgoing_labels=outgoing_labels,
        dimension_by_label=dimension_by_label,
        operand_state_by_id=operand_state_by_id,
    )


def validate_linear_periodic_chain(
    chain: LinearPeriodicChainSpec,
    *,
    issues: list[ValidationIssue],
) -> None:
    """Validate all cells and cross-cell interfaces in a periodic chain."""
    validate_metadata("linear_periodic_chain.metadata", chain.metadata, issues)

    for cell_name, cell in iter_linear_periodic_cells(chain):
        _validate_linear_periodic_cell(cell_name, cell, issues=issues)

    _validate_linear_periodic_interfaces(chain, issues=issues)
    _validate_linear_periodic_carry_mode(chain, issues=issues)


def _validate_linear_periodic_cell(
    cell_name: LinearPeriodicCellName,
    cell: LinearPeriodicCellSpec,
    *,
    issues: list[ValidationIssue],
) -> None:
    """Validate one cell using the existing plain-network validators."""
    prefix = _linear_periodic_cell_prefix(cell_name)
    cell_network = linear_periodic_cell_as_network(cell, cell_name=cell_name)

    local_issues: list[ValidationIssue] = []
    validate_network(cell_network, local_issues)
    analysis = analyze_network(cell_network)
    tensor_ids = set(analysis.tensor_map)

    for tensor in cell.tensors:
        validate_tensor(tensor, issues=local_issues)

    for group in cell.groups:
        validate_group(group, tensor_ids=tensor_ids, issues=local_issues)

    for note in cell.notes:
        validate_note(note, issues=local_issues)

    connected_indices: set[str] = set()
    for edge in cell.edges:
        validate_edge(
            edge,
            analysis_tensor_map=analysis.tensor_map,
            analysis_index_map=analysis.index_map,
            connected_indices=connected_indices,
            issues=local_issues,
        )

    if cell.contraction_plan is not None:
        validate_contraction_plan(
            cell.contraction_plan,
            tensor_ids={
                tensor.id
                for tensor in cell.tensors
                if tensor.linear_periodic_role is None
            }
            | (
                set(LINEAR_PERIODIC_RESERVED_OPERAND_IDS)
                if linear_periodic_cell_uses_carry_mode(cell)
                else set()
            ),
            issues=local_issues,
        )

    issues.extend(_prefix_validation_issues(prefix, local_issues))
    _validate_linear_periodic_boundary_roles(cell_name, cell, issues=issues)


def _validate_linear_periodic_boundary_roles(
    cell_name: LinearPeriodicCellName,
    cell: LinearPeriodicCellSpec,
    *,
    issues: list[ValidationIssue],
) -> None:
    """Ensure each cell exposes the expected virtual boundary tensors."""
    allowed_roles = set(_EXPECTED_BOUNDARY_ROLES[cell_name])
    cell_prefix = _linear_periodic_cell_prefix(cell_name)

    for tensor in cell.tensors:
        role = tensor.linear_periodic_role
        if role is None:
            continue
        if role not in allowed_roles:
            append_issue(
                issues,
                code="linear-periodic-boundary-role",
                message=(
                    f"Cell '{cell_name.value}' does not allow a boundary tensor "
                    f"with role '{role.value}'."
                ),
                path=f"{cell_prefix}.tensors.{tensor.id}.linear_periodic_role",
            )

    for role in allowed_roles:
        matching_tensors = linear_periodic_boundary_tensors(cell, role=role)
        if len(matching_tensors) != 1:
            append_issue(
                issues,
                code="linear-periodic-boundary-role",
                message=(
                    f"Cell '{cell_name.value}' must contain exactly one "
                    f"'{role.value}' boundary tensor."
                ),
                path=f"{cell_prefix}.{role.value}_boundary",
            )


def _validate_linear_periodic_interfaces(
    chain: LinearPeriodicChainSpec,
    *,
    issues: list[ValidationIssue],
) -> None:
    """Validate the ordered boundary interfaces that connect adjacent cells."""
    interface_pairs = (
        (
            LinearPeriodicCellName.INITIAL,
            LinearPeriodicTensorRole.NEXT,
            LinearPeriodicCellName.PERIODIC,
            LinearPeriodicTensorRole.PREVIOUS,
            "linear_periodic_chain.initial_cell.next_interface",
            "initial.next -> periodic.previous",
        ),
        (
            LinearPeriodicCellName.PERIODIC,
            LinearPeriodicTensorRole.NEXT,
            LinearPeriodicCellName.PERIODIC,
            LinearPeriodicTensorRole.PREVIOUS,
            "linear_periodic_chain.periodic_cell.next_interface",
            "periodic.next -> periodic.previous",
        ),
        (
            LinearPeriodicCellName.PERIODIC,
            LinearPeriodicTensorRole.NEXT,
            LinearPeriodicCellName.FINAL,
            LinearPeriodicTensorRole.PREVIOUS,
            "linear_periodic_chain.periodic_cell.next_interface",
            "periodic.next -> final.previous",
        ),
    )
    cell_by_name = dict(iter_linear_periodic_cells(chain))

    for (
        source_cell_name,
        source_role,
        target_cell_name,
        target_role,
        path,
        pair_label,
    ) in interface_pairs:
        source_cell = cell_by_name[source_cell_name]
        target_cell = cell_by_name[target_cell_name]
        if len(linear_periodic_boundary_tensors(source_cell, role=source_role)) != 1:
            continue
        if len(linear_periodic_boundary_tensors(target_cell, role=target_role)) != 1:
            continue

        source_ports = build_linear_periodic_interface_ports(
            source_cell,
            cell_name=source_cell_name,
            role=source_role,
        )
        target_ports = build_linear_periodic_interface_ports(
            target_cell,
            cell_name=target_cell_name,
            role=target_role,
        )
        source_dimensions = tuple(port.dimension for port in source_ports)
        target_dimensions = tuple(port.dimension for port in target_ports)
        if source_dimensions != target_dimensions:
            append_issue(
                issues,
                code="linear-periodic-interface-mismatch",
                message=(
                    f"Linear periodic interface mismatch for {pair_label}: "
                    f"{source_dimensions} != {target_dimensions}."
                ),
                path=path,
            )


def _validate_linear_periodic_carry_mode(
    chain: LinearPeriodicChainSpec,
    *,
    issues: list[ValidationIssue],
) -> None:
    """Validate reserved ``previous``/``next`` operands across the chain."""
    if not linear_periodic_chain_uses_carry_mode(chain):
        return

    for cell_name, cell in iter_linear_periodic_cells(chain):
        _validate_linear_periodic_carry_cell(cell_name, cell, issues=issues)


def _validate_linear_periodic_carry_cell(
    cell_name: LinearPeriodicCellName,
    cell: LinearPeriodicCellSpec,
    *,
    issues: list[ValidationIssue],
) -> None:
    """Validate carry-mode semantics inside one cell."""
    plan = cell.contraction_plan
    previous_expected, next_expected = _EXPECTED_CARRY_COUNTS[cell_name]
    cell_prefix = _linear_periodic_cell_prefix(cell_name)

    if plan is None or not plan.steps:
        append_issue(
            issues,
            code="linear-periodic-carry-boundary",
            message=(
                f"Cell '{cell_name.value}' must define carry-mode steps for "
                "'previous'/'next'."
            ),
            path=f"{cell_prefix}.contraction_plan",
        )
        return

    previous_steps = _carry_steps_using_operand(
        plan.steps,
        LINEAR_PERIODIC_PREVIOUS_OPERAND_ID,
    )
    next_steps = _carry_steps_using_operand(
        plan.steps,
        LINEAR_PERIODIC_NEXT_OPERAND_ID,
    )
    if len(previous_steps) != previous_expected:
        append_issue(
            issues,
            code="linear-periodic-carry-boundary",
            message=(
                f"Cell '{cell_name.value}' must use 'previous' exactly "
                f"{previous_expected} time(s) in carry mode."
            ),
            path=f"{cell_prefix}.contraction_plan.steps",
        )
    if len(next_steps) != next_expected:
        append_issue(
            issues,
            code="linear-periodic-carry-boundary",
            message=(
                f"Cell '{cell_name.value}' must use 'next' exactly "
                f"{next_expected} time(s) in carry mode."
            ),
            path=f"{cell_prefix}.contraction_plan.steps",
        )

    next_step_index = _find_first_carry_step_index(
        plan.steps,
        LINEAR_PERIODIC_NEXT_OPERAND_ID,
    )
    if next_step_index is not None and next_step_index < len(plan.steps) - 1:
        append_issue(
            issues,
            code="linear-periodic-carry-order",
            message=(
                f"Cell '{cell_name.value}' cannot define steps after a 'next' carry step."
            ),
            path=f"{cell_prefix}.contraction_plan.steps.{plan.steps[next_step_index + 1].id}",
        )

    context = _build_carry_validation_context(
        cell_name,
        cell,
        previous_expected=previous_expected,
        next_expected=next_expected,
    )

    for step in plan.steps:
        _validate_linear_periodic_carry_step(step, context=context, issues=issues)


def _carry_steps_using_operand(
    steps: list[ContractionStepSpec],
    operand_id: str,
) -> list[ContractionStepSpec]:
    """Return the carry-mode steps that reference one reserved operand."""
    return [
        step
        for step in steps
        if operand_id in {step.left_operand_id, step.right_operand_id}
    ]


def _find_first_carry_step_index(
    steps: list[ContractionStepSpec],
    operand_id: str,
) -> int | None:
    """Return the first step index using one reserved carry operand."""
    return next(
        (
            step_index
            for step_index, step in enumerate(steps)
            if operand_id in {step.left_operand_id, step.right_operand_id}
        ),
        None,
    )


def _validate_linear_periodic_carry_step(
    step: ContractionStepSpec,
    *,
    context: _CarryValidationContext,
    issues: list[ValidationIssue],
) -> None:
    """Validate and apply one carry-mode step."""
    step_path = f"{context.cell_prefix}.contraction_plan.steps.{step.id}"
    uses_previous = LINEAR_PERIODIC_PREVIOUS_OPERAND_ID in {
        step.left_operand_id,
        step.right_operand_id,
    }
    uses_next = LINEAR_PERIODIC_NEXT_OPERAND_ID in {
        step.left_operand_id,
        step.right_operand_id,
    }

    if uses_previous and uses_next:
        append_issue(
            issues,
            code="linear-periodic-carry-boundary",
            message=(
                f"Carry step '{step.id}' in cell '{context.cell_name.value}' cannot use "
                "'previous' and 'next' together."
            ),
            path=step_path,
        )
        return

    if uses_next:
        _validate_carry_step_using_next(
            step,
            step_path=step_path,
            context=context,
            issues=issues,
        )
        return

    if uses_previous:
        _validate_carry_step_using_previous(
            step,
            step_path=step_path,
            context=context,
            issues=issues,
        )
        return

    _validate_carry_step_without_reserved_operands(step, context=context)


def _validate_carry_step_using_next(
    step: ContractionStepSpec,
    *,
    step_path: str,
    context: _CarryValidationContext,
    issues: list[ValidationIssue],
) -> None:
    """Validate one carry step that contracts with ``next``."""
    partner_operand_id = _carry_partner_operand_id(
        step,
        LINEAR_PERIODIC_NEXT_OPERAND_ID,
    )
    next_state = context.operand_state_by_id.get(LINEAR_PERIODIC_NEXT_OPERAND_ID)
    partner_state = context.operand_state_by_id.get(partner_operand_id)
    if next_state is None or partner_state is None:
        return
    left_state, right_state = _resolve_step_operand_states(
        step=step,
        reserved_operand_id=LINEAR_PERIODIC_NEXT_OPERAND_ID,
        reserved_state=next_state,
        partner_state=partner_state,
    )
    simulation, result_axis_names = _simulate_carry_step(
        step=step,
        left_state=left_state,
        right_state=right_state,
        dimension_by_label=context.dimension_by_label,
    )
    if not any(
        label in context.outgoing_labels for label in simulation.contracted_labels
    ):
        append_issue(
            issues,
            code="linear-periodic-carry-operand",
            message=(
                f"Carry step '{step.id}' in cell '{context.cell_name.value}' must "
                "consume at least one outgoing interface label from 'next'."
            ),
            path=step_path,
        )
    _store_carry_step_result(
        context=context,
        step=step,
        removed_operand_ids=(LINEAR_PERIODIC_NEXT_OPERAND_ID, partner_operand_id),
        result_axis_names=result_axis_names,
        surviving_labels=simulation.surviving_labels,
    )


def _validate_carry_step_using_previous(
    step: ContractionStepSpec,
    *,
    step_path: str,
    context: _CarryValidationContext,
    issues: list[ValidationIssue],
) -> None:
    """Validate one carry step that contracts with ``previous``."""
    partner_operand_id = _carry_partner_operand_id(
        step,
        LINEAR_PERIODIC_PREVIOUS_OPERAND_ID,
    )
    previous_state = context.operand_state_by_id.get(
        LINEAR_PERIODIC_PREVIOUS_OPERAND_ID
    )
    partner_state = context.operand_state_by_id.get(partner_operand_id)
    if previous_state is None or partner_state is None:
        return
    left_state, right_state = _resolve_step_operand_states(
        step=step,
        reserved_operand_id=LINEAR_PERIODIC_PREVIOUS_OPERAND_ID,
        reserved_state=previous_state,
        partner_state=partner_state,
    )
    simulation, result_axis_names = _simulate_carry_step(
        step=step,
        left_state=left_state,
        right_state=right_state,
        dimension_by_label=context.dimension_by_label,
    )
    if not any(
        label in context.incoming_labels for label in simulation.contracted_labels
    ):
        append_issue(
            issues,
            code="linear-periodic-carry-operand",
            message=(
                f"Carry step '{step.id}' in cell '{context.cell_name.value}' must "
                "consume at least one incoming interface label from 'previous'."
            ),
            path=step_path,
        )
    _store_carry_step_result(
        context=context,
        step=step,
        removed_operand_ids=(LINEAR_PERIODIC_PREVIOUS_OPERAND_ID, partner_operand_id),
        result_axis_names=result_axis_names,
        surviving_labels=simulation.surviving_labels,
    )


def _validate_carry_step_without_reserved_operands(
    step: ContractionStepSpec,
    *,
    context: _CarryValidationContext,
) -> None:
    """Validate one carry step that contracts two standard operands."""
    maybe_left_state = context.operand_state_by_id.get(step.left_operand_id)
    maybe_right_state = context.operand_state_by_id.get(step.right_operand_id)
    if maybe_left_state is None or maybe_right_state is None:
        return
    simulation, result_axis_names = _simulate_carry_step(
        step=step,
        left_state=maybe_left_state,
        right_state=maybe_right_state,
        dimension_by_label=context.dimension_by_label,
    )
    _store_carry_step_result(
        context=context,
        step=step,
        removed_operand_ids=(step.left_operand_id, step.right_operand_id),
        result_axis_names=result_axis_names,
        surviving_labels=simulation.surviving_labels,
    )


def _carry_partner_operand_id(
    step: ContractionStepSpec,
    reserved_operand_id: str,
) -> str:
    """Return the non-reserved operand id used by one carry step."""
    if step.left_operand_id == reserved_operand_id:
        return step.right_operand_id
    return step.left_operand_id


def _resolve_step_operand_states(
    *,
    step: ContractionStepSpec,
    reserved_operand_id: str,
    reserved_state: _CarryOperandState,
    partner_state: _CarryOperandState,
) -> tuple[_CarryOperandState, _CarryOperandState]:
    """Return left/right operand states in the order used by ``step``."""
    if step.left_operand_id == reserved_operand_id:
        return reserved_state, partner_state
    return partner_state, reserved_state


def _store_carry_step_result(
    *,
    context: _CarryValidationContext,
    step: ContractionStepSpec,
    removed_operand_ids: tuple[str, str],
    result_axis_names: tuple[str, ...],
    surviving_labels: tuple[str, ...],
) -> None:
    """Replace the consumed operand states with the result of one carry step."""
    for operand_id in removed_operand_ids:
        context.operand_state_by_id.pop(operand_id, None)
    context.operand_state_by_id[step.id] = _CarryOperandState(
        labels=surviving_labels,
        axis_names=result_axis_names,
    )


def _prefix_validation_issues(
    prefix: str,
    issues: list[ValidationIssue],
) -> list[ValidationIssue]:
    """Return a copy of ``issues`` with every path nested below ``prefix``."""
    return [
        ValidationIssue(
            code=issue.code,
            message=issue.message,
            path=f"{prefix}.{issue.path}" if issue.path else prefix,
        )
        for issue in issues
    ]
