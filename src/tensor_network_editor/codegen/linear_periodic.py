"""Code generation helpers for typed linear periodic-chain specifications."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, replace

from .._contraction_plan import (
    SimulatedContractionStep,
    build_dimension_by_label,
    build_initial_operand_axis_names,
    build_initial_operand_labels,
    simulate_contraction_plan,
    simulate_contraction_step,
)
from .._linear_periodic import (
    LINEAR_PERIODIC_NEXT_OPERAND_ID,
    LINEAR_PERIODIC_PREVIOUS_OPERAND_ID,
    LinearPeriodicInterfacePort,
    build_internal_linear_periodic_cell_network,
    build_linear_periodic_interface_ports,
    linear_periodic_chain_uses_carry_mode,
)
from ..errors import CodeGenerationError
from ..models import (
    CodegenResult,
    ContractionStepSpec,
    EngineName,
    LinearPeriodicCellName,
    LinearPeriodicCellSpec,
    LinearPeriodicChainSpec,
    LinearPeriodicTensorRole,
    NetworkSpec,
    TensorCollectionFormat,
)
from .common import (
    PreparedNetwork,
    container_name_for_format,
    flattened_tensor_collection_expression,
    prepare_network,
    render_results_list_reference,
    render_tensor_collection_assignment,
    tensor_collection_reference_by_id,
)
from .einsum_numpy import EinsumNumpyCodeGenerator
from .einsum_torch import EinsumTorchCodeGenerator
from .quimb import QuimbCodeGenerator
from .tensorkrowch import TensorKrowchCodeGenerator
from .tensornetwork import TensorNetworkCodeGenerator

_TENSORKROWCH_AXIS_INDEX_SUFFIX_RE = re.compile(r"_\d+$")


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


def _build_interface_labels(
    *,
    ports: tuple[LinearPeriodicInterfacePort, ...],
    label_by_index_id: dict[str, str],
) -> tuple[str, ...]:
    """Resolve prepared labels for the provided interface ports."""
    return tuple(
        label_by_index_id[port.internal_index_id]
        for port in ports
        if port.internal_index_id in label_by_index_id
    )


def _build_interface_axis_names(
    *,
    ports: tuple[LinearPeriodicInterfacePort, ...],
) -> tuple[str, ...]:
    """Expose stable slot names for carry-mode boundary operands."""
    return tuple(port.boundary_index_name for port in ports)


def _build_interface_dimensions(
    *,
    ports: tuple[LinearPeriodicInterfacePort, ...],
    label_by_index_id: dict[str, str],
) -> dict[str, int]:
    """Map interface labels to the dimensions declared on the boundary tensor."""
    return {
        label_by_index_id[port.internal_index_id]: port.dimension
        for port in ports
        if port.internal_index_id in label_by_index_id
    }


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


def generate_linear_periodic_code(
    spec: NetworkSpec,
    engine: EngineName,
    *,
    collection_format: TensorCollectionFormat,
) -> CodegenResult:
    """Generate helper-based Python code for the linear periodic-chain mode."""
    if spec.linear_periodic_chain is None:
        raise CodeGenerationError(
            "Linear periodic code generation requires a chain payload."
        )
    chain = spec.linear_periodic_chain
    uses_carry_mode = linear_periodic_chain_uses_carry_mode(chain)
    if engine is EngineName.QUIMB:
        if uses_carry_mode:
            return _generate_quimb_linear_periodic_carry_code(
                chain=chain,
                collection_format=collection_format,
            )
        return _generate_quimb_linear_periodic_code(
            chain=chain,
            collection_format=collection_format,
        )
    if engine in {EngineName.EINSUM_NUMPY, EngineName.EINSUM_TORCH}:
        if uses_carry_mode:
            return _generate_einsum_linear_periodic_carry_code(
                chain=chain,
                engine=engine,
                collection_format=collection_format,
            )
        return _generate_einsum_linear_periodic_code(
            chain=chain,
            engine=engine,
            collection_format=collection_format,
        )
    if engine not in {
        EngineName.TENSORNETWORK,
        EngineName.TENSORKROWCH,
    }:
        raise CodeGenerationError(
            f"The {engine.value} backend does not support linear periodic code generation."
        )

    lines = _render_import_lines(engine)
    if not uses_carry_mode:
        lines.extend(_render_connect_helper(engine))

    carry_simulation_by_cell_name: dict[
        LinearPeriodicCellName, _CarryPlanSimulation
    ] = {}
    if uses_carry_mode:
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

    helper_signature_by_cell_name = {
        LinearPeriodicCellName.INITIAL: (
            "" if uses_carry_mode else "",
            "build_initial_cell",
        ),
        LinearPeriodicCellName.PERIODIC: (
            "cell_index, previous_payload" if uses_carry_mode else "cell_index",
            "build_periodic_cell",
        ),
        LinearPeriodicCellName.FINAL: (
            "previous_payload" if uses_carry_mode else "",
            "build_final_cell",
        ),
    }
    for cell_name in (
        LinearPeriodicCellName.INITIAL,
        LinearPeriodicCellName.PERIODIC,
        LinearPeriodicCellName.FINAL,
    ):
        helper_signature, helper_name = helper_signature_by_cell_name[cell_name]
        if uses_carry_mode:
            lines.extend(
                _render_carry_cell_helper(
                    chain=chain,
                    cell_name=cell_name,
                    helper_name=helper_name,
                    helper_signature=helper_signature,
                    engine=engine,
                    collection_format=collection_format,
                    simulation=carry_simulation_by_cell_name[cell_name],
                ).lines
            )
        else:
            lines.extend(
                _render_cell_helper(
                    chain=chain,
                    cell_name=cell_name,
                    helper_name=helper_name,
                    helper_signature=helper_signature,
                    engine=engine,
                    collection_format=collection_format,
                ).lines
            )
        if cell_name is not LinearPeriodicCellName.FINAL:
            lines.append("")

    lines.extend(_render_main_flow_lines(uses_carry_mode=uses_carry_mode))
    return CodegenResult(engine=engine, code="\n".join(lines).strip() + "\n")


def _render_import_lines(engine: EngineName) -> list[str]:
    """Render the common import prelude for one backend."""
    if engine is EngineName.TENSORNETWORK:
        return [
            "# Tensor Network Editor linear periodic mode",
            "import numpy as np",
            "import tensornetwork as tn",
            "",
        ]
    return [
        "# Tensor Network Editor linear periodic mode",
        "import torch",
        "import tensorkrowch as tk",
        "",
        "network = tk.TensorNetwork()",
        "",
    ]


def _render_connect_helper(engine: EngineName) -> list[str]:
    """Render the shared interface-connection helper for non-carry mode."""
    if engine is EngineName.TENSORNETWORK:
        return _render_tensornetwork_connect_helper()
    return _render_tensorkrowch_connect_helper()


def _render_main_flow_lines(*, uses_carry_mode: bool) -> list[str]:
    """Render the outer free-``n`` orchestration block."""
    if uses_carry_mode:
        return [
            "",
            "if n < 2:",
            "    raise ValueError('n must be at least 2 for a linear periodic chain.')",
            "",
            "remaining_operands = {}",
            "open_edges = []",
            "",
            "previous_payload = build_initial_cell()",
            "",
            "for cell_index in range(1, n - 1):",
            "    previous_payload = build_periodic_cell(cell_index, previous_payload)",
            "",
            "result = build_final_cell(previous_payload)",
            "network_nodes = list(remaining_operands.values())",
        ]
    return [
        "",
        "if n < 2:",
        "    raise ValueError('n must be at least 2 for a linear periodic chain.')",
        "",
        "initial_cell = build_initial_cell()",
        "network_nodes = list(initial_cell['nodes'])",
        "open_edges = list(initial_cell['open_edges'])",
        "previous_interface = list(initial_cell['outgoing_interface'])",
        "",
        "for cell_index in range(1, n - 1):",
        "    periodic_cell = build_periodic_cell(cell_index)",
        "    connect_cell_interfaces(previous_interface, periodic_cell['incoming_interface'])",
        "    network_nodes.extend(periodic_cell['nodes'])",
        "    open_edges.extend(periodic_cell['open_edges'])",
        "    previous_interface = list(periodic_cell['outgoing_interface'])",
        "",
        "final_cell = build_final_cell()",
        "connect_cell_interfaces(previous_interface, final_cell['incoming_interface'])",
        "network_nodes.extend(final_cell['nodes'])",
        "open_edges.extend(final_cell['open_edges'])",
        "result = network_nodes[0] if len(network_nodes) == 1 else None",
    ]


def _render_tensornetwork_connect_helper() -> list[str]:
    """Render the shared interface-connection helper for ``tensornetwork``."""
    return [
        "def connect_cell_interfaces(left_interface: list[object], right_interface: list[object]) -> None:",
        "    if len(left_interface) != len(right_interface):",
        "        raise ValueError('Cell interfaces must have matching lengths.')",
        "    for left_edge, right_edge in zip(left_interface, right_interface):",
        "        tn.connect(left_edge, right_edge)",
        "",
    ]


def _render_tensorkrowch_connect_helper() -> list[str]:
    """Render the shared interface-connection helper for ``tensorkrowch``."""
    return [
        "def connect_cell_interfaces(left_interface: list[object], right_interface: list[object]) -> None:",
        "    if len(left_interface) != len(right_interface):",
        "        raise ValueError('Cell interfaces must have matching lengths.')",
        "    for left_edge, right_edge in zip(left_interface, right_interface):",
        "        tk.connect(left_edge, right_edge)",
        "",
    ]


def _render_cell_helper(
    *,
    chain: LinearPeriodicChainSpec,
    cell_name: LinearPeriodicCellName,
    helper_name: str,
    helper_signature: str,
    engine: EngineName,
    collection_format: TensorCollectionFormat,
) -> _RenderedCellHelper:
    """Render one non-carry cell helper and the expressions it returns."""
    cell = _cell_from_chain(chain, cell_name)
    internal_spec = build_internal_linear_periodic_cell_network(
        cell,
        cell_name=cell_name,
    )
    prepared = prepare_network(internal_spec)
    collection_name = container_name_for_format(collection_format)
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
    prepared_label_by_index_id = {
        index.spec.id: index.label
        for tensor in prepared.tensors
        for index in tensor.indices
    }
    interface_index_ids = {
        port.internal_index_id for port in (*incoming_ports, *outgoing_ports)
    }

    helper_lines = [f"def {helper_name}({helper_signature}):"]
    body_lines = _render_cell_body(
        prepared=prepared,
        engine=engine,
        collection_format=collection_format,
        collection_name=collection_name,
    )
    label_expression_by_label = _build_label_expression_map(
        prepared=prepared,
        engine=engine,
        collection_format=collection_format,
        collection_name=collection_name,
    )
    incoming_expressions = [
        label_expression_by_label[prepared_label_by_index_id[port.internal_index_id]]
        for port in incoming_ports
        if port.internal_index_id in prepared_label_by_index_id
    ]
    outgoing_expressions = [
        label_expression_by_label[prepared_label_by_index_id[port.internal_index_id]]
        for port in outgoing_ports
        if port.internal_index_id in prepared_label_by_index_id
    ]
    open_edge_expressions = [
        label_expression_by_label[index.label]
        for index in prepared.open_indices
        if index.spec.id not in interface_index_ids
    ]

    body_lines.append("incoming_interface = [" + ", ".join(incoming_expressions) + "]")
    body_lines.append("outgoing_interface = [" + ", ".join(outgoing_expressions) + "]")
    body_lines.append("open_edges = [" + ", ".join(open_edge_expressions) + "]")
    body_lines.extend(
        [
            "return {",
            "    'nodes': network_nodes,",
            "    'incoming_interface': incoming_interface,",
            "    'outgoing_interface': outgoing_interface,",
            "    'open_edges': open_edges,",
            "}",
        ]
    )

    helper_lines.extend([f"    {line}" if line else "" for line in body_lines])
    return _RenderedCellHelper(lines=helper_lines)


def _render_carry_cell_helper(
    *,
    chain: LinearPeriodicChainSpec,
    cell_name: LinearPeriodicCellName,
    helper_name: str,
    helper_signature: str,
    engine: EngineName,
    collection_format: TensorCollectionFormat,
    simulation: _CarryPlanSimulation,
) -> _RenderedCellHelper:
    """Render one carry-mode helper that threads ``previous_operand``."""
    del chain
    collection_name = container_name_for_format(collection_format)
    helper_lines = [f"def {helper_name}({helper_signature}):"]
    body_lines = _render_cell_setup(
        prepared=simulation.prepared,
        engine=engine,
        collection_format=collection_format,
        collection_name=collection_name,
    )
    body_lines.extend(
        _render_carry_boundary_setup(
            simulation=simulation,
            engine=engine,
            collection_format=collection_format,
            collection_name=collection_name,
        )
    )
    body_lines.extend(
        _render_carry_plan_lines(
            simulation=simulation,
            cell_name=cell_name,
            engine=engine,
            collection_format=collection_format,
            collection_name=collection_name,
        )
    )
    helper_lines.extend([f"    {line}" if line else "" for line in body_lines])
    return _RenderedCellHelper(lines=helper_lines)


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
    incoming_labels = _build_interface_labels(
        ports=incoming_ports,
        label_by_index_id=label_by_index_id,
    )
    outgoing_labels = _build_interface_labels(
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
                _build_interface_axis_names(ports=outgoing_ports),
            ),
            dimensions=tuple(port.dimension for port in outgoing_ports),
        )
        remaining_operand_ids.append(LINEAR_PERIODIC_NEXT_OPERAND_ID)

    dimension_by_label = {
        **build_dimension_by_label(prepared),
        **_build_interface_dimensions(
            ports=incoming_ports,
            label_by_index_id=label_by_index_id,
        ),
        **_build_interface_dimensions(
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


def _render_carry_boundary_setup(
    *,
    simulation: _CarryPlanSimulation,
    engine: EngineName,
    collection_format: TensorCollectionFormat,
    collection_name: str,
) -> list[str]:
    """Render boundary-slot wiring for ``previous`` and ``next`` carry operands."""
    lines: list[str] = []
    if simulation.incoming_ports:
        previous_operand_index = simulation.previous_operand_interface_index or 0
        lines.append(
            "previous_interface = list(previous_payload['outgoing_interface'])"
        )
        lines.append("previous_operands = list(previous_payload['outgoing_operands'])")
        lines.append(
            f"if len(previous_interface) != {len(simulation.incoming_ports)} or "
            f"len(previous_operands) != {len(simulation.incoming_ports)}:"
        )
        lines.append(
            "    raise ValueError('Previous payload interface does not match this cell.')"
        )
        lines.append(f"previous_operand = previous_operands[{previous_operand_index}]")
        lines.append("incoming_edges = []")
        for port_index, port in enumerate(simulation.incoming_ports):
            internal_axis_name = _axis_name_for_engine(
                engine,
                port.internal_index_name,
            )
            local_tensor = tensor_collection_reference_by_id(
                simulation.prepared,
                port.internal_tensor_id,
                collection_format,
                collection_name,
            )
            if engine is EngineName.TENSORNETWORK:
                lines.append(
                    "incoming_edges.append(tn.connect("
                    f"previous_interface[{port_index}], "
                    f"{local_tensor}[{internal_axis_name!r}], "
                    f"name={port.boundary_index_name!r}))"
                )
            else:
                lines.append(
                    "incoming_edges.append(("
                    f"{port.boundary_index_name!r}, "
                    f"tk.connect(previous_interface[{port_index}], "
                    f"{local_tensor}[{internal_axis_name!r}])"
                    "))"
                )
        lines.append("")

    return lines


def _render_carry_plan_lines(
    *,
    simulation: _CarryPlanSimulation,
    cell_name: LinearPeriodicCellName,
    engine: EngineName,
    collection_format: TensorCollectionFormat,
    collection_name: str,
) -> list[str]:
    """Render all carry-mode contractions and helper epilogue lines."""
    if engine is EngineName.TENSORKROWCH and any(
        step.is_outer_product for step in simulation.real_steps
    ):
        raise CodeGenerationError(
            "TensorKrowch cannot emit manual outer product steps with contract_between."
        )

    base_operand_expressions = {
        tensor.spec.id: tensor_collection_reference_by_id(
            simulation.prepared,
            tensor.spec.id,
            collection_format,
            collection_name,
        )
        for tensor in simulation.prepared.tensors
    }
    if cell_name is not LinearPeriodicCellName.INITIAL:
        base_operand_expressions[LINEAR_PERIODIC_PREVIOUS_OPERAND_ID] = (
            "previous_operand"
        )

    lines = ["results_list = []", ""]
    for step_index, step in enumerate(simulation.real_steps):
        latest_result_index = step_index - 1 if step_index > 0 else None
        left_expression = _operand_expression(
            engine=engine,
            operand_id=step.left_operand_id,
            base_operand_expressions=base_operand_expressions,
            step_result_indexes=simulation.result_index_by_step_id,
            latest_result_index=latest_result_index,
        )
        right_expression = _operand_expression(
            engine=engine,
            operand_id=step.right_operand_id,
            base_operand_expressions=base_operand_expressions,
            step_result_indexes=simulation.result_index_by_step_id,
            latest_result_index=latest_result_index,
        )
        lines.append(f"# Manual step {step.step_id}")
        if engine is EngineName.TENSORNETWORK:
            output_edge_order = TensorNetworkCodeGenerator._build_output_edge_order(
                left_expression=left_expression,
                right_expression=right_expression,
                left_labels=step.left_labels,
                right_labels=step.right_labels,
                left_axis_names=step.left_axis_names,
                right_axis_names=step.right_axis_names,
                contracted_labels=step.contracted_labels,
            )
            lines.append(
                "results_list.append(tn.contract_between("
                f"{left_expression}, "
                f"{right_expression}, "
                f"name={step.step_id!r}, "
                "allow_outer_product=True, "
                f"output_edge_order={output_edge_order}, "
                f"axis_names={list(step.result_axis_names)!r}))"
            )
        else:
            lines.append(
                "results_list.append(tk.contract_between("
                f"{left_expression}, {right_expression}))"
            )
        lines.append("")

    final_result_index = (
        len(simulation.real_steps) - 1 if simulation.real_steps else None
    )
    if engine is EngineName.TENSORKROWCH and simulation.outgoing_interface_operand_ids:
        for operand_id in dict.fromkeys(simulation.outgoing_interface_operand_ids):
            if operand_id not in simulation.result_index_by_step_id:
                continue
            operand_expression = _operand_expression(
                engine=engine,
                operand_id=operand_id,
                base_operand_expressions=base_operand_expressions,
                step_result_indexes=simulation.result_index_by_step_id,
                latest_result_index=final_result_index,
            )
            lines.append(f"{operand_expression}.reattach_edges()")

    label_expression_by_label = _build_remaining_label_expression_map(
        remaining_operand_ids=simulation.remaining_operand_ids,
        remaining_operand_states=simulation.remaining_operand_states,
        engine=engine,
        base_operand_expressions=base_operand_expressions,
        step_result_indexes=simulation.result_index_by_step_id,
        latest_result_index=final_result_index,
    )
    local_open_expressions = [
        label_expression_by_label[label]
        for label in simulation.local_open_labels
        if label in label_expression_by_label
    ]
    if local_open_expressions:
        lines.append("open_edges.extend([" + ", ".join(local_open_expressions) + "])")

    local_remaining_operand_ids = [
        operand_id
        for operand_id in simulation.remaining_operand_ids
        if operand_id != simulation.carry_operand_id
    ]
    if local_remaining_operand_ids:
        lines.append(
            "cell_key_prefix = " + _carry_cell_key_prefix_expression(cell_name)
        )
        for operand_id in local_remaining_operand_ids:
            operand_expression = _operand_expression(
                engine=engine,
                operand_id=operand_id,
                base_operand_expressions=base_operand_expressions,
                step_result_indexes=simulation.result_index_by_step_id,
                latest_result_index=final_result_index,
            )
            lines.append(
                f'remaining_operands[f"{{cell_key_prefix}}:{operand_id}"] = {operand_expression}'
            )

    if simulation.carry_operand_id is not None:
        carry_expression = _operand_expression(
            engine=engine,
            operand_id=simulation.carry_operand_id,
            base_operand_expressions=base_operand_expressions,
            step_result_indexes=simulation.result_index_by_step_id,
            latest_result_index=final_result_index,
        )
        outgoing_interface_expressions = [
            label_expression_by_label[label] for label in simulation.outgoing_labels
        ]
        outgoing_operand_expressions = [
            _operand_expression(
                engine=engine,
                operand_id=operand_id,
                base_operand_expressions=base_operand_expressions,
                step_result_indexes=simulation.result_index_by_step_id,
                latest_result_index=final_result_index,
            )
            for operand_id in simulation.outgoing_interface_operand_ids
        ]
        lines.append(
            "outgoing_interface = [" + ", ".join(outgoing_interface_expressions) + "]"
        )
        lines.append(
            "outgoing_operands = [" + ", ".join(outgoing_operand_expressions) + "]"
        )
        lines.extend(
            [
                "return {",
                f"    'operand': {carry_expression},",
                "    'outgoing_interface': outgoing_interface,",
                "    'outgoing_operands': outgoing_operands,",
                "}",
            ]
        )
        return lines

    if local_remaining_operand_ids:
        final_expression = _operand_expression(
            engine=engine,
            operand_id=local_remaining_operand_ids[0],
            base_operand_expressions=base_operand_expressions,
            step_result_indexes=simulation.result_index_by_step_id,
            latest_result_index=final_result_index,
        )
        lines.append(f"return {final_expression}")
    else:
        lines.append("return None")
    return lines


def _carry_cell_key_prefix_expression(cell_name: LinearPeriodicCellName) -> str:
    """Return the runtime Python expression used to namespace remaining operands."""
    if cell_name is LinearPeriodicCellName.INITIAL:
        return "'initial'"
    if cell_name is LinearPeriodicCellName.PERIODIC:
        return "f'periodic_{cell_index}'"
    return "'final'"


def _render_cell_setup(
    *,
    prepared: PreparedNetwork,
    engine: EngineName,
    collection_format: TensorCollectionFormat,
    collection_name: str,
) -> list[str]:
    """Render tensor creation plus real intra-cell edge connections."""
    if engine is EngineName.TENSORNETWORK:
        return _render_tensornetwork_cell_setup(
            prepared=prepared,
            collection_format=collection_format,
            collection_name=collection_name,
        )
    return _render_tensorkrowch_cell_setup(
        prepared=prepared,
        collection_format=collection_format,
        collection_name=collection_name,
    )


def _render_cell_body(
    *,
    prepared: PreparedNetwork,
    engine: EngineName,
    collection_format: TensorCollectionFormat,
    collection_name: str,
) -> list[str]:
    """Render the body of one cell helper for the requested backend."""
    if engine is EngineName.TENSORNETWORK:
        return _render_tensornetwork_cell_body(
            prepared=prepared,
            collection_format=collection_format,
            collection_name=collection_name,
        )
    return _render_tensorkrowch_cell_body(
        prepared=prepared,
        collection_format=collection_format,
        collection_name=collection_name,
    )


def _render_tensornetwork_cell_setup(
    *,
    prepared: PreparedNetwork,
    collection_format: TensorCollectionFormat,
    collection_name: str,
) -> list[str]:
    """Render tensor creation and edge wiring for ``tensornetwork``."""
    lines: list[str] = []
    lines.extend(
        render_tensor_collection_assignment(
            collection_name=collection_name,
            collection_format=collection_format,
            prepared=prepared,
            tensor_value_by_id={
                tensor.spec.id: (
                    f"tn.Node(np.zeros({tensor.spec.shape!r}, dtype=float), "
                    f"name={tensor.spec.name!r}, "
                    f"axis_names={[index.spec.name for index in tensor.indices]!r})"
                )
                for tensor in prepared.tensors
            },
        )
    )
    lines.append("")

    if prepared.edges:
        lines.append("edges_list = []")
        for edge in prepared.edges:
            left_tensor = tensor_collection_reference_by_id(
                prepared,
                edge.spec.left.tensor_id,
                collection_format,
                collection_name,
            )
            right_tensor = tensor_collection_reference_by_id(
                prepared,
                edge.spec.right.tensor_id,
                collection_format,
                collection_name,
            )
            lines.append(
                "edges_list.append(tn.connect("
                f"{left_tensor}[{edge.left.spec.name!r}], "
                f"{right_tensor}[{edge.right.spec.name!r}], "
                f"name={edge.spec.name!r}))"
            )
        lines.append("")
    return lines


def _render_tensorkrowch_cell_setup(
    *,
    prepared: PreparedNetwork,
    collection_format: TensorCollectionFormat,
    collection_name: str,
) -> list[str]:
    """Render tensor creation and edge wiring for ``tensorkrowch``."""
    lines: list[str] = []
    lines.extend(
        render_tensor_collection_assignment(
            collection_name=collection_name,
            collection_format=collection_format,
            prepared=prepared,
            tensor_value_by_id={
                tensor.spec.id: (
                    f"tk.Node(tensor=torch.zeros({tensor.spec.shape!r}, dtype=torch.float32), "
                    f"axes_names={tuple(index.spec.name for index in tensor.indices)!r}, "
                    f"name={TensorKrowchCodeGenerator.node_name(tensor)!r}, "
                    "network=network)"
                )
                for tensor in prepared.tensors
            },
        )
    )
    lines.append("")

    if prepared.edges:
        lines.append("edges_list = []")
        for edge in prepared.edges:
            left_tensor = tensor_collection_reference_by_id(
                prepared,
                edge.spec.left.tensor_id,
                collection_format,
                collection_name,
            )
            right_tensor = tensor_collection_reference_by_id(
                prepared,
                edge.spec.right.tensor_id,
                collection_format,
                collection_name,
            )
            left_axis_name = _axis_name_for_engine(
                EngineName.TENSORKROWCH,
                edge.left.spec.name,
            )
            right_axis_name = _axis_name_for_engine(
                EngineName.TENSORKROWCH,
                edge.right.spec.name,
            )
            lines.append(f"# {edge.spec.name}")
            lines.append(
                "edges_list.append(("
                f"{edge.spec.name!r}, "
                f"tk.connect({left_tensor}[{left_axis_name!r}], {right_tensor}[{right_axis_name!r}])"
                "))"
            )
        lines.append("")
    return lines


def _render_tensornetwork_cell_body(
    *,
    prepared: PreparedNetwork,
    collection_format: TensorCollectionFormat,
    collection_name: str,
) -> list[str]:
    """Render a periodic-cell helper body for ``tensornetwork``."""
    generator = TensorNetworkCodeGenerator()
    lines = _render_tensornetwork_cell_setup(
        prepared=prepared,
        collection_format=collection_format,
        collection_name=collection_name,
    )

    if (
        prepared.spec.contraction_plan is not None
        and prepared.spec.contraction_plan.steps
    ):
        lines.extend(
            generator._render_manual_plan(
                prepared=prepared,
                collection_format=collection_format,
                collection_name=collection_name,
            )
        )
    else:
        lines.append(
            "network_nodes = "
            + flattened_tensor_collection_expression(collection_format, collection_name)
        )
    return lines


def _render_tensorkrowch_cell_body(
    *,
    prepared: PreparedNetwork,
    collection_format: TensorCollectionFormat,
    collection_name: str,
) -> list[str]:
    """Render a periodic-cell helper body for ``tensorkrowch``."""
    generator = TensorKrowchCodeGenerator()
    lines = _render_tensorkrowch_cell_setup(
        prepared=prepared,
        collection_format=collection_format,
        collection_name=collection_name,
    )

    if (
        prepared.spec.contraction_plan is not None
        and prepared.spec.contraction_plan.steps
    ):
        lines.extend(
            generator._render_manual_plan(
                prepared=prepared,
                collection_format=collection_format,
                collection_name=collection_name,
            )
        )
        lines.append("network_nodes = list(remaining_operands.values())")
    else:
        lines.append(
            "network_nodes = "
            + flattened_tensor_collection_expression(collection_format, collection_name)
        )
    return lines


def _build_label_expression_map(
    *,
    prepared: PreparedNetwork,
    engine: EngineName,
    collection_format: TensorCollectionFormat,
    collection_name: str,
) -> dict[str, str]:
    """Resolve every surviving open label to the generated Python expression."""
    simulation = simulate_contraction_plan(
        initial_operand_ids=tuple(tensor.spec.id for tensor in prepared.tensors),
        initial_operands=build_initial_operand_labels(prepared),
        initial_axis_names=build_initial_operand_axis_names(prepared),
        dimension_by_label=build_dimension_by_label(prepared),
        plan=prepared.spec.contraction_plan,
    )
    dimension_by_label = build_dimension_by_label(prepared)
    if engine is EngineName.TENSORKROWCH:
        remaining_operand_states = _build_tensorkrowch_remaining_operand_states(
            prepared
        )
    else:
        remaining_operand_states = {
            operand_id: _CarryOperandState(
                labels=simulation.remaining_operands[operand_id],
                axis_names=simulation.remaining_axis_names[operand_id],
                dimensions=tuple(
                    dimension_by_label[label]
                    for label in simulation.remaining_operands[operand_id]
                ),
            )
            for operand_id in simulation.remaining_operand_ids
        }
    return _build_remaining_label_expression_map(
        remaining_operand_ids=simulation.remaining_operand_ids,
        remaining_operand_states=remaining_operand_states,
        engine=engine,
        base_operand_expressions={
            tensor.spec.id: tensor_collection_reference_by_id(
                prepared,
                tensor.spec.id,
                collection_format,
                collection_name,
            )
            for tensor in prepared.tensors
        },
        step_result_indexes={
            step.step_id: result_index
            for result_index, step in enumerate(simulation.steps)
        },
        latest_result_index=len(simulation.steps) - 1 if simulation.steps else None,
    )


def _build_tensorkrowch_remaining_operand_states(
    prepared: PreparedNetwork,
) -> dict[str, _CarryOperandState]:
    """Simulate manual-plan axis names as TensorKrowch keeps them at runtime."""
    remaining_operand_states = {
        tensor.spec.id: _CarryOperandState(
            labels=tuple(index.label for index in tensor.indices),
            axis_names=_axis_names_for_engine(
                EngineName.TENSORKROWCH,
                tuple(index.spec.name for index in tensor.indices),
            ),
            dimensions=tuple(index.spec.dimension for index in tensor.indices),
        )
        for tensor in prepared.tensors
    }
    plan = prepared.spec.contraction_plan
    if plan is None or not plan.steps:
        return remaining_operand_states

    for step in plan.steps:
        left_state = remaining_operand_states.pop(step.left_operand_id)
        right_state = remaining_operand_states.pop(step.right_operand_id)
        contracted_labels = set(left_state.labels).intersection(right_state.labels)
        surviving_labels = tuple(
            label for label in left_state.labels if label not in contracted_labels
        ) + tuple(
            label for label in right_state.labels if label not in contracted_labels
        )
        surviving_axis_names = tuple(
            axis_name
            for label, axis_name in zip(
                left_state.labels,
                left_state.axis_names,
                strict=True,
            )
            if label not in contracted_labels
        ) + tuple(
            axis_name
            for label, axis_name in zip(
                right_state.labels,
                right_state.axis_names,
                strict=True,
            )
            if label not in contracted_labels
        )
        surviving_dimensions = tuple(
            dimension
            for label, dimension in zip(
                left_state.labels,
                left_state.dimensions,
                strict=True,
            )
            if label not in contracted_labels
        ) + tuple(
            dimension
            for label, dimension in zip(
                right_state.labels,
                right_state.dimensions,
                strict=True,
            )
            if label not in contracted_labels
        )
        remaining_operand_states = {
            step.id: _CarryOperandState(
                labels=surviving_labels,
                axis_names=_deduplicate_tensorkrowch_axis_names(surviving_axis_names),
                dimensions=surviving_dimensions,
            ),
            **remaining_operand_states,
        }
    return remaining_operand_states


def _deduplicate_tensorkrowch_axis_names(
    axis_names: tuple[str, ...],
) -> tuple[str, ...]:
    """Mirror TensorKrowch's suffixing for duplicate surviving axis names."""
    axis_names = tuple(
        _TENSORKROWCH_AXIS_INDEX_SUFFIX_RE.sub("", axis_name)
        for axis_name in axis_names
    )
    return _deduplicate_axis_names(axis_names)


def _deduplicate_axis_names(
    axis_names: tuple[str, ...],
) -> tuple[str, ...]:
    """Make duplicate axis names unique with stable numeric suffixes."""
    counts = Counter(axis_names)
    seen: dict[str, int] = {}
    resolved_axis_names: list[str] = []
    for axis_name in axis_names:
        if counts[axis_name] == 1:
            resolved_axis_names.append(axis_name)
            continue
        suffix = seen.get(axis_name, 0)
        seen[axis_name] = suffix + 1
        resolved_axis_names.append(f"{axis_name}_{suffix}")
    return tuple(resolved_axis_names)


def _axis_names_for_engine(
    engine: EngineName,
    axis_names: tuple[str, ...],
) -> tuple[str, ...]:
    """Return the runtime axis names produced by the requested backend."""
    if engine is EngineName.TENSORKROWCH:
        return _deduplicate_tensorkrowch_axis_names(axis_names)
    return _deduplicate_axis_names(axis_names)


def _axis_name_for_engine(engine: EngineName, axis_name: str) -> str:
    """Return one runtime axis name for the requested backend."""
    return _axis_names_for_engine(engine, (axis_name,))[0]


def _build_remaining_label_expression_map(
    *,
    remaining_operand_ids: tuple[str, ...],
    remaining_operand_states: dict[str, _CarryOperandState],
    engine: EngineName,
    base_operand_expressions: dict[str, str],
    step_result_indexes: dict[str, int],
    latest_result_index: int | None,
) -> dict[str, str]:
    """Resolve surviving labels from the current operand state mapping."""
    label_expression_by_label: dict[str, str] = {}
    for operand_id in remaining_operand_ids:
        operand_state = remaining_operand_states.get(operand_id)
        if operand_state is None:
            continue
        operand_expression = _operand_expression(
            engine=engine,
            operand_id=operand_id,
            base_operand_expressions=base_operand_expressions,
            step_result_indexes=step_result_indexes,
            latest_result_index=latest_result_index,
        )
        for label, axis_name in zip(
            operand_state.labels,
            operand_state.axis_names,
            strict=True,
        ):
            label_expression_by_label[label] = f"{operand_expression}[{axis_name!r}]"
    return label_expression_by_label


def _operand_expression(
    *,
    engine: EngineName,
    operand_id: str,
    base_operand_expressions: dict[str, str],
    step_result_indexes: dict[str, int],
    latest_result_index: int | None,
) -> str:
    """Resolve one simulated operand id to the generated Python expression."""
    if operand_id in base_operand_expressions:
        return base_operand_expressions[operand_id]
    if operand_id not in step_result_indexes:
        raise CodeGenerationError(
            f"Operand '{operand_id}' is not available while rendering linear periodic code."
        )
    return render_results_list_reference(
        step_result_indexes[operand_id],
        latest_result_index=latest_result_index,
    )


def _generate_quimb_linear_periodic_code(
    *,
    chain: LinearPeriodicChainSpec,
    collection_format: TensorCollectionFormat,
) -> CodegenResult:
    """Generate non-carry linear-periodic code for the ``quimb`` backend."""
    lines = [
        "# Tensor Network Editor linear periodic mode",
        "import numpy as np",
        "import quimb.tensor as qtn",
        "",
        "def interface_label(left_cell: int, right_cell: int, slot_index: int) -> str:",
        "    return f'lp_link_{left_cell}_{right_cell}_{slot_index}'",
        "",
        "def cell_label(cell_kind: str, cell_index: int, label_name: str) -> str:",
        "    return f'lp_{cell_kind}_{cell_index}_{label_name}'",
        "",
    ]
    helper_signature_by_cell_name: dict[LinearPeriodicCellName, str] = {
        LinearPeriodicCellName.INITIAL: "",
        LinearPeriodicCellName.PERIODIC: "cell_index: int",
        LinearPeriodicCellName.FINAL: "cell_index: int",
    }
    helper_name_by_cell_name: dict[LinearPeriodicCellName, str] = {
        LinearPeriodicCellName.INITIAL: "build_initial_cell",
        LinearPeriodicCellName.PERIODIC: "build_periodic_cell",
        LinearPeriodicCellName.FINAL: "build_final_cell",
    }
    for cell_name in (
        LinearPeriodicCellName.INITIAL,
        LinearPeriodicCellName.PERIODIC,
        LinearPeriodicCellName.FINAL,
    ):
        lines.extend(
            _render_quimb_cell_helper(
                chain=chain,
                cell_name=cell_name,
                helper_name=helper_name_by_cell_name[cell_name],
                helper_signature=helper_signature_by_cell_name[cell_name],
                collection_format=collection_format,
            ).lines
        )
        if cell_name is not LinearPeriodicCellName.FINAL:
            lines.append("")
    lines.extend(_render_quimb_main_flow_lines())
    return CodegenResult(engine=EngineName.QUIMB, code="\n".join(lines).strip() + "\n")


def _generate_einsum_linear_periodic_code(
    *,
    chain: LinearPeriodicChainSpec,
    engine: EngineName,
    collection_format: TensorCollectionFormat,
) -> CodegenResult:
    """Generate non-carry linear-periodic code for one einsum backend."""
    generator = _einsum_generator_for_engine(engine)
    lines = [
        "# Tensor Network Editor linear periodic mode",
        generator.import_line,
        "",
        "def interface_label(left_cell: int, slot_index: int) -> int:",
        "    return 1_000_000_000 + left_cell * 10_000 + slot_index",
        "",
        "def initial_label(label_offset: int) -> int:",
        "    return 2_000_000_000 + label_offset",
        "",
        "def periodic_label(cell_index: int, label_offset: int) -> int:",
        "    return 3_000_000_000 + cell_index * 10_000 + label_offset",
        "",
        "def final_label(cell_index: int, label_offset: int) -> int:",
        "    return 4_000_000_000 + cell_index * 10_000 + label_offset",
        "",
    ]
    helper_signature_by_cell_name: dict[LinearPeriodicCellName, str] = {
        LinearPeriodicCellName.INITIAL: "",
        LinearPeriodicCellName.PERIODIC: "cell_index: int",
        LinearPeriodicCellName.FINAL: "cell_index: int",
    }
    helper_name_by_cell_name: dict[LinearPeriodicCellName, str] = {
        LinearPeriodicCellName.INITIAL: "build_initial_cell",
        LinearPeriodicCellName.PERIODIC: "build_periodic_cell",
        LinearPeriodicCellName.FINAL: "build_final_cell",
    }
    for cell_name in (
        LinearPeriodicCellName.INITIAL,
        LinearPeriodicCellName.PERIODIC,
        LinearPeriodicCellName.FINAL,
    ):
        lines.extend(
            _render_einsum_cell_helper(
                chain=chain,
                cell_name=cell_name,
                helper_name=helper_name_by_cell_name[cell_name],
                helper_signature=helper_signature_by_cell_name[cell_name],
                engine=engine,
                collection_format=collection_format,
            ).lines
        )
        if cell_name is not LinearPeriodicCellName.FINAL:
            lines.append("")
    lines.extend(_render_einsum_main_flow_lines(engine))
    return CodegenResult(engine=engine, code="\n".join(lines).strip() + "\n")


def _generate_quimb_linear_periodic_carry_code(
    *,
    chain: LinearPeriodicChainSpec,
    collection_format: TensorCollectionFormat,
) -> CodegenResult:
    """Generate carry-mode linear-periodic code for the ``quimb`` backend."""
    lines = [
        "# Tensor Network Editor linear periodic mode",
        "import numpy as np",
        "import quimb.tensor as qtn",
        "",
        "def interface_label(left_cell: int, right_cell: int, slot_index: int) -> str:",
        "    return f'lp_link_{left_cell}_{right_cell}_{slot_index}'",
        "",
        "def cell_label(cell_kind: str, cell_index: int, label_name: str) -> str:",
        "    return f'lp_{cell_kind}_{cell_index}_{label_name}'",
        "",
    ]
    carry_simulation_by_cell_name = _build_carry_simulation_map(
        chain=chain,
        engine=EngineName.QUIMB,
    )
    helper_signature_by_cell_name: dict[LinearPeriodicCellName, str] = {
        LinearPeriodicCellName.INITIAL: "",
        LinearPeriodicCellName.PERIODIC: "cell_index: int, previous_payload: dict[str, object]",
        LinearPeriodicCellName.FINAL: "cell_index: int, previous_payload: dict[str, object]",
    }
    helper_name_by_cell_name: dict[LinearPeriodicCellName, str] = {
        LinearPeriodicCellName.INITIAL: "build_initial_cell",
        LinearPeriodicCellName.PERIODIC: "build_periodic_cell",
        LinearPeriodicCellName.FINAL: "build_final_cell",
    }
    for cell_name in (
        LinearPeriodicCellName.INITIAL,
        LinearPeriodicCellName.PERIODIC,
        LinearPeriodicCellName.FINAL,
    ):
        lines.extend(
            _render_quimb_carry_cell_helper(
                chain=chain,
                cell_name=cell_name,
                helper_name=helper_name_by_cell_name[cell_name],
                helper_signature=helper_signature_by_cell_name[cell_name],
                collection_format=collection_format,
                simulation=carry_simulation_by_cell_name[cell_name],
            ).lines
        )
        if cell_name is not LinearPeriodicCellName.FINAL:
            lines.append("")
    lines.extend(_render_quimb_carry_main_flow_lines())
    return CodegenResult(engine=EngineName.QUIMB, code="\n".join(lines).strip() + "\n")


def _generate_einsum_linear_periodic_carry_code(
    *,
    chain: LinearPeriodicChainSpec,
    engine: EngineName,
    collection_format: TensorCollectionFormat,
) -> CodegenResult:
    """Generate carry-mode linear-periodic code for one einsum backend."""
    generator = _einsum_generator_for_engine(engine)
    lines = [
        "# Tensor Network Editor linear periodic mode",
        generator.import_line,
        "",
        "def interface_label(left_cell: int, slot_index: int) -> int:",
        "    return 1_000_000_000 + left_cell * 10_000 + slot_index",
        "",
        "def initial_label(label_offset: int) -> int:",
        "    return 2_000_000_000 + label_offset",
        "",
        "def periodic_label(cell_index: int, label_offset: int) -> int:",
        "    return 3_000_000_000 + cell_index * 10_000 + label_offset",
        "",
        "def final_label(cell_index: int, label_offset: int) -> int:",
        "    return 4_000_000_000 + cell_index * 10_000 + label_offset",
        "",
    ]
    carry_simulation_by_cell_name = _build_carry_simulation_map(
        chain=chain,
        engine=engine,
    )
    helper_signature_by_cell_name: dict[LinearPeriodicCellName, str] = {
        LinearPeriodicCellName.INITIAL: "",
        LinearPeriodicCellName.PERIODIC: "cell_index: int, previous_payload: dict[str, object]",
        LinearPeriodicCellName.FINAL: "cell_index: int, previous_payload: dict[str, object]",
    }
    helper_name_by_cell_name: dict[LinearPeriodicCellName, str] = {
        LinearPeriodicCellName.INITIAL: "build_initial_cell",
        LinearPeriodicCellName.PERIODIC: "build_periodic_cell",
        LinearPeriodicCellName.FINAL: "build_final_cell",
    }
    for cell_name in (
        LinearPeriodicCellName.INITIAL,
        LinearPeriodicCellName.PERIODIC,
        LinearPeriodicCellName.FINAL,
    ):
        lines.extend(
            _render_einsum_carry_cell_helper(
                chain=chain,
                cell_name=cell_name,
                helper_name=helper_name_by_cell_name[cell_name],
                helper_signature=helper_signature_by_cell_name[cell_name],
                engine=engine,
                collection_format=collection_format,
                simulation=carry_simulation_by_cell_name[cell_name],
            ).lines
        )
        if cell_name is not LinearPeriodicCellName.FINAL:
            lines.append("")
    lines.extend(_render_einsum_carry_main_flow_lines())
    return CodegenResult(engine=engine, code="\n".join(lines).strip() + "\n")


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


def _render_quimb_cell_helper(
    *,
    chain: LinearPeriodicChainSpec,
    cell_name: LinearPeriodicCellName,
    helper_name: str,
    helper_signature: str,
    collection_format: TensorCollectionFormat,
) -> _RenderedCellHelper:
    """Render one non-carry cell helper for ``quimb``."""
    cell = _cell_from_chain(chain, cell_name)
    internal_spec = build_internal_linear_periodic_cell_network(
        cell,
        cell_name=cell_name,
    )
    prepared = prepare_network(internal_spec)
    collection_name = container_name_for_format(collection_format)
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
    label_expression_by_label = _build_quimb_label_expression_map(
        prepared=prepared,
        cell_name=cell_name,
        incoming_ports=incoming_ports,
        outgoing_ports=outgoing_ports,
    )
    interface_index_ids = {
        port.internal_index_id for port in (*incoming_ports, *outgoing_ports)
    }
    generator = QuimbCodeGenerator()
    tensor_value_by_id = {
        tensor.spec.id: (
            f"qtn.Tensor(data=np.zeros({tensor.spec.shape!r}, dtype=float), "
            f"inds={_render_python_tuple_expression([label_expression_by_label[index.label] for index in tensor.indices])}, "
            f"tags={(tensor.spec.name, generator._operand_tag(tensor.spec.id))!r})"
        )
        for tensor in prepared.tensors
    }
    body_lines = render_tensor_collection_assignment(
        collection_name=collection_name,
        collection_format=collection_format,
        prepared=prepared,
        tensor_value_by_id=tensor_value_by_id,
    )
    body_lines.append("")
    body_lines.append(
        "network_tensors = "
        + flattened_tensor_collection_expression(collection_format, collection_name)
    )
    body_lines.append("network = qtn.TensorNetwork(network_tensors)")
    if (
        prepared.spec.contraction_plan is not None
        and prepared.spec.contraction_plan.steps
    ):
        body_lines.extend(
            generator._render_manual_plan(
                prepared=prepared,
                collection_format=collection_format,
                collection_name=collection_name,
            )
        )
        body_lines.append("cell_tensors = list(remaining_operands.values())")
    else:
        body_lines.append("cell_tensors = list(network_tensors)")
    local_open_expressions = [
        label_expression_by_label[index.label]
        for index in prepared.open_indices
        if index.spec.id not in interface_index_ids
    ]
    body_lines.append(
        "open_inds = " + _render_python_tuple_expression(local_open_expressions)
    )
    body_lines.append("result = cell_tensors[0] if len(cell_tensors) == 1 else None")
    body_lines.extend(
        [
            "return {",
            "    'tensors': cell_tensors,",
            "    'open_inds': open_inds,",
            "    'result': result,",
            "}",
        ]
    )
    helper_lines = [f"def {helper_name}({helper_signature}) -> dict[str, object]:"]
    helper_lines.extend([f"    {line}" if line else "" for line in body_lines])
    return _RenderedCellHelper(lines=helper_lines)


def _render_quimb_carry_cell_helper(
    *,
    chain: LinearPeriodicChainSpec,
    cell_name: LinearPeriodicCellName,
    helper_name: str,
    helper_signature: str,
    collection_format: TensorCollectionFormat,
    simulation: _CarryPlanSimulation,
) -> _RenderedCellHelper:
    """Render one carry-mode cell helper for ``quimb``."""
    del chain
    prepared = simulation.prepared
    collection_name = container_name_for_format(collection_format)
    label_expression_by_label = _build_quimb_label_expression_map(
        prepared=prepared,
        cell_name=cell_name,
        incoming_ports=simulation.incoming_ports,
        outgoing_ports=simulation.outgoing_ports,
    )
    generator = QuimbCodeGenerator()
    tensor_value_by_id = {
        tensor.spec.id: (
            f"qtn.Tensor(data=np.zeros({tensor.spec.shape!r}, dtype=float), "
            f"inds={_render_python_tuple_expression([label_expression_by_label[index.label] for index in tensor.indices])}, "
            f"tags={(tensor.spec.name, generator._operand_tag(tensor.spec.id))!r})"
        )
        for tensor in prepared.tensors
    }
    body_lines = render_tensor_collection_assignment(
        collection_name=collection_name,
        collection_format=collection_format,
        prepared=prepared,
        tensor_value_by_id=tensor_value_by_id,
    )
    body_lines.append("")
    body_lines.append(
        "network_tensors = "
        + flattened_tensor_collection_expression(collection_format, collection_name)
    )
    base_operand_expressions = {
        tensor.spec.id: tensor_collection_reference_by_id(
            prepared,
            tensor.spec.id,
            collection_format,
            collection_name,
        )
        for tensor in prepared.tensors
    }
    if simulation.incoming_ports:
        previous_operand_index = simulation.previous_operand_interface_index or 0
        expected_interface = [
            label_expression_by_label[label] for label in simulation.incoming_labels
        ]
        body_lines.extend(
            [
                "previous_interface = list(previous_payload['outgoing_interface'])",
                "previous_operands = list(previous_payload['outgoing_operands'])",
                "expected_interface = "
                + _render_python_list_expression(expected_interface),
                f"if previous_interface != expected_interface or len(previous_operands) != {len(simulation.incoming_ports)}:",
                "    raise ValueError('Previous payload interface does not match this cell.')",
                f"previous_operand = previous_operands[{previous_operand_index}]",
                f"previous_operand.add_tag({generator._operand_tag(LINEAR_PERIODIC_PREVIOUS_OPERAND_ID)!r})",
                "network_tensors = [previous_operand, *network_tensors]",
            ]
        )
        base_operand_expressions[LINEAR_PERIODIC_PREVIOUS_OPERAND_ID] = (
            "previous_operand"
        )
    body_lines.append("network = qtn.TensorNetwork(network_tensors)")
    body_lines.append("results_list = []")
    body_lines.append("")
    for step in simulation.real_steps:
        left_tag = generator._operand_tag(step.left_operand_id)
        right_tag = generator._operand_tag(step.right_operand_id)
        step_tag = generator._operand_tag(step.step_id)
        body_lines.append(f"# Manual step {step.step_id}")
        body_lines.append(f"network.contract_between({left_tag!r}, {right_tag!r})")
        body_lines.append(f"network[{left_tag!r}].add_tag({step_tag!r})")
        body_lines.append(f"results_list.append(network[{step_tag!r}])")
        body_lines.append("")
    latest_result_index = (
        len(simulation.real_steps) - 1 if simulation.real_steps else None
    )
    local_open_expressions = [
        label_expression_by_label[label] for label in simulation.local_open_labels
    ]
    if local_open_expressions:
        body_lines.append(
            "open_inds.extend("
            + _render_python_list_expression(local_open_expressions)
            + ")"
        )
    local_remaining_operand_ids = [
        operand_id
        for operand_id in simulation.remaining_operand_ids
        if operand_id != simulation.carry_operand_id
    ]
    if local_remaining_operand_ids:
        body_lines.append(
            "cell_key_prefix = " + _carry_cell_key_prefix_expression(cell_name)
        )
        for operand_id in local_remaining_operand_ids:
            body_lines.append(
                f'remaining_operands[f"{{cell_key_prefix}}:{operand_id}"] = '
                + _operand_expression(
                    engine=EngineName.QUIMB,
                    operand_id=operand_id,
                    base_operand_expressions=base_operand_expressions,
                    step_result_indexes=simulation.result_index_by_step_id,
                    latest_result_index=latest_result_index,
                )
            )
    if simulation.carry_operand_id is not None:
        outgoing_interface_expressions = [
            label_expression_by_label[label] for label in simulation.outgoing_labels
        ]
        outgoing_operand_expressions = [
            _operand_expression(
                engine=EngineName.QUIMB,
                operand_id=operand_id,
                base_operand_expressions=base_operand_expressions,
                step_result_indexes=simulation.result_index_by_step_id,
                latest_result_index=latest_result_index,
            )
            for operand_id in simulation.outgoing_interface_operand_ids
        ]
        body_lines.extend(
            [
                "outgoing_interface = "
                + _render_python_list_expression(outgoing_interface_expressions),
                "outgoing_operands = "
                + _render_python_list_expression(outgoing_operand_expressions),
                "return {",
                "    'operand': "
                + _operand_expression(
                    engine=EngineName.QUIMB,
                    operand_id=simulation.carry_operand_id,
                    base_operand_expressions=base_operand_expressions,
                    step_result_indexes=simulation.result_index_by_step_id,
                    latest_result_index=latest_result_index,
                )
                + ",",
                "    'outgoing_interface': outgoing_interface,",
                "    'outgoing_operands': outgoing_operands,",
                "}",
            ]
        )
    elif local_remaining_operand_ids:
        body_lines.append(
            "return "
            + _operand_expression(
                engine=EngineName.QUIMB,
                operand_id=local_remaining_operand_ids[0],
                base_operand_expressions=base_operand_expressions,
                step_result_indexes=simulation.result_index_by_step_id,
                latest_result_index=latest_result_index,
            )
        )
    else:
        body_lines.append("return None")
    helper_lines = [
        f"def {helper_name}({helper_signature}) -> dict[str, object] | object | None:"
    ]
    helper_lines.extend([f"    {line}" if line else "" for line in body_lines])
    return _RenderedCellHelper(lines=helper_lines)


def _render_einsum_cell_helper(
    *,
    chain: LinearPeriodicChainSpec,
    cell_name: LinearPeriodicCellName,
    helper_name: str,
    helper_signature: str,
    engine: EngineName,
    collection_format: TensorCollectionFormat,
) -> _RenderedCellHelper:
    """Render one non-carry cell helper for an einsum backend."""
    cell = _cell_from_chain(chain, cell_name)
    internal_spec = build_internal_linear_periodic_cell_network(
        cell,
        cell_name=cell_name,
    )
    prepared = prepare_network(internal_spec)
    collection_name = container_name_for_format(collection_format)
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
    label_expression_by_label = _build_einsum_label_expression_map(
        prepared=prepared,
        cell_name=cell_name,
        incoming_ports=incoming_ports,
        outgoing_ports=outgoing_ports,
    )
    interface_index_ids = {
        port.internal_index_id for port in (*incoming_ports, *outgoing_ports)
    }
    generator = _einsum_generator_for_engine(engine)
    tensor_value_by_id = {
        tensor.spec.id: (
            f"{generator.module_alias}.zeros({tensor.spec.shape!r}"
            f"{generator.zero_initializer_suffix})"
        )
        for tensor in prepared.tensors
    }
    body_lines = render_tensor_collection_assignment(
        collection_name=collection_name,
        collection_format=collection_format,
        prepared=prepared,
        tensor_value_by_id=tensor_value_by_id,
    )
    body_lines.append("")
    if (
        prepared.spec.contraction_plan is not None
        and prepared.spec.contraction_plan.steps
    ):
        body_lines.extend(
            _render_einsum_manual_plan_lines(
                prepared=prepared,
                engine=engine,
                collection_format=collection_format,
                collection_name=collection_name,
                label_expression_by_label=label_expression_by_label,
            )
        )
    else:
        body_lines.extend(
            [
                "cell_operands = []",
                "cell_operand_labels = []",
            ]
        )
        for tensor in prepared.tensors:
            body_lines.append(
                "cell_operands.append("
                + tensor_collection_reference_by_id(
                    prepared,
                    tensor.spec.id,
                    collection_format,
                    collection_name,
                )
                + ")"
            )
            body_lines.append(
                "cell_operand_labels.append("
                + _render_python_list_expression(
                    [label_expression_by_label[index.label] for index in tensor.indices]
                )
                + ")"
            )
    local_open_expressions = [
        label_expression_by_label[index.label]
        for index in prepared.open_indices
        if index.spec.id not in interface_index_ids
    ]
    body_lines.append(
        "open_labels = " + _render_python_list_expression(local_open_expressions)
    )
    body_lines.extend(
        [
            "return {",
            "    'operands': cell_operands,",
            "    'operand_labels': cell_operand_labels,",
            "    'open_labels': open_labels,",
            "}",
        ]
    )
    helper_lines = [f"def {helper_name}({helper_signature}) -> dict[str, object]:"]
    helper_lines.extend([f"    {line}" if line else "" for line in body_lines])
    return _RenderedCellHelper(lines=helper_lines)


def _render_einsum_carry_cell_helper(
    *,
    chain: LinearPeriodicChainSpec,
    cell_name: LinearPeriodicCellName,
    helper_name: str,
    helper_signature: str,
    engine: EngineName,
    collection_format: TensorCollectionFormat,
    simulation: _CarryPlanSimulation,
) -> _RenderedCellHelper:
    """Render one carry-mode cell helper for an einsum backend."""
    del chain
    prepared = simulation.prepared
    collection_name = container_name_for_format(collection_format)
    label_expression_by_label = _build_einsum_label_expression_map(
        prepared=prepared,
        cell_name=cell_name,
        incoming_ports=simulation.incoming_ports,
        outgoing_ports=simulation.outgoing_ports,
    )
    generator = _einsum_generator_for_engine(engine)
    tensor_value_by_id = {
        tensor.spec.id: (
            f"{generator.module_alias}.zeros({tensor.spec.shape!r}"
            f"{generator.zero_initializer_suffix})"
        )
        for tensor in prepared.tensors
    }
    body_lines = render_tensor_collection_assignment(
        collection_name=collection_name,
        collection_format=collection_format,
        prepared=prepared,
        tensor_value_by_id=tensor_value_by_id,
    )
    body_lines.append("")
    base_operand_expressions = {
        tensor.spec.id: tensor_collection_reference_by_id(
            prepared,
            tensor.spec.id,
            collection_format,
            collection_name,
        )
        for tensor in prepared.tensors
    }
    if simulation.incoming_ports:
        previous_operand_index = simulation.previous_operand_interface_index or 0
        expected_interface = [
            label_expression_by_label[label] for label in simulation.incoming_labels
        ]
        body_lines.extend(
            [
                "previous_interface = list(previous_payload['outgoing_interface'])",
                "previous_operands = list(previous_payload['outgoing_operands'])",
                "expected_interface = "
                + _render_python_list_expression(expected_interface),
                "if previous_interface != expected_interface "
                + f"or len(previous_operands) != {len(simulation.incoming_ports)}:",
                "    raise ValueError('Previous payload interface does not match this cell.')",
                f"previous_operand = previous_operands[{previous_operand_index}]",
            ]
        )
        base_operand_expressions[LINEAR_PERIODIC_PREVIOUS_OPERAND_ID] = (
            "previous_operand"
        )
    label_order = list(
        dict.fromkeys(
            index.label for tensor in prepared.tensors for index in tensor.indices
        )
    )
    for step in simulation.real_steps:
        for label in (*step.left_labels, *step.right_labels, *step.surviving_labels):
            if label not in label_order:
                label_order.append(label)
    label_to_int = {label: offset for offset, label in enumerate(label_order)}
    lines_for_plan = ["results_list = []", ""]
    for step_index, step in enumerate(simulation.real_steps):
        latest_result_index = step_index - 1 if step_index > 0 else None
        lines_for_plan.append(f"# Manual step {step.step_id}")
        lines_for_plan.append(
            "results_list.append("
            + generator._render_manual_step_call(
                left_expression=generator._operand_expression(
                    step.left_operand_id,
                    base_operand_expressions=base_operand_expressions,
                    step_result_indexes=simulation.result_index_by_step_id,
                    latest_result_index=latest_result_index,
                ),
                right_expression=generator._operand_expression(
                    step.right_operand_id,
                    base_operand_expressions=base_operand_expressions,
                    step_result_indexes=simulation.result_index_by_step_id,
                    latest_result_index=latest_result_index,
                ),
                left_labels=step.left_labels,
                right_labels=step.right_labels,
                output_labels=step.surviving_labels,
                use_string_labels=False,
                symbol_map={},
                label_to_int=label_to_int,
            )
            + ")"
        )
        lines_for_plan.append("")
    body_lines.extend(lines_for_plan)
    latest_result_index = (
        len(simulation.real_steps) - 1 if simulation.real_steps else None
    )
    local_open_expressions = [
        label_expression_by_label[label] for label in simulation.local_open_labels
    ]
    if local_open_expressions:
        body_lines.append(
            "open_labels.extend("
            + _render_python_list_expression(local_open_expressions)
            + ")"
        )
    local_remaining_operand_ids = [
        operand_id
        for operand_id in simulation.remaining_operand_ids
        if operand_id != simulation.carry_operand_id
    ]
    if local_remaining_operand_ids:
        body_lines.append(
            "cell_key_prefix = " + _carry_cell_key_prefix_expression(cell_name)
        )
        for operand_id in local_remaining_operand_ids:
            body_lines.append(
                f'remaining_operands[f"{{cell_key_prefix}}:{operand_id}"] = '
                + _operand_expression(
                    engine=engine,
                    operand_id=operand_id,
                    base_operand_expressions=base_operand_expressions,
                    step_result_indexes=simulation.result_index_by_step_id,
                    latest_result_index=latest_result_index,
                )
            )
    if simulation.carry_operand_id is not None:
        outgoing_interface_expressions = [
            label_expression_by_label[label] for label in simulation.outgoing_labels
        ]
        outgoing_operand_expressions = [
            _operand_expression(
                engine=engine,
                operand_id=operand_id,
                base_operand_expressions=base_operand_expressions,
                step_result_indexes=simulation.result_index_by_step_id,
                latest_result_index=latest_result_index,
            )
            for operand_id in simulation.outgoing_interface_operand_ids
        ]
        body_lines.extend(
            [
                "outgoing_interface = "
                + _render_python_list_expression(outgoing_interface_expressions),
                "outgoing_operands = "
                + _render_python_list_expression(outgoing_operand_expressions),
                "return {",
                "    'operand': "
                + _operand_expression(
                    engine=engine,
                    operand_id=simulation.carry_operand_id,
                    base_operand_expressions=base_operand_expressions,
                    step_result_indexes=simulation.result_index_by_step_id,
                    latest_result_index=latest_result_index,
                )
                + ",",
                "    'outgoing_interface': outgoing_interface,",
                "    'outgoing_operands': outgoing_operands,",
                "}",
            ]
        )
    elif local_remaining_operand_ids:
        body_lines.append(
            "return "
            + _operand_expression(
                engine=engine,
                operand_id=local_remaining_operand_ids[0],
                base_operand_expressions=base_operand_expressions,
                step_result_indexes=simulation.result_index_by_step_id,
                latest_result_index=latest_result_index,
            )
        )
    else:
        body_lines.append("return None")
    helper_lines = [
        f"def {helper_name}({helper_signature}) -> dict[str, object] | object | None:"
    ]
    helper_lines.extend([f"    {line}" if line else "" for line in body_lines])
    return _RenderedCellHelper(lines=helper_lines)


def _render_quimb_main_flow_lines() -> list[str]:
    """Render the non-carry outer flow for the ``quimb`` backend."""
    return [
        "",
        "if n < 2:",
        "    raise ValueError('n must be at least 2 for a linear periodic chain.')",
        "",
        "initial_cell = build_initial_cell()",
        "network_tensors = list(initial_cell['tensors'])",
        "open_inds = list(initial_cell['open_inds'])",
        "",
        "for cell_index in range(1, n - 1):",
        "    periodic_cell = build_periodic_cell(cell_index)",
        "    network_tensors.extend(periodic_cell['tensors'])",
        "    open_inds.extend(periodic_cell['open_inds'])",
        "",
        "final_cell = build_final_cell(n - 1)",
        "network_tensors.extend(final_cell['tensors'])",
        "open_inds.extend(final_cell['open_inds'])",
        "open_inds = tuple(open_inds)",
        "network = qtn.TensorNetwork(network_tensors)",
        "result = network_tensors[0] if len(network_tensors) == 1 else None",
    ]


def _render_quimb_carry_main_flow_lines() -> list[str]:
    """Render the carry-mode outer flow for the ``quimb`` backend."""
    return [
        "",
        "if n < 2:",
        "    raise ValueError('n must be at least 2 for a linear periodic chain.')",
        "",
        "remaining_operands = {}",
        "open_inds = []",
        "",
        "previous_payload = build_initial_cell()",
        "",
        "for cell_index in range(1, n - 1):",
        "    previous_payload = build_periodic_cell(cell_index, previous_payload)",
        "",
        "result = build_final_cell(n - 1, previous_payload)",
        "network_tensors = list(remaining_operands.values())",
        "if result is not None:",
        "    network_tensors.append(result)",
        "open_inds = tuple(open_inds)",
        "network = qtn.TensorNetwork(network_tensors) if network_tensors else None",
    ]


def _render_einsum_main_flow_lines(engine: EngineName) -> list[str]:
    """Render the non-carry outer flow for one einsum backend."""
    module_alias = _einsum_generator_for_engine(engine).module_alias
    return [
        "",
        "if n < 2:",
        "    raise ValueError('n must be at least 2 for a linear periodic chain.')",
        "",
        "initial_cell = build_initial_cell()",
        "einsum_operands: list[object] = []",
        "output_labels = list(initial_cell['open_labels'])",
        "for operand, operand_labels in zip(",
        "    initial_cell['operands'],",
        "    initial_cell['operand_labels'],",
        "    strict=True,",
        "):",
        "    einsum_operands.append(operand)",
        "    einsum_operands.append(operand_labels)",
        "",
        "for cell_index in range(1, n - 1):",
        "    periodic_cell = build_periodic_cell(cell_index)",
        "    output_labels.extend(periodic_cell['open_labels'])",
        "    for operand, operand_labels in zip(",
        "        periodic_cell['operands'],",
        "        periodic_cell['operand_labels'],",
        "        strict=True,",
        "    ):",
        "        einsum_operands.append(operand)",
        "        einsum_operands.append(operand_labels)",
        "",
        "final_cell = build_final_cell(n - 1)",
        "output_labels.extend(final_cell['open_labels'])",
        "for operand, operand_labels in zip(",
        "    final_cell['operands'],",
        "    final_cell['operand_labels'],",
        "    strict=True,",
        "):",
        "    einsum_operands.append(operand)",
        "    einsum_operands.append(operand_labels)",
        "",
        "dense_label_values: list[int] = []",
        "for operand_labels in einsum_operands[1::2]:",
        "    for label in operand_labels:",
        "        if label not in dense_label_values:",
        "            dense_label_values.append(label)",
        "for label in output_labels:",
        "    if label not in dense_label_values:",
        "        dense_label_values.append(label)",
        "dense_label_by_value = {",
        "    label: offset for offset, label in enumerate(dense_label_values)",
        "}",
        "dense_einsum_operands: list[object] = []",
        "for operand_index, operand in enumerate(einsum_operands):",
        "    if operand_index % 2 == 0:",
        "        dense_einsum_operands.append(operand)",
        "        continue",
        "    dense_einsum_operands.append(",
        "        [dense_label_by_value[label] for label in operand]",
        "    )",
        "dense_output_labels = [dense_label_by_value[label] for label in output_labels]",
        f"result = {module_alias}.einsum(*dense_einsum_operands, dense_output_labels)",
    ]


def _render_einsum_carry_main_flow_lines() -> list[str]:
    """Render the carry-mode outer flow for einsum backends."""
    return [
        "",
        "if n < 2:",
        "    raise ValueError('n must be at least 2 for a linear periodic chain.')",
        "",
        "remaining_operands = {}",
        "open_labels = []",
        "",
        "previous_payload = build_initial_cell()",
        "",
        "for cell_index in range(1, n - 1):",
        "    previous_payload = build_periodic_cell(cell_index, previous_payload)",
        "",
        "result = build_final_cell(n - 1, previous_payload)",
    ]


def _build_quimb_label_expression_map(
    *,
    prepared: PreparedNetwork,
    cell_name: LinearPeriodicCellName,
    incoming_ports: tuple[LinearPeriodicInterfacePort, ...],
    outgoing_ports: tuple[LinearPeriodicInterfacePort, ...],
) -> dict[str, str]:
    """Map prepared labels to runtime ``quimb`` index-label expressions."""
    prepared_label_by_index_id = {
        index.spec.id: index.label
        for tensor in prepared.tensors
        for index in tensor.indices
    }
    incoming_slot_by_label = {
        prepared_label_by_index_id[port.internal_index_id]: slot_index
        for slot_index, port in enumerate(incoming_ports)
        if port.internal_index_id in prepared_label_by_index_id
    }
    outgoing_slot_by_label = {
        prepared_label_by_index_id[port.internal_index_id]: slot_index
        for slot_index, port in enumerate(outgoing_ports)
        if port.internal_index_id in prepared_label_by_index_id
    }
    label_expression_by_label: dict[str, str] = {}
    for tensor in prepared.tensors:
        for index in tensor.indices:
            label_expression_by_label[index.label] = _quimb_label_expression(
                cell_name=cell_name,
                label=index.label,
                incoming_slot_by_label=incoming_slot_by_label,
                outgoing_slot_by_label=outgoing_slot_by_label,
            )
    return label_expression_by_label


def _build_einsum_label_expression_map(
    *,
    prepared: PreparedNetwork,
    cell_name: LinearPeriodicCellName,
    incoming_ports: tuple[LinearPeriodicInterfacePort, ...],
    outgoing_ports: tuple[LinearPeriodicInterfacePort, ...],
) -> dict[str, str]:
    """Map prepared labels to runtime integer-label expressions for einsum."""
    prepared_label_by_index_id = {
        index.spec.id: index.label
        for tensor in prepared.tensors
        for index in tensor.indices
    }
    incoming_slot_by_label = {
        prepared_label_by_index_id[port.internal_index_id]: slot_index
        for slot_index, port in enumerate(incoming_ports)
        if port.internal_index_id in prepared_label_by_index_id
    }
    outgoing_slot_by_label = {
        prepared_label_by_index_id[port.internal_index_id]: slot_index
        for slot_index, port in enumerate(outgoing_ports)
        if port.internal_index_id in prepared_label_by_index_id
    }
    local_label_offsets = {
        label: offset
        for offset, label in enumerate(
            dict.fromkeys(
                index.label
                for tensor in prepared.tensors
                for index in tensor.indices
                if index.label not in incoming_slot_by_label
                and index.label not in outgoing_slot_by_label
            )
        )
    }
    label_expression_by_label: dict[str, str] = {}
    for tensor in prepared.tensors:
        for index in tensor.indices:
            label_expression_by_label[index.label] = _einsum_label_expression(
                cell_name=cell_name,
                label=index.label,
                incoming_slot_by_label=incoming_slot_by_label,
                outgoing_slot_by_label=outgoing_slot_by_label,
                local_label_offsets=local_label_offsets,
            )
    return label_expression_by_label


def _render_einsum_manual_plan_lines(
    *,
    prepared: PreparedNetwork,
    engine: EngineName,
    collection_format: TensorCollectionFormat,
    collection_name: str,
    label_expression_by_label: dict[str, str],
) -> list[str]:
    """Render one cell-local manual plan for an einsum backend."""
    generator = _einsum_generator_for_engine(engine)
    label_order = list(
        dict.fromkeys(
            index.label for tensor in prepared.tensors for index in tensor.indices
        )
    )
    label_to_int = {label: offset for offset, label in enumerate(label_order)}
    simulation = simulate_contraction_plan(
        initial_operand_ids=tuple(tensor.spec.id for tensor in prepared.tensors),
        initial_operands=build_initial_operand_labels(prepared),
        initial_axis_names=build_initial_operand_axis_names(prepared),
        dimension_by_label=build_dimension_by_label(prepared),
        plan=prepared.spec.contraction_plan,
    )
    step_result_indexes = {
        step.step_id: result_index for result_index, step in enumerate(simulation.steps)
    }
    base_operand_expressions = {
        tensor.spec.id: tensor_collection_reference_by_id(
            prepared,
            tensor.spec.id,
            collection_format,
            collection_name,
        )
        for tensor in prepared.tensors
    }
    lines = ["results_list = []", "", "cell_operands = []", "cell_operand_labels = []"]
    for step_index, step in enumerate(simulation.steps):
        latest_result_index = step_index - 1 if step_index > 0 else None
        lines.append(f"# Manual step {step.step_id}")
        lines.append(
            "results_list.append("
            + generator._render_manual_step_call(
                left_expression=generator._operand_expression(
                    step.left_operand_id,
                    base_operand_expressions=base_operand_expressions,
                    step_result_indexes=step_result_indexes,
                    latest_result_index=latest_result_index,
                ),
                right_expression=generator._operand_expression(
                    step.right_operand_id,
                    base_operand_expressions=base_operand_expressions,
                    step_result_indexes=step_result_indexes,
                    latest_result_index=latest_result_index,
                ),
                left_labels=step.left_labels,
                right_labels=step.right_labels,
                output_labels=step.surviving_labels,
                use_string_labels=False,
                symbol_map={},
                label_to_int=label_to_int,
            )
            + ")"
        )
        lines.append("")
    latest_result_index = len(simulation.steps) - 1 if simulation.steps else None
    for operand_id in simulation.remaining_operand_ids:
        lines.append(
            "cell_operands.append("
            + generator._operand_expression(
                operand_id,
                base_operand_expressions=base_operand_expressions,
                step_result_indexes=step_result_indexes,
                latest_result_index=latest_result_index,
            )
            + ")"
        )
        lines.append(
            "cell_operand_labels.append("
            + _render_python_list_expression(
                [
                    label_expression_by_label[label]
                    for label in simulation.remaining_operands[operand_id]
                ]
            )
            + ")"
        )
    return lines


def _quimb_label_expression(
    *,
    cell_name: LinearPeriodicCellName,
    label: str,
    incoming_slot_by_label: dict[str, int],
    outgoing_slot_by_label: dict[str, int],
) -> str:
    """Render one runtime ``quimb`` label expression."""
    cell_index_expression = _runtime_cell_index_expression(cell_name)
    if label in incoming_slot_by_label:
        return (
            f"interface_label({cell_index_expression} - 1, "
            f"{cell_index_expression}, {incoming_slot_by_label[label]})"
        )
    if label in outgoing_slot_by_label:
        if cell_name is LinearPeriodicCellName.INITIAL:
            return f"interface_label(0, 1, {outgoing_slot_by_label[label]})"
        return (
            f"interface_label({cell_index_expression}, "
            f"{cell_index_expression} + 1, {outgoing_slot_by_label[label]})"
        )
    return f"cell_label({cell_name.value!r}, {cell_index_expression}, {label!r})"


def _einsum_label_expression(
    *,
    cell_name: LinearPeriodicCellName,
    label: str,
    incoming_slot_by_label: dict[str, int],
    outgoing_slot_by_label: dict[str, int],
    local_label_offsets: dict[str, int],
) -> str:
    """Render one runtime integer-label expression for einsum."""
    cell_index_expression = _runtime_cell_index_expression(cell_name)
    if label in incoming_slot_by_label:
        return f"interface_label({cell_index_expression} - 1, {incoming_slot_by_label[label]})"
    if label in outgoing_slot_by_label:
        if cell_name is LinearPeriodicCellName.INITIAL:
            return f"interface_label(0, {outgoing_slot_by_label[label]})"
        return (
            f"interface_label({cell_index_expression}, {outgoing_slot_by_label[label]})"
        )
    label_offset = local_label_offsets[label]
    if cell_name is LinearPeriodicCellName.INITIAL:
        return f"initial_label({label_offset})"
    if cell_name is LinearPeriodicCellName.PERIODIC:
        return f"periodic_label({cell_index_expression}, {label_offset})"
    return f"final_label({cell_index_expression}, {label_offset})"


def _runtime_cell_index_expression(cell_name: LinearPeriodicCellName) -> str:
    """Return the runtime cell-index expression for the given helper."""
    if cell_name is LinearPeriodicCellName.INITIAL:
        return "0"
    return "cell_index"


def _render_python_tuple_expression(values: list[str]) -> str:
    """Render a Python tuple literal from pre-rendered item expressions."""
    if not values:
        return "()"
    if len(values) == 1:
        return f"({values[0]},)"
    return "(" + ", ".join(values) + ")"


def _render_python_list_expression(values: list[str]) -> str:
    """Render a Python list literal from pre-rendered item expressions."""
    return "[" + ", ".join(values) + "]"


def _einsum_generator_for_engine(
    engine: EngineName,
) -> EinsumNumpyCodeGenerator | EinsumTorchCodeGenerator:
    """Return the concrete einsum generator instance for ``engine``."""
    if engine is EngineName.EINSUM_NUMPY:
        return EinsumNumpyCodeGenerator()
    if engine is EngineName.EINSUM_TORCH:
        return EinsumTorchCodeGenerator()
    raise CodeGenerationError(f"The {engine.value} backend is not an einsum engine.")


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
