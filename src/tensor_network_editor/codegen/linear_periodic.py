"""Code generation helpers for typed linear periodic-chain specifications."""

from __future__ import annotations

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
from .tensorkrowch import TensorKrowchCodeGenerator
from .tensornetwork import TensorNetworkCodeGenerator


@dataclass(slots=True)
class _RenderedCellHelper:
    """Generated helper function together with interface expressions."""

    lines: list[str]


@dataclass(slots=True, frozen=True)
class _CarryOperandState:
    """Track the current labels and axis names of one carry operand."""

    labels: tuple[str, ...]
    axis_names: tuple[str, ...]


@dataclass(slots=True)
class _CarryPlanSimulation:
    """Prepared rendering state for one carry-mode cell helper."""

    prepared: PreparedNetwork
    real_steps: list[SimulatedContractionStep]
    result_index_by_step_id: dict[str, int]
    remaining_operand_ids: tuple[str, ...]
    remaining_operand_states: dict[str, _CarryOperandState]
    carry_operand_id: str | None
    local_open_labels: tuple[str, ...]
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
) -> tuple[SimulatedContractionStep, tuple[str, ...]]:
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
    return replace(simulation, result_axis_names=result_axis_names), result_axis_names


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
    if engine not in {
        EngineName.TENSORNETWORK,
        EngineName.TENSORKROWCH,
    }:
        raise CodeGenerationError(
            f"The {engine.value} backend does not support linear periodic code generation."
        )

    chain = spec.linear_periodic_chain
    uses_carry_mode = linear_periodic_chain_uses_carry_mode(chain)
    lines = _render_import_lines(engine)
    if not uses_carry_mode:
        lines.extend(_render_connect_helper(engine))

    helper_signature_by_cell_name = {
        LinearPeriodicCellName.INITIAL: (
            "" if uses_carry_mode else "",
            "build_initial_cell",
        ),
        LinearPeriodicCellName.PERIODIC: (
            "cell_index, previous_operand" if uses_carry_mode else "cell_index",
            "build_periodic_cell",
        ),
        LinearPeriodicCellName.FINAL: (
            "previous_operand" if uses_carry_mode else "",
            "build_final_cell",
        ),
    }
    for cell_name in (
        LinearPeriodicCellName.INITIAL,
        LinearPeriodicCellName.PERIODIC,
        LinearPeriodicCellName.FINAL,
    ):
        helper_signature, helper_name = helper_signature_by_cell_name[cell_name]
        helper_renderer = (
            _render_carry_cell_helper if uses_carry_mode else _render_cell_helper
        )
        lines.extend(
            helper_renderer(
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
            "previous_operand = build_initial_cell()",
            "",
            "for cell_index in range(1, n - 1):",
            "    previous_operand = build_periodic_cell(cell_index, previous_operand)",
            "",
            "result = build_final_cell(previous_operand)",
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
) -> _RenderedCellHelper:
    """Render one carry-mode helper that threads ``previous_operand``."""
    cell = _cell_from_chain(chain, cell_name)
    simulation = _simulate_carry_cell(
        cell=cell,
        cell_name=cell_name,
    )
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
        operand_id: _CarryOperandState(
            labels=labels,
            axis_names=axis_names,
        )
        for operand_id, labels, axis_names in zip(
            initial_operand_ids,
            build_initial_operand_labels(prepared).values(),
            build_initial_operand_axis_names(prepared).values(),
            strict=True,
        )
    }
    remaining_operand_ids = list(initial_operand_ids)
    if incoming_labels:
        remaining_operand_states[LINEAR_PERIODIC_PREVIOUS_OPERAND_ID] = (
            _CarryOperandState(
                labels=incoming_labels,
                axis_names=_build_interface_axis_names(ports=incoming_ports),
            )
        )
        remaining_operand_ids.insert(0, LINEAR_PERIODIC_PREVIOUS_OPERAND_ID)
    if outgoing_labels:
        remaining_operand_states[LINEAR_PERIODIC_NEXT_OPERAND_ID] = _CarryOperandState(
            labels=outgoing_labels,
            axis_names=_build_interface_axis_names(ports=outgoing_ports),
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

    for step in cell.contraction_plan.steps:
        left_state = remaining_operand_states.pop(step.left_operand_id, None)
        right_state = remaining_operand_states.pop(step.right_operand_id, None)
        if left_state is None or right_state is None:
            raise CodeGenerationError(
                f"Carry step '{step.id}' in cell '{cell_name.value}' references an unavailable operand."
            )
        simulated_step, result_axis_names = _simulate_carry_step(
            step=step,
            left_state=left_state,
            right_state=right_state,
            dimension_by_label=dimension_by_label,
        )
        remaining_operand_states = {
            step.id: _CarryOperandState(
                labels=simulated_step.surviving_labels,
                axis_names=result_axis_names,
            ),
            **remaining_operand_states,
        }
        remaining_operand_ids = [
            step.id,
            *[
                operand_id
                for operand_id in remaining_operand_ids
                if operand_id not in {step.left_operand_id, step.right_operand_id}
            ],
        ]
        if LINEAR_PERIODIC_NEXT_OPERAND_ID in {
            step.left_operand_id,
            step.right_operand_id,
        }:
            carry_operand_id = step.id
        result_index_by_step_id[step.id] = len(real_steps)
        real_steps.append(simulated_step)

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
        local_open_labels=local_open_labels,
        incoming_ports=incoming_ports,
        outgoing_ports=outgoing_ports,
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
        lines.append("incoming_edges = []")
        for port in simulation.incoming_ports:
            local_tensor = tensor_collection_reference_by_id(
                simulation.prepared,
                port.internal_tensor_id,
                collection_format,
                collection_name,
            )
            if engine is EngineName.TENSORNETWORK:
                lines.append(
                    "incoming_edges.append(tn.connect("
                    f"previous_operand[{port.boundary_index_name!r}], "
                    f"{local_tensor}[{port.internal_index_name!r}], "
                    f"name={port.boundary_index_name!r}))"
                )
            else:
                lines.append(
                    "incoming_edges.append(("
                    f"{port.boundary_index_name!r}, "
                    f"tk.connect(previous_operand[{port.boundary_index_name!r}], "
                    f"{local_tensor}[{port.internal_index_name!r}])"
                    "))"
                )
        lines.append("")

    if simulation.outgoing_ports:
        next_shape = tuple(port.dimension for port in simulation.outgoing_ports)
        next_axis_names = tuple(
            port.boundary_index_name for port in simulation.outgoing_ports
        )
        if engine is EngineName.TENSORNETWORK:
            lines.append(
                "next_boundary_operand = "
                f"tn.Node(np.zeros({next_shape!r}, dtype=float), "
                "name='Next cell', "
                f"axis_names={list(next_axis_names)!r})"
            )
        else:
            lines.append(
                "next_boundary_operand = "
                f"tk.Node(tensor=torch.zeros({next_shape!r}, dtype=torch.float32), "
                f"axes_names={next_axis_names!r}, "
                "name='Next cell', network=network)"
            )
        lines.append("outgoing_edges = []")
        for port in simulation.outgoing_ports:
            local_tensor = tensor_collection_reference_by_id(
                simulation.prepared,
                port.internal_tensor_id,
                collection_format,
                collection_name,
            )
            if engine is EngineName.TENSORNETWORK:
                lines.append(
                    "outgoing_edges.append(tn.connect("
                    f"{local_tensor}[{port.internal_index_name!r}], "
                    f"next_boundary_operand[{port.boundary_index_name!r}], "
                    f"name={port.boundary_index_name!r}))"
                )
            else:
                lines.append(
                    "outgoing_edges.append(("
                    f"{port.boundary_index_name!r}, "
                    f"tk.connect({local_tensor}[{port.internal_index_name!r}], "
                    f"next_boundary_operand[{port.boundary_index_name!r}])"
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
    if simulation.outgoing_ports:
        base_operand_expressions[LINEAR_PERIODIC_NEXT_OPERAND_ID] = (
            "next_boundary_operand"
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
        lines.append(f"return {carry_expression}")
        return lines

    lines.append(
        "return list(remaining_operands.values())[0] if len(remaining_operands) == 1 else None"
    )
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
                    f"name={tensor.spec.name!r}, network=network)"
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
            lines.append(f"# {edge.spec.name}")
            lines.append(
                "edges_list.append(("
                f"{edge.spec.name!r}, "
                f"tk.connect({left_tensor}[{edge.left.spec.name!r}], {right_tensor}[{edge.right.spec.name!r}])"
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
    return _build_remaining_label_expression_map(
        remaining_operand_ids=simulation.remaining_operand_ids,
        remaining_operand_states={
            operand_id: _CarryOperandState(
                labels=simulation.remaining_operands[operand_id],
                axis_names=simulation.remaining_axis_names[operand_id],
            )
            for operand_id in simulation.remaining_operand_ids
        },
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
