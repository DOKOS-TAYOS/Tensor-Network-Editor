"""Graph-backend renderers for linear periodic code generation."""

from __future__ import annotations

from .._contraction_plan import (
    prepare_contraction_inputs,
    simulate_contraction_plan,
)
from .._linear_periodic import (
    LINEAR_PERIODIC_PREVIOUS_OPERAND_ID,
    LinearPeriodicTensorRole,
    build_internal_linear_periodic_cell_network,
    build_linear_periodic_interface_ports,
)
from ..errors import CodeGenerationError
from ..models import (
    CodegenResult,
    EngineName,
    LinearPeriodicCellName,
    LinearPeriodicChainSpec,
    TensorCollectionFormat,
)
from ._linear_periodic_expressions import (
    _axis_name_for_engine,
    _axis_names_for_engine,
    _build_remaining_label_expression_map,
    _carry_cell_key_prefix_expression,
    _deduplicate_tensorkrowch_axis_names,
    _operand_expression,
)
from ._linear_periodic_shared import (
    _build_carry_simulation_map,
    _CarryOperandState,
    _CarryPlanSimulation,
    _cell_from_chain,
    _RenderedCellHelper,
    render_linear_periodic_helper,
    render_linear_periodic_script,
    render_linear_periodic_shared_helpers,
)
from .common import (
    CodeSection,
    PreparedNetwork,
    container_name_for_format,
    flattened_tensor_collection_expression,
    prepare_network,
    render_tensor_collection_assignment,
    render_tensor_collection_initialization,
    tensor_collection_reference_by_id,
)
from .tensorkrowch import TensorKrowchCodeGenerator
from .tensornetwork import TensorNetworkCodeGenerator


def generate_graph_linear_periodic_code(
    *,
    chain: LinearPeriodicChainSpec,
    engine: EngineName,
    collection_format: TensorCollectionFormat,
    uses_carry_mode: bool,
) -> CodegenResult:
    """Generate linear-periodic code for TensorNetwork and TensorKrowch backends."""
    import_lines = _render_import_lines(engine)
    shared_helper_lines = _render_shared_helper_lines(
        engine=engine,
        uses_carry_mode=uses_carry_mode,
    )

    carry_simulation_by_cell_name: dict[
        LinearPeriodicCellName, _CarryPlanSimulation
    ] = {}
    if uses_carry_mode:
        carry_simulation_by_cell_name = _build_carry_simulation_map(
            chain=chain,
            engine=engine,
        )

    helper_signature_by_cell_name = {
        LinearPeriodicCellName.INITIAL: ("", "build_initial_cell"),
        LinearPeriodicCellName.PERIODIC: (
            (
                "cell_index: int, previous_payload: dict[str, object]"
                if uses_carry_mode
                else "cell_index: int"
            ),
            "build_periodic_cell",
        ),
        LinearPeriodicCellName.FINAL: (
            "previous_payload: dict[str, object]" if uses_carry_mode else "",
            "build_final_cell",
        ),
    }
    main_loop_lines, output_lines = _render_main_flow_lines(
        uses_carry_mode=uses_carry_mode
    )

    def _render_one_cell(cell_name: LinearPeriodicCellName) -> list[str]:
        """Render helper lines for one chain cell in the selected mode."""
        helper_signature, helper_name = helper_signature_by_cell_name[cell_name]
        if uses_carry_mode:
            return _render_carry_cell_helper(
                chain=chain,
                cell_name=cell_name,
                helper_name=helper_name,
                helper_signature=helper_signature,
                engine=engine,
                collection_format=collection_format,
                simulation=carry_simulation_by_cell_name[cell_name],
            ).lines
        return _render_cell_helper(
            chain=chain,
            cell_name=cell_name,
            helper_name=helper_name,
            helper_signature=helper_signature,
            engine=engine,
            collection_format=collection_format,
        ).lines

    return CodegenResult(
        engine=engine,
        code=render_linear_periodic_script(
            import_lines=import_lines,
            shared_helper_lines=shared_helper_lines,
            initial_cell_lines=_render_one_cell(LinearPeriodicCellName.INITIAL),
            periodic_cell_lines=_render_one_cell(LinearPeriodicCellName.PERIODIC),
            final_cell_lines=_render_one_cell(LinearPeriodicCellName.FINAL),
            main_loop_lines=main_loop_lines,
            output_lines=output_lines,
        ),
    )


def _render_shared_helper_lines(
    *,
    engine: EngineName,
    uses_carry_mode: bool,
) -> list[str]:
    """Render shared top-level helpers for graph backends."""
    extra_lines: list[str] = []
    if not uses_carry_mode:
        extra_lines = _render_connect_helper(engine)
    return render_linear_periodic_shared_helpers(extra_lines=extra_lines)


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


def _render_main_flow_lines(*, uses_carry_mode: bool) -> tuple[list[str], list[str]]:
    """Render the outer free-``n`` orchestration block."""
    if uses_carry_mode:
        return (
            [
                "validate_chain_length(n)",
                "remaining_operands = {}",
                "open_edges = []",
                "",
                "previous_payload = build_initial_cell()",
                "",
                "for cell_index in range(1, n - 1):",
                "    previous_payload = build_periodic_cell(cell_index, previous_payload)",
            ],
            [
                "result = build_final_cell(previous_payload)",
                "network_nodes = list(remaining_operands.values())",
            ],
        )
    return (
        [
            "validate_chain_length(n)",
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
        ],
        [
            "final_cell = build_final_cell()",
            "connect_cell_interfaces(previous_interface, final_cell['incoming_interface'])",
            "network_nodes.extend(final_cell['nodes'])",
            "open_edges.extend(final_cell['open_edges'])",
            "result = network_nodes[0] if len(network_nodes) == 1 else None",
        ],
    )


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
    (
        tensor_collection_lines,
        tensor_construction_lines,
        network_connection_lines,
    ) = _render_cell_setup_sections(
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
    output_lines: list[str]
    if (
        prepared.spec.contraction_plan is not None
        and prepared.spec.contraction_plan.steps
    ):
        generator = (
            TensorNetworkCodeGenerator()
            if engine is EngineName.TENSORNETWORK
            else TensorKrowchCodeGenerator()
        )
        contraction_lines, output_lines = generator._render_manual_plan(
            prepared=prepared,
            collection_format=collection_format,
            collection_name=collection_name,
        )
        if engine is EngineName.TENSORKROWCH:
            output_lines.append("network_nodes = list(remaining_operands.values())")
    else:
        contraction_lines = [
            "network_nodes = "
            + flattened_tensor_collection_expression(collection_format, collection_name)
        ]
        output_lines = []

    output_lines.append(
        "incoming_interface = [" + ", ".join(incoming_expressions) + "]"
    )
    output_lines.append(
        "outgoing_interface = [" + ", ".join(outgoing_expressions) + "]"
    )
    output_lines.append("open_edges = [" + ", ".join(open_edge_expressions) + "]")
    output_lines.extend(
        [
            "return {",
            "    'nodes': network_nodes,",
            "    'incoming_interface': incoming_interface,",
            "    'outgoing_interface': outgoing_interface,",
            "    'open_edges': open_edges,",
            "}",
        ]
    )
    sections = [
        CodeSection(title="Tensor collection", lines=tensor_collection_lines),
        CodeSection(title="Tensor construction", lines=tensor_construction_lines),
        CodeSection(title="Network connections", lines=network_connection_lines),
    ]
    if (
        prepared.spec.contraction_plan is not None
        and prepared.spec.contraction_plan.steps
    ):
        sections.append(
            CodeSection(title="Manual contraction", lines=contraction_lines)
        )
    else:
        sections.append(CodeSection(title="Cell assembly", lines=contraction_lines))
    sections.append(CodeSection(title="Outputs", lines=output_lines))
    return render_linear_periodic_helper(
        helper_name=helper_name,
        helper_signature=helper_signature,
        return_annotation="dict[str, object]",
        sections=sections,
    )


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
    (
        tensor_collection_lines,
        tensor_construction_lines,
        network_connection_lines,
    ) = _render_cell_setup_sections(
        prepared=simulation.prepared,
        engine=engine,
        collection_format=collection_format,
        collection_name=collection_name,
    )
    previous_interface_lines = _render_carry_boundary_setup(
        simulation=simulation,
        engine=engine,
        collection_format=collection_format,
        collection_name=collection_name,
    )
    contraction_lines, output_lines = _render_carry_plan_lines(
        simulation=simulation,
        cell_name=cell_name,
        engine=engine,
        collection_format=collection_format,
        collection_name=collection_name,
    )
    return render_linear_periodic_helper(
        helper_name=helper_name,
        helper_signature=helper_signature,
        return_annotation="dict[str, object] | object | None",
        sections=[
            CodeSection(title="Tensor collection", lines=tensor_collection_lines),
            CodeSection(title="Tensor construction", lines=tensor_construction_lines),
            CodeSection(title="Network connections", lines=network_connection_lines),
            CodeSection(title="Previous interface", lines=previous_interface_lines),
            CodeSection(title="Manual contraction", lines=contraction_lines),
            CodeSection(title="Outputs", lines=output_lines),
        ],
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
) -> tuple[list[str], list[str]]:
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

    contraction_lines: list[str] = []
    for step_index, step in enumerate(simulation.real_steps):
        latest_result_index = step_index - 1 if step_index > 0 else None
        left_expression = _operand_expression(
            operand_id=step.left_operand_id,
            base_operand_expressions=base_operand_expressions,
            step_result_indexes=simulation.result_index_by_step_id,
            latest_result_index=latest_result_index,
        )
        right_expression = _operand_expression(
            operand_id=step.right_operand_id,
            base_operand_expressions=base_operand_expressions,
            step_result_indexes=simulation.result_index_by_step_id,
            latest_result_index=latest_result_index,
        )
        if not contraction_lines:
            contraction_lines.extend(["results_list = []", ""])
        contraction_lines.append(f"# Manual step {step.step_id}")
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
            contraction_lines.append(
                "results_list.append(tn.contract_between("
                f"{left_expression}, "
                f"{right_expression}, "
                f"name={step.step_id!r}, "
                "allow_outer_product=True, "
                f"output_edge_order={output_edge_order}, "
                f"axis_names={list(step.result_axis_names)!r}))"
            )
        else:
            contraction_lines.append(
                "results_list.append(tk.contract_between("
                f"{left_expression}, {right_expression}))"
            )
        contraction_lines.append("")

    final_result_index = (
        len(simulation.real_steps) - 1 if simulation.real_steps else None
    )
    output_lines: list[str] = []
    if engine is EngineName.TENSORKROWCH and simulation.outgoing_interface_operand_ids:
        for operand_id in dict.fromkeys(simulation.outgoing_interface_operand_ids):
            if operand_id not in simulation.result_index_by_step_id:
                continue
            operand_expression = _operand_expression(
                operand_id=operand_id,
                base_operand_expressions=base_operand_expressions,
                step_result_indexes=simulation.result_index_by_step_id,
                latest_result_index=final_result_index,
            )
            output_lines.append(f"{operand_expression}.reattach_edges()")

    label_expression_by_label = _build_remaining_label_expression_map(
        remaining_operand_ids=simulation.remaining_operand_ids,
        remaining_operand_states=simulation.remaining_operand_states,
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
        output_lines.append(
            "open_edges.extend([" + ", ".join(local_open_expressions) + "])"
        )

    local_remaining_operand_ids = [
        operand_id
        for operand_id in simulation.remaining_operand_ids
        if operand_id != simulation.carry_operand_id
    ]
    if local_remaining_operand_ids:
        output_lines.append(
            "cell_key_prefix = " + _carry_cell_key_prefix_expression(cell_name)
        )
        for operand_id in local_remaining_operand_ids:
            operand_expression = _operand_expression(
                operand_id=operand_id,
                base_operand_expressions=base_operand_expressions,
                step_result_indexes=simulation.result_index_by_step_id,
                latest_result_index=final_result_index,
            )
            output_lines.append(
                f'remaining_operands[f"{{cell_key_prefix}}:{operand_id}"] = {operand_expression}'
            )

    if simulation.carry_operand_id is not None:
        carry_expression = _operand_expression(
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
                operand_id=operand_id,
                base_operand_expressions=base_operand_expressions,
                step_result_indexes=simulation.result_index_by_step_id,
                latest_result_index=final_result_index,
            )
            for operand_id in simulation.outgoing_interface_operand_ids
        ]
        output_lines.append(
            "outgoing_interface = [" + ", ".join(outgoing_interface_expressions) + "]"
        )
        output_lines.append(
            "outgoing_operands = [" + ", ".join(outgoing_operand_expressions) + "]"
        )
        output_lines.extend(
            [
                "return {",
                f"    'operand': {carry_expression},",
                "    'outgoing_interface': outgoing_interface,",
                "    'outgoing_operands': outgoing_operands,",
                "}",
            ]
        )
        return contraction_lines, output_lines

    if local_remaining_operand_ids:
        final_expression = _operand_expression(
            operand_id=local_remaining_operand_ids[0],
            base_operand_expressions=base_operand_expressions,
            step_result_indexes=simulation.result_index_by_step_id,
            latest_result_index=final_result_index,
        )
        output_lines.append(f"return {final_expression}")
    else:
        output_lines.append("return None")
    return contraction_lines, output_lines


def _render_cell_setup_sections(
    *,
    prepared: PreparedNetwork,
    engine: EngineName,
    collection_format: TensorCollectionFormat,
    collection_name: str,
) -> tuple[list[str], list[str], list[str]]:
    """Render collection, tensor construction, and edge sections for one backend."""
    if engine is EngineName.TENSORNETWORK:
        return _render_tensornetwork_cell_setup_sections(
            prepared=prepared,
            collection_format=collection_format,
            collection_name=collection_name,
        )
    return _render_tensorkrowch_cell_setup_sections(
        prepared=prepared,
        collection_format=collection_format,
        collection_name=collection_name,
    )


def _render_tensornetwork_cell_setup_sections(
    *,
    prepared: PreparedNetwork,
    collection_format: TensorCollectionFormat,
    collection_name: str,
) -> tuple[list[str], list[str], list[str]]:
    """Render tensor collection and edge sections for ``tensornetwork``."""
    tensor_collection_lines = render_tensor_collection_initialization(
        collection_name,
        collection_format,
    )
    tensor_construction_lines = render_tensor_collection_assignment(
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
        include_initialization=False,
    )
    network_connection_lines: list[str] = []
    if prepared.edges:
        network_connection_lines.append("edges_list = []")
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
            network_connection_lines.append(
                "edges_list.append(tn.connect("
                f"{left_tensor}[{edge.left.spec.name!r}], "
                f"{right_tensor}[{edge.right.spec.name!r}], "
                f"name={edge.spec.name!r}))"
            )
    return tensor_collection_lines, tensor_construction_lines, network_connection_lines


def _render_tensorkrowch_cell_setup_sections(
    *,
    prepared: PreparedNetwork,
    collection_format: TensorCollectionFormat,
    collection_name: str,
) -> tuple[list[str], list[str], list[str]]:
    """Render tensor collection and edge sections for ``tensorkrowch``."""
    tensor_collection_lines = render_tensor_collection_initialization(
        collection_name,
        collection_format,
    )
    tensor_construction_lines = render_tensor_collection_assignment(
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
        include_initialization=False,
    )
    network_connection_lines: list[str] = []
    if prepared.edges:
        network_connection_lines.append("edges_list = []")
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
            network_connection_lines.append(f"# {edge.spec.name}")
            network_connection_lines.append(
                "edges_list.append(("
                f"{edge.spec.name!r}, "
                f"tk.connect({left_tensor}[{left_axis_name!r}], {right_tensor}[{right_axis_name!r}])"
                "))"
            )
    return tensor_collection_lines, tensor_construction_lines, network_connection_lines


def _build_label_expression_map(
    *,
    prepared: PreparedNetwork,
    engine: EngineName,
    collection_format: TensorCollectionFormat,
    collection_name: str,
) -> dict[str, str]:
    """Resolve every surviving open label to the generated Python expression."""
    contraction_inputs = prepare_contraction_inputs(prepared)
    simulation = simulate_contraction_plan(
        initial_operand_ids=contraction_inputs.initial_operand_ids,
        initial_operands=contraction_inputs.initial_operands,
        initial_axis_names=contraction_inputs.initial_axis_names,
        dimension_by_label=contraction_inputs.dimension_by_label,
        plan=prepared.spec.contraction_plan,
    )
    dimension_by_label = contraction_inputs.dimension_by_label
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
