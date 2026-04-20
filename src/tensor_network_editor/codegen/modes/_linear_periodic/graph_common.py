"""Shared graph-backend helpers for linear-periodic code generation."""

from __future__ import annotations

from ....internal.analysis._contraction_plan import (
    prepare_contraction_inputs,
    simulate_contraction_plan,
)
from ....models import (
    CodegenResult,
    EngineName,
    LinearPeriodicCellName,
    LinearPeriodicChainSpec,
    TensorCollectionFormat,
)
from ...backends.tensorkrowch import TensorKrowchCodeGenerator
from ...shared._linear_periodic_expressions import (
    _axis_name_for_engine,
    _axis_names_for_engine,
    _build_remaining_label_expression_map,
    _deduplicate_tensorkrowch_axis_names,
)
from ...shared.common import (
    PreparedNetwork,
    render_tensor_collection_assignment,
    render_tensor_collection_initialization,
    tensor_collection_reference_by_id,
)
from .carry import _build_carry_simulation_map, _CarryOperandState, _CarryPlanSimulation
from .common import render_linear_periodic_script, render_linear_periodic_shared_helpers


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
        helper_signature, helper_name = helper_signature_by_cell_name[cell_name]
        if uses_carry_mode:
            from .graph_carry import _render_carry_cell_helper

            return _render_carry_cell_helper(
                cell_name=cell_name,
                helper_name=helper_name,
                helper_signature=helper_signature,
                engine=engine,
                collection_format=collection_format,
                simulation=carry_simulation_by_cell_name[cell_name],
            ).lines
        from .graph_standard import _render_cell_helper

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
