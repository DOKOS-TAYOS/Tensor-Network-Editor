"""Code generation helpers for typed linear periodic-chain specifications."""

from __future__ import annotations

from dataclasses import dataclass

from .._contraction_plan import (
    build_dimension_by_label,
    build_initial_operand_axis_names,
    build_initial_operand_labels,
    simulate_contraction_plan,
)
from .._linear_periodic import (
    build_internal_linear_periodic_cell_network,
    build_linear_periodic_interface_ports,
)
from ..errors import CodeGenerationError
from ..models import (
    CodegenResult,
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
    render_tensor_collection_assignment,
    tensor_collection_reference_by_id,
)
from .tensorkrowch import TensorKrowchCodeGenerator
from .tensornetwork import TensorNetworkCodeGenerator


@dataclass(slots=True)
class _RenderedCellHelper:
    """Generated helper function together with interface expressions."""

    lines: list[str]


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
    if engine is EngineName.TENSORNETWORK:
        lines = [
            "# Tensor Network Editor linear periodic mode",
            "import numpy as np",
            "import tensornetwork as tn",
            "",
        ]
        lines.extend(_render_tensornetwork_connect_helper())
    else:
        lines = [
            "# Tensor Network Editor linear periodic mode",
            "import torch",
            "import tensorkrowch as tk",
            "",
            "network = tk.TensorNetwork()",
            "",
        ]
        lines.extend(_render_tensorkrowch_connect_helper())

    lines.extend(
        _render_cell_helper(
            chain=chain,
            cell_name=LinearPeriodicCellName.INITIAL,
            helper_name="build_initial_cell",
            helper_signature="",
            engine=engine,
            collection_format=collection_format,
        ).lines
    )
    lines.append("")
    lines.extend(
        _render_cell_helper(
            chain=chain,
            cell_name=LinearPeriodicCellName.PERIODIC,
            helper_name="build_periodic_cell",
            helper_signature="cell_index",
            engine=engine,
            collection_format=collection_format,
        ).lines
    )
    lines.append("")
    lines.extend(
        _render_cell_helper(
            chain=chain,
            cell_name=LinearPeriodicCellName.FINAL,
            helper_name="build_final_cell",
            helper_signature="",
            engine=engine,
            collection_format=collection_format,
        ).lines
    )
    lines.extend(
        [
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
    )
    return CodegenResult(engine=engine, code="\n".join(lines).strip() + "\n")


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
    """Render one cell helper and the expressions it returns."""
    cell = _cell_from_chain(chain, cell_name)
    internal_spec = build_internal_linear_periodic_cell_network(
        cell, cell_name=cell_name
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


def _render_tensornetwork_cell_body(
    *,
    prepared: PreparedNetwork,
    collection_format: TensorCollectionFormat,
    collection_name: str,
) -> list[str]:
    """Render a periodic-cell helper body for ``tensornetwork``."""
    generator = TensorNetworkCodeGenerator()
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
    latest_result_index = len(simulation.steps) - 1 if simulation.steps else None
    label_expression_by_label: dict[str, str] = {}
    for operand_id in simulation.remaining_operand_ids:
        operand_expression = _operand_expression(
            engine=engine,
            operand_id=operand_id,
            base_operand_expressions=base_operand_expressions,
            step_result_indexes=step_result_indexes,
            latest_result_index=latest_result_index,
        )
        axis_names = simulation.remaining_axis_names[operand_id]
        for label, axis_name in zip(
            simulation.remaining_operands[operand_id],
            axis_names,
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
    if engine is EngineName.TENSORNETWORK:
        return TensorNetworkCodeGenerator._operand_expression(
            operand_id,
            base_operand_expressions=base_operand_expressions,
            step_result_indexes=step_result_indexes,
            latest_result_index=latest_result_index,
        )
    return TensorKrowchCodeGenerator._operand_expression(
        operand_id,
        base_operand_expressions=base_operand_expressions,
        step_result_indexes=step_result_indexes,
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
