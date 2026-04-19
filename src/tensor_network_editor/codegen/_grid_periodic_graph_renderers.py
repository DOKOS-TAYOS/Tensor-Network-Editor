"""Graph-backend renderers for bidimensional periodic-grid code generation."""

from __future__ import annotations

from .._grid_periodic import (
    build_grid_periodic_interface_ports,
    build_internal_grid_periodic_cell_network,
)
from ..models import (
    CodegenResult,
    EngineName,
    GridPeriodicCellName,
    GridPeriodicGridSpec,
    GridPeriodicTensorRole,
    LinearPeriodicCellSpec,
    TensorCollectionFormat,
)
from ._grid_periodic_shared import (
    GRID_PERIODIC_CELL_ORDER,
    _RenderedCellHelper,
    grid_periodic_helper_name,
    grid_periodic_helper_signature,
    render_grid_periodic_helper,
    render_grid_periodic_script,
    render_grid_periodic_shared_helpers,
)
from ._linear_periodic_expressions import _axis_name_for_engine
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


def generate_graph_grid_periodic_code(
    *,
    grid: GridPeriodicGridSpec,
    engine: EngineName,
    collection_format: TensorCollectionFormat,
) -> CodegenResult:
    """Generate grid-periodic code for TensorNetwork and TensorKrowch backends."""
    import_lines = _render_import_lines(engine)
    shared_helper_lines = render_grid_periodic_shared_helpers(
        extra_lines=_render_connect_helper(engine)
    )
    cell_lines_by_name = {
        cell_name: _render_cell_helper(
            grid=grid,
            cell_name=cell_name,
            helper_name=grid_periodic_helper_name(cell_name),
            helper_signature=grid_periodic_helper_signature(cell_name),
            engine=engine,
            collection_format=collection_format,
        ).lines
        for cell_name in GRID_PERIODIC_CELL_ORDER
    }
    main_loop_lines, output_lines = _render_main_flow_lines()
    return CodegenResult(
        engine=engine,
        code=render_grid_periodic_script(
            import_lines=import_lines,
            shared_helper_lines=shared_helper_lines,
            cell_lines_by_name=cell_lines_by_name,
            main_loop_lines=main_loop_lines,
            output_lines=output_lines,
        ),
    )


def _render_import_lines(engine: EngineName) -> list[str]:
    """Render the common import prelude for one graph backend."""
    if engine is EngineName.TENSORNETWORK:
        return [
            "# Tensor Network Editor grid periodic mode",
            "import numpy as np",
            "import tensornetwork as tn",
            "",
        ]
    return [
        "# Tensor Network Editor grid periodic mode",
        "import torch",
        "import tensorkrowch as tk",
        "",
        "network = tk.TensorNetwork()",
        "",
    ]


def _render_connect_helper(engine: EngineName) -> list[str]:
    """Render the shared interface-connection helper for one graph backend."""
    connect_call = "tn.connect" if engine is EngineName.TENSORNETWORK else "tk.connect"
    return [
        "def connect_cell_interfaces(source_interface: list[object], target_interface: list[object]) -> None:",
        "    if len(source_interface) != len(target_interface):",
        "        raise ValueError('Cell interfaces must have matching lengths.')",
        "    for source_edge, target_edge in zip(source_interface, target_interface, strict=True):",
        f"        {connect_call}(source_edge, target_edge)",
        "",
    ]


def _render_main_flow_lines() -> tuple[list[str], list[str]]:
    """Render the outer free-``n``/``m`` orchestration block."""
    return (
        [
            "validate_grid_shape(n, m)",
            "cells: dict[tuple[int, int], dict[str, object]] = {}",
            "network_nodes: list[object] = []",
            "open_edges: list[object] = []",
            "",
            "top_left_cell = build_top_left_cell()",
            "cells[(0, 0)] = top_left_cell",
            "network_nodes.extend(top_left_cell['nodes'])",
            "open_edges.extend(top_left_cell['open_edges'])",
            "",
            "for column_index in range(1, n - 1):",
            "    top_cell = build_top_cell(column_index)",
            "    connect_cell_interfaces(cells[(column_index - 1, 0)]['right_interface'], top_cell['left_interface'])",
            "    cells[(column_index, 0)] = top_cell",
            "    network_nodes.extend(top_cell['nodes'])",
            "    open_edges.extend(top_cell['open_edges'])",
            "",
            "top_right_cell = build_top_right_cell(n - 1)",
            "connect_cell_interfaces(cells[(n - 2, 0)]['right_interface'], top_right_cell['left_interface'])",
            "cells[(n - 1, 0)] = top_right_cell",
            "network_nodes.extend(top_right_cell['nodes'])",
            "open_edges.extend(top_right_cell['open_edges'])",
            "",
            "for row_index in range(1, m - 1):",
            "    left_cell = build_left_cell(row_index)",
            "    connect_cell_interfaces(cells[(0, row_index - 1)]['down_interface'], left_cell['up_interface'])",
            "    cells[(0, row_index)] = left_cell",
            "    network_nodes.extend(left_cell['nodes'])",
            "    open_edges.extend(left_cell['open_edges'])",
            "",
            "    for column_index in range(1, n - 1):",
            "        center_cell = build_center_cell(column_index, row_index)",
            "        connect_cell_interfaces(cells[(column_index - 1, row_index)]['right_interface'], center_cell['left_interface'])",
            "        connect_cell_interfaces(cells[(column_index, row_index - 1)]['down_interface'], center_cell['up_interface'])",
            "        cells[(column_index, row_index)] = center_cell",
            "        network_nodes.extend(center_cell['nodes'])",
            "        open_edges.extend(center_cell['open_edges'])",
            "",
            "    right_cell = build_right_cell(n - 1, row_index)",
            "    connect_cell_interfaces(cells[(n - 2, row_index)]['right_interface'], right_cell['left_interface'])",
            "    connect_cell_interfaces(cells[(n - 1, row_index - 1)]['down_interface'], right_cell['up_interface'])",
            "    cells[(n - 1, row_index)] = right_cell",
            "    network_nodes.extend(right_cell['nodes'])",
            "    open_edges.extend(right_cell['open_edges'])",
            "",
            "bottom_left_cell = build_bottom_left_cell(m - 1)",
            "connect_cell_interfaces(cells[(0, m - 2)]['down_interface'], bottom_left_cell['up_interface'])",
            "cells[(0, m - 1)] = bottom_left_cell",
            "network_nodes.extend(bottom_left_cell['nodes'])",
            "open_edges.extend(bottom_left_cell['open_edges'])",
            "",
            "for column_index in range(1, n - 1):",
            "    bottom_cell = build_bottom_cell(column_index, m - 1)",
            "    connect_cell_interfaces(cells[(column_index - 1, m - 1)]['right_interface'], bottom_cell['left_interface'])",
            "    connect_cell_interfaces(cells[(column_index, m - 2)]['down_interface'], bottom_cell['up_interface'])",
            "    cells[(column_index, m - 1)] = bottom_cell",
            "    network_nodes.extend(bottom_cell['nodes'])",
            "    open_edges.extend(bottom_cell['open_edges'])",
            "",
            "bottom_right_cell = build_bottom_right_cell(n - 1, m - 1)",
            "connect_cell_interfaces(cells[(n - 2, m - 1)]['right_interface'], bottom_right_cell['left_interface'])",
            "connect_cell_interfaces(cells[(n - 1, m - 2)]['down_interface'], bottom_right_cell['up_interface'])",
            "cells[(n - 1, m - 1)] = bottom_right_cell",
            "network_nodes.extend(bottom_right_cell['nodes'])",
            "open_edges.extend(bottom_right_cell['open_edges'])",
        ],
        ["result = network_nodes[0] if len(network_nodes) == 1 else None"],
    )


def _render_cell_helper(
    *,
    grid: GridPeriodicGridSpec,
    cell_name: GridPeriodicCellName,
    helper_name: str,
    helper_signature: str,
    engine: EngineName,
    collection_format: TensorCollectionFormat,
) -> _RenderedCellHelper:
    """Render one non-contracting grid cell helper."""
    cell = _cell_from_grid(grid, cell_name)
    internal_spec = build_internal_grid_periodic_cell_network(
        cell,
        cell_name=cell_name,
    )
    prepared = prepare_network(internal_spec)
    collection_name = container_name_for_format(collection_format)
    ports_by_role = {
        role: build_grid_periodic_interface_ports(
            cell,
            cell_name=cell_name,
            role=role,
        )
        for role in GridPeriodicTensorRole
    }
    prepared_label_by_index_id = {
        index.spec.id: index.label
        for tensor in prepared.tensors
        for index in tensor.indices
    }
    interface_index_ids = {
        port.internal_index_id for ports in ports_by_role.values() for port in ports
    }
    tensor_collection_lines = render_tensor_collection_initialization(
        collection_name,
        collection_format,
    )
    tensor_construction_lines = render_tensor_collection_assignment(
        collection_name=collection_name,
        collection_format=collection_format,
        prepared=prepared,
        tensor_value_by_id={
            tensor.spec.id: _tensor_value_expression(
                prepared=prepared,
                tensor_id=tensor.spec.id,
                engine=engine,
            )
            for tensor in prepared.tensors
        },
        include_initialization=False,
    )
    network_connection_lines = _render_network_connection_lines(
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
    output_lines = [
        "network_nodes = "
        + flattened_tensor_collection_expression(collection_format, collection_name)
    ]
    for role in (
        GridPeriodicTensorRole.UP,
        GridPeriodicTensorRole.RIGHT,
        GridPeriodicTensorRole.DOWN,
        GridPeriodicTensorRole.LEFT,
    ):
        interface_expressions = [
            label_expression_by_label[
                prepared_label_by_index_id[port.internal_index_id]
            ]
            for port in ports_by_role[role]
            if port.internal_index_id in prepared_label_by_index_id
        ]
        output_lines.append(
            f"{role.value}_interface = [" + ", ".join(interface_expressions) + "]"
        )
    open_edge_expressions = [
        label_expression_by_label[index.label]
        for index in prepared.open_indices
        if index.spec.id not in interface_index_ids
    ]
    output_lines.append("open_edges = [" + ", ".join(open_edge_expressions) + "]")
    output_lines.extend(
        [
            "return {",
            "    'nodes': network_nodes,",
            "    'up_interface': up_interface,",
            "    'right_interface': right_interface,",
            "    'down_interface': down_interface,",
            "    'left_interface': left_interface,",
            "    'open_edges': open_edges,",
            "}",
        ]
    )
    return render_grid_periodic_helper(
        helper_name=helper_name,
        helper_signature=helper_signature,
        return_annotation="dict[str, object]",
        sections=[
            CodeSection(title="Tensor collection", lines=tensor_collection_lines),
            CodeSection(title="Tensor construction", lines=tensor_construction_lines),
            CodeSection(title="Network connections", lines=network_connection_lines),
            CodeSection(title="Outputs", lines=output_lines),
        ],
    )


def _tensor_value_expression(
    *,
    prepared: PreparedNetwork,
    tensor_id: str,
    engine: EngineName,
) -> str:
    """Render the backend-specific tensor constructor for one tensor id."""
    tensor = next(item for item in prepared.tensors if item.spec.id == tensor_id)
    if engine is EngineName.TENSORNETWORK:
        return (
            f"tn.Node(np.zeros({tensor.spec.shape!r}, dtype=float), "
            f"name={tensor.spec.name!r}, "
            f"axis_names={[index.spec.name for index in tensor.indices]!r})"
        )
    return (
        f"tk.Node(tensor=torch.zeros({tensor.spec.shape!r}, dtype=torch.float32), "
        f"axes_names={tuple(index.spec.name for index in tensor.indices)!r}, "
        f"name={tensor.spec.name!r}, "
        "network=network)"
    )


def _render_network_connection_lines(
    *,
    prepared: PreparedNetwork,
    engine: EngineName,
    collection_format: TensorCollectionFormat,
    collection_name: str,
) -> list[str]:
    """Render the internal edge construction section for one graph backend."""
    if not prepared.edges:
        return []
    lines = ["edges_list = []"]
    connect_prefix = (
        "tn.connect" if engine is EngineName.TENSORNETWORK else "tk.connect"
    )
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
        if engine is EngineName.TENSORNETWORK:
            lines.append(
                "edges_list.append(tn.connect("
                f"{left_tensor}[{edge.left.spec.name!r}], "
                f"{right_tensor}[{edge.right.spec.name!r}], "
                f"name={edge.spec.name!r}))"
            )
            continue
        left_axis_name = _axis_name_for_engine(
            EngineName.TENSORKROWCH, edge.left.spec.name
        )
        right_axis_name = _axis_name_for_engine(
            EngineName.TENSORKROWCH,
            edge.right.spec.name,
        )
        lines.append(
            "edges_list.append(("
            f"{edge.spec.name!r}, "
            f"{connect_prefix}({left_tensor}[{left_axis_name!r}], {right_tensor}[{right_axis_name!r}])"
            "))"
        )
    return lines


def _build_label_expression_map(
    *,
    prepared: PreparedNetwork,
    engine: EngineName,
    collection_format: TensorCollectionFormat,
    collection_name: str,
) -> dict[str, str]:
    """Resolve every open label to the generated Python expression."""
    label_expression_by_label: dict[str, str] = {}
    for tensor in prepared.tensors:
        tensor_reference = tensor_collection_reference_by_id(
            prepared,
            tensor.spec.id,
            collection_format,
            collection_name,
        )
        for index in tensor.indices:
            axis_name = (
                index.spec.name
                if engine is EngineName.TENSORNETWORK
                else _axis_name_for_engine(engine, index.spec.name)
            )
            label_expression_by_label[index.label] = (
                f"{tensor_reference}[{axis_name!r}]"
            )
    return label_expression_by_label


def _cell_from_grid(
    grid: GridPeriodicGridSpec,
    cell_name: GridPeriodicCellName,
) -> LinearPeriodicCellSpec:
    """Return the matching cell from ``grid``."""
    if cell_name is GridPeriodicCellName.TOP_LEFT:
        return grid.top_left_cell
    if cell_name is GridPeriodicCellName.TOP:
        return grid.top_cell
    if cell_name is GridPeriodicCellName.TOP_RIGHT:
        return grid.top_right_cell
    if cell_name is GridPeriodicCellName.LEFT:
        return grid.left_cell
    if cell_name is GridPeriodicCellName.CENTER:
        return grid.center_cell
    if cell_name is GridPeriodicCellName.RIGHT:
        return grid.right_cell
    if cell_name is GridPeriodicCellName.BOTTOM_LEFT:
        return grid.bottom_left_cell
    if cell_name is GridPeriodicCellName.BOTTOM:
        return grid.bottom_cell
    return grid.bottom_right_cell
