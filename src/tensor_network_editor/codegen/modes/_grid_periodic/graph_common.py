"""Shared graph orchestration for periodic-grid code generation."""

from __future__ import annotations

from ....models import (
    CodegenResult,
    EngineName,
    GridPeriodicGridSpec,
    TensorCollectionFormat,
)
from .common import _manual_plan_step_ids_for_grid, _render_partial_network_output_lines
from .graph_cells import _render_cell_helper
from .shared import (
    GRID_PERIODIC_CELL_ORDER,
    grid_periodic_helper_name,
    grid_periodic_helper_signature,
    render_grid_periodic_script,
    render_grid_periodic_shared_helpers,
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
    main_loop_lines, output_lines = _render_main_flow_lines(
        manual_step_ids=_manual_plan_step_ids_for_grid(grid)
    )
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


def _render_main_flow_lines(
    *,
    manual_step_ids: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Render the outer free-``n``/``m`` orchestration block."""
    output_lines = (
        _render_partial_network_output_lines(
            operand_expression="network_nodes",
            step_ids=manual_step_ids,
            key_prefix="grid_node",
            mode_message="Manual grid cell plans may leave a partial network.",
        )
        if manual_step_ids
        else ["result = network_nodes[0] if len(network_nodes) == 1 else None"]
    )
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
        output_lines,
    )
