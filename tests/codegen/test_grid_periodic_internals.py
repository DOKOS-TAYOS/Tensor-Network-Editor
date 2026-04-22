from __future__ import annotations

from tensor_network_editor.models import GridPeriodicCellName
from tests.factories import build_grid_periodic_grid_spec


def test_grid_periodic_array_renderer_facade_reexports_internal_entrypoint() -> None:
    from tensor_network_editor.codegen.modes import (
        _grid_periodic_array_renderers as facade,
    )
    from tensor_network_editor.codegen.modes._grid_periodic.array_common import (
        generate_array_grid_periodic_code as implementation,
    )

    assert facade.generate_array_grid_periodic_code is implementation


def test_grid_periodic_graph_renderer_facade_reexports_internal_entrypoint() -> None:
    from tensor_network_editor.codegen.modes import (
        _grid_periodic_graph_renderers as facade,
    )
    from tensor_network_editor.codegen.modes._grid_periodic.graph_common import (
        generate_graph_grid_periodic_code as implementation,
    )

    assert facade.generate_graph_grid_periodic_code is implementation


def test_grid_periodic_common_cell_lookup_matches_named_cells() -> None:
    from tensor_network_editor.codegen.modes._grid_periodic.common import (
        _cell_from_grid,
    )

    grid = build_grid_periodic_grid_spec().grid_periodic_grid
    assert grid is not None

    assert _cell_from_grid(grid, GridPeriodicCellName.TOP_LEFT) is grid.top_left_cell
    assert _cell_from_grid(grid, GridPeriodicCellName.CENTER) is grid.center_cell
    assert _cell_from_grid(grid, GridPeriodicCellName.BOTTOM_RIGHT) is (
        grid.bottom_right_cell
    )


def test_grid_periodic_internal_helpers_keep_shared_labels_and_main_flow() -> None:
    from tensor_network_editor.codegen.modes._grid_periodic.array_common import (
        _render_einsum_main_loop_lines,
        _render_einsum_shared_helper_lines,
        _render_quimb_shared_helper_lines,
    )
    from tensor_network_editor.codegen.modes._grid_periodic.graph_common import (
        _render_main_flow_lines,
    )

    quimb_shared = _render_quimb_shared_helper_lines()
    einsum_shared = _render_einsum_shared_helper_lines()
    graph_main_lines, graph_output_lines = _render_main_flow_lines()
    einsum_main_lines = _render_einsum_main_loop_lines()

    assert (
        "def horizontal_label(column_index: int, row_index: int, slot_index: int) -> str:"
        in quimb_shared
    )
    assert (
        "def local_label(kind_offset: int, column_index: int, row_index: int, label_offset: int) -> int:"
        in einsum_shared
    )
    assert "validate_grid_shape(n, m)" in graph_main_lines
    assert (
        "bottom_right_cell = build_bottom_right_cell(n - 1, m - 1)" in graph_main_lines
    )
    assert graph_output_lines == [
        "result = network_nodes[0] if len(network_nodes) == 1 else None"
    ]
    assert "output_labels.extend(bottom_right_cell['open_labels'])" in einsum_main_lines
