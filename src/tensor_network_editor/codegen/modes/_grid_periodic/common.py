"""Common grid-periodic cell lookups for code generation."""

from __future__ import annotations

from ....models import (
    GridPeriodicCellName,
    GridPeriodicGridSpec,
    LinearPeriodicCellSpec,
)
from .shared import GRID_PERIODIC_CELL_ORDER


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


def _manual_plan_step_ids_for_grid(grid: GridPeriodicGridSpec) -> list[str]:
    """Return saved manual step ids from all grid cells in row-major order."""
    step_ids: list[str] = []
    for cell_name in GRID_PERIODIC_CELL_ORDER:
        plan = _cell_from_grid(grid, cell_name).contraction_plan
        if plan is None:
            continue
        step_ids.extend(step.id for step in plan.steps)
    return step_ids


def _render_partial_network_output_lines(
    *,
    operand_expression: str,
    step_ids: list[str],
    key_prefix: str,
    mode_message: str,
) -> list[str]:
    """Render a stable ``remaining_operands`` export for partial For outputs."""
    lines = [
        f"# {mode_message}",
        *[f"# Manual plan step: {step_id}" for step_id in step_ids],
        "remaining_operands = {",
        f'    f"{key_prefix}:{{operand_index}}": operand',
        f"    for operand_index, operand in enumerate({operand_expression})",
        "}",
        "result = next(iter(remaining_operands.values())) if len(remaining_operands) == 1 else None",
    ]
    return lines
