"""Common grid-periodic cell lookups for code generation."""

from __future__ import annotations

from ....models import (
    GridPeriodicCellName,
    GridPeriodicGridSpec,
    LinearPeriodicCellSpec,
)


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
