"""Shared rendering helpers for bidimensional periodic-grid codegen."""

from __future__ import annotations

from dataclasses import dataclass

from ...models import GridPeriodicCellName
from ..shared.common import CodeSection, render_code_section_lines, render_code_sections

GRID_PERIODIC_CELL_ORDER: tuple[GridPeriodicCellName, ...] = (
    GridPeriodicCellName.TOP_LEFT,
    GridPeriodicCellName.TOP,
    GridPeriodicCellName.TOP_RIGHT,
    GridPeriodicCellName.LEFT,
    GridPeriodicCellName.CENTER,
    GridPeriodicCellName.RIGHT,
    GridPeriodicCellName.BOTTOM_LEFT,
    GridPeriodicCellName.BOTTOM,
    GridPeriodicCellName.BOTTOM_RIGHT,
)

_GRID_PERIODIC_CELL_TITLES: dict[GridPeriodicCellName, str] = {
    GridPeriodicCellName.TOP_LEFT: "Top left cell",
    GridPeriodicCellName.TOP: "Top cell",
    GridPeriodicCellName.TOP_RIGHT: "Top right cell",
    GridPeriodicCellName.LEFT: "Left cell",
    GridPeriodicCellName.CENTER: "Center cell",
    GridPeriodicCellName.RIGHT: "Right cell",
    GridPeriodicCellName.BOTTOM_LEFT: "Bottom left cell",
    GridPeriodicCellName.BOTTOM: "Bottom cell",
    GridPeriodicCellName.BOTTOM_RIGHT: "Bottom right cell",
}

_GRID_PERIODIC_HELPER_NAMES: dict[GridPeriodicCellName, str] = {
    GridPeriodicCellName.TOP_LEFT: "build_top_left_cell",
    GridPeriodicCellName.TOP: "build_top_cell",
    GridPeriodicCellName.TOP_RIGHT: "build_top_right_cell",
    GridPeriodicCellName.LEFT: "build_left_cell",
    GridPeriodicCellName.CENTER: "build_center_cell",
    GridPeriodicCellName.RIGHT: "build_right_cell",
    GridPeriodicCellName.BOTTOM_LEFT: "build_bottom_left_cell",
    GridPeriodicCellName.BOTTOM: "build_bottom_cell",
    GridPeriodicCellName.BOTTOM_RIGHT: "build_bottom_right_cell",
}

_GRID_PERIODIC_HELPER_SIGNATURES: dict[GridPeriodicCellName, str] = {
    GridPeriodicCellName.TOP_LEFT: "",
    GridPeriodicCellName.TOP: "column_index: int",
    GridPeriodicCellName.TOP_RIGHT: "column_index: int",
    GridPeriodicCellName.LEFT: "row_index: int",
    GridPeriodicCellName.CENTER: "column_index: int, row_index: int",
    GridPeriodicCellName.RIGHT: "column_index: int, row_index: int",
    GridPeriodicCellName.BOTTOM_LEFT: "row_index: int",
    GridPeriodicCellName.BOTTOM: "column_index: int, row_index: int",
    GridPeriodicCellName.BOTTOM_RIGHT: "column_index: int, row_index: int",
}


@dataclass(slots=True)
class _RenderedCellHelper:
    """Generated helper function body for one grid cell."""

    lines: list[str]


def grid_periodic_cell_title(cell_name: GridPeriodicCellName) -> str:
    """Return the user-facing section title for one grid cell."""
    return _GRID_PERIODIC_CELL_TITLES[cell_name]


def grid_periodic_helper_name(cell_name: GridPeriodicCellName) -> str:
    """Return the generated helper name for one grid cell."""
    return _GRID_PERIODIC_HELPER_NAMES[cell_name]


def grid_periodic_helper_signature(cell_name: GridPeriodicCellName) -> str:
    """Return the generated helper signature for one grid cell."""
    return _GRID_PERIODIC_HELPER_SIGNATURES[cell_name]


def render_grid_periodic_shared_helpers(*, extra_lines: list[str]) -> list[str]:
    """Render shared top-level helpers plus backend-specific extras."""
    return [
        "def validate_grid_shape(n: int, m: int) -> None:",
        "    if n < 2:",
        "        raise ValueError('n must be at least 2 for a bidimensional periodic grid.')",
        "    if m < 2:",
        "        raise ValueError('m must be at least 2 for a bidimensional periodic grid.')",
        "",
        *extra_lines,
    ]


def render_grid_periodic_helper(
    *,
    helper_name: str,
    helper_signature: str,
    return_annotation: str,
    sections: list[CodeSection],
) -> _RenderedCellHelper:
    """Render one generated helper function with titled body sections."""
    helper_lines = [f"def {helper_name}({helper_signature}) -> {return_annotation}:"]
    body_lines = render_code_section_lines(*sections)
    helper_lines.extend([f"    {line}" if line else "" for line in body_lines])
    return _RenderedCellHelper(lines=helper_lines)


def render_grid_periodic_script(
    *,
    import_lines: list[str],
    shared_helper_lines: list[str],
    cell_lines_by_name: dict[GridPeriodicCellName, list[str]],
    main_loop_lines: list[str],
    output_lines: list[str],
) -> str:
    """Render one grid-periodic script with a fixed top-level section order."""
    sections = [
        CodeSection(title=None, lines=import_lines),
        CodeSection(title="Shared helpers", lines=shared_helper_lines),
    ]
    sections.extend(
        CodeSection(
            title=grid_periodic_cell_title(cell_name),
            lines=cell_lines_by_name[cell_name],
        )
        for cell_name in GRID_PERIODIC_CELL_ORDER
    )
    sections.extend(
        [
            CodeSection(title="Main loop", lines=main_loop_lines),
            CodeSection(title="Outputs", lines=output_lines),
        ]
    )
    return render_code_sections(*sections)
