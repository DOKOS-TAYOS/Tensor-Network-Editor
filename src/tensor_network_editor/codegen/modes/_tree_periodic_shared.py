"""Shared rendering helpers for tree periodic codegen."""

from __future__ import annotations

from dataclasses import dataclass

from ...models import TreePeriodicCellName
from ..shared.common import CodeSection, render_code_section_lines, render_code_sections

TREE_PERIODIC_CELL_ORDER: tuple[TreePeriodicCellName, ...] = (
    TreePeriodicCellName.ROOT,
    TreePeriodicCellName.BRANCH,
    TreePeriodicCellName.LEAF,
)

_TREE_PERIODIC_CELL_TITLES: dict[TreePeriodicCellName, str] = {
    TreePeriodicCellName.ROOT: "Root cell",
    TreePeriodicCellName.BRANCH: "Branch cell",
    TreePeriodicCellName.LEAF: "Leaf cell",
}

_TREE_PERIODIC_HELPER_NAMES: dict[TreePeriodicCellName, str] = {
    TreePeriodicCellName.ROOT: "build_root_cell",
    TreePeriodicCellName.BRANCH: "build_branch_cell",
    TreePeriodicCellName.LEAF: "build_leaf_cell",
}


@dataclass(slots=True)
class _RenderedTreeCellHelper:
    """Generated helper function body for one tree cell."""

    lines: list[str]


def tree_periodic_cell_title(cell_name: TreePeriodicCellName) -> str:
    """Return the user-facing section title for one tree cell."""
    return _TREE_PERIODIC_CELL_TITLES[cell_name]


def tree_periodic_helper_name(cell_name: TreePeriodicCellName) -> str:
    """Return the generated helper name for one tree cell."""
    return _TREE_PERIODIC_HELPER_NAMES[cell_name]


def tree_periodic_helper_signature(
    cell_name: TreePeriodicCellName,
    *,
    interface_annotation: str,
) -> str:
    """Return the generated helper signature for one tree cell."""
    if cell_name is TreePeriodicCellName.ROOT:
        return ""
    return f"level: int, node_index: int, parent_interface: {interface_annotation}"


def render_tree_periodic_shared_helpers(*, extra_lines: list[str]) -> list[str]:
    """Render shared top-level helpers plus backend-specific extras."""
    return [
        "def validate_tree_depth(n: int) -> None:",
        "    if n < 3:",
        "        raise ValueError('n must be at least 3 for a tree periodic network.')",
        "",
        *extra_lines,
    ]


def render_tree_periodic_helper(
    *,
    helper_name: str,
    helper_signature: str,
    return_annotation: str,
    sections: list[CodeSection],
) -> _RenderedTreeCellHelper:
    """Render one generated helper function with titled body sections."""
    helper_lines = [f"def {helper_name}({helper_signature}) -> {return_annotation}:"]
    body_lines = render_code_section_lines(*sections)
    helper_lines.extend([f"    {line}" if line else "" for line in body_lines])
    return _RenderedTreeCellHelper(lines=helper_lines)


def render_tree_periodic_script(
    *,
    import_lines: list[str],
    shared_helper_lines: list[str],
    cell_lines_by_name: dict[TreePeriodicCellName, list[str]],
    main_loop_lines: list[str],
    output_lines: list[str],
) -> str:
    """Render one tree-periodic script with a fixed top-level section order."""
    sections = [
        CodeSection(title=None, lines=import_lines),
        CodeSection(title="Shared helpers", lines=shared_helper_lines),
    ]
    sections.extend(
        CodeSection(
            title=tree_periodic_cell_title(cell_name),
            lines=cell_lines_by_name[cell_name],
        )
        for cell_name in TREE_PERIODIC_CELL_ORDER
    )
    sections.extend(
        [
            CodeSection(title="Main loop", lines=main_loop_lines),
            CodeSection(title="Outputs", lines=output_lines),
        ]
    )
    return render_code_sections(*sections)
