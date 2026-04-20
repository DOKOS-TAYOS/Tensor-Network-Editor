"""Small shared render helpers for linear-periodic codegen."""

from __future__ import annotations

from dataclasses import dataclass

from ....models import (
    LinearPeriodicCellName,
    LinearPeriodicCellSpec,
    LinearPeriodicChainSpec,
)
from ...shared.common import (
    CodeSection,
    render_code_section_lines,
    render_code_sections,
)


@dataclass(slots=True)
class _RenderedCellHelper:
    """Generated helper function together with its rendered lines."""

    lines: list[str]


_LINEAR_PERIODIC_CHAIN_LENGTH_ERROR = (
    "n must be at least 2 for a linear periodic chain."
)


def render_linear_periodic_shared_helpers(*, extra_lines: list[str]) -> list[str]:
    """Render shared top-level helpers plus backend-specific extras."""
    return [
        "def validate_chain_length(n: int) -> None:",
        "    if n < 2:",
        f"        raise ValueError({_LINEAR_PERIODIC_CHAIN_LENGTH_ERROR!r})",
        "",
        *extra_lines,
    ]


def render_linear_periodic_helper(
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


def render_linear_periodic_script(
    *,
    import_lines: list[str],
    shared_helper_lines: list[str],
    initial_cell_lines: list[str],
    periodic_cell_lines: list[str],
    final_cell_lines: list[str],
    main_loop_lines: list[str],
    output_lines: list[str],
) -> str:
    """Render one linear-periodic script with a fixed top-level section order."""
    return render_code_sections(
        CodeSection(title=None, lines=import_lines),
        CodeSection(title="Shared helpers", lines=shared_helper_lines),
        CodeSection(title="Initial cell", lines=initial_cell_lines),
        CodeSection(title="Periodic cell", lines=periodic_cell_lines),
        CodeSection(title="Final cell", lines=final_cell_lines),
        CodeSection(title="Main loop", lines=main_loop_lines),
        CodeSection(title="Outputs", lines=output_lines),
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
