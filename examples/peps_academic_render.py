"""Render a small PEPS template to TikZ and Graphviz/DOT."""

from __future__ import annotations

from pathlib import Path

from tensor_network_editor import render_spec_dot, render_spec_tikz
from tensor_network_editor.templates import (
    build_template_spec,
    parse_template_parameters,
)


def main() -> None:
    """Write paper-friendly render outputs for a small PEPS design."""
    parameters = parse_template_parameters(
        "peps_2x2",
        {
            "graph_size": 2,
            "bond_dimension": 3,
            "physical_dimension": 2,
        },
    )
    spec = build_template_spec("peps_2x2", parameters=parameters)
    tikz_path = Path("peps_2x2.tex")
    dot_path = Path("peps_2x2.dot")

    render_spec_tikz(spec, output_path=tikz_path)
    render_spec_dot(spec, output_path=dot_path)

    print(f"OK: rendered {spec.name!r} to {tikz_path} and {dot_path}")


if __name__ == "__main__":
    main()
