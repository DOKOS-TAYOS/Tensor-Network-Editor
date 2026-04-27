"""Build an MPS template and generate readable NumPy-einsum code."""

from __future__ import annotations

from tensor_network_editor import EngineName, generate_code
from tensor_network_editor.templates import (
    build_template_spec,
    parse_template_parameters,
)


def main() -> None:
    """Create a small MPS design and print a compact code-generation summary."""
    parameters = parse_template_parameters(
        "mps",
        {
            "graph_size": 4,
            "bond_dimension": 3,
            "physical_dimension": 2,
        },
    )
    spec = build_template_spec("mps", parameters=parameters)
    result = generate_code(spec, engine=EngineName.EINSUM_NUMPY)

    print(f"OK: built {spec.name!r} with {len(spec.tensors)} tensors")
    print(f"OK: generated {result.engine.value} code with {len(result.code)} chars")


if __name__ == "__main__":
    main()
