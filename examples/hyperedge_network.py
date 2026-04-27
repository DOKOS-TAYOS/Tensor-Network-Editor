"""Create a normal-mode network with one first-class hyperedge."""

from __future__ import annotations

from tensor_network_editor import NetworkBuilder, NetworkSpec, render_spec_svg


def build_star_hyperedge_spec() -> NetworkSpec:
    """Build a three-tensor design sharing one logical multiway bond."""
    builder = NetworkBuilder("hyperedge-demo", id="network_hyperedge_demo")
    left = builder.tensor("A", id="tensor_a", position=(120.0, 160.0))
    left.index("i", 2, id="tensor_a_i")
    left.index("h", 3, id="tensor_a_h")

    middle = builder.tensor("B", id="tensor_b", position=(320.0, 80.0))
    middle.index("h", 3, id="tensor_b_h")
    middle.index("j", 5, id="tensor_b_j")

    right = builder.tensor("C", id="tensor_c", position=(520.0, 160.0))
    right.index("h", 3, id="tensor_c_h")
    right.index("k", 7, id="tensor_c_k")

    builder.hyperedge(
        [left["h"], middle["h"], right["h"]],
        id="hyperedge_h",
        name="shared_h",
        hub_offset=(0.0, 32.0),
        metadata={"role": "copy-bond"},
    )
    builder.note(
        "A hyperedge represents one logical bond shared by several indices.",
        id="note_hyperedge",
        position=(80.0, 280.0),
    )
    return builder.build()


def main() -> None:
    """Render the hyperedge network to SVG text and print a short summary."""
    spec = build_star_hyperedge_spec()
    svg = render_spec_svg(spec)

    print(f"OK: built {spec.name!r} with {len(spec.hyperedges)} hyperedge")
    print(f"OK: SVG contains shared_h = {'shared_h' in svg}")


if __name__ == "__main__":
    main()
