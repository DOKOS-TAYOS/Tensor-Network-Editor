"""Compare a saved manual contraction plan with available automatic rows."""

from __future__ import annotations

from tensor_network_editor import NetworkBuilder, NetworkSpec, analyze_contraction
from tensor_network_editor.models import ContractionPlanSpec, ContractionStepSpec


def build_manual_chain_spec() -> NetworkSpec:
    """Build a four-tensor chain with an explicit left-to-right manual plan."""
    builder = NetworkBuilder("benchmark-chain", id="network_benchmark_chain")
    first = builder.tensor("A", id="tensor_a", position=(80.0, 120.0))
    first.index("i", 2, id="tensor_a_i")
    first.index("x", 16, id="tensor_a_x")

    second = builder.tensor("B", id="tensor_b", position=(240.0, 120.0))
    second.index("x", 16, id="tensor_b_x")
    second.index("y", 4, id="tensor_b_y")

    third = builder.tensor("C", id="tensor_c", position=(400.0, 120.0))
    third.index("y", 4, id="tensor_c_y")
    third.index("z", 16, id="tensor_c_z")

    fourth = builder.tensor("D", id="tensor_d", position=(560.0, 120.0))
    fourth.index("z", 16, id="tensor_d_z")
    fourth.index("j", 2, id="tensor_d_j")

    builder.connect(first["x"], second["x"], id="edge_x", name="bond_x")
    builder.connect(second["y"], third["y"], id="edge_y", name="bond_y")
    builder.connect(third["z"], fourth["z"], id="edge_z", name="bond_z")

    spec = builder.build()
    spec.contraction_plan = ContractionPlanSpec(
        id="plan_left_to_right",
        name="Left-to-right plan",
        steps=[
            ContractionStepSpec(
                id="step_ab",
                left_operand_id="tensor_a",
                right_operand_id="tensor_b",
            ),
            ContractionStepSpec(
                id="step_abc",
                left_operand_id="step_ab",
                right_operand_id="tensor_c",
            ),
            ContractionStepSpec(
                id="step_abcd",
                left_operand_id="step_abc",
                right_operand_id="tensor_d",
            ),
        ],
    )
    return spec


def main() -> None:
    """Print manual contraction metrics and automatic availability."""
    analysis = analyze_contraction(build_manual_chain_spec(), memory_dtype="float32")
    manual = analysis.manual.summary

    print(f"OK: manual status = {analysis.manual.status}")
    print(f"OK: manual FLOP = {manual.total_estimated_flops}")
    print(f"OK: auto full status = {analysis.automatic_full.status}")


if __name__ == "__main__":
    main()
