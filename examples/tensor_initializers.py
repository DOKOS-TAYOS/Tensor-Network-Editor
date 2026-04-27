"""Attach portable tensor initializers and generate backend code."""

from __future__ import annotations

from tensor_network_editor import EngineName, NetworkBuilder, generate_code
from tensor_network_editor.models import (
    TensorDataDType,
    TensorDataMode,
    TensorDataRandomDistribution,
    TensorDataSpec,
)


def main() -> None:
    """Build a small network that demonstrates built-in tensor-data modes."""
    builder = NetworkBuilder("initializer-demo", id="network_initializer_demo")
    state = builder.tensor(
        "State",
        id="tensor_state",
        position=(120.0, 140.0),
        tensor_data=TensorDataSpec(
            mode=TensorDataMode.ONES,
            dtype=TensorDataDType.FLOAT64,
        ),
    )
    state.index("left", 2, id="tensor_state_left")
    state.index("bond", 3, id="tensor_state_bond")

    gate = builder.tensor(
        "Gate",
        id="tensor_gate",
        position=(340.0, 140.0),
        tensor_data=TensorDataSpec(
            mode=TensorDataMode.FILL,
            fill_value=0.5,
            dtype=TensorDataDType.FLOAT64,
        ),
    )
    gate.index("bond", 3, id="tensor_gate_bond")
    gate.index("right", 4, id="tensor_gate_right")

    probe = builder.tensor(
        "Probe",
        id="tensor_probe",
        position=(560.0, 140.0),
        tensor_data=TensorDataSpec(
            mode=TensorDataMode.RANDOM,
            seed=123,
            distribution=TensorDataRandomDistribution.NORMAL,
            dtype=TensorDataDType.FLOAT32,
        ),
    )
    probe.index("right", 4, id="tensor_probe_right")
    probe.index("out", 2, id="tensor_probe_out")

    builder.connect(state["bond"], gate["bond"], id="edge_bond", name="bond")
    builder.connect(gate["right"], probe["right"], id="edge_right", name="right")

    spec = builder.build()
    result = generate_code(spec, engine=EngineName.EINSUM_NUMPY)

    print(f"OK: built {spec.name!r} with tensor initializers")
    print(f"OK: generated code mentions seed = {'123' in result.code}")


if __name__ == "__main__":
    main()
