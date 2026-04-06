from __future__ import annotations

import pytest

from tensor_network_editor.api import generate_code
from tensor_network_editor.models import (
    CanvasPosition,
    EngineName,
    IndexSpec,
    NetworkSpec,
    TensorSpec,
)
from tests.factories import build_sample_spec


def build_many_label_spec() -> NetworkSpec:
    tensors = []
    for tensor_index in range(18):
        tensors.append(
            TensorSpec(
                id=f"tensor_{tensor_index}",
                name=f"T{tensor_index}",
                position=CanvasPosition(x=float(tensor_index * 120), y=0.0),
                indices=[
                    IndexSpec(
                        id=f"tensor_{tensor_index}_i{index_offset}",
                        name=f"i{index_offset}",
                        dimension=2,
                    )
                    for index_offset in range(3)
                ],
            )
        )
    return NetworkSpec(id="many_labels", name="many labels", tensors=tensors)


@pytest.mark.parametrize(
    ("engine", "expected_snippets"),
    [
        (
            EngineName.TENSORNETWORK,
            ["import tensornetwork as tn", "axis_names=['i', 'x']", "tn.connect("],
        ),
        (
            EngineName.QUIMB,
            ["import quimb.tensor as qtn", "qtn.Tensor(", "qtn.TensorNetwork(["],
        ),
        (
            EngineName.TENSORKROWCH,
            ["import tensorkrowch as tk", "network = tk.TensorNetwork()", "tk.connect("],
        ),
        (
            EngineName.EINSUM_NUMPY,
            ["import numpy as np", "np.zeros((2, 3)", "result = np.einsum("],
        ),
        (
            EngineName.EINSUM_TORCH,
            [
                "import torch",
                "torch.zeros((2, 3), dtype=torch.float32)",
                "result = torch.einsum(",
            ],
        ),
    ],
)
def test_generate_code_emits_engine_specific_contracts(
    engine: EngineName,
    expected_snippets: list[str],
) -> None:
    result = generate_code(build_sample_spec(), engine=engine)

    assert result.engine is engine
    assert result.code.endswith("\n")
    for snippet in expected_snippets:
        assert snippet in result.code


@pytest.mark.parametrize(
    "engine",
    [EngineName.EINSUM_NUMPY, EngineName.EINSUM_TORCH],
)
def test_einsum_codegen_uses_integer_sublist_form_for_many_labels(
    engine: EngineName,
) -> None:
    result = generate_code(build_many_label_spec(), engine=engine)

    module_alias = "np" if engine is EngineName.EINSUM_NUMPY else "torch"

    assert "integer-sublist form because the network uses many labels" in result.code
    assert f"result = {module_alias}.einsum(" in result.code
    assert "# Einsum equation:" not in result.code
