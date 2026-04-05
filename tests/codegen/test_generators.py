from __future__ import annotations

import unittest

from tensor_network_editor.api import generate_code
from tensor_network_editor.models import (
    CanvasPosition,
    EngineName,
    IndexSpec,
    NetworkSpec,
    TensorSpec,
)
from tests.test_api import build_sample_spec


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


class CodegenTests(unittest.TestCase):
    def test_tensornetwork_codegen_uses_axis_names_and_connections(self) -> None:
        result = generate_code(build_sample_spec(), engine=EngineName.TENSORNETWORK)

        self.assertIn("import tensornetwork as tn", result.code)
        self.assertIn("axis_names=['i', 'x']", result.code)
        self.assertIn("tn.connect(", result.code)

    def test_quimb_codegen_builds_tensors_from_inds(self) -> None:
        result = generate_code(build_sample_spec(), engine=EngineName.QUIMB)

        self.assertIn("import quimb.tensor as qtn", result.code)
        self.assertIn("qtn.Tensor(", result.code)
        self.assertIn("qtn.TensorNetwork([", result.code)

    def test_tensorkrowch_codegen_builds_nodes_and_connections(self) -> None:
        result = generate_code(build_sample_spec(), engine=EngineName.TENSORKROWCH)

        self.assertIn("import tensorkrowch as tk", result.code)
        self.assertIn("network = tk.TensorNetwork()", result.code)
        self.assertIn("tk.connect(", result.code)

    def test_numpy_einsum_codegen_emits_equation_and_zero_arrays(self) -> None:
        result = generate_code(build_sample_spec(), engine=EngineName.EINSUM_NUMPY)

        self.assertIn("import numpy as np", result.code)
        self.assertIn("np.zeros((2, 3)", result.code)
        self.assertIn("np.einsum", result.code)

    def test_torch_einsum_codegen_emits_equation_and_zero_tensors(self) -> None:
        result = generate_code(build_sample_spec(), engine=EngineName.EINSUM_TORCH)

        self.assertIn("import torch", result.code)
        self.assertIn("torch.zeros((2, 3), dtype=torch.float32)", result.code)
        self.assertIn("torch.einsum", result.code)

    def test_numpy_einsum_codegen_uses_integer_sublist_form_for_many_labels(
        self,
    ) -> None:
        result = generate_code(
            build_many_label_spec(),
            engine=EngineName.EINSUM_NUMPY,
        )

        self.assertIn(
            "integer-sublist form because the network uses many labels",
            result.code,
        )
        self.assertIn("result = np.einsum(", result.code)
        self.assertNotIn("# Einsum equation:", result.code)

    def test_torch_einsum_codegen_uses_integer_sublist_form_for_many_labels(
        self,
    ) -> None:
        result = generate_code(
            build_many_label_spec(),
            engine=EngineName.EINSUM_TORCH,
        )

        self.assertIn(
            "integer-sublist form because the network uses many labels",
            result.code,
        )
        self.assertIn("result = torch.einsum(", result.code)
        self.assertNotIn("# Einsum equation:", result.code)


if __name__ == "__main__":
    unittest.main()
