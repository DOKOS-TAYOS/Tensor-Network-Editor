from __future__ import annotations

import unittest

from tensor_network_editor.api import generate_code
from tensor_network_editor.models import EngineName
from tests.test_api import build_sample_spec


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

    def test_einsum_codegen_emits_equation_and_zero_arrays(self) -> None:
        result = generate_code(build_sample_spec(), engine=EngineName.EINSUM)

        self.assertIn("import numpy as np", result.code)
        self.assertIn("np.zeros((2, 3)", result.code)
        self.assertIn("np.einsum", result.code)


if __name__ == "__main__":
    unittest.main()
