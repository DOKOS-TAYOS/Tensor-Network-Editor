from __future__ import annotations

from ..models import CodegenResult, EngineName, NetworkSpec
from .base import CodeGenerator
from .common import prepare_network, tensor_variable_name


class TensorNetworkCodeGenerator(CodeGenerator):
    engine = EngineName.TENSORNETWORK

    def generate(self, spec: NetworkSpec) -> CodegenResult:
        prepared = prepare_network(spec)
        lines = [
            "import numpy as np",
            "import tensornetwork as tn",
            "",
        ]

        for tensor in prepared.tensors:
            axis_names = [index.spec.name for index in tensor.indices]
            lines.extend(
                [
                    f"{tensor.data_variable_name} = np.zeros({tensor.spec.shape!r}, dtype=float)",
                    (
                        f"{tensor.variable_name} = tn.Node("
                        f"{tensor.data_variable_name}, name={tensor.spec.name!r}, axis_names={axis_names!r})"
                    ),
                    "",
                ]
            )

        for edge in prepared.edges:
            left_tensor = tensor_variable_name(prepared, edge.spec.left.tensor_id)
            right_tensor = tensor_variable_name(prepared, edge.spec.right.tensor_id)
            lines.append(
                f"{edge.variable_name} = tn.connect("
                f"{left_tensor}[{edge.left.spec.name!r}], "
                f"{right_tensor}[{edge.right.spec.name!r}], "
                f"name={edge.spec.name!r})"
            )

        if prepared.edges:
            lines.append("")
        lines.append(
            "network_nodes = ["
            + ", ".join(tensor.variable_name for tensor in prepared.tensors)
            + "]"
        )
        lines.append(
            "open_edges = ["
            + ", ".join(
                f"{tensor_variable_name(prepared, index.tensor.id)}[{index.spec.name!r}]"
                for index in prepared.open_indices
            )
            + "]"
        )

        return CodegenResult(engine=self.engine, code="\n".join(lines).strip() + "\n")
