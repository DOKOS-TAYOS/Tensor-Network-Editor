from __future__ import annotations

from ..models import CodegenResult, EngineName, NetworkSpec
from .base import CodeGenerator
from .common import prepare_network, tensor_variable_name


class TensorKrowchCodeGenerator(CodeGenerator):
    engine = EngineName.TENSORKROWCH

    def generate(self, spec: NetworkSpec) -> CodegenResult:
        prepared = prepare_network(spec)
        lines = [
            "import torch",
            "import tensorkrowch as tk",
            "",
            "network = tk.TensorNetwork()",
            "",
        ]

        for tensor in prepared.tensors:
            axis_names = tuple(index.spec.name for index in tensor.indices)
            lines.extend(
                [
                    f"{tensor.data_variable_name} = torch.zeros({tensor.spec.shape!r}, dtype=torch.float32)",
                    f"{tensor.variable_name} = tk.Node(",
                    f"    tensor={tensor.data_variable_name},",
                    f"    axes_names={axis_names!r},",
                    f"    name={tensor.spec.name!r},",
                    "    network=network,",
                    ")",
                    "",
                ]
            )

        for edge in prepared.edges:
            left_tensor = tensor_variable_name(prepared, edge.spec.left.tensor_id)
            right_tensor = tensor_variable_name(prepared, edge.spec.right.tensor_id)
            lines.append(f"# {edge.spec.name}")
            lines.append(
                f"{edge.variable_name} = tk.connect("
                f"{left_tensor}[{edge.left.spec.name!r}], {right_tensor}[{edge.right.spec.name!r}])"
            )

        if prepared.edges:
            lines.append("")
        lines.append(
            "open_edges = ("
            + ", ".join(
                f"{tensor_variable_name(prepared, index.tensor.id)}[{index.spec.name!r}]"
                for index in prepared.open_indices
            )
            + ("," if prepared.open_indices else "")
            + ")"
        )

        return CodegenResult(engine=self.engine, code="\n".join(lines).strip() + "\n")
