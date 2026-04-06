from __future__ import annotations

from ..models import CodegenResult, EngineName, NetworkSpec, TensorCollectionFormat
from .base import CodeGenerator
from .common import (
    container_name_for_format,
    prepare_network,
    render_tensor_collection_assignment,
    tensor_collection_reference_by_id,
)


class TensorKrowchCodeGenerator(CodeGenerator):
    engine = EngineName.TENSORKROWCH

    def generate(
        self,
        spec: NetworkSpec,
        collection_format: TensorCollectionFormat = TensorCollectionFormat.LIST,
    ) -> CodegenResult:
        prepared = prepare_network(spec)
        collection_name = container_name_for_format(collection_format)
        lines = [
            "import torch",
            "import tensorkrowch as tk",
            "",
            "network = tk.TensorNetwork()",
            "",
        ]

        lines.extend(
            render_tensor_collection_assignment(
                collection_name=collection_name,
                collection_format=collection_format,
                prepared=prepared,
                tensor_value_by_id={
                    tensor.spec.id: (
                        f"tk.Node(tensor=torch.zeros({tensor.spec.shape!r}, dtype=torch.float32), "
                        f"axes_names={tuple(index.spec.name for index in tensor.indices)!r}, "
                        f"name={tensor.spec.name!r}, network=network)"
                    )
                    for tensor in prepared.tensors
                },
            )
        )
        lines.append("")

        for edge in prepared.edges:
            left_tensor = tensor_collection_reference_by_id(
                prepared, edge.spec.left.tensor_id, collection_format, collection_name
            )
            right_tensor = tensor_collection_reference_by_id(
                prepared, edge.spec.right.tensor_id, collection_format, collection_name
            )
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
                f"{tensor_collection_reference_by_id(prepared, index.tensor.id, collection_format, collection_name)}[{index.spec.name!r}]"
                for index in prepared.open_indices
            )
            + ("," if prepared.open_indices else "")
            + ")"
        )

        return CodegenResult(engine=self.engine, code="\n".join(lines).strip() + "\n")
