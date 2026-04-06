from __future__ import annotations

from ..models import CodegenResult, EngineName, NetworkSpec, TensorCollectionFormat
from .base import CodeGenerator
from .common import (
    container_name_for_format,
    flattened_tensor_collection_expression,
    prepare_network,
    render_tensor_collection_assignment,
    tensor_collection_reference_by_id,
)


class TensorNetworkCodeGenerator(CodeGenerator):
    engine = EngineName.TENSORNETWORK

    def generate(
        self,
        spec: NetworkSpec,
        collection_format: TensorCollectionFormat = TensorCollectionFormat.LIST,
    ) -> CodegenResult:
        prepared = prepare_network(spec)
        collection_name = container_name_for_format(collection_format)
        lines = [
            "import numpy as np",
            "import tensornetwork as tn",
            "",
        ]

        lines.extend(
            render_tensor_collection_assignment(
                collection_name=collection_name,
                collection_format=collection_format,
                prepared=prepared,
                tensor_value_by_id={
                    tensor.spec.id: (
                        f"tn.Node(np.zeros({tensor.spec.shape!r}, dtype=float), "
                        f"name={tensor.spec.name!r}, "
                        f"axis_names={[index.spec.name for index in tensor.indices]!r})"
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
            lines.append(
                f"{edge.variable_name} = tn.connect("
                f"{left_tensor}[{edge.left.spec.name!r}], "
                f"{right_tensor}[{edge.right.spec.name!r}], "
                f"name={edge.spec.name!r})"
            )

        if prepared.edges:
            lines.append("")
        lines.append(
            "network_nodes = "
            + flattened_tensor_collection_expression(collection_format, collection_name)
        )
        lines.append(
            "open_edges = ["
            + ", ".join(
                f"{tensor_collection_reference_by_id(prepared, index.tensor.id, collection_format, collection_name)}[{index.spec.name!r}]"
                for index in prepared.open_indices
            )
            + "]"
        )

        return CodegenResult(engine=self.engine, code="\n".join(lines).strip() + "\n")
