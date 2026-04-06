from __future__ import annotations

from ..models import CodegenResult, EngineName, NetworkSpec, TensorCollectionFormat
from .base import CodeGenerator
from .common import (
    container_name_for_format,
    flattened_tensor_collection_expression,
    prepare_network,
    render_tensor_collection_assignment,
)


class QuimbCodeGenerator(CodeGenerator):
    engine = EngineName.QUIMB

    def generate(
        self,
        spec: NetworkSpec,
        collection_format: TensorCollectionFormat = TensorCollectionFormat.LIST,
    ) -> CodegenResult:
        prepared = prepare_network(spec)
        collection_name = container_name_for_format(collection_format)
        lines = [
            "import numpy as np",
            "import quimb.tensor as qtn",
            "",
        ]

        lines.extend(
            render_tensor_collection_assignment(
                collection_name=collection_name,
                collection_format=collection_format,
                prepared=prepared,
                tensor_value_by_id={
                    tensor.spec.id: (
                        f"qtn.Tensor(data=np.zeros({tensor.spec.shape!r}, dtype=float), "
                        f"inds={tuple(index.label for index in tensor.indices)!r}, "
                        f"tags={(tensor.spec.name,)!r})"
                    )
                    for tensor in prepared.tensors
                },
            )
        )
        lines.append("")
        lines.append(
            "network_tensors = "
            + flattened_tensor_collection_expression(collection_format, collection_name)
        )
        lines.append("network = qtn.TensorNetwork(network_tensors)")
        if prepared.open_indices:
            lines.append(
                "open_inds = ("
                + ", ".join(repr(index.label) for index in prepared.open_indices)
                + ",)"
            )
        else:
            lines.append("open_inds = ()")

        return CodegenResult(engine=self.engine, code="\n".join(lines).strip() + "\n")
