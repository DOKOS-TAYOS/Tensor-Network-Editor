from __future__ import annotations

from ..models import CodegenResult, EngineName, NetworkSpec
from .base import CodeGenerator
from .common import prepare_network


class QuimbCodeGenerator(CodeGenerator):
    engine = EngineName.QUIMB

    def generate(self, spec: NetworkSpec) -> CodegenResult:
        prepared = prepare_network(spec)
        lines = [
            "import numpy as np",
            "import quimb.tensor as qtn",
            "",
        ]

        for tensor in prepared.tensors:
            inds = tuple(index.label for index in tensor.indices)
            tags = (tensor.spec.name,)
            lines.extend(
                [
                    f"{tensor.data_variable_name} = np.zeros({tensor.spec.shape!r}, dtype=float)",
                    (
                        f"{tensor.variable_name} = qtn.Tensor("
                        f"data={tensor.data_variable_name}, inds={inds!r}, tags={tags!r})"
                    ),
                    "",
                ]
            )

        lines.append(
            "network = qtn.TensorNetwork(["
            + ", ".join(tensor.variable_name for tensor in prepared.tensors)
            + "])"
        )
        if prepared.open_indices:
            lines.append(
                "open_inds = ("
                + ", ".join(repr(index.label) for index in prepared.open_indices)
                + ",)"
            )
        else:
            lines.append("open_inds = ()")

        return CodegenResult(engine=self.engine, code="\n".join(lines).strip() + "\n")
