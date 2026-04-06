from __future__ import annotations

from abc import ABC
from string import ascii_letters

from ..models import CodegenResult, EngineName, NetworkSpec, TensorCollectionFormat
from .base import CodeGenerator
from .common import (
    PreparedTensor,
    container_name_for_format,
    prepare_network,
    render_tensor_collection_assignment,
    tensor_collection_reference,
)


class BaseEinsumCodeGenerator(CodeGenerator, ABC):
    engine: EngineName
    import_line: str
    module_alias: str
    zero_initializer_suffix: str = ""

    def generate(
        self,
        spec: NetworkSpec,
        collection_format: TensorCollectionFormat = TensorCollectionFormat.LIST,
    ) -> CodegenResult:
        prepared = prepare_network(spec)
        label_order: list[str] = []
        for tensor in prepared.tensors:
            for index in tensor.indices:
                if index.label not in label_order:
                    label_order.append(index.label)

        label_to_int = {label: offset for offset, label in enumerate(label_order)}
        output_labels = [index.label for index in prepared.open_indices]

        use_string_equation = len(label_order) <= len(ascii_letters)
        symbol_map = {
            label: ascii_letters[offset]
            for offset, label in enumerate(label_order[: len(ascii_letters)])
        }
        collection_name = container_name_for_format(collection_format)

        lines = [
            self.import_line,
            "",
        ]

        lines.extend(
            render_tensor_collection_assignment(
                collection_name=collection_name,
                collection_format=collection_format,
                prepared=prepared,
                tensor_value_by_id={
                    tensor.spec.id: (
                        f"{self.module_alias}.zeros({tensor.spec.shape!r}"
                        f"{self.zero_initializer_suffix})"
                    )
                    for tensor in prepared.tensors
                },
            )
        )
        lines.append("")

        if use_string_equation:
            equation = self._build_equation(
                tensors=prepared.tensors,
                output_labels=output_labels,
                symbol_map=symbol_map,
            )
            operand_names = ", ".join(
                tensor_collection_reference(tensor, collection_format, collection_name)
                for tensor in prepared.tensors
            )
            lines.append(f"# Einsum equation: {equation}")
            lines.append(
                f"result = {self.module_alias}.einsum({equation!r}, {operand_names})"
            )
        else:
            lines.append(
                "# Einsum uses the integer-sublist form because the network uses many labels."
            )
            sublist_args: list[str] = []
            for tensor in prepared.tensors:
                sublist_args.append(
                    tensor_collection_reference(
                        tensor, collection_format, collection_name
                    )
                )
                sublist_args.append(
                    str([label_to_int[index.label] for index in tensor.indices])
                )
            sublist_args.append(str([label_to_int[label] for label in output_labels]))
            lines.append(
                f"result = {self.module_alias}.einsum(" + ", ".join(sublist_args) + ")"
            )

        return CodegenResult(engine=self.engine, code="\n".join(lines).strip() + "\n")

    @staticmethod
    def _build_equation(
        tensors: list[PreparedTensor],
        output_labels: list[str],
        symbol_map: dict[str, str],
    ) -> str:
        input_terms = [
            "".join(symbol_map[index.label] for index in tensor.indices)
            for tensor in tensors
        ]
        output_term = "".join(symbol_map[label] for label in output_labels)
        return ",".join(input_terms) + "->" + output_term
