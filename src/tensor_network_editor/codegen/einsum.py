from __future__ import annotations

from abc import ABC
from string import ascii_letters

from ..models import CodegenResult, EngineName, NetworkSpec
from .base import CodeGenerator
from .common import PreparedTensor, prepare_network


class BaseEinsumCodeGenerator(CodeGenerator, ABC):
    engine: EngineName
    import_line: str
    module_alias: str
    zero_initializer_suffix: str = ""

    def generate(self, spec: NetworkSpec) -> CodegenResult:
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

        lines = [
            self.import_line,
            "",
        ]

        for tensor in prepared.tensors:
            lines.append(
                f"{tensor.data_variable_name} = {self.module_alias}.zeros({tensor.spec.shape!r}{self.zero_initializer_suffix})"
            )
        lines.append("")

        if use_string_equation:
            equation = self._build_equation(
                tensors=prepared.tensors,
                output_labels=output_labels,
                symbol_map=symbol_map,
            )
            operand_names = ", ".join(
                tensor.data_variable_name for tensor in prepared.tensors
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
                sublist_args.append(tensor.data_variable_name)
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
