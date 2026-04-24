"""Shared preparation and rendering helpers for backend code generators."""

from __future__ import annotations

from dataclasses import dataclass

from ...internal.analysis._prepared_network import (
    PreparedEdge,
    PreparedIndex,
    PreparedNetwork,
    PreparedTensor,
    group_tensors_by_visual_rows,
    make_unique_identifiers,
    prepare_analyzed_network,
    prepare_network,
    sanitize_identifier,
)
from ...models import TensorCollectionFormat, TensorDataMode

__all__ = [
    "CodeSection",
    "PreparedEdge",
    "PreparedIndex",
    "PreparedNetwork",
    "PreparedTensor",
    "container_name_for_format",
    "flattened_tensor_collection_expression",
    "group_tensors_by_visual_rows",
    "joined_tensor_display_name",
    "make_unique_identifiers",
    "prepare_analyzed_network",
    "prepare_network",
    "render_code_section_lines",
    "render_code_sections",
    "render_tensor_data_assignments",
    "render_tensor_data_expression",
    "render_manual_step_comment",
    "render_operand_expression",
    "render_remaining_operands_mapping",
    "render_results_list_reference",
    "render_tensor_collection_assignment",
    "render_tensor_collection_initialization",
    "sanitize_identifier",
    "tensor_collection_reference",
    "tensor_collection_reference_by_id",
    "tensor_display_name_by_id",
    "tensor_variable_name",
]


@dataclass(slots=True)
class CodeSection:
    """One titled section in a generated code listing."""

    title: str | None
    lines: list[str]


def tensor_variable_name(prepared: PreparedNetwork, tensor_id: str) -> str:
    """Return the generated variable name for ``tensor_id``."""
    try:
        return prepared.tensor_by_id[tensor_id].variable_name
    except KeyError as exc:
        raise KeyError(tensor_id) from exc


def tensor_display_name_by_id(prepared: PreparedNetwork) -> dict[str, str]:
    """Return a readable tensor-display name for each tensor id."""
    return {
        tensor.spec.id: (tensor.spec.name or tensor.variable_name or tensor.spec.id)
        for tensor in prepared.tensors
    }


def joined_tensor_display_name(
    source_tensor_ids: tuple[str, ...],
    tensor_names_by_id: dict[str, str],
) -> str:
    """Join source tensor display names into one readable operand label."""
    return "-".join(
        tensor_names_by_id.get(tensor_id, tensor_id) for tensor_id in source_tensor_ids
    )


def render_results_list_reference(
    result_index: int,
    *,
    latest_result_index: int | None,
) -> str:
    """Render a compact ``results_list`` reference for a step result."""
    if latest_result_index is not None and result_index == latest_result_index:
        return "results_list[-1]"
    return f"results_list[{result_index}]"


def render_operand_expression(
    operand_id: str,
    *,
    base_operand_expressions: dict[str, str],
    step_result_indexes: dict[str, int],
    latest_result_index: int | None,
) -> str:
    """Resolve an operand id to its generated Python expression."""
    if operand_id in base_operand_expressions:
        return base_operand_expressions[operand_id]
    return render_results_list_reference(
        step_result_indexes[operand_id],
        latest_result_index=latest_result_index,
    )


def render_remaining_operands_mapping(
    *,
    operand_ids: tuple[str, ...],
    source_tensor_ids_by_operand_id: dict[str, tuple[str, ...]],
    tensor_names_by_id: dict[str, str],
    base_operand_expressions: dict[str, str],
    step_result_indexes: dict[str, int],
    latest_result_index: int | None,
) -> list[str]:
    """Render the ``remaining_operands`` mapping for partial plans."""
    lines = ["remaining_operands = {"]
    for operand_id in operand_ids:
        operand_expression = render_operand_expression(
            operand_id,
            base_operand_expressions=base_operand_expressions,
            step_result_indexes=step_result_indexes,
            latest_result_index=latest_result_index,
        )
        operand_name = joined_tensor_display_name(
            source_tensor_ids_by_operand_id[operand_id],
            tensor_names_by_id,
        )
        lines.append(f"    {operand_name!r}: {operand_expression},")
    lines.append("}")
    return lines


def render_manual_step_comment(
    step_id: str,
    left_operand_id: str,
    right_operand_id: str,
) -> str:
    """Render a structured manual-step comment for round-trip parsing."""
    return (
        f"# Manual step {step_id} | left={left_operand_id} | right={right_operand_id}"
    )


def container_name_for_format(collection_format: TensorCollectionFormat) -> str:
    """Return the default container variable name for ``collection_format``."""
    if collection_format is TensorCollectionFormat.MATRIX:
        return "tensor_rows"
    if collection_format is TensorCollectionFormat.DICT:
        return "tensors_dict"
    return "tensors"


def render_code_sections(*sections: CodeSection) -> str:
    """Render titled sections into one formatted Python source string."""
    return "\n".join(render_code_section_lines(*sections)).strip() + "\n"


def render_code_section_lines(*sections: CodeSection) -> list[str]:
    """Render titled sections into formatted Python source lines."""
    rendered_lines: list[str] = []
    for section in sections:
        section_lines = _trim_blank_lines(section.lines)
        if not section_lines:
            continue
        if rendered_lines:
            rendered_lines.append("")
        if section.title:
            rendered_lines.append(f"# {section.title}")
        rendered_lines.extend(section_lines)
    return rendered_lines


def render_tensor_collection_initialization(
    collection_name: str,
    collection_format: TensorCollectionFormat,
) -> list[str]:
    """Render the container initialization for the chosen collection format."""
    if collection_format is TensorCollectionFormat.DICT:
        return [f"{collection_name} = {{}}"]
    return [f"{collection_name} = []"]


def tensor_collection_reference(
    tensor: PreparedTensor,
    collection_format: TensorCollectionFormat,
    collection_name: str | None = None,
) -> str:
    """Return the Python expression that references ``tensor`` in the container."""
    resolved_collection_name = collection_name or container_name_for_format(
        collection_format
    )
    if collection_format is TensorCollectionFormat.MATRIX:
        return f"{resolved_collection_name}[{tensor.row_index}][{tensor.column_index}]"
    if collection_format is TensorCollectionFormat.DICT:
        return f"{resolved_collection_name}[{tensor.variable_name!r}]"
    return f"{resolved_collection_name}[{tensor.flat_index}]"


def tensor_collection_reference_by_id(
    prepared: PreparedNetwork,
    tensor_id: str,
    collection_format: TensorCollectionFormat,
    collection_name: str | None = None,
) -> str:
    """Return the Python expression that references a tensor by id."""
    try:
        tensor = prepared.tensor_by_id[tensor_id]
    except KeyError as exc:
        raise KeyError(tensor_id) from exc
    return tensor_collection_reference(tensor, collection_format, collection_name)


def flattened_tensor_collection_expression(
    collection_format: TensorCollectionFormat,
    collection_name: str | None = None,
) -> str:
    """Return an expression that flattens the chosen tensor collection layout."""
    resolved_collection_name = collection_name or container_name_for_format(
        collection_format
    )
    if collection_format is TensorCollectionFormat.MATRIX:
        return f"[tensor for row in {resolved_collection_name} for tensor in row]"
    if collection_format is TensorCollectionFormat.DICT:
        return f"list({resolved_collection_name}.values())"
    return resolved_collection_name


def render_tensor_collection_assignment(
    collection_name: str,
    collection_format: TensorCollectionFormat,
    prepared: PreparedNetwork,
    tensor_value_by_id: dict[str, str],
    *,
    include_initialization: bool = True,
) -> list[str]:
    """Render assignment lines for the requested tensor collection layout."""
    if collection_format is TensorCollectionFormat.MATRIX:
        lines = (
            render_tensor_collection_initialization(collection_name, collection_format)
            if include_initialization
            else []
        )
        for row_index, tensor_row in enumerate(prepared.tensor_rows):
            lines.append(f"{collection_name}.append([])")
            for tensor in tensor_row:
                lines.append(f"# Tensor {_tensor_display_name(tensor)}")
                lines.append(
                    f"{collection_name}[{row_index}].append("
                    f"{tensor_value_by_id[tensor.spec.id]})"
                )
        return lines

    if collection_format is TensorCollectionFormat.DICT:
        lines = (
            render_tensor_collection_initialization(collection_name, collection_format)
            if include_initialization
            else []
        )
        for tensor in prepared.tensors:
            lines.append(f"# Tensor {_tensor_display_name(tensor)}")
            lines.append(
                f"{collection_name}[{tensor.variable_name!r}] = "
                f"{tensor_value_by_id[tensor.spec.id]}"
            )
        return lines

    lines = (
        render_tensor_collection_initialization(collection_name, collection_format)
        if include_initialization
        else []
    )
    for tensor in prepared.tensors:
        lines.append(f"# Tensor {_tensor_display_name(tensor)}")
        lines.append(f"{collection_name}.append({tensor_value_by_id[tensor.spec.id]})")
    return lines


def render_tensor_data_expression(
    tensor: PreparedTensor,
    *,
    module_alias: str,
    zeros_initializer_suffix: str = "",
    literal_constructor_name: str,
) -> str:
    """Render one backend-specific tensor-data initializer expression."""
    tensor_data = tensor.spec.tensor_data
    if tensor_data is None:
        return f"{module_alias}.zeros({tensor.spec.shape!r}{zeros_initializer_suffix})"
    if tensor_data.mode is TensorDataMode.ONES:
        return f"{module_alias}.ones({tensor.spec.shape!r}{zeros_initializer_suffix})"
    if tensor_data.mode is TensorDataMode.FILL:
        return (
            f"{module_alias}.full({tensor.spec.shape!r}, {tensor_data.fill_value!r}"
            f"{zeros_initializer_suffix})"
        )
    return (
        f"{module_alias}.{literal_constructor_name}({tensor_data.values!r}"
        f"{zeros_initializer_suffix})"
    )


def _hyperedge_copy_tensor_signature(tensor: PreparedTensor) -> tuple[int, int] | None:
    """Return ``(dimension, rank)`` for generated hyperedge copy tensors."""
    tensor_data = tensor.spec.tensor_data
    if tensor_data is None or tensor_data.mode is not TensorDataMode.LITERAL:
        return None
    metadata = tensor.spec.metadata
    if (
        metadata.get("generated_by") != "hyperedge_lowering"
        or "generated_for_hyperedge" not in metadata
    ):
        return None
    shape = tensor.spec.shape
    if not shape:
        return None
    dimension = shape[0]
    if any(axis_dimension != dimension for axis_dimension in shape[1:]):
        return None
    return dimension, len(shape)


def _render_hyperedge_copy_tensor_data_assignments(
    tensor: PreparedTensor,
    *,
    module_alias: str,
    zeros_initializer_suffix: str = "",
) -> list[str] | None:
    """Render compact generated code for one lowered hyperedge copy tensor."""
    signature = _hyperedge_copy_tensor_signature(tensor)
    if signature is None or module_alias not in {"np", "torch"}:
        return None
    dimension, rank = signature
    repeated_shape = f"({dimension},) * {rank}"
    repeated_indices = f"({module_alias}.arange({dimension}),) * {rank}"
    lines = [
        f"{tensor.data_variable_name} = {module_alias}.zeros({repeated_shape}{zeros_initializer_suffix})"
    ]
    if module_alias == "torch":
        lines.append(
            f"{tensor.data_variable_name}.index_put_("
            f"{repeated_indices}, "
            f"{module_alias}.ones({dimension}{zeros_initializer_suffix}))"
        )
    else:
        lines.append(f"{tensor.data_variable_name}[{repeated_indices}] = 1")
    return lines


def render_tensor_data_assignments(
    prepared: PreparedNetwork,
    *,
    module_alias: str,
    zeros_initializer_suffix: str = "",
    literal_constructor_name: str,
) -> list[str]:
    """Render one data-variable assignment per tensor in display order."""
    lines: list[str] = []
    for tensor in prepared.tensors:
        lines.append(f"# Tensor {_tensor_display_name(tensor)} data")
        hyperedge_copy_tensor_lines = _render_hyperedge_copy_tensor_data_assignments(
            tensor,
            module_alias=module_alias,
            zeros_initializer_suffix=zeros_initializer_suffix,
        )
        if hyperedge_copy_tensor_lines is not None:
            lines.extend(hyperedge_copy_tensor_lines)
            continue
        tensor_data_expression = render_tensor_data_expression(
            tensor,
            module_alias=module_alias,
            zeros_initializer_suffix=zeros_initializer_suffix,
            literal_constructor_name=literal_constructor_name,
        )
        lines.append(f"{tensor.data_variable_name} = {tensor_data_expression}")
    return lines


def _tensor_display_name(tensor: PreparedTensor) -> str:
    """Return the readable tensor label used in generated comments."""
    return tensor.spec.name or tensor.variable_name


def _trim_blank_lines(lines: list[str]) -> list[str]:
    """Trim blank lines at the edges of one rendered section."""
    start = 0
    end = len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return lines[start:end]
