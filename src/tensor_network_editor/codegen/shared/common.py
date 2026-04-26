"""Shared preparation and rendering helpers for backend code generators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

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
from ...models import (
    TensorCollectionFormat,
    TensorDataDType,
    TensorDataMode,
    TensorDataRandomDistribution,
    TensorDataSpec,
)

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
    "uses_external_tensor_data",
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
    dtype_suffix = _render_tensor_data_dtype_suffix(
        tensor_data,
        module_alias=module_alias,
        default_suffix=zeros_initializer_suffix,
    )
    if tensor_data is None or tensor_data.mode is TensorDataMode.ZEROS:
        return f"{module_alias}.zeros({tensor.spec.shape!r}{dtype_suffix})"
    if tensor_data.mode is TensorDataMode.ONES:
        return f"{module_alias}.ones({tensor.spec.shape!r}{dtype_suffix})"
    if tensor_data.mode is TensorDataMode.FILL:
        return (
            f"{module_alias}.full({tensor.spec.shape!r}, "
            f"{_render_python_literal(tensor_data.fill_value)}{dtype_suffix})"
        )
    if tensor_data.mode is TensorDataMode.IDENTITY:
        dimension = tensor.spec.shape[0] if tensor.spec.shape else 1
        return f"{module_alias}.eye({dimension}{dtype_suffix})"
    if tensor_data.mode is TensorDataMode.EXTERNAL:
        return _render_external_tensor_data_expression(
            tensor,
            module_alias=module_alias,
        )
    return (
        f"{module_alias}.{literal_constructor_name}("
        f"{_render_python_literal(tensor_data.values)}{dtype_suffix})"
    )


def uses_external_tensor_data(prepared: PreparedNetwork) -> bool:
    """Return whether any prepared tensor loads data from an external file."""
    return any(
        tensor.spec.tensor_data is not None
        and tensor.spec.tensor_data.mode is TensorDataMode.EXTERNAL
        for tensor in prepared.tensors
    )


def _render_external_tensor_data_expression(
    tensor: PreparedTensor,
    *,
    module_alias: str,
) -> str:
    """Render a backend-specific external NumPy-file data initializer."""
    tensor_data = cast(TensorDataSpec, tensor.spec.tensor_data)
    call_expression = (
        "_load_external_tensor_data("
        f"{tensor_data.file_path!r}, "
        f"expected_shape={tensor.spec.shape!r}, "
        f"array_key={tensor_data.array_key!r})"
    )
    dtype_expression = _render_tensor_data_dtype_expression(
        tensor_data,
        module_alias=module_alias,
        default_suffix="",
    )
    if module_alias == "torch":
        if dtype_expression:
            return f"torch.as_tensor({call_expression}, dtype={dtype_expression})"
        return f"torch.as_tensor({call_expression})"
    if dtype_expression:
        return f"{call_expression}.astype({dtype_expression}, copy=False)"
    return call_expression


def _render_external_tensor_data_helper_lines() -> list[str]:
    """Render the shared helper used by generated code to load .npy/.npz data."""
    return [
        "def _load_external_tensor_data(path, *, expected_shape, array_key=None):",
        "    loaded = np.load(path)",
        "    if hasattr(loaded, 'files'):",
        "        if array_key is None:",
        "            raise ValueError('External .npz tensor data requires array_key.')",
        "        data = loaded[array_key]",
        "    else:",
        "        if array_key is not None:",
        "            raise ValueError('External .npy tensor data does not use array_key.')",
        "        data = loaded",
        "    if tuple(data.shape) != tuple(expected_shape):",
        "        raise ValueError(",
        "            f'External tensor data shape {tuple(data.shape)!r} does not match '",
        "            f'expected shape {tuple(expected_shape)!r}.'",
        "        )",
        "    return data",
        "",
    ]


def _hyperedge_copy_tensor_signature(tensor: PreparedTensor) -> tuple[int, int] | None:
    """Return ``(dimension, rank)`` for generated hyperedge copy tensors."""
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
        _render_hyperedge_copy_tensor_comment(tensor),
        f"{tensor.data_variable_name} = {module_alias}.zeros({repeated_shape}{zeros_initializer_suffix})",
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


def _render_hyperedge_copy_tensor_comment(tensor: PreparedTensor) -> str:
    """Render structured metadata for lowered hyperedge reconstruction."""
    hyperedge_id = str(tensor.spec.metadata["generated_for_hyperedge"])
    hyperedge_name = tensor.spec.name.removeprefix("Copy ")
    return (
        "# Hyperedge copy tensor | "
        f"id={hyperedge_id} | "
        f"name={hyperedge_name} | "
        f"data={tensor.data_variable_name}"
    )


def _render_copy_tensor_data_assignments(
    tensor: PreparedTensor,
    *,
    module_alias: str,
    zeros_initializer_suffix: str = "",
) -> list[str] | None:
    """Render compact generated code for one explicit copy tensor initializer."""
    tensor_data = tensor.spec.tensor_data
    if tensor_data is None or tensor_data.mode is not TensorDataMode.COPY:
        return None
    if not tensor.spec.shape or module_alias not in {"np", "torch"}:
        return None
    dimension = tensor.spec.shape[0]
    rank = len(tensor.spec.shape)
    dtype_suffix = _render_tensor_data_dtype_suffix(
        tensor_data,
        module_alias=module_alias,
        default_suffix=zeros_initializer_suffix,
    )
    repeated_shape = f"({dimension},) * {rank}"
    repeated_indices = f"({module_alias}.arange({dimension}),) * {rank}"
    lines = [
        f"{tensor.data_variable_name} = {module_alias}.zeros({repeated_shape}{dtype_suffix})"
    ]
    if module_alias == "torch":
        lines.append(
            f"{tensor.data_variable_name}.index_put_("
            f"{repeated_indices}, "
            f"{module_alias}.ones({dimension}{dtype_suffix}))"
        )
    else:
        lines.append(f"{tensor.data_variable_name}[{repeated_indices}] = 1")
    return lines


def _render_random_tensor_data_assignments(
    tensor: PreparedTensor,
    *,
    module_alias: str,
    zeros_initializer_suffix: str = "",
) -> list[str] | None:
    """Render deterministic seeded random initializer lines."""
    tensor_data = tensor.spec.tensor_data
    if tensor_data is None or tensor_data.mode is not TensorDataMode.RANDOM:
        return None
    seed = cast(int, tensor_data.seed)
    distribution = cast(TensorDataRandomDistribution, tensor_data.distribution)
    dtype_expression = _render_tensor_data_dtype_expression(
        tensor_data,
        module_alias=module_alias,
        default_suffix=zeros_initializer_suffix,
    )
    variable_prefix = tensor.data_variable_name.removesuffix("_data")
    if module_alias == "np":
        rng_name = f"{variable_prefix}_rng"
        random_method = (
            "uniform"
            if distribution is TensorDataRandomDistribution.UNIFORM
            else "normal"
        )
        sample_expression = f"{rng_name}.{random_method}(size={tensor.spec.shape!r})"
        if _is_complex_dtype(tensor_data.dtype):
            sample_expression = (
                f"({sample_expression} + 1j * "
                f"{rng_name}.{random_method}(size={tensor.spec.shape!r}))"
            )
        return [
            f"{rng_name} = np.random.default_rng({seed})",
            f"{tensor.data_variable_name} = {sample_expression}.astype({dtype_expression})",
        ]
    if module_alias == "torch":
        generator_name = f"{variable_prefix}_generator"
        random_function = (
            "rand" if distribution is TensorDataRandomDistribution.UNIFORM else "randn"
        )
        if _is_complex_dtype(tensor_data.dtype):
            real_dtype = (
                "torch.float32"
                if tensor_data.dtype is TensorDataDType.COMPLEX64
                else "torch.float64"
            )
            return [
                f"{generator_name} = torch.Generator()",
                f"{generator_name}.manual_seed({seed})",
                (
                    f"{variable_prefix}_real = torch.{random_function}("
                    f"{tensor.spec.shape!r}, generator={generator_name}, dtype={real_dtype})"
                ),
                (
                    f"{variable_prefix}_imag = torch.{random_function}("
                    f"{tensor.spec.shape!r}, generator={generator_name}, dtype={real_dtype})"
                ),
                (
                    f"{tensor.data_variable_name} = torch.complex("
                    f"{variable_prefix}_real, {variable_prefix}_imag).to(dtype={dtype_expression})"
                ),
            ]
        return [
            f"{generator_name} = torch.Generator()",
            f"{generator_name}.manual_seed({seed})",
            (
                f"{tensor.data_variable_name} = torch.{random_function}("
                f"{tensor.spec.shape!r}, generator={generator_name}, dtype={dtype_expression})"
            ),
        ]
    return None


def render_tensor_data_assignments(
    prepared: PreparedNetwork,
    *,
    module_alias: str,
    zeros_initializer_suffix: str = "",
    literal_constructor_name: str,
) -> list[str]:
    """Render one data-variable assignment per tensor in display order."""
    lines: list[str] = []
    if uses_external_tensor_data(prepared):
        lines.extend(_render_external_tensor_data_helper_lines())
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
        copy_tensor_lines = _render_copy_tensor_data_assignments(
            tensor,
            module_alias=module_alias,
            zeros_initializer_suffix=zeros_initializer_suffix,
        )
        if copy_tensor_lines is not None:
            lines.extend(copy_tensor_lines)
            continue
        random_tensor_lines = _render_random_tensor_data_assignments(
            tensor,
            module_alias=module_alias,
            zeros_initializer_suffix=zeros_initializer_suffix,
        )
        if random_tensor_lines is not None:
            lines.extend(random_tensor_lines)
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


def _render_tensor_data_dtype_suffix(
    tensor_data: TensorDataSpec | None,
    *,
    module_alias: str,
    default_suffix: str,
) -> str:
    """Render the ``, dtype=...`` suffix for one backend initializer."""
    dtype_expression = _render_tensor_data_dtype_expression(
        tensor_data,
        module_alias=module_alias,
        default_suffix=default_suffix,
    )
    return f", dtype={dtype_expression}" if dtype_expression else default_suffix


def _render_tensor_data_dtype_expression(
    tensor_data: TensorDataSpec | None,
    *,
    module_alias: str,
    default_suffix: str,
) -> str:
    """Render the backend dtype expression for one tensor-data payload."""
    dtype = tensor_data.dtype if tensor_data is not None else None
    if (
        dtype is None
        and tensor_data is not None
        and _tensor_data_contains_complex(tensor_data)
    ):
        dtype = (
            TensorDataDType.COMPLEX64
            if module_alias == "torch"
            else TensorDataDType.COMPLEX128
        )
    if dtype is None:
        return _dtype_expression_from_suffix(default_suffix)
    if module_alias == "torch":
        return {
            TensorDataDType.FLOAT32: "torch.float32",
            TensorDataDType.FLOAT64: "torch.float64",
            TensorDataDType.COMPLEX64: "torch.complex64",
            TensorDataDType.COMPLEX128: "torch.complex128",
        }[dtype]
    return {
        TensorDataDType.FLOAT32: "np.float32",
        TensorDataDType.FLOAT64: "np.float64",
        TensorDataDType.COMPLEX64: "np.complex64",
        TensorDataDType.COMPLEX128: "np.complex128",
    }[dtype]


def _dtype_expression_from_suffix(default_suffix: str) -> str:
    """Extract a default dtype expression from an existing initializer suffix."""
    marker = "dtype="
    if marker not in default_suffix:
        return ""
    return default_suffix.split(marker, maxsplit=1)[1].strip()


def _is_complex_dtype(dtype: TensorDataDType | None) -> bool:
    """Return whether one portable dtype is complex-valued."""
    return dtype in {TensorDataDType.COMPLEX64, TensorDataDType.COMPLEX128}


def _tensor_data_contains_complex(tensor_data: TensorDataSpec) -> bool:
    """Return whether one tensor-data payload contains portable complex values."""
    if tensor_data.fill_value is not None:
        return _literal_contains_complex(tensor_data.fill_value)
    if tensor_data.values is not None:
        return _literal_contains_complex(tensor_data.values)
    return False


def _literal_contains_complex(value: object) -> bool:
    """Return whether a recursive literal tree contains complex scalar mappings."""
    if isinstance(value, dict):
        return set(value) == {"real", "imag"}
    if isinstance(value, list):
        return any(_literal_contains_complex(item) for item in value)
    return False


def _render_python_literal(value: object) -> str:
    """Render a scalar or nested literal tree as Python source."""
    if isinstance(value, dict) and set(value) == {"real", "imag"}:
        return _render_complex_literal(value)
    if isinstance(value, list):
        return "[" + ", ".join(_render_python_literal(item) for item in value) + "]"
    return repr(value)


def _render_complex_literal(value: dict[object, object]) -> str:
    """Render one portable complex scalar mapping as a Python complex literal."""
    real_text = repr(value["real"])
    imag_value = value["imag"]
    imag_text = repr(imag_value)
    sign = "" if str(imag_text).startswith("-") else "+"
    return f"({real_text}{sign}{imag_text}j)"


def _trim_blank_lines(lines: list[str]) -> list[str]:
    """Trim blank lines at the edges of one rendered section."""
    start = 0
    end = len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return lines[start:end]
