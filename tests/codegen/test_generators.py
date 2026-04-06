from __future__ import annotations

import pytest

from tensor_network_editor.api import generate_code
from tensor_network_editor.errors import CodeGenerationError
from tensor_network_editor.models import (
    CanvasPosition,
    EngineName,
    IndexSpec,
    NetworkSpec,
    TensorCollectionFormat,
    TensorSpec,
)
from tests.factories import (
    build_outer_product_plan_spec,
    build_sample_spec,
    build_sample_spec_without_plan,
    build_three_tensor_spec,
)


def build_many_label_spec() -> NetworkSpec:
    tensors = []
    for tensor_index in range(18):
        tensors.append(
            TensorSpec(
                id=f"tensor_{tensor_index}",
                name=f"T{tensor_index}",
                position=CanvasPosition(x=float(tensor_index * 120), y=0.0),
                indices=[
                    IndexSpec(
                        id=f"tensor_{tensor_index}_i{index_offset}",
                        name=f"i{index_offset}",
                        dimension=2,
                    )
                    for index_offset in range(3)
                ],
            )
        )
    return NetworkSpec(id="many_labels", name="many labels", tensors=tensors)


def build_matrix_layout_spec() -> NetworkSpec:
    return NetworkSpec(
        id="matrix_layout",
        name="matrix layout",
        tensors=[
            TensorSpec(
                id="tensor_a",
                name="A",
                position=CanvasPosition(x=80.0, y=100.0),
                indices=[
                    IndexSpec(id="tensor_a_i", name="i", dimension=2),
                    IndexSpec(id="tensor_a_x", name="x", dimension=3),
                ],
            ),
            TensorSpec(
                id="tensor_b",
                name="B",
                position=CanvasPosition(x=260.0, y=108.0),
                indices=[
                    IndexSpec(id="tensor_b_x", name="x", dimension=3),
                    IndexSpec(id="tensor_b_y", name="y", dimension=5),
                ],
            ),
            TensorSpec(
                id="tensor_c",
                name="C",
                position=CanvasPosition(x=120.0, y=260.0),
                indices=[
                    IndexSpec(id="tensor_c_y", name="y", dimension=5),
                    IndexSpec(id="tensor_c_j", name="j", dimension=7),
                ],
            ),
        ],
    )


@pytest.mark.parametrize(
    ("engine", "expected_snippets"),
    [
        (
            EngineName.TENSORNETWORK,
            ["import tensornetwork as tn", "axis_names=['i', 'x']", "tn.connect("],
        ),
        (
            EngineName.QUIMB,
            [
                "import quimb.tensor as qtn",
                "qtn.Tensor(",
                "network = qtn.TensorNetwork(",
            ],
        ),
        (
            EngineName.TENSORKROWCH,
            [
                "import tensorkrowch as tk",
                "network = tk.TensorNetwork()",
                "tk.connect(",
            ],
        ),
        (
            EngineName.EINSUM_NUMPY,
            ["import numpy as np", "np.zeros((2, 3)", "result = np.einsum("],
        ),
        (
            EngineName.EINSUM_TORCH,
            [
                "import torch",
                "torch.zeros((2, 3), dtype=torch.float32)",
                "result = torch.einsum(",
            ],
        ),
    ],
)
def test_generate_code_emits_engine_specific_contracts(
    engine: EngineName,
    expected_snippets: list[str],
) -> None:
    result = generate_code(build_sample_spec_without_plan(), engine=engine)

    assert result.engine is engine
    assert result.code.endswith("\n")
    for snippet in expected_snippets:
        assert snippet in result.code


@pytest.mark.parametrize("engine", list(EngineName))
@pytest.mark.parametrize(
    ("collection_format", "container_name", "expected_snippets"),
    [
        (
            TensorCollectionFormat.LIST,
            "tensors",
            ["tensors = []", "tensors.append("],
        ),
        (
            TensorCollectionFormat.MATRIX,
            "tensor_rows",
            ["tensor_rows = []", "tensor_rows.append([])", "tensor_rows[0].append("],
        ),
        (
            TensorCollectionFormat.DICT,
            "tensors_dict",
            ["tensors_dict = {}", "tensors_dict["],
        ),
    ],
)
def test_generate_code_supports_all_collection_formats(
    engine: EngineName,
    collection_format: TensorCollectionFormat,
    container_name: str,
    expected_snippets: list[str],
) -> None:
    spec = build_matrix_layout_spec()

    result = generate_code(spec, engine=engine, collection_format=collection_format)

    assert result.engine is engine
    for snippet in expected_snippets:
        assert snippet in result.code
    assert container_name in result.code
    assert "# Tensor A" in result.code
    assert "# Tensor B" in result.code
    assert "_TNE_SPEC" not in result.code
    assert "_data =" not in result.code


def test_matrix_collection_format_groups_tensors_by_visual_rows() -> None:
    result = generate_code(
        build_matrix_layout_spec(),
        engine=EngineName.EINSUM_NUMPY,
        collection_format=TensorCollectionFormat.MATRIX,
    )

    assignment_start = result.code.index("tensor_rows = []")
    assignment_end = result.code.index("# Einsum equation:")
    assignment = result.code[assignment_start:assignment_end]

    assert "tensor_rows = []" in assignment
    assert "tensor_rows.append([])" in assignment
    assert assignment.index("# Tensor A") < assignment.index("tensor_rows[0].append(")
    assert assignment.index("# Tensor B") < assignment.index(
        "tensor_rows[0].append(", assignment.index("# Tensor B")
    )
    assert assignment.index("# Tensor C") < assignment.index("tensor_rows[1].append(")
    assert (
        assignment.index("np.zeros((2, 3), dtype=float)")
        < assignment.index("np.zeros((3, 5), dtype=float)")
        < assignment.index("np.zeros((5, 7), dtype=float)")
    )
    assert "tensor_rows[0][0]" in result.code
    assert "tensor_rows[0][1]" in result.code
    assert "tensor_rows[1][0]" in result.code


def test_generate_code_does_not_emit_roundtrip_metadata() -> None:
    result = generate_code(
        build_sample_spec_without_plan(),
        engine=EngineName.TENSORNETWORK,
    )

    assert "_TNE_SPEC" not in result.code
    assert "_data =" not in result.code


@pytest.mark.parametrize(
    ("engine", "expected_snippet"),
    [
        (EngineName.TENSORNETWORK, "results_list.append(tn.contract_between("),
        (EngineName.QUIMB, "results_list.append(network["),
        (EngineName.TENSORKROWCH, "results_list.append(tk.contract_between("),
        (EngineName.EINSUM_NUMPY, "results_list.append(np.einsum("),
        (EngineName.EINSUM_TORCH, "results_list.append(torch.einsum("),
    ],
)
def test_generate_code_respects_manual_plan_steps(
    engine: EngineName,
    expected_snippet: str,
) -> None:
    result = generate_code(build_sample_spec(), engine=engine)

    assert expected_snippet in result.code
    assert "results_list = []" in result.code
    assert "remaining_operands = {" in result.code
    assert "'A-B': results_list[-1]" in result.code
    assert "result = results_list[-1]" in result.code


@pytest.mark.parametrize("engine", list(EngineName))
def test_generate_code_keeps_partial_manual_plan_as_prefix(
    engine: EngineName,
) -> None:
    result = generate_code(build_three_tensor_spec(), engine=engine)

    assert "results_list = []" in result.code
    assert "remaining_operands = {" in result.code
    assert "'A-B': results_list[-1]" in result.code
    assert "'C': tensors[2]" in result.code
    assert "result =" not in result.code


def test_tensorkrowch_codegen_rejects_manual_outer_product_plan() -> None:
    with pytest.raises(CodeGenerationError, match="outer product"):
        generate_code(build_outer_product_plan_spec(), engine=EngineName.TENSORKROWCH)


def test_tensorkrowch_codegen_uses_edges_list_for_connections() -> None:
    result = generate_code(
        build_sample_spec_without_plan(),
        engine=EngineName.TENSORKROWCH,
    )

    assert "edges_list = []" in result.code
    assert "edges_list.append((" in result.code
    assert "tk.connect(" in result.code


def test_tensornetwork_codegen_uses_edges_list_for_connections() -> None:
    result = generate_code(
        build_sample_spec_without_plan(),
        engine=EngineName.TENSORNETWORK,
    )

    assert "edges_list = []" in result.code
    assert "edges_list.append(tn.connect(" in result.code
    assert "name='bond_x'" in result.code


@pytest.mark.parametrize(
    "engine",
    [EngineName.EINSUM_NUMPY, EngineName.EINSUM_TORCH],
)
def test_einsum_codegen_uses_integer_sublist_form_for_many_labels(
    engine: EngineName,
) -> None:
    result = generate_code(build_many_label_spec(), engine=engine)

    module_alias = "np" if engine is EngineName.EINSUM_NUMPY else "torch"

    assert "integer-sublist form because the network uses many labels" in result.code
    assert f"result = {module_alias}.einsum(" in result.code
    assert "# Einsum equation:" not in result.code
