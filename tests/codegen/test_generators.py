from __future__ import annotations

from collections.abc import Callable
from unittest.mock import patch

import pytest

from tensor_network_editor.api import generate_code
from tensor_network_editor.codegen.common import render_remaining_operands_mapping
from tensor_network_editor.errors import CodeGenerationError
from tensor_network_editor.models import (
    CanvasPosition,
    EdgeEndpointRef,
    EdgeSpec,
    EngineName,
    IndexSpec,
    NetworkSpec,
    TensorCollectionFormat,
    TensorSpec,
)
from tests.factories import (
    build_grid_periodic_grid_spec,
    build_linear_periodic_carry_chain_spec,
    build_linear_periodic_chain_spec,
    build_linear_periodic_partial_carry_chain_spec,
    build_outer_product_plan_spec,
    build_sample_spec,
    build_sample_spec_without_plan,
    build_three_tensor_spec,
    build_three_tensor_spec_without_plan,
    build_tree_periodic_tree_spec,
)
from tests.optional_backends import require_engine_backend


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


def build_disconnected_pairs_spec() -> NetworkSpec:
    return NetworkSpec(
        id="disconnected_pairs",
        name="disconnected pairs",
        tensors=[
            TensorSpec(
                id="tensor_a",
                name="A",
                position=CanvasPosition(x=80.0, y=120.0),
                indices=[
                    IndexSpec(id="tensor_a_i", name="i", dimension=2),
                    IndexSpec(id="tensor_a_x", name="x", dimension=3),
                ],
            ),
            TensorSpec(
                id="tensor_b",
                name="B",
                position=CanvasPosition(x=240.0, y=120.0),
                indices=[
                    IndexSpec(id="tensor_b_x", name="x", dimension=3),
                    IndexSpec(id="tensor_b_j", name="j", dimension=5),
                ],
            ),
            TensorSpec(
                id="tensor_c",
                name="C",
                position=CanvasPosition(x=440.0, y=120.0),
                indices=[
                    IndexSpec(id="tensor_c_k", name="k", dimension=7),
                    IndexSpec(id="tensor_c_y", name="y", dimension=11),
                ],
            ),
            TensorSpec(
                id="tensor_d",
                name="D",
                position=CanvasPosition(x=600.0, y=120.0),
                indices=[
                    IndexSpec(id="tensor_d_y", name="y", dimension=11),
                    IndexSpec(id="tensor_d_l", name="l", dimension=13),
                ],
            ),
        ],
        edges=[
            EdgeSpec(
                id="edge_x",
                name="bond_x",
                left=EdgeEndpointRef(tensor_id="tensor_a", index_id="tensor_a_x"),
                right=EdgeEndpointRef(tensor_id="tensor_b", index_id="tensor_b_x"),
            ),
            EdgeSpec(
                id="edge_y",
                name="bond_y",
                left=EdgeEndpointRef(tensor_id="tensor_c", index_id="tensor_c_y"),
                right=EdgeEndpointRef(tensor_id="tensor_d", index_id="tensor_d_y"),
            ),
        ],
    )


def build_random_better_chain_spec() -> NetworkSpec:
    return NetworkSpec(
        id="random_better_chain",
        name="random better chain",
        tensors=[
            TensorSpec(
                id="tensor_a",
                name="A",
                position=CanvasPosition(x=80.0, y=120.0),
                indices=[
                    IndexSpec(id="tensor_a_i", name="i", dimension=100),
                    IndexSpec(id="tensor_a_x", name="x", dimension=2),
                ],
            ),
            TensorSpec(
                id="tensor_b",
                name="B",
                position=CanvasPosition(x=240.0, y=120.0),
                indices=[
                    IndexSpec(id="tensor_b_x", name="x", dimension=2),
                    IndexSpec(id="tensor_b_y", name="y", dimension=100),
                ],
            ),
            TensorSpec(
                id="tensor_c",
                name="C",
                position=CanvasPosition(x=400.0, y=120.0),
                indices=[
                    IndexSpec(id="tensor_c_y", name="y", dimension=100),
                    IndexSpec(id="tensor_c_j", name="j", dimension=2),
                ],
            ),
        ],
        edges=[
            EdgeSpec(
                id="edge_x",
                name="bond_x",
                left=EdgeEndpointRef(tensor_id="tensor_a", index_id="tensor_a_x"),
                right=EdgeEndpointRef(tensor_id="tensor_b", index_id="tensor_b_x"),
            ),
            EdgeSpec(
                id="edge_y",
                name="bond_y",
                left=EdgeEndpointRef(tensor_id="tensor_b", index_id="tensor_b_y"),
                right=EdgeEndpointRef(tensor_id="tensor_c", index_id="tensor_c_y"),
            ),
        ],
    )


def build_empty_spec() -> NetworkSpec:
    return NetworkSpec(id="empty_network", name="empty network")


def test_render_remaining_operands_mapping_renders_joined_display_names() -> None:
    lines = render_remaining_operands_mapping(
        operand_ids=("step_ab", "tensor_c"),
        source_tensor_ids_by_operand_id={
            "step_ab": ("tensor_a", "tensor_b"),
            "tensor_c": ("tensor_c",),
        },
        tensor_names_by_id={
            "tensor_a": "A",
            "tensor_b": "B",
            "tensor_c": "C",
        },
        base_operand_expressions={
            "tensor_a": "tensors[0]",
            "tensor_b": "tensors[1]",
            "tensor_c": "tensors[2]",
        },
        step_result_indexes={"step_ab": 0},
        latest_result_index=0,
    )

    assert lines == [
        "remaining_operands = {",
        "    'A-B': results_list[-1],",
        "    'C': tensors[2],",
        "}",
    ]


def _import_required_backend(engine: EngineName) -> None:
    """Skip execution tests when an optional backend is not installed."""
    require_engine_backend(engine)


def _execute_generated_code(
    code: str,
    *,
    n: int | None = None,
    m: int | None = None,
) -> dict[str, object]:
    """Execute generated code in a shared namespace and return that namespace."""
    namespace: dict[str, object] = {}
    if n is not None:
        namespace["n"] = n
    if m is not None:
        namespace["m"] = m
    exec(code, namespace, namespace)
    return namespace


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
            [
                "import numpy as np",
                "np.zeros((2, 3)",
                "results_list.append(np.einsum(",
                "result = results_list[-1]",
            ],
        ),
        (
            EngineName.EINSUM_TORCH,
            [
                "import torch",
                "torch.zeros((2, 3), dtype=torch.float32)",
                "results_list.append(torch.einsum(",
                "result = results_list[-1]",
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
    assignment_end = result.code.index("# Contraction")
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


@pytest.mark.parametrize("engine", list(EngineName))
def test_generate_code_labels_shared_normal_sections(engine: EngineName) -> None:
    result = generate_code(build_sample_spec_without_plan(), engine=engine)

    assert "# Tensor collection" in result.code
    assert "# Tensor construction" in result.code
    assert "# Outputs" in result.code
    assert result.code.index("# Tensor collection") < result.code.index(
        "# Tensor construction"
    )
    assert result.code.index("# Tensor construction") < result.code.index("# Outputs")


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

    assert "# Manual contraction" in result.code
    assert expected_snippet in result.code
    assert "results_list = []" in result.code
    assert result.code.index("# Manual contraction") < result.code.index(
        "results_list = []"
    )
    assert result.code.index("results_list = []") < result.code.index("# Outputs")
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


@pytest.mark.parametrize(
    ("engine", "append_snippet"),
    [
        (EngineName.EINSUM_NUMPY, "results_list.append(np.einsum("),
        (EngineName.EINSUM_TORCH, "results_list.append(torch.einsum("),
    ],
)
def test_einsum_codegen_without_plan_emits_pairwise_steps(
    engine: EngineName,
    append_snippet: str,
) -> None:
    result = generate_code(build_three_tensor_spec_without_plan(), engine=engine)

    assert "# Contraction" in result.code
    assert "results_list = []" in result.code
    assert result.code.count(append_snippet) == 2
    assert "result = results_list[-1]" in result.code
    assert (
        f"result = {'np' if engine is EngineName.EINSUM_NUMPY else 'torch'}.einsum("
        not in result.code
    )


@pytest.mark.parametrize(
    ("engine", "append_snippet"),
    [
        (EngineName.EINSUM_NUMPY, "results_list.append(np.einsum("),
        (EngineName.EINSUM_TORCH, "results_list.append(torch.einsum("),
    ],
)
def test_einsum_codegen_without_plan_contracts_connected_pairs_first(
    engine: EngineName,
    append_snippet: str,
) -> None:
    with patch(
        "tensor_network_editor.codegen.backends.einsum._load_random_optimizer_tools",
        return_value=None,
    ):
        result = generate_code(build_disconnected_pairs_spec(), engine=engine)

    assert result.code.count(append_snippet) == 3
    first_connected_step = f"{append_snippet}'ab,bc->ac', tensors[0], tensors[1]))"
    second_connected_step = f"{append_snippet}'de,ef->df', tensors[2], tensors[3]))"

    assert first_connected_step in result.code
    assert second_connected_step in result.code
    assert result.code.index(first_connected_step) < result.code.index(
        second_connected_step
    )


@pytest.mark.parametrize(
    ("engine", "module_name"),
    [
        (EngineName.EINSUM_NUMPY, "np"),
        (EngineName.EINSUM_TORCH, "torch"),
    ],
)
def test_einsum_codegen_without_plan_prefers_better_random_route_when_available(
    engine: EngineName,
    module_name: str,
) -> None:
    class FakeRandomGreedy:
        def __init__(
            self,
            *,
            max_time: float | None = None,
            minimize: str = "flops",
        ) -> None:
            self.max_time = max_time
            self.minimize = minimize

    def fake_contract_path(
        equation: str,
        *operand_shapes: tuple[int, ...],
        shapes: bool = True,
        optimize: object | None = None,
    ) -> tuple[list[tuple[int, int]], object]:
        assert equation == "ab,bc,cd->ad"
        assert shapes is True
        assert operand_shapes == ((100, 2), (2, 100), (100, 2))
        assert isinstance(optimize, FakeRandomGreedy)
        assert optimize.max_time == 0.05
        assert optimize.minimize == "flops"
        return [(1, 2), (0, 1)], object()

    with patch(
        "tensor_network_editor.codegen.backends.einsum._load_random_optimizer_tools",
        return_value=(fake_contract_path, FakeRandomGreedy),
    ):
        result = generate_code(build_random_better_chain_spec(), engine=engine)

    first_step = f"results_list.append({module_name}.einsum('bc,cd->bd', tensors[1], tensors[2]))"
    second_step = f"results_list.append({module_name}.einsum('ab,bd->ad', tensors[0], results_list[-1]))"

    assert first_step in result.code
    assert second_step in result.code
    assert result.code.index(first_step) < result.code.index(second_step)


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


@pytest.mark.heavy_backend
def test_tensorkrowch_codegen_executes_when_tensor_names_contain_spaces() -> None:
    _import_required_backend(EngineName.TENSORKROWCH)
    spec = NetworkSpec(
        id="space_names",
        name="space names",
        tensors=[
            TensorSpec(
                id="tensor_a",
                name="Tensor A",
                position=CanvasPosition(x=0.0, y=0.0),
                indices=[IndexSpec(id="tensor_a_i", name="i", dimension=2)],
            )
        ],
    )

    result = generate_code(spec, engine=EngineName.TENSORKROWCH)
    namespace = _execute_generated_code(result.code)

    assert "tensors" in namespace


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
    [EngineName.TENSORNETWORK, EngineName.TENSORKROWCH],
)
def test_linear_periodic_codegen_uses_cell_helpers_and_free_n_loop(
    engine: EngineName,
) -> None:
    result = generate_code(build_linear_periodic_chain_spec(), engine=engine)

    assert "def build_initial_cell(" in result.code
    assert "def build_periodic_cell(cell_index" in result.code
    assert "def build_final_cell(" in result.code
    assert "if n < 2:" in result.code
    assert "for cell_index in range(1, n - 1):" in result.code
    assert "connect_cell_interfaces(" in result.code
    assert "periodic_contract_internal" in result.code


@pytest.mark.parametrize("engine", list(EngineName))
def test_linear_periodic_codegen_labels_shared_for_sections(
    engine: EngineName,
) -> None:
    result = generate_code(build_linear_periodic_chain_spec(), engine=engine)

    assert "# Shared helpers" in result.code
    assert "# Initial cell" in result.code
    assert "# Periodic cell" in result.code
    assert "# Final cell" in result.code
    assert "# Main loop" in result.code
    assert "# Tensor collection" in result.code
    assert "# Tensor construction" in result.code
    assert "# Outputs" in result.code
    assert result.code.index("# Shared helpers") < result.code.index("# Initial cell")
    assert result.code.index("# Initial cell") < result.code.index("# Periodic cell")
    assert result.code.index("# Periodic cell") < result.code.index("# Final cell")
    assert result.code.index("# Final cell") < result.code.index("# Main loop")
    assert "def build_initial_cell() -> " in result.code
    assert "def build_periodic_cell(" in result.code
    assert ") -> " in result.code
    assert "def build_final_cell(" in result.code


@pytest.mark.parametrize("engine", list(EngineName))
def test_linear_periodic_carry_codegen_labels_shared_for_sections(
    engine: EngineName,
) -> None:
    result = generate_code(build_linear_periodic_carry_chain_spec(), engine=engine)

    assert "# Shared helpers" in result.code
    assert "# Initial cell" in result.code
    assert "# Periodic cell" in result.code
    assert "# Final cell" in result.code
    assert "# Main loop" in result.code
    assert "# Tensor collection" in result.code
    assert "# Tensor construction" in result.code
    assert "# Previous interface" in result.code
    assert "# Manual contraction" in result.code
    assert "# Outputs" in result.code
    assert "def build_initial_cell() -> " in result.code
    assert "previous_payload: dict[str, object]" in result.code


@pytest.mark.parametrize("engine", list(EngineName))
def test_linear_periodic_codegen_does_not_stringify_manual_blocks(
    engine: EngineName,
) -> None:
    result = generate_code(build_linear_periodic_chain_spec(), engine=engine)

    assert "['results_list = []'" not in result.code
    assert "['remaining_operands = {" not in result.code


@pytest.mark.parametrize(
    ("engine", "expected_snippet"),
    [
        (EngineName.QUIMB, "import quimb.tensor as qtn"),
        (EngineName.EINSUM_NUMPY, "result = np.einsum("),
        (EngineName.EINSUM_TORCH, "result = torch.einsum("),
    ],
)
def test_linear_periodic_codegen_supports_remaining_backends(
    engine: EngineName,
    expected_snippet: str,
) -> None:
    result = generate_code(build_linear_periodic_chain_spec(), engine=engine)

    assert "if n < 2:" in result.code
    assert "for cell_index in range(1, n - 1):" in result.code
    assert expected_snippet in result.code


@pytest.mark.parametrize(
    "engine",
    [
        pytest.param(EngineName.TENSORNETWORK, marks=pytest.mark.optional_backend),
        pytest.param(EngineName.TENSORKROWCH, marks=pytest.mark.heavy_backend),
    ],
)
def test_linear_periodic_codegen_executes_for_supported_backends(
    engine: EngineName,
) -> None:
    _import_required_backend(engine)

    result = generate_code(build_linear_periodic_chain_spec(), engine=engine)
    namespace = _execute_generated_code(result.code, n=3)

    assert "network_nodes" in namespace
    assert "open_edges" in namespace


@pytest.mark.parametrize(
    "engine",
    [
        pytest.param(EngineName.TENSORNETWORK, marks=pytest.mark.optional_backend),
        pytest.param(EngineName.TENSORKROWCH, marks=pytest.mark.heavy_backend),
    ],
)
@pytest.mark.parametrize(
    "spec_factory",
    [
        build_linear_periodic_carry_chain_spec,
        build_linear_periodic_partial_carry_chain_spec,
    ],
)
def test_linear_periodic_carry_codegen_executes_for_supported_backends(
    engine: EngineName,
    spec_factory: Callable[[], NetworkSpec],
) -> None:
    _import_required_backend(engine)

    result = generate_code(spec_factory(), engine=engine)
    namespace = _execute_generated_code(result.code, n=3)

    assert "network_nodes" in namespace
    assert "open_edges" in namespace
    assert "result" in namespace


def test_linear_periodic_carry_codegen_threads_interface_payloads() -> None:
    result = generate_code(
        build_linear_periodic_carry_chain_spec(),
        engine=EngineName.TENSORNETWORK,
    )

    assert "previous_payload = build_initial_cell()" in result.code
    assert "'operand':" in result.code
    assert "'outgoing_interface':" in result.code
    assert "next_boundary_operand" not in result.code


@pytest.mark.parametrize(
    ("engine", "expected_names"),
    [
        pytest.param(
            EngineName.QUIMB,
            {"network", "open_inds"},
            marks=pytest.mark.optional_backend,
        ),
        pytest.param(
            EngineName.EINSUM_NUMPY,
            {"result"},
            marks=pytest.mark.optional_backend,
        ),
        pytest.param(
            EngineName.EINSUM_TORCH,
            {"result"},
            marks=pytest.mark.heavy_backend,
        ),
    ],
)
def test_linear_periodic_codegen_executes_for_remaining_backends(
    engine: EngineName,
    expected_names: set[str],
) -> None:
    _import_required_backend(engine)

    result = generate_code(build_linear_periodic_chain_spec(), engine=engine)
    namespace = _execute_generated_code(result.code, n=3)

    for expected_name in expected_names:
        assert expected_name in namespace


@pytest.mark.parametrize(
    ("engine", "spec_factory", "expected_snippet"),
    [
        (
            EngineName.QUIMB,
            build_linear_periodic_carry_chain_spec,
            "network.contract_between(",
        ),
        (
            EngineName.QUIMB,
            build_linear_periodic_partial_carry_chain_spec,
            "network.contract_between(",
        ),
        (
            EngineName.EINSUM_NUMPY,
            build_linear_periodic_carry_chain_spec,
            "results_list.append(np.einsum(",
        ),
        (
            EngineName.EINSUM_NUMPY,
            build_linear_periodic_partial_carry_chain_spec,
            "results_list.append(np.einsum(",
        ),
        (
            EngineName.EINSUM_TORCH,
            build_linear_periodic_carry_chain_spec,
            "results_list.append(torch.einsum(",
        ),
        (
            EngineName.EINSUM_TORCH,
            build_linear_periodic_partial_carry_chain_spec,
            "results_list.append(torch.einsum(",
        ),
    ],
)
def test_linear_periodic_carry_codegen_supports_remaining_backends(
    engine: EngineName,
    spec_factory: Callable[[], NetworkSpec],
    expected_snippet: str,
) -> None:
    result = generate_code(spec_factory(), engine=engine)

    assert "previous_payload = build_initial_cell()" in result.code
    assert expected_snippet in result.code


@pytest.mark.parametrize(
    (
        "engine",
        "spec_factory",
        "expected_names",
        "expect_non_empty_remaining_operands",
    ),
    [
        pytest.param(
            EngineName.QUIMB,
            build_linear_periodic_carry_chain_spec,
            {"network", "open_inds", "result"},
            False,
            marks=pytest.mark.optional_backend,
        ),
        pytest.param(
            EngineName.QUIMB,
            build_linear_periodic_partial_carry_chain_spec,
            {"network", "open_inds", "result"},
            False,
            marks=pytest.mark.optional_backend,
        ),
        pytest.param(
            EngineName.EINSUM_NUMPY,
            build_linear_periodic_carry_chain_spec,
            {"result", "remaining_operands"},
            False,
            marks=pytest.mark.optional_backend,
        ),
        pytest.param(
            EngineName.EINSUM_NUMPY,
            build_linear_periodic_partial_carry_chain_spec,
            {"result", "remaining_operands"},
            True,
            marks=pytest.mark.optional_backend,
        ),
        pytest.param(
            EngineName.EINSUM_TORCH,
            build_linear_periodic_carry_chain_spec,
            {"result", "remaining_operands"},
            False,
            marks=pytest.mark.heavy_backend,
        ),
        pytest.param(
            EngineName.EINSUM_TORCH,
            build_linear_periodic_partial_carry_chain_spec,
            {"result", "remaining_operands"},
            True,
            marks=pytest.mark.heavy_backend,
        ),
    ],
)
def test_linear_periodic_carry_codegen_executes_for_remaining_backends(
    engine: EngineName,
    spec_factory: Callable[[], NetworkSpec],
    expected_names: set[str],
    expect_non_empty_remaining_operands: bool,
) -> None:
    _import_required_backend(engine)

    result = generate_code(spec_factory(), engine=engine)
    namespace = _execute_generated_code(result.code, n=3)

    for expected_name in expected_names:
        assert expected_name in namespace
    if "remaining_operands" in expected_names:
        remaining_operands = namespace["remaining_operands"]
        assert isinstance(remaining_operands, dict)
        if expect_non_empty_remaining_operands:
            assert remaining_operands


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
def test_quimb_linear_periodic_codegen_supports_collection_formats(
    collection_format: TensorCollectionFormat,
    container_name: str,
    expected_snippets: list[str],
) -> None:
    result = generate_code(
        build_linear_periodic_chain_spec(),
        engine=EngineName.QUIMB,
        collection_format=collection_format,
    )

    for snippet in expected_snippets:
        assert snippet in result.code
    assert container_name in result.code
    assert "def build_initial_cell()" in result.code
    assert "network_tensors = list(initial_cell['tensors'])" in result.code


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
            ["tensor_rows = []", "tensor_rows.append([])"],
        ),
        (
            TensorCollectionFormat.DICT,
            "tensors_dict",
            ["tensors_dict = {}", "tensors_dict["],
        ),
    ],
)
def test_linear_periodic_codegen_supports_all_collection_formats(
    engine: EngineName,
    collection_format: TensorCollectionFormat,
    container_name: str,
    expected_snippets: list[str],
) -> None:
    result = generate_code(
        build_linear_periodic_chain_spec(),
        engine=engine,
        collection_format=collection_format,
    )

    assert container_name in result.code
    for snippet in expected_snippets:
        assert snippet in result.code


@pytest.mark.parametrize("engine", list(EngineName))
def test_grid_periodic_codegen_uses_cell_helpers_and_free_n_m_loops(
    engine: EngineName,
) -> None:
    result = generate_code(build_grid_periodic_grid_spec(), engine=engine)

    assert "def build_top_left_cell(" in result.code
    assert "def build_top_cell(column_index" in result.code
    assert "def build_center_cell(column_index: int, row_index: int)" in result.code
    assert (
        "def build_bottom_right_cell(column_index: int, row_index: int)" in result.code
    )
    assert "validate_grid_shape(n, m)" in result.code
    assert "if n < 2:" in result.code
    assert "if m < 2:" in result.code
    assert "for column_index in range(1, n - 1):" in result.code
    assert "for row_index in range(1, m - 1):" in result.code


@pytest.mark.parametrize(
    "engine",
    [
        EngineName.TENSORNETWORK,
        EngineName.TENSORKROWCH,
    ],
)
def test_grid_periodic_codegen_supports_graph_backends_without_execution(
    engine: EngineName,
) -> None:
    result = generate_code(build_grid_periodic_grid_spec(), engine=engine)

    assert "# Tensor Network Editor grid periodic mode" in result.code
    assert "connect_cell_interfaces(" in result.code
    assert "network_nodes.extend(" in result.code
    assert "open_edges.extend(" in result.code
    assert "build_bottom_right_cell(n - 1, m - 1)" in result.code
    if engine is EngineName.TENSORNETWORK:
        assert "import tensornetwork as tn" in result.code
        assert "tn.connect(" in result.code
    else:
        assert "import tensorkrowch as tk" in result.code
        assert "tk.connect(" in result.code


@pytest.mark.parametrize(
    "engine",
    [
        EngineName.QUIMB,
        EngineName.EINSUM_NUMPY,
        EngineName.EINSUM_TORCH,
    ],
)
def test_grid_periodic_codegen_supports_remaining_backends_without_execution(
    engine: EngineName,
) -> None:
    result = generate_code(build_grid_periodic_grid_spec(), engine=engine)

    assert "# Tensor Network Editor grid periodic mode" in result.code
    assert "horizontal_label(" in result.code
    assert "vertical_label(" in result.code
    assert "build_bottom_right_cell(n - 1, m - 1)" in result.code
    assert "for column_index in range(1, n - 1):" in result.code
    assert "for row_index in range(1, m - 1):" in result.code
    if engine is EngineName.QUIMB:
        assert "import quimb.tensor as qtn" in result.code
        assert "network = qtn.TensorNetwork(network_tensors)" in result.code
    if engine is EngineName.EINSUM_NUMPY:
        assert "import numpy as np" in result.code
        assert "result = np.einsum(" in result.code
    if engine is EngineName.EINSUM_TORCH:
        assert "import torch" in result.code
        assert "result = torch.einsum(" in result.code


@pytest.mark.parametrize("engine", list(EngineName))
def test_tree_periodic_codegen_uses_tree_helpers_and_total_depth_loops(
    engine: EngineName,
) -> None:
    result = generate_code(build_tree_periodic_tree_spec(), engine=engine)

    assert "def build_root_cell(" in result.code
    assert (
        "def build_branch_cell(level: int, node_index: int, parent_interface:"
        in result.code
    )
    assert (
        "def build_leaf_cell(level: int, node_index: int, parent_interface:"
        in result.code
    )
    assert "validate_tree_depth(n)" in result.code
    assert "if n < 3:" in result.code
    assert "branching_factor = 3" in result.code
    assert "frontier = list(root_cell['child_interfaces'])" in result.code
    assert "for level in range(1, n - 1):" in result.code
    assert "build_leaf_cell(n - 1, node_index, parent_interface)" in result.code


@pytest.mark.parametrize(
    "engine",
    [EngineName.TENSORNETWORK, EngineName.TENSORKROWCH],
)
def test_tree_periodic_codegen_supports_graph_backends(
    engine: EngineName,
) -> None:
    result = generate_code(build_tree_periodic_tree_spec(), engine=engine)

    assert "# Tensor Network Editor tree periodic mode" in result.code
    assert "def connect_tree_interfaces(" in result.code
    assert "network_nodes.extend(branch_cell['nodes'])" in result.code
    assert "open_edges.extend(leaf_cell['open_edges'])" in result.code
    if engine is EngineName.TENSORNETWORK:
        assert "import tensornetwork as tn" in result.code
        assert "tn.connect(" in result.code
    else:
        assert "import tensorkrowch as tk" in result.code
        assert "tk.connect(" in result.code


@pytest.mark.parametrize(
    "engine",
    [EngineName.QUIMB, EngineName.EINSUM_NUMPY, EngineName.EINSUM_TORCH],
)
def test_tree_periodic_codegen_supports_array_backends(
    engine: EngineName,
) -> None:
    result = generate_code(build_tree_periodic_tree_spec(), engine=engine)

    assert "# Tensor Network Editor tree periodic mode" in result.code
    assert "child_interfaces = []" in result.code
    assert "branching_factor = 3" in result.code
    if engine is EngineName.QUIMB:
        assert "import quimb.tensor as qtn" in result.code
        assert "network = qtn.TensorNetwork(network_tensors)" in result.code
        assert (
            "child_label(level: int, node_index: int, child_index: int, slot_index: int) -> str"
            in result.code
        )
    if engine is EngineName.EINSUM_NUMPY:
        assert "import numpy as np" in result.code
        assert "result = np.einsum(" in result.code
        assert (
            "child_label(level: int, node_index: int, child_index: int, slot_index: int) -> int"
            in result.code
        )
    if engine is EngineName.EINSUM_TORCH:
        assert "import torch" in result.code
        assert "result = torch.einsum(" in result.code
        assert (
            "child_label(level: int, node_index: int, child_index: int, slot_index: int) -> int"
            in result.code
        )


@pytest.mark.parametrize(
    "engine",
    [EngineName.EINSUM_NUMPY, EngineName.EINSUM_TORCH],
)
def test_einsum_codegen_uses_integer_sublist_form_for_many_labels(
    engine: EngineName,
) -> None:
    result = generate_code(build_many_label_spec(), engine=engine)

    append_snippet = (
        "results_list.append(np.einsum("
        if engine is EngineName.EINSUM_NUMPY
        else "results_list.append(torch.einsum("
    )

    assert "integer-sublist form because the network uses many labels" in result.code
    assert append_snippet in result.code
    assert "result = results_list[-1]" in result.code
    assert "# Einsum equation:" not in result.code


@pytest.mark.parametrize(
    "engine",
    [
        pytest.param(EngineName.EINSUM_NUMPY, marks=pytest.mark.optional_backend),
        pytest.param(EngineName.EINSUM_TORCH, marks=pytest.mark.heavy_backend),
    ],
)
def test_einsum_codegen_executes_for_empty_network(engine: EngineName) -> None:
    result = generate_code(build_empty_spec(), engine=engine)
    namespace: dict[str, object] = {}
    _import_required_backend(engine)

    exec(result.code, namespace, namespace)

    assert "result" in namespace
