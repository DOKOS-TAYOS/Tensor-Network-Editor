from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from tensor_network_editor.codegen.shared.common import (
    group_tensors_by_visual_rows,
    make_unique_identifiers,
    prepare_network,
    render_helper_function_lines,
    sanitize_identifier,
    tensor_collection_reference_by_id,
    tensor_variable_name,
)
from tensor_network_editor.models import (
    CanvasPosition,
    CodegenResult,
    EngineName,
    NetworkSpec,
    TensorCollectionFormat,
    TensorSpec,
)
from tests.factories import build_three_tensor_hyperedge_spec


class _CountingPosition:
    def __init__(self, *, x: float, y: float) -> None:
        self.x = x
        self._y = y
        self.y_reads = 0

    @property
    def y(self) -> float:
        self.y_reads += 1
        return self._y


def test_sanitize_identifier_normalizes_empty_and_numeric_names() -> None:
    assert sanitize_identifier("  Tensor A  ", "tensor") == "tensor_a"
    assert sanitize_identifier("123 bond", "edge") == "edge_123_bond"
    assert sanitize_identifier("!!!", "tensor") == "tensor"


def test_make_unique_identifiers_deduplicates_collisions() -> None:
    assert make_unique_identifiers(
        ["Tensor A", "tensor-a", "123", "123"],
        "tensor",
    ) == ["tensor_a", "tensor_a_2", "tensor_123", "tensor_123_2"]


def test_group_tensors_by_visual_rows_preserves_row_grouping_and_x_order() -> None:
    tensors = [
        TensorSpec(id="top_right", position=CanvasPosition(x=300.0, y=100.0)),
        TensorSpec(id="bottom_left", position=CanvasPosition(x=100.0, y=240.0)),
        TensorSpec(id="top_left", position=CanvasPosition(x=100.0, y=104.0)),
        TensorSpec(id="bottom_right", position=CanvasPosition(x=300.0, y=244.0)),
    ]

    rows = group_tensors_by_visual_rows(tensors)

    assert [[tensor.id for tensor in row] for row in rows] == [
        ["top_left", "top_right"],
        ["bottom_left", "bottom_right"],
    ]


def test_group_tensors_by_visual_rows_uses_linear_row_center_work() -> None:
    tensor_count = 150
    positions: list[_CountingPosition] = []
    tensors: list[TensorSpec] = []
    for index in range(tensor_count):
        tensor = TensorSpec(id=f"tensor_{index}", position=CanvasPosition())
        position = _CountingPosition(x=float(index), y=120.0)
        cast(Any, tensor).position = position
        positions.append(position)
        tensors.append(tensor)

    rows = group_tensors_by_visual_rows(tensors)

    assert len(rows) == 1
    assert [tensor.id for tensor in rows[0]] == [
        f"tensor_{index}" for index in range(tensor_count)
    ]
    assert sum(position.y_reads for position in positions) <= tensor_count * 5


def test_prepare_network_assigns_stable_labels(sample_spec: NetworkSpec) -> None:
    prepared = prepare_network(sample_spec)

    assert [tensor.variable_name for tensor in prepared.tensors] == ["a", "b"]
    assert [tensor.data_variable_name for tensor in prepared.tensors] == [
        "a_data",
        "b_data",
    ]
    assert [edge.label for edge in prepared.edges] == ["bond_x"]
    assert [edge.variable_name for edge in prepared.edges] == ["bond_x_edge"]
    assert [index.label for index in prepared.open_indices] == ["a_i", "b_j"]
    assert prepared.edges[0].left.label == "bond_x"
    assert prepared.edges[0].right.label == "bond_x"


def test_tensor_variable_name_resolves_known_tensors(sample_spec: NetworkSpec) -> None:
    prepared = prepare_network(sample_spec)

    assert tensor_variable_name(prepared, "tensor_a") == "a"
    with pytest.raises(KeyError, match="missing_tensor"):
        tensor_variable_name(prepared, "missing_tensor")


def test_tensor_collection_reference_by_id_uses_prepared_tensor_lookup() -> None:
    prepared = cast(
        Any,
        SimpleNamespace(
            tensor_by_id={
                "tensor_a": SimpleNamespace(
                    row_index=0,
                    column_index=0,
                    flat_index=0,
                )
            },
            tensors=None,
        ),
    )

    assert (
        tensor_collection_reference_by_id(
            prepared,
            "tensor_a",
            TensorCollectionFormat.LIST,
            "tensors",
        )
        == "tensors[0]"
    )


def test_prepare_network_lowers_hyperedges_to_copy_tensors_for_codegen() -> None:
    prepared = prepare_network(build_three_tensor_hyperedge_spec())

    assert prepared.spec.hyperedges == []
    assert len(prepared.tensors) == 4
    assert len(prepared.edges) == 3
    copy_tensors = [
        tensor
        for tensor in prepared.tensors
        if tensor.spec.metadata.get("generated_for_hyperedge") == "hyperedge_h"
    ]
    assert len(copy_tensors) == 1
    assert copy_tensors[0].spec.tensor_data is None
    assert copy_tensors[0].spec.shape == (3, 3, 3)


def test_render_helper_function_lines_indents_rendered_sections() -> None:
    helper_lines = render_helper_function_lines(
        helper_name="build_cell",
        helper_signature="slot_index: int",
        return_annotation="dict[str, object]",
        sections=[],
    )

    assert helper_lines == ["def build_cell(slot_index: int) -> dict[str, object]:"]


def test_dispatch_periodic_codegen_routes_supported_backends_and_roundtrip() -> None:
    from tensor_network_editor.codegen.modes._periodic_codegen import (
        dispatch_periodic_codegen,
    )

    seen_calls: list[tuple[str, str]] = []

    def render_array(payload: str) -> CodegenResult:
        seen_calls.append(("array", payload))
        return CodegenResult(engine=EngineName.EINSUM_NUMPY, code="array_result = 1\n")

    def render_graph(payload: str) -> CodegenResult:
        seen_calls.append(("graph", payload))
        return CodegenResult(engine=EngineName.TENSORNETWORK, code="graph_result = 1\n")

    spec = build_three_tensor_hyperedge_spec()

    array_result = dispatch_periodic_codegen(
        spec=spec,
        payload="array-payload",
        missing_payload_message="missing payload",
        unsupported_backend_label="periodic",
        engine=EngineName.EINSUM_NUMPY,
        include_roundtrip_metadata=True,
        array_renderer=render_array,
        graph_renderer=render_graph,
    )
    graph_result = dispatch_periodic_codegen(
        spec=spec,
        payload="graph-payload",
        missing_payload_message="missing payload",
        unsupported_backend_label="periodic",
        engine=EngineName.TENSORNETWORK,
        include_roundtrip_metadata=False,
        array_renderer=render_array,
        graph_renderer=render_graph,
    )

    assert seen_calls == [("array", "array-payload"), ("graph", "graph-payload")]
    assert "# TNE_SPEC_B64:" in array_result.code
    assert graph_result.code == "graph_result = 1\n"


def test_dispatch_periodic_codegen_rejects_missing_payload() -> None:
    from tensor_network_editor.codegen.modes._periodic_codegen import (
        dispatch_periodic_codegen,
    )
    from tensor_network_editor.errors import CodeGenerationError

    with pytest.raises(CodeGenerationError, match="grid payload"):
        dispatch_periodic_codegen(
            spec=NetworkSpec(name="missing payload"),
            payload=None,
            missing_payload_message="Grid periodic code generation requires a grid payload.",
            unsupported_backend_label="grid periodic",
            engine=EngineName.EINSUM_NUMPY,
            include_roundtrip_metadata=False,
            array_renderer=lambda payload: CodegenResult(
                engine=EngineName.EINSUM_NUMPY,
                code=f"{payload}\n",
            ),
            graph_renderer=lambda payload: CodegenResult(
                engine=EngineName.TENSORNETWORK,
                code=f"{payload}\n",
            ),
        )
