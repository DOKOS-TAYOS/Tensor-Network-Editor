from __future__ import annotations

from typing import cast

import pytest

from tensor_network_editor.codegen.shared.common import prepare_network
from tensor_network_editor.internal.analysis._contraction_analysis import (
    analyze_contraction,
)
from tensor_network_editor.internal.analysis._contraction_plan import (
    prepare_contraction_inputs,
    simulate_contraction_plan,
)
from tensor_network_editor.models import (
    CanvasPosition,
    ContractionPlanSpec,
    ContractionStepSpec,
    EdgeEndpointRef,
    EdgeSpec,
    IndexSpec,
    LinearPeriodicCellName,
    NetworkSpec,
    TensorSpec,
    TreePeriodicCellName,
)
from tensor_network_editor.types import JSONValue
from tensor_network_editor.validation import validate_spec
from tests.factories import (
    build_grid_periodic_grid_spec,
    build_linear_periodic_partial_carry_chain_spec,
    build_three_tensor_hyperedge_spec,
    build_three_tensor_spec,
    build_tree_periodic_tree_spec,
)


def build_four_tensor_chain_spec() -> NetworkSpec:
    return NetworkSpec(
        id="network_chain_four",
        name="chain-four",
        tensors=[
            TensorSpec(
                id="tensor_a",
                name="A",
                position=CanvasPosition(x=40.0, y=120.0),
                indices=[
                    IndexSpec(id="tensor_a_i", name="i", dimension=2),
                    IndexSpec(id="tensor_a_x", name="x", dimension=50),
                ],
            ),
            TensorSpec(
                id="tensor_b",
                name="B",
                position=CanvasPosition(x=180.0, y=120.0),
                indices=[
                    IndexSpec(id="tensor_b_x", name="x", dimension=50),
                    IndexSpec(id="tensor_b_y", name="y", dimension=3),
                ],
            ),
            TensorSpec(
                id="tensor_c",
                name="C",
                position=CanvasPosition(x=320.0, y=120.0),
                indices=[
                    IndexSpec(id="tensor_c_y", name="y", dimension=3),
                    IndexSpec(id="tensor_c_z", name="z", dimension=50),
                ],
            ),
            TensorSpec(
                id="tensor_d",
                name="D",
                position=CanvasPosition(x=460.0, y=120.0),
                indices=[
                    IndexSpec(id="tensor_d_z", name="z", dimension=50),
                    IndexSpec(id="tensor_d_j", name="j", dimension=2),
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
            EdgeSpec(
                id="edge_z",
                name="bond_z",
                left=EdgeEndpointRef(tensor_id="tensor_c", index_id="tensor_c_z"),
                right=EdgeEndpointRef(tensor_id="tensor_d", index_id="tensor_d_z"),
            ),
        ],
    )


def build_long_tensor_chain_spec(tensor_count: int) -> NetworkSpec:
    tensors: list[TensorSpec] = []
    edges: list[EdgeSpec] = []
    steps: list[ContractionStepSpec] = []
    for index in range(tensor_count):
        tensors.append(
            TensorSpec(
                id=f"tensor_{index}",
                name=f"T{index}",
                position=CanvasPosition(x=80.0 + index * 140.0, y=120.0),
                indices=[
                    IndexSpec(id=f"tensor_{index}_left", name="left", dimension=2),
                    IndexSpec(id=f"tensor_{index}_right", name="right", dimension=2),
                ],
            )
        )
        if index <= 0:
            continue
        edges.append(
            EdgeSpec(
                id=f"edge_{index}",
                name=f"bond_{index}",
                left=EdgeEndpointRef(
                    tensor_id=f"tensor_{index - 1}",
                    index_id=f"tensor_{index - 1}_right",
                ),
                right=EdgeEndpointRef(
                    tensor_id=f"tensor_{index}",
                    index_id=f"tensor_{index}_left",
                ),
            )
        )
        steps.append(
            ContractionStepSpec(
                id=f"step_{index}",
                left_operand_id="tensor_0" if index == 1 else f"step_{index - 1}",
                right_operand_id=f"tensor_{index}",
            )
        )

    return NetworkSpec(
        id="network_long_chain",
        name="long-chain",
        tensors=tensors,
        edges=edges,
        contraction_plan=ContractionPlanSpec(
            id="plan_long_chain",
            name="Long chain",
            steps=steps,
        ),
    )


def test_analyze_contraction_reports_manual_pairwise_costs(
    sample_spec: NetworkSpec,
) -> None:
    result = analyze_contraction(sample_spec)

    assert result.memory_dtype == "float64"
    assert result.network_output_shape == (2, 4)
    assert result.manual.status == "complete"
    assert len(result.manual.steps) == 1
    assert result.manual.steps[0].estimated_flops == 48
    assert result.manual.steps[0].estimated_macs == 24
    assert result.manual.steps[0].intermediate_size == 8
    assert result.manual.summary.total_estimated_flops == 48
    assert result.manual.summary.total_estimated_macs == 24
    assert result.manual.summary.final_shape == (2, 4)
    assert result.manual.summary.peak_intermediate_size == 8
    assert result.manual.summary.peak_intermediate_bytes == 64
    assert result.automatic_future is not None
    assert result.automatic_past is not None


def test_analyze_contraction_lowers_hyperedges_to_synthetic_operands() -> None:
    result = analyze_contraction(build_three_tensor_hyperedge_spec())

    assert result.manual.status == "incomplete"
    assert result.network_output_shape == (2, 5, 7)
    assert result.warnings == [
        "Hyperedges are analyzed as generated copy tensors; the visual model is unchanged."
    ]
    assert [operand.operand_id for operand in result.synthetic_operands] == [
        "hyperedge_copy_hyperedge_h"
    ]
    assert result.synthetic_operands[0].source_hyperedge_id == "hyperedge_h"
    payload = result.to_dict()
    synthetic_operands = cast(list[dict[str, JSONValue]], payload["synthetic_operands"])
    assert synthetic_operands[0]["operand_id"] == ("hyperedge_copy_hyperedge_h")


def test_validate_spec_accepts_hyperedge_synthetic_contraction_operands() -> None:
    spec = build_three_tensor_hyperedge_spec()
    spec.contraction_plan = ContractionPlanSpec(
        id="plan_hyperedge",
        name="Hyperedge plan",
        steps=[
            ContractionStepSpec(
                id="step_a_copy",
                left_operand_id="tensor_a",
                right_operand_id="hyperedge_copy_hyperedge_h",
            ),
            ContractionStepSpec(
                id="step_ab",
                left_operand_id="step_a_copy",
                right_operand_id="tensor_b",
            ),
        ],
    )

    assert validate_spec(spec) == []


def test_validate_spec_rejects_unknown_hyperedge_contraction_operands() -> None:
    spec = build_three_tensor_hyperedge_spec()
    spec.contraction_plan = ContractionPlanSpec(
        id="plan_hyperedge",
        name="Hyperedge plan",
        steps=[
            ContractionStepSpec(
                id="step_bad",
                left_operand_id="tensor_a",
                right_operand_id="missing_copy_operand",
            )
        ],
    )

    issues = validate_spec(spec)

    assert [issue.code for issue in issues] == ["invalid-contraction-operand"]


def test_analyze_contraction_uses_active_linear_periodic_cell_plan() -> None:
    spec = build_linear_periodic_partial_carry_chain_spec()

    result = analyze_contraction(spec)

    assert result.network_output_shape == (11, 19, 13, 29)
    assert result.manual.status == "incomplete"
    assert [step.step_id for step in result.manual.steps] == [
        "periodic_from_previous_partial",
        "periodic_partial_carry",
    ]
    assert result.manual.summary.remaining_operand_ids == (
        "periodic_partial_carry",
        "periodic_from_previous_partial",
        "periodic_previous_right_tensor",
        "periodic_next_right_tensor",
    )
    assert result.automatic_full.status == "complete"
    assert result.automatic_full.steps
    assert result.comparisons["manual_vs_automatic_full"].status == "complete"


def test_analyze_contraction_respects_linear_periodic_active_cell() -> None:
    spec = build_linear_periodic_partial_carry_chain_spec()
    assert spec.linear_periodic_chain is not None
    spec.linear_periodic_chain.active_cell = LinearPeriodicCellName.FINAL

    result = analyze_contraction(spec)

    assert result.network_output_shape == (31, 37)
    assert [step.step_id for step in result.manual.steps] == [
        "final_from_previous_partial"
    ]
    assert result.manual.steps[0].left_operand_id == "__linear_previous__"


def test_analyze_contraction_uses_grid_periodic_reserved_border_operands() -> None:
    spec = build_grid_periodic_grid_spec()
    assert spec.grid_periodic_grid is not None
    spec.grid_periodic_grid.center_cell.contraction_plan = ContractionPlanSpec(
        id="grid_plan",
        name="Grid plan",
        steps=[
            ContractionStepSpec(
                id="grid_step",
                left_operand_id="__grid_left__",
                right_operand_id="center_tensor",
            )
        ],
    )

    result = analyze_contraction(spec)

    assert result.manual.status == "incomplete"
    assert [step.left_operand_id for step in result.manual.steps] == ["__grid_left__"]
    assert result.manual.summary.remaining_operand_ids[0] == "grid_step"
    assert set(result.manual.summary.remaining_operand_ids[1:]) == {
        "__grid_up__",
        "__grid_right__",
        "__grid_down__",
    }


def test_analyze_contraction_uses_tree_periodic_reserved_border_operands() -> None:
    spec = build_tree_periodic_tree_spec()
    assert spec.tree_periodic_tree is not None
    spec.tree_periodic_tree.active_cell = TreePeriodicCellName.BRANCH
    spec.tree_periodic_tree.branch_cell.contraction_plan = ContractionPlanSpec(
        id="tree_plan",
        name="Tree plan",
        steps=[
            ContractionStepSpec(
                id="tree_step",
                left_operand_id="__tree_parent__",
                right_operand_id="branch_tensor",
            )
        ],
    )

    result = analyze_contraction(spec)

    assert result.manual.status == "incomplete"
    assert [step.left_operand_id for step in result.manual.steps] == ["__tree_parent__"]
    assert result.manual.summary.remaining_operand_ids[0] == "tree_step"
    assert set(result.manual.summary.remaining_operand_ids[1:]) == {
        "__tree_child_0__",
        "__tree_child_1__",
        "__tree_child_2__",
    }


def test_analyze_contraction_marks_incomplete_manual_plan() -> None:
    result = analyze_contraction(build_three_tensor_spec())

    assert result.network_output_shape == (2, 7)
    assert result.manual.status == "incomplete"
    assert result.manual.steps[0].estimated_flops == 60
    assert result.manual.steps[0].estimated_macs == 30
    assert result.manual.summary.total_estimated_flops == 60
    assert result.manual.summary.total_estimated_macs == 30
    assert result.manual.summary.final_shape == (2, 5)
    assert result.manual.summary.peak_intermediate_size == 10
    assert result.manual.summary.peak_intermediate_bytes == 80
    assert result.manual.summary.remaining_operand_ids == ("step_ab", "tensor_c")
    assert result.automatic_future.status in {"complete", "unavailable"}
    assert result.automatic_past.status in {"complete", "unavailable"}
    if result.automatic_future.status == "complete":
        assert len(result.automatic_future.steps) == 1
        assert result.automatic_future.summary.total_estimated_flops == 140
        assert result.automatic_future.summary.total_estimated_macs == 70
        assert result.automatic_future.summary.peak_intermediate_bytes == 112
        assert {
            result.automatic_future.steps[0].left_operand_id,
            result.automatic_future.steps[0].right_operand_id,
        } == {"step_ab", "tensor_c"}
    if result.automatic_past.status == "complete":
        assert len(result.automatic_past.steps) == 1
        assert result.automatic_past.steps[0].result_operand_id == "step_ab"
        assert result.automatic_past.summary.total_estimated_flops == 60
        assert result.automatic_past.summary.total_estimated_macs == 30
        assert {
            result.automatic_past.steps[0].left_operand_id,
            result.automatic_past.steps[0].right_operand_id,
        } == {"tensor_a", "tensor_b"}


def test_analyze_contraction_preserves_placeholder_future_comparison() -> None:
    result = analyze_contraction(build_three_tensor_spec())
    comparison = result.comparisons["manual_remaining_vs_automatic_future"]

    assert comparison.status == "unavailable"
    assert comparison.baseline_label == "manual_remaining"
    assert comparison.candidate_label == "automatic_future"
    assert comparison.memory_dtype == "float64"
    assert (
        comparison.message
        == "The saved manual plan does not expose a separate remaining suffix to compare yet."
    )


def test_analyze_contraction_marks_planner_value_errors_as_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakePlannerModule:
        @staticmethod
        def contract_path(
            equation: str,
            *operand_shapes: tuple[int, ...],
            shapes: bool,
            optimize: str,
        ) -> tuple[tuple[tuple[int, int], ...], object]:
            del equation, operand_shapes, shapes, optimize
            raise ValueError("planner shape mismatch")

    def fake_import_module(name: str) -> object:
        assert name == "opt_einsum"
        return FakePlannerModule

    monkeypatch.setattr(
        "tensor_network_editor.internal.analysis._contraction_analysis_automatic.import_module",
        fake_import_module,
    )

    result = analyze_contraction(build_three_tensor_spec())

    assert result.automatic_full.status == "unavailable"
    assert result.automatic_full.message == (
        "Automatic greedy path analysis failed: planner shape mismatch"
    )
    assert result.comparisons["manual_vs_automatic_full"].status == "unavailable"


def test_analyze_contraction_reports_missing_required_opt_einsum_clearly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_import_module(name: str) -> object:
        assert name == "opt_einsum"
        raise ImportError("opt_einsum is not installed")

    monkeypatch.setattr(
        "tensor_network_editor.internal.analysis._contraction_analysis_automatic.import_module",
        fake_import_module,
    )

    result = analyze_contraction(build_three_tensor_spec())

    assert result.automatic_full.status == "unavailable"
    assert result.automatic_full.message == (
        "The required opt_einsum dependency is not available in the current .venv. Reinstall tensor-network-editor in this environment to enable Auto full, Auto future, and Auto past."
    )
    assert result.automatic_future.status == "unavailable"
    assert result.automatic_future.message == result.automatic_full.message


def test_analyze_contraction_does_not_hide_unexpected_planner_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakePlannerModule:
        @staticmethod
        def contract_path(
            equation: str,
            *operand_shapes: tuple[int, ...],
            shapes: bool,
            optimize: str,
        ) -> tuple[tuple[tuple[int, int], ...], object]:
            del equation, operand_shapes, shapes, optimize
            raise RuntimeError("planner exploded")

    def fake_import_module(name: str) -> object:
        assert name == "opt_einsum"
        return FakePlannerModule

    monkeypatch.setattr(
        "tensor_network_editor.internal.analysis._contraction_analysis_automatic.import_module",
        fake_import_module,
    )

    with pytest.raises(RuntimeError, match="planner exploded"):
        analyze_contraction(build_three_tensor_spec())


def test_analyze_contraction_accepts_multi_step_manual_plan() -> None:
    spec = build_three_tensor_spec()
    spec.contraction_plan = ContractionPlanSpec(
        id="plan_chain_complete",
        name="Chain complete",
        steps=[
            ContractionStepSpec(
                id="step_ab",
                left_operand_id="tensor_a",
                right_operand_id="tensor_b",
            ),
            ContractionStepSpec(
                id="step_abc",
                left_operand_id="step_ab",
                right_operand_id="tensor_c",
            ),
        ],
    )

    result = analyze_contraction(spec)

    assert result.manual.status == "complete"
    assert len(result.manual.steps) == 2
    assert result.manual.steps[1].left_operand_id == "step_ab"
    assert result.manual.steps[1].right_operand_id == "tensor_c"
    assert result.manual.steps[1].estimated_flops == 140
    assert result.manual.steps[1].estimated_macs == 70
    assert result.manual.summary.total_estimated_flops == 200
    assert result.manual.summary.total_estimated_macs == 100
    assert result.manual.summary.peak_intermediate_size == 14
    assert result.manual.summary.peak_intermediate_bytes == 112
    assert result.manual.summary.final_shape == (2, 7)
    assert result.manual.summary.remaining_operand_ids == ("step_abc",)


def test_automatic_summaries_do_not_expose_final_shape() -> None:
    result = analyze_contraction(build_three_tensor_spec())

    assert not hasattr(result.automatic_future.summary, "final_shape")
    assert not hasattr(result.automatic_past.summary, "final_shape")


def test_matrix_multiplication_counts_two_flops_per_mac() -> None:
    spec = NetworkSpec(
        id="network_mm",
        name="matrix multiply",
        tensors=[
            TensorSpec(
                id="tensor_a",
                name="A",
                position=CanvasPosition(x=80.0, y=120.0),
                indices=[
                    IndexSpec(id="tensor_a_i", name="i", dimension=2),
                    IndexSpec(id="tensor_a_k", name="k", dimension=2),
                ],
            ),
            TensorSpec(
                id="tensor_b",
                name="B",
                position=CanvasPosition(x=240.0, y=120.0),
                indices=[
                    IndexSpec(id="tensor_b_k", name="k", dimension=2),
                    IndexSpec(id="tensor_b_j", name="j", dimension=2),
                ],
            ),
        ],
        edges=[
            EdgeSpec(
                id="edge_k",
                name="bond_k",
                left=EdgeEndpointRef(tensor_id="tensor_a", index_id="tensor_a_k"),
                right=EdgeEndpointRef(tensor_id="tensor_b", index_id="tensor_b_k"),
            )
        ],
        contraction_plan=ContractionPlanSpec(
            id="plan_mm",
            name="Matrix multiply",
            steps=[
                ContractionStepSpec(
                    id="step_ab",
                    left_operand_id="tensor_a",
                    right_operand_id="tensor_b",
                )
            ],
        ),
    )

    result = analyze_contraction(spec)

    assert result.manual.steps[0].estimated_macs == 8
    assert result.manual.steps[0].estimated_flops == 16


def test_analyze_contraction_past_preserves_existing_root_step_id() -> None:
    spec = build_three_tensor_spec()
    spec.contraction_plan = ContractionPlanSpec(
        id="plan_chain_complete",
        name="Chain complete",
        steps=[
            ContractionStepSpec(
                id="step_ab",
                left_operand_id="tensor_a",
                right_operand_id="tensor_b",
            ),
            ContractionStepSpec(
                id="step_abc",
                left_operand_id="step_ab",
                right_operand_id="tensor_c",
            ),
        ],
    )

    result = analyze_contraction(spec)

    assert result.automatic_past.status in {"complete", "unavailable"}
    if result.automatic_past.status == "complete":
        assert result.automatic_past.steps
        assert result.automatic_past.steps[-1].result_operand_id == "step_abc"
        assert result.automatic_past.steps[-1].step_id == "step_abc"


def test_analyze_contraction_future_is_complete_when_manual_path_is_complete() -> None:
    spec = build_three_tensor_spec()
    spec.contraction_plan = ContractionPlanSpec(
        id="plan_chain_complete",
        name="Chain complete",
        steps=[
            ContractionStepSpec(
                id="step_ab",
                left_operand_id="tensor_a",
                right_operand_id="tensor_b",
            ),
            ContractionStepSpec(
                id="step_abc",
                left_operand_id="step_ab",
                right_operand_id="tensor_c",
            ),
        ],
    )

    result = analyze_contraction(spec)

    assert result.automatic_future.status == "complete"
    assert result.automatic_future.steps == []


def test_analyze_contraction_reuses_one_manual_plan_simulation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = build_three_tensor_spec()
    spec.contraction_plan = ContractionPlanSpec(
        id="plan_chain_complete",
        name="Chain complete",
        steps=[
            ContractionStepSpec(
                id="step_ab",
                left_operand_id="tensor_a",
                right_operand_id="tensor_b",
            ),
            ContractionStepSpec(
                id="step_abc",
                left_operand_id="step_ab",
                right_operand_id="tensor_c",
            ),
        ],
    )
    original_simulate = __import__(
        "tensor_network_editor.internal.analysis._contraction_analysis_manual",
        fromlist=["simulate_contraction_plan"],
    ).simulate_contraction_plan
    call_count = 0

    def counting_simulate_contraction_plan(*args: object, **kwargs: object) -> object:
        nonlocal call_count
        call_count += 1
        return original_simulate(*args, **kwargs)

    monkeypatch.setattr(
        "tensor_network_editor.internal.analysis._contraction_analysis_manual.simulate_contraction_plan",
        counting_simulate_contraction_plan,
    )

    result = analyze_contraction(spec)

    assert result.manual.status == "complete"
    assert call_count == 1


def test_analyze_contraction_reuses_equivalent_automatic_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = build_three_tensor_spec()
    spec.contraction_plan = ContractionPlanSpec(
        id="plan_chain_complete",
        name="Chain complete",
        steps=[
            ContractionStepSpec(
                id="step_ab",
                left_operand_id="tensor_a",
                right_operand_id="tensor_b",
            ),
            ContractionStepSpec(
                id="step_abc",
                left_operand_id="step_ab",
                right_operand_id="tensor_c",
            ),
        ],
    )
    call_count = 0

    class FakePlannerModule:
        @staticmethod
        def contract_path(
            equation: str,
            *operand_shapes: tuple[int, ...],
            shapes: bool,
            optimize: str,
        ) -> tuple[tuple[tuple[int, int], ...], object]:
            nonlocal call_count
            del equation, operand_shapes, shapes, optimize
            call_count += 1
            return ((0, 1), (0, 1)), object()

    def fake_import_module(name: str) -> object:
        assert name == "opt_einsum"
        return FakePlannerModule

    monkeypatch.setattr(
        "tensor_network_editor.internal.analysis._contraction_analysis_automatic.import_module",
        fake_import_module,
    )

    result = analyze_contraction(spec)

    assert result.automatic_full.status == "complete"
    assert result.automatic_past.status == "complete"
    assert call_count == 1


def test_long_manual_chain_preserves_remaining_ids_shapes_and_source_tensors() -> None:
    spec = build_long_tensor_chain_spec(40)
    prepared = prepare_network(spec)
    contraction_inputs = prepare_contraction_inputs(prepared)

    simulation = simulate_contraction_plan(
        initial_operand_ids=contraction_inputs.initial_operand_ids,
        initial_operands=contraction_inputs.initial_operands,
        initial_axis_names=contraction_inputs.initial_axis_names,
        dimension_by_label=contraction_inputs.dimension_by_label,
        plan=spec.contraction_plan,
    )
    expected_final_labels = (
        contraction_inputs.initial_operands["tensor_0"][0],
        contraction_inputs.initial_operands["tensor_39"][1],
    )

    assert simulation.remaining_operand_ids == ("step_39",)
    assert simulation.remaining_operands["step_39"] == expected_final_labels
    assert simulation.steps[-1].result_shape == (2, 2)
    assert simulation.source_tensor_ids_by_operand_id["step_39"] == tuple(
        f"tensor_{index}" for index in range(40)
    )


def test_analyze_contraction_reports_full_automatic_plan_and_deltas() -> None:
    spec = build_four_tensor_chain_spec()
    spec.contraction_plan = ContractionPlanSpec(
        id="plan_chain_manual_reverse",
        name="Reverse chain",
        steps=[
            ContractionStepSpec(
                id="step_cd",
                left_operand_id="tensor_c",
                right_operand_id="tensor_c",
            )
        ],
    )
    spec.contraction_plan.steps[0].right_operand_id = "tensor_d"
    spec.contraction_plan.steps.extend(
        [
            ContractionStepSpec(
                id="step_bcd",
                left_operand_id="tensor_b",
                right_operand_id="step_cd",
            ),
            ContractionStepSpec(
                id="step_abcd",
                left_operand_id="tensor_a",
                right_operand_id="step_bcd",
            ),
        ]
    )

    result = analyze_contraction(spec)
    comparison = result.comparisons["manual_vs_automatic_full"]

    assert result.automatic_full.status in {"complete", "unavailable"}
    assert comparison.memory_dtype == "float64"
    if result.automatic_full.status == "complete":
        assert len(result.automatic_full.steps) == 3
        assert result.automatic_full.summary.total_estimated_flops == 1224
        assert result.automatic_full.summary.total_estimated_macs == 612
        assert result.automatic_full.summary.peak_intermediate_size == 6
        assert comparison.status == "complete"
        assert comparison.baseline_label == "manual"
        assert comparison.candidate_label == "automatic_full"
        assert comparison.delta_total_estimated_flops == -376
        assert comparison.delta_total_estimated_macs == -188
        assert comparison.delta_peak_intermediate_size == -94
        assert comparison.baseline_peak_intermediate_bytes == 800
        assert comparison.candidate_peak_intermediate_bytes == 48
        assert comparison.delta_peak_intermediate_bytes == -752
        assert comparison.baseline_peak_step_id == "step_bcd"
        assert comparison.candidate_peak_step_id in {
            result.automatic_full.steps[0].step_id,
            result.automatic_full.steps[1].step_id,
        }
        assert comparison.candidate_bottleneck_labels
    else:
        assert comparison.status == "unavailable"


def test_analyze_contraction_peak_bytes_respect_requested_dtype() -> None:
    spec = build_four_tensor_chain_spec()
    spec.contraction_plan = ContractionPlanSpec(
        id="plan_chain_manual_reverse",
        name="Reverse chain",
        steps=[
            ContractionStepSpec(
                id="step_cd",
                left_operand_id="tensor_c",
                right_operand_id="tensor_c",
            )
        ],
    )
    spec.contraction_plan.steps[0].right_operand_id = "tensor_d"
    spec.contraction_plan.steps.extend(
        [
            ContractionStepSpec(
                id="step_bcd",
                left_operand_id="tensor_b",
                right_operand_id="step_cd",
            ),
            ContractionStepSpec(
                id="step_abcd",
                left_operand_id="tensor_a",
                right_operand_id="step_bcd",
            ),
        ]
    )

    float64_result = analyze_contraction(spec, memory_dtype="float64")
    float32_result = analyze_contraction(spec, memory_dtype="float32")

    float64_comparison = float64_result.comparisons["manual_vs_automatic_full"]
    float32_comparison = float32_result.comparisons["manual_vs_automatic_full"]

    assert float64_comparison.memory_dtype == "float64"
    assert float32_comparison.memory_dtype == "float32"
    assert float64_result.manual.summary.peak_intermediate_bytes == 800
    assert float32_result.manual.summary.peak_intermediate_bytes == 400
    if (
        float64_comparison.status == "complete"
        and float32_comparison.status == "complete"
    ):
        assert float64_result.automatic_full.summary.peak_intermediate_bytes == 48
        assert float32_result.automatic_full.summary.peak_intermediate_bytes == 24
        assert float64_comparison.baseline_peak_intermediate_bytes == 800
        assert float32_comparison.baseline_peak_intermediate_bytes == 400
        assert float64_comparison.candidate_peak_intermediate_bytes == 48
        assert float32_comparison.candidate_peak_intermediate_bytes == 24
