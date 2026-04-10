from __future__ import annotations

from tensor_network_editor._contraction_analysis import analyze_contraction
from tensor_network_editor.models import (
    CanvasPosition,
    ContractionPlanSpec,
    ContractionStepSpec,
    EdgeEndpointRef,
    EdgeSpec,
    IndexSpec,
    NetworkSpec,
    TensorSpec,
)
from tests.factories import build_three_tensor_spec


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
