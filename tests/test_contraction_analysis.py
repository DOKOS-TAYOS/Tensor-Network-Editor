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


def test_analyze_contraction_reports_manual_pairwise_costs(
    sample_spec: NetworkSpec,
) -> None:
    result = analyze_contraction(sample_spec)

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
    assert result.automatic_global is not None
    assert result.automatic_local is not None


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
    assert result.manual.summary.remaining_operand_ids == ("step_ab", "tensor_c")
    assert result.automatic_global.status in {"complete", "unavailable"}
    assert result.automatic_local.status in {"complete", "unavailable"}
    if result.automatic_local.status == "complete":
        assert len(result.automatic_local.steps) == 1
        assert result.automatic_local.summary.total_estimated_flops == 60
        assert result.automatic_local.summary.total_estimated_macs == 30
        assert {
            result.automatic_local.steps[0].left_operand_id,
            result.automatic_local.steps[0].right_operand_id,
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
    assert result.manual.summary.final_shape == (2, 7)
    assert result.manual.summary.remaining_operand_ids == ("step_abc",)


def test_automatic_summaries_do_not_expose_final_shape() -> None:
    result = analyze_contraction(build_three_tensor_spec())

    assert not hasattr(result.automatic_global.summary, "final_shape")
    assert not hasattr(result.automatic_local.summary, "final_shape")


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
