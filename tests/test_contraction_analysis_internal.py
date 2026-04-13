from __future__ import annotations

from tensor_network_editor._contraction_analysis_types import (
    AutomaticContractionPlanAnalysis,
    AutomaticContractionSummary,
    ContractionStepAnalysis,
    ManualContractionPlanAnalysis,
    ManualContractionSummary,
)


def test_contraction_analysis_internal_types_preserve_public_payload_shape() -> None:
    manual = ManualContractionSummary(
        total_estimated_flops=12,
        total_estimated_macs=6,
        peak_intermediate_size=4,
        peak_intermediate_bytes=32,
        final_shape=(2, 2),
        completion_status="complete",
        remaining_operand_ids=("step_ab",),
    )
    automatic = AutomaticContractionSummary(
        total_estimated_flops=20,
        total_estimated_macs=10,
        peak_intermediate_size=8,
        peak_intermediate_bytes=64,
    )

    assert manual.to_dict()["completion_status"] == "complete"
    assert manual.to_dict()["remaining_operand_ids"] == ["step_ab"]
    assert manual.to_dict()["peak_intermediate_bytes"] == 32
    assert automatic.to_dict()["peak_intermediate_size"] == 8
    assert automatic.to_dict()["peak_intermediate_bytes"] == 64


def test_contraction_analysis_internal_compare_module_reports_deltas() -> None:
    from tensor_network_editor._contraction_analysis_compare import (
        _compare_plan_analyses,
    )

    manual = ManualContractionPlanAnalysis(
        status="complete",
        steps=[
            ContractionStepAnalysis(
                step_id="manual_step",
                left_operand_id="left",
                right_operand_id="right",
                result_operand_id="manual_step",
                contracted_labels=("bond",),
                surviving_labels=("left", "right"),
                result_shape=(2, 2),
                result_rank=2,
                estimated_flops=48,
                estimated_macs=24,
                intermediate_size=16,
            )
        ],
        summary=ManualContractionSummary(
            total_estimated_flops=48,
            total_estimated_macs=24,
            peak_intermediate_size=16,
            peak_intermediate_bytes=128,
            final_shape=(2, 2),
            completion_status="complete",
            remaining_operand_ids=("manual_step",),
        ),
    )
    automatic = AutomaticContractionPlanAnalysis(
        status="complete",
        steps=[
            ContractionStepAnalysis(
                step_id="auto_step",
                left_operand_id="left",
                right_operand_id="right",
                result_operand_id="auto_step",
                contracted_labels=("bond",),
                surviving_labels=("left",),
                result_shape=(2,),
                result_rank=1,
                estimated_flops=24,
                estimated_macs=12,
                intermediate_size=8,
            )
        ],
        summary=AutomaticContractionSummary(
            total_estimated_flops=24,
            total_estimated_macs=12,
            peak_intermediate_size=8,
            peak_intermediate_bytes=64,
        ),
    )

    comparison = _compare_plan_analyses(
        baseline_label="manual",
        baseline_analysis=manual,
        candidate_label="automatic_full",
        candidate_analysis=automatic,
        memory_dtype="float64",
    )

    assert comparison.status == "complete"
    assert comparison.delta_total_estimated_flops == -24
    assert comparison.delta_peak_intermediate_bytes == -64
    assert comparison.baseline_peak_step_id == "manual_step"
    assert comparison.candidate_peak_step_id == "auto_step"
    assert comparison.baseline_bottleneck_labels == ("bond", "left", "right")
