from __future__ import annotations

from tensor_network_editor._contraction_analysis_types import (
    AutomaticContractionSummary,
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
