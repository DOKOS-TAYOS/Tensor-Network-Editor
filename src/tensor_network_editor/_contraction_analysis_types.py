"""Structured results returned by contraction analysis endpoints and helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

from ._memory_dtypes import DEFAULT_MEMORY_DTYPE
from .types import JSONValue


@dataclass(slots=True)
class ContractionStepAnalysis:
    """Estimated metrics for one contraction step."""

    step_id: str
    left_operand_id: str
    right_operand_id: str
    result_operand_id: str
    contracted_labels: tuple[str, ...]
    surviving_labels: tuple[str, ...]
    result_shape: tuple[int, ...]
    result_rank: int
    estimated_flops: int
    estimated_macs: int
    intermediate_size: int

    def to_dict(self) -> dict[str, JSONValue]:
        """Serialize the step analysis to a JSON-compatible mapping."""
        return {
            "step_id": self.step_id,
            "left_operand_id": self.left_operand_id,
            "right_operand_id": self.right_operand_id,
            "result_operand_id": self.result_operand_id,
            "contracted_labels": cast(JSONValue, list(self.contracted_labels)),
            "surviving_labels": cast(JSONValue, list(self.surviving_labels)),
            "result_shape": cast(JSONValue, list(self.result_shape)),
            "result_rank": self.result_rank,
            "estimated_flops": self.estimated_flops,
            "estimated_macs": self.estimated_macs,
            "intermediate_size": self.intermediate_size,
        }


@dataclass(slots=True)
class ManualContractionSummary:
    """Summary metrics for the saved manual contraction plan."""

    total_estimated_flops: int
    total_estimated_macs: int
    peak_intermediate_size: int
    final_shape: tuple[int, ...] | None
    completion_status: str
    remaining_operand_ids: tuple[str, ...] = field(default_factory=tuple)
    peak_intermediate_bytes: int = 0

    def to_dict(self) -> dict[str, JSONValue]:
        """Serialize the manual summary to a JSON-compatible mapping."""
        return {
            "total_estimated_flops": self.total_estimated_flops,
            "total_estimated_macs": self.total_estimated_macs,
            "peak_intermediate_size": self.peak_intermediate_size,
            "peak_intermediate_bytes": self.peak_intermediate_bytes,
            "final_shape": (
                cast(JSONValue, list(self.final_shape))
                if self.final_shape is not None
                else None
            ),
            "completion_status": self.completion_status,
            "remaining_operand_ids": cast(JSONValue, list(self.remaining_operand_ids)),
        }


@dataclass(slots=True)
class AutomaticContractionSummary:
    """Summary metrics for an automatically suggested contraction path."""

    total_estimated_flops: int
    total_estimated_macs: int
    peak_intermediate_size: int
    peak_intermediate_bytes: int = 0

    def to_dict(self) -> dict[str, JSONValue]:
        """Serialize the automatic summary to a JSON-compatible mapping."""
        return {
            "total_estimated_flops": self.total_estimated_flops,
            "total_estimated_macs": self.total_estimated_macs,
            "peak_intermediate_size": self.peak_intermediate_size,
            "peak_intermediate_bytes": self.peak_intermediate_bytes,
        }


@dataclass(slots=True)
class ManualContractionPlanAnalysis:
    """Detailed analysis for the saved manual contraction plan."""

    status: str
    steps: list[ContractionStepAnalysis]
    summary: ManualContractionSummary
    message: str | None = None

    def to_dict(self) -> dict[str, JSONValue]:
        """Serialize the manual plan analysis to a JSON-compatible mapping."""
        payload: dict[str, JSONValue] = {
            "status": self.status,
            "steps": [step.to_dict() for step in self.steps],
            "summary": self.summary.to_dict(),
        }
        if self.message is not None:
            payload["message"] = self.message
        return payload


@dataclass(slots=True)
class AutomaticContractionPlanAnalysis:
    """Detailed analysis for an automatically derived contraction plan."""

    status: str
    steps: list[ContractionStepAnalysis]
    summary: AutomaticContractionSummary
    message: str | None = None

    def to_dict(self) -> dict[str, JSONValue]:
        """Serialize the automatic plan analysis to a JSON-compatible mapping."""
        payload: dict[str, JSONValue] = {
            "status": self.status,
            "steps": [step.to_dict() for step in self.steps],
            "summary": self.summary.to_dict(),
        }
        if self.message is not None:
            payload["message"] = self.message
        return payload


@dataclass(slots=True)
class ContractionComparison:
    """Comparison metrics between two contraction plans or summaries."""

    status: str
    baseline_label: str
    candidate_label: str
    memory_dtype: str
    baseline_peak_intermediate_bytes: int = 0
    candidate_peak_intermediate_bytes: int = 0
    delta_total_estimated_flops: int = 0
    delta_total_estimated_macs: int = 0
    delta_peak_intermediate_size: int = 0
    delta_peak_intermediate_bytes: int = 0
    baseline_peak_step_id: str | None = None
    candidate_peak_step_id: str | None = None
    baseline_bottleneck_labels: tuple[str, ...] = field(default_factory=tuple)
    candidate_bottleneck_labels: tuple[str, ...] = field(default_factory=tuple)
    message: str | None = None

    def to_dict(self) -> dict[str, JSONValue]:
        """Serialize the comparison metrics to a JSON-compatible mapping."""
        payload: dict[str, JSONValue] = {
            "status": self.status,
            "baseline_label": self.baseline_label,
            "candidate_label": self.candidate_label,
            "memory_dtype": self.memory_dtype,
            "baseline_peak_intermediate_bytes": self.baseline_peak_intermediate_bytes,
            "candidate_peak_intermediate_bytes": self.candidate_peak_intermediate_bytes,
            "delta_total_estimated_flops": self.delta_total_estimated_flops,
            "delta_total_estimated_macs": self.delta_total_estimated_macs,
            "delta_peak_intermediate_size": self.delta_peak_intermediate_size,
            "delta_peak_intermediate_bytes": self.delta_peak_intermediate_bytes,
            "baseline_peak_step_id": self.baseline_peak_step_id,
            "candidate_peak_step_id": self.candidate_peak_step_id,
            "baseline_bottleneck_labels": cast(
                JSONValue, list(self.baseline_bottleneck_labels)
            ),
            "candidate_bottleneck_labels": cast(
                JSONValue, list(self.candidate_bottleneck_labels)
            ),
        }
        if self.message is not None:
            payload["message"] = self.message
        return payload


@dataclass(slots=True)
class ContractionAnalysisResult:
    """Top-level contraction analysis payload for the editor UI."""

    network_output_shape: tuple[int, ...]
    manual: ManualContractionPlanAnalysis
    automatic_full: AutomaticContractionPlanAnalysis
    automatic_future: AutomaticContractionPlanAnalysis
    automatic_past: AutomaticContractionPlanAnalysis
    memory_dtype: str = DEFAULT_MEMORY_DTYPE
    comparisons: dict[str, ContractionComparison] = field(default_factory=dict)
    automatic_strategy: str = "greedy"
    message: str | None = None

    def to_dict(self) -> dict[str, JSONValue]:
        """Serialize the full contraction analysis result."""
        payload: dict[str, JSONValue] = {
            "network_output_shape": cast(JSONValue, list(self.network_output_shape)),
            "manual": self.manual.to_dict(),
            "automatic_full": self.automatic_full.to_dict(),
            "automatic_future": self.automatic_future.to_dict(),
            "automatic_past": self.automatic_past.to_dict(),
            "memory_dtype": self.memory_dtype,
            "comparisons": cast(
                JSONValue,
                {
                    key: comparison.to_dict()
                    for key, comparison in self.comparisons.items()
                },
            ),
            "automatic_strategy": self.automatic_strategy,
        }
        if self.message is not None:
            payload["message"] = self.message
        return payload
