"""Automatic-path helpers for contraction analysis."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache
from importlib import import_module
from string import ascii_letters
from typing import Any, TypeAlias, cast

from ...models import ContractionStepSpec, NetworkSpec
from ._contraction_analysis_manual import ManualOperandState
from ._contraction_analysis_types import (
    AutomaticContractionPlanAnalysis,
    AutomaticContractionSummary,
    ContractionStepAnalysis,
)
from ._contraction_plan import simulate_contraction_step

MISSING_OPT_EINSUM_MESSAGE = (
    "The required opt_einsum dependency is not available in the current .venv. "
    "Reinstall tensor-network-editor in this environment to enable Auto full, "
    "Auto future, and Auto past."
)

_AutomaticPath: TypeAlias = tuple[tuple[int, ...], ...]
_AutomaticPathCacheKey: TypeAlias = tuple[
    tuple[str, tuple[str, ...], tuple[int, ...]],
    ...,
]
_AutomaticPathCache: TypeAlias = dict[
    _AutomaticPathCacheKey,
    "_AutomaticPathCacheEntry",
]


@dataclass(slots=True, frozen=True)
class _AutomaticPathCacheEntry:
    """Cached opt_einsum path lookup for one operand signature."""

    path: _AutomaticPath | None = None
    message: str | None = None


def _contract_operands(
    *,
    step_id: str,
    left_operand_id: str,
    right_operand_id: str,
    left_labels: tuple[str, ...],
    right_labels: tuple[str, ...],
    dimension_by_label: dict[str, int],
) -> ContractionStepAnalysis:
    """Estimate the metrics for one pairwise contraction."""
    simulated_step = simulate_contraction_step(
        step=ContractionStepSpec(
            id=step_id,
            left_operand_id=left_operand_id,
            right_operand_id=right_operand_id,
        ),
        left_labels=left_labels,
        right_labels=right_labels,
        left_axis_names=left_labels,
        right_axis_names=right_labels,
        dimension_by_label=dimension_by_label,
    )
    return ContractionStepAnalysis(
        step_id=simulated_step.step_id,
        left_operand_id=simulated_step.left_operand_id,
        right_operand_id=simulated_step.right_operand_id,
        result_operand_id=simulated_step.step_id,
        contracted_labels=simulated_step.contracted_labels,
        surviving_labels=simulated_step.surviving_labels,
        result_shape=simulated_step.result_shape,
        result_rank=simulated_step.result_rank,
        estimated_flops=simulated_step.estimated_flops,
        estimated_macs=simulated_step.estimated_macs,
        intermediate_size=simulated_step.intermediate_size,
    )


def _build_automatic_summary(
    *,
    total_estimated_flops: int,
    total_estimated_macs: int,
    peak_intermediate_size: int,
    bytes_per_element: int,
) -> AutomaticContractionSummary:
    """Build a summary payload for automatic path analysis."""
    return AutomaticContractionSummary(
        total_estimated_flops=total_estimated_flops,
        total_estimated_macs=total_estimated_macs,
        peak_intermediate_size=peak_intermediate_size,
        peak_intermediate_bytes=peak_intermediate_size * bytes_per_element,
    )


def _analyze_future_automatic_plan(
    *,
    initial_operands: dict[str, tuple[str, ...]],
    manual_operand_state: ManualOperandState,
    dimension_by_label: dict[str, int],
    bytes_per_element: int,
    path_cache: _AutomaticPathCache | None = None,
) -> AutomaticContractionPlanAnalysis:
    """Analyze the greedy path that continues from the current manual state."""
    del initial_operands
    return _analyze_automatic_operands(
        operand_order=manual_operand_state.active_operand_ids,
        operands=manual_operand_state.remaining_operands,
        dimension_by_label=dimension_by_label,
        step_id_prefix="auto_future_step_",
        bytes_per_element=bytes_per_element,
        path_cache=path_cache,
    )


def _analyze_past_automatic_plan(
    *,
    spec: NetworkSpec,
    initial_operands: dict[str, tuple[str, ...]],
    manual_operand_state: ManualOperandState,
    dimension_by_label: dict[str, int],
    bytes_per_element: int,
    path_cache: _AutomaticPathCache | None = None,
) -> AutomaticContractionPlanAnalysis:
    """Analyze greedy paths for already contracted manual subtrees."""
    del spec
    base_tensor_ids = set(initial_operands)
    contracted_root_operand_ids = [
        operand_id
        for operand_id in manual_operand_state.active_operand_ids
        if operand_id not in base_tensor_ids
        and len(
            manual_operand_state.source_tensor_ids_by_operand_id.get(operand_id, ())
        )
        > 1
    ]
    if not contracted_root_operand_ids:
        return _unavailable_automatic_analysis(
            "Contract at least one tensor pair to unlock the auto past preview.",
            bytes_per_element=bytes_per_element,
        )

    all_steps: list[ContractionStepAnalysis] = []
    total_estimated_flops = 0
    total_estimated_macs = 0
    peak_intermediate_size = 0

    for root_operand_id in contracted_root_operand_ids:
        root_tensor_ids = manual_operand_state.source_tensor_ids_by_operand_id.get(
            root_operand_id, ()
        )
        analysis = _analyze_automatic_operands(
            operand_order=root_tensor_ids,
            operands={
                tensor_id: initial_operands[tensor_id] for tensor_id in root_tensor_ids
            },
            dimension_by_label=dimension_by_label,
            step_id_prefix=f"{root_operand_id}__auto_past_",
            final_step_id=root_operand_id,
            bytes_per_element=bytes_per_element,
            path_cache=path_cache,
        )
        if analysis.status == "unavailable":
            return analysis
        all_steps.extend(analysis.steps)
        total_estimated_flops += analysis.summary.total_estimated_flops
        total_estimated_macs += analysis.summary.total_estimated_macs
        peak_intermediate_size = max(
            peak_intermediate_size, analysis.summary.peak_intermediate_size
        )

    return AutomaticContractionPlanAnalysis(
        status="complete",
        steps=all_steps,
        summary=_build_automatic_summary(
            total_estimated_flops=total_estimated_flops,
            total_estimated_macs=total_estimated_macs,
            peak_intermediate_size=peak_intermediate_size,
            bytes_per_element=bytes_per_element,
        ),
    )


def _analyze_automatic_operands(
    *,
    operand_order: Sequence[str],
    operands: dict[str, tuple[str, ...]],
    dimension_by_label: dict[str, int],
    step_id_prefix: str,
    bytes_per_element: int,
    final_step_id: str | None = None,
    path_cache: _AutomaticPathCache | None = None,
) -> AutomaticContractionPlanAnalysis:
    """Run automatic greedy analysis for the provided operand set."""
    if len(operand_order) <= 1:
        return AutomaticContractionPlanAnalysis(
            status="complete",
            steps=[],
            summary=_build_automatic_summary(
                total_estimated_flops=0,
                total_estimated_macs=0,
                peak_intermediate_size=0,
                bytes_per_element=bytes_per_element,
            ),
        )

    operand_ids = tuple(operand_order)
    path_entry = _resolve_automatic_path(
        operand_ids=operand_ids,
        operands=operands,
        dimension_by_label=dimension_by_label,
        path_cache=path_cache,
    )
    if path_entry.message is not None:
        return _unavailable_automatic_analysis(
            path_entry.message,
            bytes_per_element=bytes_per_element,
        )
    if path_entry.path is None:
        return _unavailable_automatic_analysis(
            "Automatic greedy path analysis did not return a contraction path.",
            bytes_per_element=bytes_per_element,
        )
    path = path_entry.path

    remaining_order = list(operand_ids)
    remaining_operands = dict(operands)
    steps: list[ContractionStepAnalysis] = []
    total_estimated_flops = 0
    total_estimated_macs = 0
    peak_intermediate_size = 0

    for step_index, raw_indices in enumerate(path, start=1):
        indices = tuple(int(value) for value in raw_indices)
        if len(indices) != 2:
            return _unavailable_automatic_analysis(
                "Automatic greedy path produced a non-pairwise contraction step.",
                bytes_per_element=bytes_per_element,
            )
        left_operand_id = remaining_order[indices[0]]
        right_operand_id = remaining_order[indices[1]]
        step_id = (
            final_step_id
            if final_step_id is not None and step_index == len(path)
            else f"{step_id_prefix}{step_index}"
        )
        step_result = _contract_operands(
            step_id=step_id,
            left_operand_id=left_operand_id,
            right_operand_id=right_operand_id,
            left_labels=remaining_operands.pop(left_operand_id),
            right_labels=remaining_operands.pop(right_operand_id),
            dimension_by_label=dimension_by_label,
        )
        steps.append(step_result)
        remaining_operands[step_id] = step_result.surviving_labels
        total_estimated_flops += step_result.estimated_flops
        total_estimated_macs += step_result.estimated_macs
        peak_intermediate_size = max(
            peak_intermediate_size, step_result.intermediate_size
        )
        if indices[0] > indices[1]:
            higher_index, lower_index = indices[0], indices[1]
        else:
            higher_index, lower_index = indices[1], indices[0]
        remaining_order.pop(higher_index)
        remaining_order.pop(lower_index)
        remaining_order.append(step_id)

    status = "complete" if len(remaining_operands) <= 1 else "incomplete"
    return AutomaticContractionPlanAnalysis(
        status=status,
        steps=steps,
        summary=_build_automatic_summary(
            total_estimated_flops=total_estimated_flops,
            total_estimated_macs=total_estimated_macs,
            peak_intermediate_size=peak_intermediate_size,
            bytes_per_element=bytes_per_element,
        ),
    )


def _unavailable_automatic_analysis(
    message: str,
    *,
    bytes_per_element: int,
) -> AutomaticContractionPlanAnalysis:
    """Return a standardized unavailable-analysis payload."""
    return AutomaticContractionPlanAnalysis(
        status="unavailable",
        steps=[],
        summary=_build_automatic_summary(
            total_estimated_flops=0,
            total_estimated_macs=0,
            peak_intermediate_size=0,
            bytes_per_element=bytes_per_element,
        ),
        message=message,
    )


def _resolve_automatic_path(
    *,
    operand_ids: tuple[str, ...],
    operands: dict[str, tuple[str, ...]],
    dimension_by_label: dict[str, int],
    path_cache: _AutomaticPathCache | None,
) -> _AutomaticPathCacheEntry:
    """Return a cached or freshly resolved greedy path for one operand set."""
    cache_key = _build_automatic_path_cache_key(
        operand_ids=operand_ids,
        operands=operands,
        dimension_by_label=dimension_by_label,
    )
    if path_cache is not None and cache_key in path_cache:
        return path_cache[cache_key]

    entry = _build_automatic_path_cache_entry(
        operand_ids=operand_ids,
        operands=operands,
        dimension_by_label=dimension_by_label,
    )
    if path_cache is not None:
        path_cache[cache_key] = entry
    return entry


def _build_automatic_path_cache_key(
    *,
    operand_ids: tuple[str, ...],
    operands: dict[str, tuple[str, ...]],
    dimension_by_label: dict[str, int],
) -> _AutomaticPathCacheKey:
    """Build the stable in-call cache key for an automatic path request."""
    return tuple(
        (
            operand_id,
            operands[operand_id],
            tuple(dimension_by_label[label] for label in operands[operand_id]),
        )
        for operand_id in operand_ids
    )


def _build_automatic_path_cache_entry(
    *,
    operand_ids: tuple[str, ...],
    operands: dict[str, tuple[str, ...]],
    dimension_by_label: dict[str, int],
) -> _AutomaticPathCacheEntry:
    """Resolve one opt_einsum path and normalize expected unavailability."""
    contract_path = _load_contract_path(import_module)
    if contract_path is None:
        return _AutomaticPathCacheEntry(message=MISSING_OPT_EINSUM_MESSAGE)

    label_order = list(
        dict.fromkeys(
            label for operand_id in operand_ids for label in operands[operand_id]
        )
    )
    if len(label_order) > len(ascii_letters):
        return _AutomaticPathCacheEntry(
            message="Automatic greedy path analysis currently supports up to 52 distinct labels."
        )

    symbol_map = {
        label: ascii_letters[offset]
        for offset, label in enumerate(label_order[: len(ascii_letters)])
    }
    label_counts = {label: 0 for label in label_order}
    for operand_id in operand_ids:
        for label in operands[operand_id]:
            label_counts[label] += 1
    output_labels = [label for label in label_order if label_counts[label] == 1]
    equation = (
        ",".join(
            "".join(symbol_map[label] for label in operands[operand_id])
            for operand_id in operand_ids
        )
        + "->"
        + "".join(symbol_map[label] for label in output_labels)
    )
    shapes = [
        tuple(dimension_by_label[label] for label in operands[operand_id])
        for operand_id in operand_ids
    ]

    try:
        raw_path, _ = contract_path(
            equation,
            *shapes,
            shapes=True,
            optimize="greedy",
        )
    except ValueError as exc:
        return _AutomaticPathCacheEntry(
            message=f"Automatic greedy path analysis failed: {exc}"
        )

    return _AutomaticPathCacheEntry(
        path=tuple(
            tuple(int(value) for value in raw_indices) for raw_indices in raw_path
        )
    )


@cache
def _load_contract_path(
    importer: Callable[[str], Any],
) -> Any | None:
    """Resolve ``opt_einsum.contract_path`` once per importer function."""
    try:
        return cast(
            Any,
            cast(Any, importer("opt_einsum")).contract_path,
        )
    except ImportError:
        return None
