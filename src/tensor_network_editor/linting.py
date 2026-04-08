"""Public linter helpers for soft tensor-network diagnostics."""

from __future__ import annotations

from math import prod
from typing import Protocol

from ._analysis import NetworkAnalysis, analyze_network
from ._headless_models import LintIssue, LintReport
from .models import ContractionStepSpec, GroupSpec, NetworkSpec


class _NamedEntity(Protocol):
    """Protocol for readable entities that expose stable ids and names."""

    id: str
    name: str


def lint_spec(
    spec: NetworkSpec,
    *,
    max_tensor_rank: int = 6,
    max_tensor_cardinality: int = 4096,
) -> LintReport:
    """Return soft diagnostics for ``spec`` without treating them as hard errors."""
    analysis = analyze_network(spec)
    issues: list[LintIssue] = []
    issues.extend(_lint_disconnected_components(spec))
    issues.extend(_lint_open_indices(analysis))
    issues.extend(
        _lint_tensor_sizes(
            spec,
            max_tensor_rank=max_tensor_rank,
            max_tensor_cardinality=max_tensor_cardinality,
        )
    )
    issues.extend(_lint_groups(spec.groups))
    issues.extend(_lint_names(spec))
    issues.extend(_lint_manual_plan(spec))
    return LintReport(issues=issues)


def _lint_disconnected_components(spec: NetworkSpec) -> list[LintIssue]:
    """Warn when the tensor graph contains multiple disconnected components."""
    tensor_ids = [tensor.id for tensor in spec.tensors]
    if len(tensor_ids) <= 1:
        return []

    adjacency: dict[str, set[str]] = {tensor_id: set() for tensor_id in tensor_ids}
    valid_tensor_ids = set(tensor_ids)
    for edge in spec.edges:
        if (
            edge.left.tensor_id in valid_tensor_ids
            and edge.right.tensor_id in valid_tensor_ids
            and edge.left.tensor_id != edge.right.tensor_id
        ):
            adjacency[edge.left.tensor_id].add(edge.right.tensor_id)
            adjacency[edge.right.tensor_id].add(edge.left.tensor_id)

    visited: set[str] = set()
    component_count = 0
    for tensor_id in tensor_ids:
        if tensor_id in visited:
            continue
        component_count += 1
        stack = [tensor_id]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            stack.extend(
                neighbor for neighbor in adjacency[current] if neighbor not in visited
            )

    if component_count <= 1:
        return []
    return [
        LintIssue(
            severity="warning",
            code="disconnected-components",
            message=f"The network is split into {component_count} disconnected tensor components.",
            path="tensors",
            suggestion="Connect the components or split them into separate specs if they are independent.",
        )
    ]


def _lint_open_indices(analysis: NetworkAnalysis) -> list[LintIssue]:
    """Warn about open indices whose names look like accidental dangling legs."""
    suspicious_names = {
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "x",
        "y",
        "z",
        "left",
        "right",
        "up",
        "down",
        "shared",
        "bond",
    }
    issues: list[LintIssue] = []
    for tensor, index in analysis.open_indices:
        if index.name.strip().lower() not in suspicious_names:
            continue
        issues.append(
            LintIssue(
                severity="warning",
                code="suspicious-open-index",
                message=(
                    f"Index '{index.name}' on tensor '{tensor.name}' is open and looks like a missing connection."
                ),
                path=f"tensors.{tensor.id}.indices.{index.id}",
                suggestion="Connect it, rename it to reflect an output leg, or document it in metadata.",
            )
        )
    return issues


def _lint_tensor_sizes(
    spec: NetworkSpec,
    *,
    max_tensor_rank: int,
    max_tensor_cardinality: int,
) -> list[LintIssue]:
    """Warn about unusually large tensor rank or cardinality."""
    issues: list[LintIssue] = []
    for tensor in spec.tensors:
        if len(tensor.indices) > max_tensor_rank:
            issues.append(
                LintIssue(
                    severity="warning",
                    code="large-tensor-rank",
                    message=(
                        f"Tensor '{tensor.name}' has rank {len(tensor.indices)}, above the configured threshold {max_tensor_rank}."
                    ),
                    path=f"tensors.{tensor.id}",
                    suggestion="Check whether this tensor should be decomposed or the threshold increased.",
                )
            )
        cardinality = (
            prod(index.dimension for index in tensor.indices) if tensor.indices else 1
        )
        if cardinality > max_tensor_cardinality:
            issues.append(
                LintIssue(
                    severity="warning",
                    code="large-tensor-cardinality",
                    message=(
                        f"Tensor '{tensor.name}' spans {cardinality} elements, above the configured threshold {max_tensor_cardinality}."
                    ),
                    path=f"tensors.{tensor.id}",
                    suggestion="Check dimensions, decomposition choices, or raise the threshold for this workflow.",
                )
            )
    return issues


def _lint_groups(groups: list[GroupSpec]) -> list[LintIssue]:
    """Warn when a visual group exists but contains no tensors."""
    issues: list[LintIssue] = []
    for group in groups:
        if group.tensor_ids:
            continue
        issues.append(
            LintIssue(
                severity="warning",
                code="empty-group",
                message=f"Group '{group.name}' does not contain any tensors.",
                path=f"groups.{group.id}.tensor_ids",
                suggestion="Remove the group or add the tensors it is supposed to organize.",
            )
        )
    return issues


def _lint_names(spec: NetworkSpec) -> list[LintIssue]:
    """Warn when names look like untouched generic defaults."""
    issues: list[LintIssue] = []
    generic_names = {
        "tensor",
        "group",
        "edge",
        "note",
        "index",
    }
    for entity_path, entity in _iter_named_entities(spec):
        normalized = entity.name.strip().lower()
        if normalized not in generic_names:
            continue
        issues.append(
            LintIssue(
                severity="info",
                code="uninformative-name",
                message=f"Name '{entity.name}' is very generic and may make the network harder to read.",
                path=entity_path,
                suggestion="Rename it to reflect its role in the network or contraction plan.",
            )
        )
    return issues


def _iter_named_entities(spec: NetworkSpec) -> list[tuple[str, _NamedEntity]]:
    """Return named entities that benefit from readability linting."""
    entities: list[tuple[str, _NamedEntity]] = []
    entities.extend((f"tensors.{tensor.id}.name", tensor) for tensor in spec.tensors)
    entities.extend((f"groups.{group.id}.name", group) for group in spec.groups)
    entities.extend((f"edges.{edge.id}.name", edge) for edge in spec.edges)
    return entities


def _lint_manual_plan(spec: NetworkSpec) -> list[LintIssue]:
    """Warn about incomplete or partially invalid manual contraction plans."""
    plan = spec.contraction_plan
    if plan is None or not plan.steps:
        return []

    valid_prefix_length, invalid_step, remaining_operand_ids = (
        _analyze_manual_plan_operands(spec)
    )
    issues: list[LintIssue] = []
    if invalid_step is not None and valid_prefix_length > 0:
        issues.append(
            LintIssue(
                severity="warning",
                code="invalidated-manual-suffix",
                message=(
                    f"The saved manual plan stops being valid at step '{invalid_step.id}' after a valid prefix."
                ),
                path=f"contraction_plan.steps.{invalid_step.id}",
                suggestion="Trim the invalid suffix or rebuild the remaining steps from the current frontier.",
            )
        )
    if len(remaining_operand_ids) > 1:
        issues.append(
            LintIssue(
                severity="warning",
                code="incomplete-manual-plan",
                message=(
                    f"The manual plan leaves {len(remaining_operand_ids)} active operands without finishing the contraction."
                ),
                path="contraction_plan.steps",
                suggestion="Complete the remaining contractions or rely on automatic suggestions for the suffix.",
            )
        )
    return issues


def _analyze_manual_plan_operands(
    spec: NetworkSpec,
) -> tuple[int, ContractionStepSpec | None, tuple[str, ...]]:
    """Simulate manual operand ids without requiring a fully valid plan."""
    plan = spec.contraction_plan
    if plan is None:
        return 0, None, tuple(tensor.id for tensor in spec.tensors)

    active_operand_ids = [tensor.id for tensor in spec.tensors]
    active_operand_set = set(active_operand_ids)
    reserved_operand_ids = set(active_operand_ids)
    valid_prefix_length = 0
    for step in plan.steps:
        if (
            step.left_operand_id == step.right_operand_id
            or step.left_operand_id not in active_operand_set
            or step.right_operand_id not in active_operand_set
            or step.id in reserved_operand_ids
        ):
            return valid_prefix_length, step, tuple(active_operand_ids)
        active_operand_set.remove(step.left_operand_id)
        active_operand_set.remove(step.right_operand_id)
        active_operand_ids = [
            step.id,
            *[
                operand_id
                for operand_id in active_operand_ids
                if operand_id not in {step.left_operand_id, step.right_operand_id}
            ],
        ]
        active_operand_set.add(step.id)
        reserved_operand_ids.add(step.id)
        valid_prefix_length += 1
    return valid_prefix_length, None, tuple(active_operand_ids)


__all__ = ["LintIssue", "LintReport", "lint_spec"]
