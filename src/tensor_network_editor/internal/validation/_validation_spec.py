"""Internal orchestration helpers for full spec validation."""

from __future__ import annotations

from ...errors import SpecValidationError
from ...models import NetworkSpec, ValidationIssue
from ..analysis._analysis import NetworkAnalysis, analyze_network
from ..analysis._hyperedge_lowering import lower_hyperedges_to_pairwise_spec
from ._validation_common import append_issue
from ._validation_contraction import validate_contraction_plan
from ._validation_edges import validate_edge
from ._validation_entities import (
    validate_group,
    validate_network,
    validate_note,
    validate_tensor,
)
from ._validation_grid_periodic import validate_grid_periodic_grid
from ._validation_hyperedges import validate_hyperedge
from ._validation_linear_periodic import validate_linear_periodic_chain
from ._validation_tree_periodic import validate_tree_periodic_tree


def validate_spec(spec: NetworkSpec) -> list[ValidationIssue]:
    """Collect all validation issues found in ``spec``."""
    issues, _ = validate_spec_with_analysis(spec)
    return issues


def validate_spec_with_analysis(
    spec: NetworkSpec,
) -> tuple[list[ValidationIssue], NetworkAnalysis]:
    """Validate ``spec`` and return both issues and the computed analysis."""
    issues: list[ValidationIssue] = []
    _validate_periodic_mode_exclusivity(spec, issues=issues)
    _validate_hyperedge_mode_compatibility(spec, issues=issues)
    validate_network(spec, issues)
    analysis = analyze_network(spec)
    tensor_ids = set(analysis.tensor_map)

    for tensor in spec.tensors:
        validate_tensor(tensor, issues=issues)

    for group in spec.groups:
        validate_group(group, tensor_ids=tensor_ids, issues=issues)

    for note in spec.notes:
        validate_note(note, issues=issues)

    connected_indices: set[str] = set()
    for edge in spec.edges:
        validate_edge(
            edge,
            analysis_tensor_map=analysis.tensor_map,
            analysis_index_map=analysis.index_map,
            connected_indices=connected_indices,
            issues=issues,
        )
    for hyperedge in spec.hyperedges:
        validate_hyperedge(
            hyperedge,
            analysis_tensor_map=analysis.tensor_map,
            analysis_index_map=analysis.index_map,
            connected_indices=connected_indices,
            issues=issues,
        )

    if spec.contraction_plan is not None:
        validate_contraction_plan(
            spec.contraction_plan,
            tensor_ids=_contraction_operand_ids_for_validation(spec, tensor_ids),
            issues=issues,
        )

    if spec.linear_periodic_chain is not None:
        validate_linear_periodic_chain(spec.linear_periodic_chain, issues=issues)
    if spec.grid_periodic_grid is not None:
        validate_grid_periodic_grid(spec.grid_periodic_grid, issues=issues)
        if spec.contraction_plan is not None:
            append_issue(
                issues,
                code="grid-periodic-contraction-plan",
                message=(
                    "The active cell cannot define a contraction plan in "
                    "bidimensional For mode."
                ),
                path="contraction_plan",
            )
    if spec.tree_periodic_tree is not None:
        validate_tree_periodic_tree(spec.tree_periodic_tree, issues=issues)

    return issues, analysis


def _contraction_operand_ids_for_validation(
    spec: NetworkSpec,
    visible_tensor_ids: set[str],
) -> set[str]:
    """Return operand ids that a contraction plan may reference."""
    if not _spec_allows_hyperedge_analysis(spec):
        return visible_tensor_ids
    try:
        lowered_spec = lower_hyperedges_to_pairwise_spec(
            spec,
            preserve_contraction_plan=True,
        )
    except KeyError:
        return visible_tensor_ids
    return {tensor.id for tensor in lowered_spec.tensors}


def _spec_allows_hyperedge_analysis(spec: NetworkSpec) -> bool:
    """Return whether hyperedges can be lowered for normal-mode analysis."""
    return (
        bool(spec.hyperedges)
        and spec.linear_periodic_chain is None
        and spec.grid_periodic_grid is None
        and spec.tree_periodic_tree is None
    )


def ensure_valid_spec(spec: NetworkSpec) -> NetworkSpec:
    """Return ``spec`` or raise ``SpecValidationError`` if it is invalid."""
    issues, _ = validate_spec_with_analysis(spec)
    if issues:
        raise SpecValidationError(issues)
    return spec


def _validate_periodic_mode_exclusivity(
    spec: NetworkSpec,
    *,
    issues: list[ValidationIssue],
) -> None:
    """Reject payloads that mix the supported periodic editor modes."""
    active_modes = [
        ("linear_periodic_chain", spec.linear_periodic_chain),
        ("grid_periodic_grid", spec.grid_periodic_grid),
        ("tree_periodic_tree", spec.tree_periodic_tree),
    ]
    populated_modes = [
        field_name for field_name, payload in active_modes if payload is not None
    ]
    if len(populated_modes) <= 1:
        return
    append_issue(
        issues,
        code="periodic-mode-conflict",
        message=(
            "Specs cannot mix 'linear_periodic_chain', "
            "'grid_periodic_grid', and 'tree_periodic_tree'."
        ),
        path=populated_modes[-1],
    )


def _validate_hyperedge_mode_compatibility(
    spec: NetworkSpec,
    *,
    issues: list[ValidationIssue],
) -> None:
    """Reject hyperedges in the current periodic editor modes."""
    if not spec.hyperedges:
        return
    if (
        spec.linear_periodic_chain is None
        and spec.grid_periodic_grid is None
        and spec.tree_periodic_tree is None
    ):
        return
    append_issue(
        issues,
        code="hyperedges-not-supported-in-for-mode",
        message=(
            "Hyperedges are only supported in normal mode in this version of the "
            "editor."
        ),
        path="hyperedges",
    )
