"""Validate abstract tensor-network specifications."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._analysis import analyze_network
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
from ._validation_linear_periodic import validate_linear_periodic_chain
from ._validation_tree_periodic import validate_tree_periodic_tree
from .errors import SpecValidationError
from .models import NetworkSpec, ValidationIssue

if TYPE_CHECKING:
    from ._analysis import NetworkAnalysis


def validate_spec(spec: NetworkSpec) -> list[ValidationIssue]:
    """Collect all validation issues found in ``spec``."""
    issues, _ = _validate_spec_with_analysis(spec)
    return issues


def _validate_spec_with_analysis(
    spec: NetworkSpec,
) -> tuple[list[ValidationIssue], NetworkAnalysis]:
    """Validate ``spec`` and return both issues and the computed analysis."""
    issues: list[ValidationIssue] = []
    _validate_periodic_mode_exclusivity(spec, issues=issues)
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

    if spec.contraction_plan is not None:
        validate_contraction_plan(
            spec.contraction_plan,
            tensor_ids=tensor_ids,
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


def _validate_periodic_mode_exclusivity(
    spec: NetworkSpec,
    *,
    issues: list[ValidationIssue],
) -> None:
    """Reject payloads that mix the linear and bidimensional For modes."""
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


def ensure_valid_spec(spec: NetworkSpec) -> NetworkSpec:
    """Return ``spec`` or raise ``SpecValidationError`` if it is invalid."""
    issues, _ = _validate_spec_with_analysis(spec)
    if issues:
        raise SpecValidationError(issues)
    return spec
