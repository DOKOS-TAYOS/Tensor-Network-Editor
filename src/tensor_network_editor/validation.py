from __future__ import annotations

from ._analysis import analyze_network
from ._validation_contraction import validate_contraction_plan
from ._validation_edges import validate_edge
from ._validation_entities import (
    validate_group,
    validate_network,
    validate_note,
    validate_tensor,
)
from .errors import SpecValidationError
from .models import NetworkSpec, ValidationIssue


def validate_spec(spec: NetworkSpec) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
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

    return issues


def ensure_valid_spec(spec: NetworkSpec) -> NetworkSpec:
    issues = validate_spec(spec)
    if issues:
        raise SpecValidationError(issues)
    return spec
