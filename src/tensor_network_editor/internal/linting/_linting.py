"""Public linter helpers for soft tensor-network diagnostics."""

from __future__ import annotations

from difflib import get_close_matches
from math import prod
from typing import Protocol

from ...models import ContractionStepSpec, GroupSpec, NetworkSpec
from ..analysis._analysis import NetworkAnalysis, analyze_network
from ..models._headless_models import LintIssue, LintReport
from ..templates._annotation_catalog import AnnotationScope, annotation_keys_by_scope

_GUIDED_METADATA_KEYS_BY_SCOPE = annotation_keys_by_scope()
_TENSOR_GUIDED_METADATA_KEYS = frozenset(
    _GUIDED_METADATA_KEYS_BY_SCOPE[AnnotationScope.TENSOR]
)
_INDEX_GUIDED_METADATA_KEYS = frozenset(
    _GUIDED_METADATA_KEYS_BY_SCOPE[AnnotationScope.INDEX]
)


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
    issues.extend(_lint_guided_metadata(spec, analysis))
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


def _lint_guided_metadata(
    spec: NetworkSpec,
    analysis: NetworkAnalysis,
) -> list[LintIssue]:
    """Return soft diagnostics derived from guided metadata conventions."""
    issues: list[LintIssue] = []
    issues.extend(_lint_guided_metadata_hygiene(spec))
    issues.extend(_lint_index_semantics(spec, analysis))
    issues.extend(_lint_connected_symmetry(spec, analysis))
    issues.extend(_lint_tensor_index_symmetry(spec))
    return issues


def _lint_guided_metadata_hygiene(spec: NetworkSpec) -> list[LintIssue]:
    """Warn about malformed or suspicious guided metadata keys and values."""
    issues: list[LintIssue] = []
    for tensor in spec.tensors:
        issues.extend(
            _lint_guided_metadata_entries(
                metadata=tensor.metadata,
                path_prefix=f"tensors.{tensor.id}.metadata",
                entity_label=f"tensor '{tensor.name}'",
                guided_keys=_TENSOR_GUIDED_METADATA_KEYS,
            )
        )
        for index in tensor.indices:
            issues.extend(
                _lint_guided_metadata_entries(
                    metadata=index.metadata,
                    path_prefix=f"tensors.{tensor.id}.indices.{index.id}.metadata",
                    entity_label=f"index '{index.name}' on tensor '{tensor.name}'",
                    guided_keys=_INDEX_GUIDED_METADATA_KEYS,
                )
            )
    return issues


def _lint_guided_metadata_entries(
    *,
    metadata: object,
    path_prefix: str,
    entity_label: str,
    guided_keys: frozenset[str],
) -> list[LintIssue]:
    """Lint guided metadata keys and values for one metadata mapping."""
    if not isinstance(metadata, dict):
        return []
    issues: list[LintIssue] = []
    for metadata_key, metadata_value in metadata.items():
        if not isinstance(metadata_key, str):
            continue
        canonical_key, key_is_noncanonical = _resolve_guided_metadata_key(
            metadata_key,
            guided_keys=guided_keys,
        )
        if canonical_key is None:
            continue
        if key_is_noncanonical:
            issues.append(
                LintIssue(
                    severity="info",
                    code="noncanonical-guided-metadata-key",
                    message=(
                        f"Metadata key '{metadata_key}' on {entity_label} looks like guided key '{canonical_key}'."
                    ),
                    path=f"{path_prefix}.{metadata_key}",
                    suggestion=(
                        f"Rename it to '{canonical_key}' to match the guided metadata convention."
                    ),
                )
            )
        if isinstance(metadata_value, bool) or not isinstance(metadata_value, str):
            issues.append(
                LintIssue(
                    severity="info",
                    code="guided-metadata-non-string-value",
                    message=(
                        f"Guided metadata key '{canonical_key}' on {entity_label} should use a short text value."
                    ),
                    path=f"{path_prefix}.{metadata_key}",
                    suggestion="Replace it with a non-empty string value.",
                )
            )
            continue
        if not metadata_value.strip():
            issues.append(
                LintIssue(
                    severity="info",
                    code="guided-metadata-empty-value",
                    message=(
                        f"Guided metadata key '{canonical_key}' on {entity_label} is empty."
                    ),
                    path=f"{path_prefix}.{metadata_key}",
                    suggestion="Fill in a short text value or remove the key.",
                )
            )
    return issues


def _lint_index_semantics(
    spec: NetworkSpec,
    analysis: NetworkAnalysis,
) -> list[LintIssue]:
    """Warn when index metadata conflicts with the graph structure."""
    open_index_ids = {index.id for _, index in analysis.open_indices}
    issues: list[LintIssue] = []
    for tensor in spec.tensors:
        for index in tensor.indices:
            leg_kind = _guided_metadata_text(
                index.metadata,
                canonical_key="leg_kind",
                guided_keys=_INDEX_GUIDED_METADATA_KEYS,
            )
            observable = _guided_metadata_text(
                index.metadata,
                canonical_key="observable",
                guided_keys=_INDEX_GUIDED_METADATA_KEYS,
            )
            if (
                leg_kind is not None
                and leg_kind[1] == "bond"
                and index.id in open_index_ids
            ):
                issues.append(
                    LintIssue(
                        severity="warning",
                        code="bond-leg-open-index",
                        message=(
                            f"Index '{index.name}' on tensor '{tensor.name}' is marked as a bond leg but is still open."
                        ),
                        path=f"tensors.{tensor.id}.indices.{index.id}.metadata.leg_kind",
                        suggestion="Connect the leg or change 'leg_kind' to match its open role.",
                    )
                )
            if observable is not None and (
                index.id in analysis.connected_index_ids
                or (leg_kind is not None and leg_kind[1] == "bond")
            ):
                issues.append(
                    LintIssue(
                        severity="warning",
                        code="observable-connected-index",
                        message=(
                            f"Index '{index.name}' on tensor '{tensor.name}' is annotated as observable but behaves like a connected bond."
                        ),
                        path=f"tensors.{tensor.id}.indices.{index.id}.metadata.observable",
                        suggestion="Keep observable annotations on open measurement legs, or remove the bond-style metadata.",
                    )
                )
    return issues


def _lint_connected_symmetry(
    spec: NetworkSpec,
    analysis: NetworkAnalysis,
) -> list[LintIssue]:
    """Warn when connected indices declare conflicting symmetries."""
    issues: list[LintIssue] = []
    for edge in spec.edges:
        left_entry = analysis.index_map.get(edge.left.index_id)
        right_entry = analysis.index_map.get(edge.right.index_id)
        if left_entry is None or right_entry is None:
            continue
        left_tensor, left_index = left_entry
        right_tensor, right_index = right_entry
        left_symmetry = _guided_metadata_text(
            left_index.metadata,
            canonical_key="symmetry",
            guided_keys=_INDEX_GUIDED_METADATA_KEYS,
        )
        right_symmetry = _guided_metadata_text(
            right_index.metadata,
            canonical_key="symmetry",
            guided_keys=_INDEX_GUIDED_METADATA_KEYS,
        )
        if (
            left_symmetry is None
            or right_symmetry is None
            or left_symmetry[1] == right_symmetry[1]
        ):
            continue
        issues.append(
            LintIssue(
                severity="warning",
                code="connected-symmetry-conflict",
                message=(
                    f"Edge '{edge.name}' connects symmetry '{left_symmetry[0]}' on '{left_tensor.name}.{left_index.name}' "
                    f"to '{right_symmetry[0]}' on '{right_tensor.name}.{right_index.name}'."
                ),
                path=f"edges.{edge.id}",
                suggestion="Align the symmetry metadata on both sides of the connection.",
            )
        )
    for hyperedge in spec.hyperedges:
        endpoint_symmetries: list[tuple[str, str, str]] = []
        for endpoint in hyperedge.endpoints:
            index_entry = analysis.index_map.get(endpoint.index_id)
            if index_entry is None:
                continue
            tensor, index = index_entry
            symmetry = _guided_metadata_text(
                index.metadata,
                canonical_key="symmetry",
                guided_keys=_INDEX_GUIDED_METADATA_KEYS,
            )
            if symmetry is None:
                continue
            endpoint_symmetries.append(
                (f"{tensor.name}.{index.name}", symmetry[0], symmetry[1])
            )
        distinct_symmetries = {normalized for _, _, normalized in endpoint_symmetries}
        if len(distinct_symmetries) <= 1:
            continue
        issues.append(
            LintIssue(
                severity="warning",
                code="connected-symmetry-conflict",
                message=(
                    f"Hyperedge '{hyperedge.name}' mixes incompatible symmetry metadata: "
                    + ", ".join(
                        f"{endpoint_name}={raw_value}"
                        for endpoint_name, raw_value, _ in endpoint_symmetries
                    )
                    + "."
                ),
                path=f"hyperedges.{hyperedge.id}",
                suggestion="Align the symmetry metadata across all hyperedge endpoints.",
            )
        )
    return issues


def _lint_tensor_index_symmetry(spec: NetworkSpec) -> list[LintIssue]:
    """Warn when a tensor symmetry disagrees with one of its index symmetries."""
    issues: list[LintIssue] = []
    for tensor in spec.tensors:
        tensor_symmetry = _guided_metadata_text(
            tensor.metadata,
            canonical_key="symmetry",
            guided_keys=_TENSOR_GUIDED_METADATA_KEYS,
        )
        if tensor_symmetry is None:
            continue
        for index in tensor.indices:
            index_symmetry = _guided_metadata_text(
                index.metadata,
                canonical_key="symmetry",
                guided_keys=_INDEX_GUIDED_METADATA_KEYS,
            )
            if index_symmetry is None or index_symmetry[1] == tensor_symmetry[1]:
                continue
            issues.append(
                LintIssue(
                    severity="warning",
                    code="tensor-index-symmetry-conflict",
                    message=(
                        f"Tensor '{tensor.name}' uses symmetry '{tensor_symmetry[0]}' but index '{index.name}' is annotated as '{index_symmetry[0]}'."
                    ),
                    path=f"tensors.{tensor.id}.indices.{index.id}.metadata.symmetry",
                    suggestion="Align the tensor and index symmetry metadata, or remove the conflicting annotation.",
                )
            )
    return issues


def _guided_metadata_text(
    metadata: object,
    *,
    canonical_key: str,
    guided_keys: frozenset[str],
) -> tuple[str, str] | None:
    """Return the raw and normalized text value for one guided metadata key."""
    if not isinstance(metadata, dict):
        return None
    for metadata_key, metadata_value in metadata.items():
        if not isinstance(metadata_key, str):
            continue
        normalized_key = _normalize_metadata_key(metadata_key)
        if metadata_key != canonical_key and normalized_key != canonical_key:
            continue
        if isinstance(metadata_value, bool) or not isinstance(metadata_value, str):
            return None
        normalized_value = _normalize_guided_text(metadata_value)
        if not normalized_value:
            return None
        resolved_key, _ = _resolve_guided_metadata_key(
            metadata_key,
            guided_keys=guided_keys,
        )
        if resolved_key != canonical_key:
            continue
        return metadata_value.strip(), normalized_value
    return None


def _resolve_guided_metadata_key(
    metadata_key: str,
    *,
    guided_keys: frozenset[str],
) -> tuple[str | None, bool]:
    """Resolve a metadata key to a canonical guided key when it looks related."""
    if metadata_key in guided_keys:
        return metadata_key, False
    normalized_key = _normalize_metadata_key(metadata_key)
    if normalized_key in guided_keys:
        return normalized_key, True
    close_matches = get_close_matches(
        normalized_key, list(guided_keys), n=1, cutoff=0.75
    )
    if close_matches:
        return close_matches[0], True
    return None, False


def _normalize_metadata_key(value: str) -> str:
    """Normalize a metadata key for conservative guided-key matching."""
    normalized_characters = [
        character.lower() if character.isalnum() else "_" for character in value.strip()
    ]
    normalized_value = "".join(normalized_characters).strip("_")
    while "__" in normalized_value:
        normalized_value = normalized_value.replace("__", "_")
    return normalized_value


def _normalize_guided_text(value: str) -> str:
    """Normalize a free-form guided metadata value for comparisons."""
    return value.strip().lower()


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
