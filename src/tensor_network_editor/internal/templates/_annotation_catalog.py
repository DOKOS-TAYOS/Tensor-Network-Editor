"""Catalog metadata for guided tensor and index annotations in the editor."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ...types import JSONValue


class AnnotationScope(StrEnum):
    """Supported editor scopes for guided metadata annotations."""

    TENSOR = "tensor"
    INDEX = "index"


@dataclass(frozen=True)
class AnnotationDefinition:
    """One guided metadata field exposed by the editor UI."""

    key: str
    label: str
    placeholder: str
    suggestions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, JSONValue]:
        """Serialize the annotation definition for frontend bootstrap payloads."""
        return {
            "key": self.key,
            "label": self.label,
            "placeholder": self.placeholder,
            "suggestions": list(self.suggestions),
        }


ANNOTATION_DEFINITIONS: dict[AnnotationScope, tuple[AnnotationDefinition, ...]] = {
    AnnotationScope.TENSOR: (
        AnnotationDefinition(
            key="role",
            label="Tensor role",
            placeholder="observable",
            suggestions=("state", "operator", "observable", "environment"),
        ),
        AnnotationDefinition(
            key="state",
            label="State",
            placeholder="ground",
            suggestions=("ground", "excited", "thermal", "variational"),
        ),
        AnnotationDefinition(
            key="provenance",
            label="Provenance",
            placeholder="imported",
            suggestions=("imported", "measured", "simulated", "derived"),
        ),
        AnnotationDefinition(
            key="symmetry",
            label="Symmetry",
            placeholder="u1",
            suggestions=("u1", "su2", "zn", "parity"),
        ),
    ),
    AnnotationScope.INDEX: (
        AnnotationDefinition(
            key="leg_kind",
            label="Leg kind",
            placeholder="physical",
            suggestions=("physical", "logical", "bond", "auxiliary"),
        ),
        AnnotationDefinition(
            key="symmetry",
            label="Symmetry",
            placeholder="u1",
            suggestions=("u1", "su2", "zn", "parity"),
        ),
        AnnotationDefinition(
            key="observable",
            label="Observable",
            placeholder="sx",
            suggestions=("sx", "sy", "sz", "number"),
        ),
    ),
}


def serialize_annotation_definitions() -> dict[str, list[dict[str, JSONValue]]]:
    """Serialize all guided annotation definitions for the browser bootstrap."""
    return {
        scope.value: [definition.to_dict() for definition in definitions]
        for scope, definitions in ANNOTATION_DEFINITIONS.items()
    }


def annotation_keys_by_scope() -> dict[AnnotationScope, tuple[str, ...]]:
    """Return the canonical guided metadata keys grouped by supported scope."""
    return {
        scope: tuple(definition.key for definition in definitions)
        for scope, definitions in ANNOTATION_DEFINITIONS.items()
    }
