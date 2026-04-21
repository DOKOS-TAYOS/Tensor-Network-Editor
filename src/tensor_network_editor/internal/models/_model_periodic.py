"""Periodic layout containers for graph models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Self

from ...types import JSONValue, MetadataDict
from ..io._payloads import coerce_int, coerce_metadata, require_dict, require_list
from ._model_contraction import ContractionPlanSpec
from ._model_entities import CanvasNoteSpec, EdgeSpec, GroupSpec, TensorSpec
from ._model_periodic_types import (
    GridPeriodicCellName,
    LinearPeriodicCellName,
    TreePeriodicCellName,
    coerce_grid_periodic_cell_name,
    coerce_linear_periodic_cell_name,
    coerce_tree_periodic_cell_name,
)


@dataclass(slots=True)
class LinearPeriodicCellSpec:
    """One editable cell inside the linear periodic-chain editor mode."""

    tensors: list[TensorSpec] = field(default_factory=list)
    groups: list[GroupSpec] = field(default_factory=list)
    edges: list[EdgeSpec] = field(default_factory=list)
    notes: list[CanvasNoteSpec] = field(default_factory=list)
    contraction_plan: ContractionPlanSpec | None = None
    metadata: MetadataDict = field(default_factory=dict)

    def to_dict(self) -> dict[str, JSONValue]:
        """Serialize the cell to a JSON-compatible mapping."""
        return {
            "tensors": [tensor.to_dict() for tensor in self.tensors],
            "groups": [group.to_dict() for group in self.groups],
            "edges": [edge.to_dict() for edge in self.edges],
            "notes": [note.to_dict() for note in self.notes],
            "contraction_plan": (
                self.contraction_plan.to_dict()
                if self.contraction_plan is not None
                else None
            ),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        """Build a cell from a serialized mapping."""
        tensors_payload = require_list(payload.get("tensors", []), field_name="tensors")
        groups_payload = require_list(payload.get("groups", []), field_name="groups")
        edges_payload = require_list(payload.get("edges", []), field_name="edges")
        notes_payload = require_list(payload.get("notes", []), field_name="notes")
        contraction_plan_payload = payload.get("contraction_plan")
        return cls(
            tensors=[
                TensorSpec.from_dict(require_dict(tensor, field_name="tensor"))
                for tensor in tensors_payload
            ],
            groups=[
                GroupSpec.from_dict(require_dict(group, field_name="group"))
                for group in groups_payload
            ],
            edges=[
                EdgeSpec.from_dict(require_dict(edge, field_name="edge"))
                for edge in edges_payload
            ],
            notes=[
                CanvasNoteSpec.from_dict(require_dict(note, field_name="note"))
                for note in notes_payload
            ],
            contraction_plan=(
                ContractionPlanSpec.from_dict(
                    require_dict(
                        contraction_plan_payload, field_name="contraction_plan"
                    )
                )
                if contraction_plan_payload is not None
                else None
            ),
            metadata=coerce_metadata(
                payload.get("metadata", {}), field_name="metadata"
            ),
        )


@dataclass(slots=True)
class LinearPeriodicChainSpec:
    """Typed payload that stores the three-cell linear periodic mode."""

    active_cell: LinearPeriodicCellName = LinearPeriodicCellName.INITIAL
    initial_cell: LinearPeriodicCellSpec = field(default_factory=LinearPeriodicCellSpec)
    periodic_cell: LinearPeriodicCellSpec = field(
        default_factory=LinearPeriodicCellSpec
    )
    final_cell: LinearPeriodicCellSpec = field(default_factory=LinearPeriodicCellSpec)
    metadata: MetadataDict = field(default_factory=dict)

    def to_dict(self) -> dict[str, JSONValue]:
        """Serialize the linear periodic-chain payload."""
        return {
            "active_cell": self.active_cell.value,
            "initial_cell": self.initial_cell.to_dict(),
            "periodic_cell": self.periodic_cell.to_dict(),
            "final_cell": self.final_cell.to_dict(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        """Build the periodic-chain payload from a serialized mapping."""
        return cls(
            active_cell=coerce_linear_periodic_cell_name(
                payload.get("active_cell", LinearPeriodicCellName.INITIAL.value),
                field_name="active_cell",
            ),
            initial_cell=LinearPeriodicCellSpec.from_dict(
                require_dict(payload.get("initial_cell", {}), field_name="initial_cell")
            ),
            periodic_cell=LinearPeriodicCellSpec.from_dict(
                require_dict(
                    payload.get("periodic_cell", {}), field_name="periodic_cell"
                )
            ),
            final_cell=LinearPeriodicCellSpec.from_dict(
                require_dict(payload.get("final_cell", {}), field_name="final_cell")
            ),
            metadata=coerce_metadata(
                payload.get("metadata", {}), field_name="metadata"
            ),
        )


@dataclass(slots=True)
class GridPeriodicGridSpec:
    """Typed payload that stores the nine-cell bidimensional periodic mode."""

    active_cell: GridPeriodicCellName = GridPeriodicCellName.CENTER
    top_left_cell: LinearPeriodicCellSpec = field(
        default_factory=LinearPeriodicCellSpec
    )
    top_cell: LinearPeriodicCellSpec = field(default_factory=LinearPeriodicCellSpec)
    top_right_cell: LinearPeriodicCellSpec = field(
        default_factory=LinearPeriodicCellSpec
    )
    left_cell: LinearPeriodicCellSpec = field(default_factory=LinearPeriodicCellSpec)
    center_cell: LinearPeriodicCellSpec = field(default_factory=LinearPeriodicCellSpec)
    right_cell: LinearPeriodicCellSpec = field(default_factory=LinearPeriodicCellSpec)
    bottom_left_cell: LinearPeriodicCellSpec = field(
        default_factory=LinearPeriodicCellSpec
    )
    bottom_cell: LinearPeriodicCellSpec = field(default_factory=LinearPeriodicCellSpec)
    bottom_right_cell: LinearPeriodicCellSpec = field(
        default_factory=LinearPeriodicCellSpec
    )
    metadata: MetadataDict = field(default_factory=dict)

    def to_dict(self) -> dict[str, JSONValue]:
        """Serialize the grid periodic payload."""
        return {
            "active_cell": self.active_cell.value,
            "top_left_cell": self.top_left_cell.to_dict(),
            "top_cell": self.top_cell.to_dict(),
            "top_right_cell": self.top_right_cell.to_dict(),
            "left_cell": self.left_cell.to_dict(),
            "center_cell": self.center_cell.to_dict(),
            "right_cell": self.right_cell.to_dict(),
            "bottom_left_cell": self.bottom_left_cell.to_dict(),
            "bottom_cell": self.bottom_cell.to_dict(),
            "bottom_right_cell": self.bottom_right_cell.to_dict(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        """Build the grid periodic payload from a serialized mapping."""
        return cls(
            active_cell=coerce_grid_periodic_cell_name(
                payload.get("active_cell", GridPeriodicCellName.CENTER.value),
                field_name="active_cell",
            ),
            top_left_cell=LinearPeriodicCellSpec.from_dict(
                require_dict(
                    payload.get("top_left_cell", {}), field_name="top_left_cell"
                )
            ),
            top_cell=LinearPeriodicCellSpec.from_dict(
                require_dict(payload.get("top_cell", {}), field_name="top_cell")
            ),
            top_right_cell=LinearPeriodicCellSpec.from_dict(
                require_dict(
                    payload.get("top_right_cell", {}), field_name="top_right_cell"
                )
            ),
            left_cell=LinearPeriodicCellSpec.from_dict(
                require_dict(payload.get("left_cell", {}), field_name="left_cell")
            ),
            center_cell=LinearPeriodicCellSpec.from_dict(
                require_dict(payload.get("center_cell", {}), field_name="center_cell")
            ),
            right_cell=LinearPeriodicCellSpec.from_dict(
                require_dict(payload.get("right_cell", {}), field_name="right_cell")
            ),
            bottom_left_cell=LinearPeriodicCellSpec.from_dict(
                require_dict(
                    payload.get("bottom_left_cell", {}),
                    field_name="bottom_left_cell",
                )
            ),
            bottom_cell=LinearPeriodicCellSpec.from_dict(
                require_dict(payload.get("bottom_cell", {}), field_name="bottom_cell")
            ),
            bottom_right_cell=LinearPeriodicCellSpec.from_dict(
                require_dict(
                    payload.get("bottom_right_cell", {}),
                    field_name="bottom_right_cell",
                )
            ),
            metadata=coerce_metadata(
                payload.get("metadata", {}), field_name="metadata"
            ),
        )


@dataclass(slots=True)
class TreePeriodicTreeSpec:
    """Typed payload that stores the three-cell tree periodic mode."""

    active_cell: TreePeriodicCellName = TreePeriodicCellName.ROOT
    branching_factor: int = 2
    root_cell: LinearPeriodicCellSpec = field(default_factory=LinearPeriodicCellSpec)
    branch_cell: LinearPeriodicCellSpec = field(default_factory=LinearPeriodicCellSpec)
    leaf_cell: LinearPeriodicCellSpec = field(default_factory=LinearPeriodicCellSpec)
    metadata: MetadataDict = field(default_factory=dict)

    def to_dict(self) -> dict[str, JSONValue]:
        """Serialize the tree periodic payload."""
        return {
            "active_cell": self.active_cell.value,
            "branching_factor": self.branching_factor,
            "root_cell": self.root_cell.to_dict(),
            "branch_cell": self.branch_cell.to_dict(),
            "leaf_cell": self.leaf_cell.to_dict(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        """Build the tree periodic payload from a serialized mapping."""
        return cls(
            active_cell=coerce_tree_periodic_cell_name(
                payload.get("active_cell", TreePeriodicCellName.ROOT.value),
                field_name="active_cell",
            ),
            branching_factor=coerce_int(
                payload.get("branching_factor", 2),
                field_name="branching_factor",
            ),
            root_cell=LinearPeriodicCellSpec.from_dict(
                require_dict(payload.get("root_cell", {}), field_name="root_cell")
            ),
            branch_cell=LinearPeriodicCellSpec.from_dict(
                require_dict(payload.get("branch_cell", {}), field_name="branch_cell")
            ),
            leaf_cell=LinearPeriodicCellSpec.from_dict(
                require_dict(payload.get("leaf_cell", {}), field_name="leaf_cell")
            ),
            metadata=coerce_metadata(
                payload.get("metadata", {}), field_name="metadata"
            ),
        )
